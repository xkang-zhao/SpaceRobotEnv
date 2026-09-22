"""Measure exclusive CPU env.step stages on successful Cube grasp episodes."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
from unittest.mock import patch


class StepTimer:
    def __init__(self):
        self.active = False
        self.values = {}

    def wrap(self, method, name):
        def timed(*args, **kwargs):
            if not self.active:
                return method(*args, **kwargs)
            start = perf_counter_ns()
            try:
                return method(*args, **kwargs)
            finally:
                self.values[name] = self.values.get(name, 0) + perf_counter_ns() - start
        return timed

    def bind(self, stack, obj, attr, name):
        stack.enter_context(patch.object(obj, attr, self.wrap(getattr(obj, attr), name)))


def profile_mode(mode, seeds):
    import gymnasium as gym
    import numpy as np
    import envs  # noqa: F401
    from planners.auto_grasp.tasks import AutoGraspSession

    env = gym.make('SpaceUR10e-Cube-v0', observation_mode=mode, physics_backend='mujoco', render_mode=None)
    base = env.unwrapped
    timer = StepTimer()
    rows, episodes = [], []
    cameras = [] if base.sensor is None else [name for _, name in base.sensor.cameras_id]
    try:
        with ExitStack() as stack:
            for attr in ('update_target_pose', 'update_target_gripper', 'get_target_pose'):
                timer.bind(stack, base.controller, attr, 'action')
            timer.bind(stack, base, '_compute_ik_control', 'ik_control')
            timer.bind(stack, base.kinematics, 'ik', 'ik')
            timer.bind(stack, base._physics, 'step', 'physics')
            timer.bind(stack, base, '_get_obs', 'observation')
            timer.bind(stack, base, 'compute_reward', 'reward')
            timer.bind(stack, base, '_check_gripper_target_contacts', 'contacts')
            if base.sensor is not None:
                timer.bind(stack, base.sensor, 'update_camera_view', 'camera')
                timer.bind(stack, base.sensor.renderer, 'update_scene', 'camera_scene')
                timer.bind(stack, base.sensor.renderer, 'render', 'camera_render')
            session = AutoGraspSession('cube', env)
            session.reset(seeds[0])
            # Warm IK, physics and RGB rendering; excluded from all measurements.
            for _ in range(3):
                env.step(np.zeros(7, dtype=np.float32))
            for seed in seeds:
                obs = session.reset(seed)
                first = len(rows)
                for _ in range(400):
                    planner_obs = {**obs, 'target_pose': base._get_target_pose()}
                    action = session.compute_action(planner_obs)
                    timer.values = {}
                    timer.active = True
                    start = perf_counter_ns()
                    try:
                        transition = env.step(action)
                    finally:
                        total = perf_counter_ns() - start
                        timer.active = False
                    v = timer.values
                    major = sum(v.get(key, 0) for key in ('action', 'ik_control', 'physics', 'observation', 'reward'))
                    row = {
                        'total': total,
                        'action': v['action'],
                        'ik': v['ik'],
                        'ik_conversion_control': v['ik_control'] - v['ik'],
                        'physics_100_substeps': v['physics'],
                        'state_observation': v['observation'] - v.get('camera', 0),
                        'camera_scene_update': v.get('camera_scene', 0),
                        'camera_render_readback': v.get('camera_render', 0),
                        'camera_other': v.get('camera', 0) - v.get('camera_scene', 0) - v.get('camera_render', 0),
                        'reward_and_contact': v['reward'],
                        'other_including_termination_and_wrappers': total - major,
                        'contact_detection_subset': v.get('contacts', 0),
                    }
                    rows.append({key: value / 1e6 for key, value in row.items()})
                    session.update(*transition)
                    obs = transition[0]
                    if session.is_done:
                        break
                episodes.append({'seed': seed, 'steps': len(rows) - first, 'success': session.is_success})
                if not session.is_success:
                    raise RuntimeError(f'{mode} seed={seed}: {session.failure_reason or "step limit"}')
        stats = {}
        for key in rows[0]:
            values = np.array([row[key] for row in rows])
            stats[key] = {'mean_ms': float(values.mean()), 'median_ms': float(np.median(values)),
                          'p95_ms': float(np.quantile(values, 0.95))}
        for value in stats.values():
            value['fraction_of_total_percent'] = 100 * value['mean_ms'] / stats['total']['mean_ms']
        return {'mode': mode, 'samples': len(rows), 'cameras': cameras,
                'episodes': episodes, 'stats': stats, 'samples_ms': rows}
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', choices=('state', 'rgb'), default=['state', 'rgb'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 8, 9])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    if any(seed < 0 for seed in args.seeds):
        parser.error('seeds must be nonnegative')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    results = []
    for mode in args.modes:
        print(f'Profiling CPU physics, observation_mode={mode}...', file=sys.stderr, flush=True)
        results.append(profile_mode(mode, args.seeds))
    import pinocchio
    payload = {'platform': platform.platform(), 'mujoco_gl': os.environ['MUJOCO_GL'],
               'versions': {name: importlib.metadata.version(name) for name in ('mujoco', 'gymnasium', 'numpy')},
               'pinocchio': pinocchio.__version__, 'render_mode': None,
               'notes': 'Warmup and reset excluded. Cameras use EGL; CPU physics does not imply CPU rendering. Contact detection is a subset of reward, not an additive stage.',
               'results': results}
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write('\n')
    for result in results:
        print(result['mode'], result['samples'], json.dumps(result['stats'], indent=2))


if __name__ == '__main__':
    main()

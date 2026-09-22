"""Time complete CPU state-only grasp episodes, including planner execution."""
import argparse
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import gymnasium as gym
import envs  # noqa: F401
from planners.auto_grasp.tasks import AUTO_GRASP_TASKS, AutoGraspSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7, 8, 9])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    rows = []
    for name, spec in AUTO_GRASP_TASKS.items():
        start = perf_counter()
        env = gym.make(spec.env_id, physics_backend='mujoco',
                       observation_mode='state', render_mode=None)
        creation = perf_counter() - start
        try:
            for seed in args.seeds:
                session = AutoGraspSession(name, env)
                with contextlib.redirect_stdout(io.StringIO()):
                    start = perf_counter()
                    obs = session.reset(seed)
                    reset = perf_counter() - start
                    simulated_start = float(env.unwrapped.data.time)
                    env_step_seconds = 0.0
                    start = perf_counter()
                    confirmed = False
                    for index in range(session.max_steps):
                        planner_obs = {**obs, 'target_pose': env.unwrapped._get_target_pose()}
                        action = session.compute_action(planner_obs)
                        step_start = perf_counter()
                        transition = env.step(action)
                        env_step_seconds += perf_counter() - step_start
                        session.update(*transition)
                        obs = transition[0]
                        confirmed = bool(transition[2] and transition[4].get('is_success'))
                        if session.is_done or transition[2] or transition[3]:
                            break
                    elapsed = perf_counter() - start
                row = dict(task=name, seed=seed, steps=index+1, confirmed=confirmed,
                           failure=session.failure_reason, creation_s=creation,
                           reset_s=reset, execution_s=elapsed, env_steps_s=env_step_seconds,
                           planner_and_loop_s=elapsed-env_step_seconds,
                           simulated_s=float(env.unwrapped.data.time)-simulated_start)
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            env.close()
    payload = dict(versions={n: importlib.metadata.version(n) for n in ('mujoco','numpy','gymnasium')},
                   mode='CPU physics, state observations, no GUI, no pacing',
                   notes='Execution includes planner, env.step, and loop. Reset includes planner setup. '
                         'Creation occurs once per task, repeated in rows for reference. '
                         'Python startup/imports, teardown, camera, dataset writes excluded. '
                         'Planner logs formatted into memory instead of terminal. No extra warmup.',
                   rows=rows)
    with args.output.open('x') as file:
        json.dump(payload, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    main()

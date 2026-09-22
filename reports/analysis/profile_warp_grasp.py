"""Measure state-only CUDA Cube grasps with the selected IK backend."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from envs.space_ur10e_env import SpaceUR10eEnv
from envs.space_ur10e_warp_vector_env import SpaceUR10eWarpVectorEnv
from planners.auto_grasp.warp_cube import run_vector_cube
from mujoco_robot.hybrid_gpu_ik import HybridGPUCandidates


def measure(n, seed, ik_backend="cpu"):
    times = {'ik_s': 0., 'gpu_ik_s': 0., 'step_s': 0., 'step_calls': 0, 'reset_s': [], 'step_samples_s': []}
    active = False
    original_ik = SpaceUR10eEnv._compute_ik_control
    original_step = SpaceUR10eWarpVectorEnv.step
    original_reset = SpaceUR10eWarpVectorEnv.reset
    original_gpu = HybridGPUCandidates.solve
    def gpu_ik(solver, *args, **kwargs):
        start = perf_counter()
        try:
            return original_gpu(solver, *args, **kwargs)
        finally:
            times['gpu_ik_s'] += perf_counter() - start
    def ik(env, *args, **kwargs):
        start = perf_counter()
        try:
            return original_ik(env, *args, **kwargs)
        finally:
            if active:
                times['ik_s'] += perf_counter() - start
    def step(env, *args, **kwargs):
        nonlocal active
        active = True
        start = perf_counter()
        try:
            return original_step(env, *args, **kwargs)
        finally:
            active = False
            elapsed = perf_counter() - start
            times['step_s'] += elapsed
            times['step_samples_s'].append(elapsed)
            times['step_calls'] += 1
    def reset(env, *args, **kwargs):
        start = perf_counter()
        try:
            return original_reset(env, *args, **kwargs)
        finally:
            times['reset_s'].append(perf_counter() - start)
    start = perf_counter()
    with patch.object(SpaceUR10eEnv, '_compute_ik_control', ik), \
         patch.object(SpaceUR10eWarpVectorEnv, 'step', step), \
         patch.object(SpaceUR10eWarpVectorEnv, 'reset', reset), \
         patch.object(HybridGPUCandidates, 'solve', gpu_ik):
        result = run_vector_cube(n, seed=seed, ik_backend=ik_backend)
    result['wall_s_including_setup_and_close'] = perf_counter() - start
    result['timing'] = times
    result['mean_batch_step_ms'] = 1000 * times['step_s'] / times['step_calls']
    result['mean_cpu_ik_per_batch_ms'] = 1000 * times['ik_s'] / times['step_calls']
    result['mean_gpu_ik_per_batch_ms'] = 1000 * times['gpu_ik_s'] / times['step_calls']
    result['mean_other_step_ms'] = 1000 * (times['step_s']-times['ik_s']-times['gpu_ik_s']) / times['step_calls']
    result['first_attempt_successes'] = sum(r['success'] for r in result['results'])
    result['first_attempts_per_second'] = n / result['seconds_including_cpu_ik_and_partial_resets']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worlds', nargs='+', type=int, default=[1, 4, 16, 32])
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--ik-backend', choices=('cpu','gpu','shadow'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    rows=[]
    for n in args.worlds:
        result = measure(n, args.seed, args.ik_backend)
        rows.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('timing','results')}), flush=True)
    with args.output.open('x') as f:
        json.dump({'notes':'Independent seed=seed+world; selected IK backend, GPU physics. CPU IK timing counts applied CPU solves, including fallback. GPU IK timing includes candidate upload/solve/readback. Includes first-step graph capture and partial resets. Completed worlds run disposable episodes until all first attempts finish. No RGB or GUI. Other step time excludes both measured IK paths and includes GPU physics, transfers, observations, checks, Python overhead and physics graph capture.', 'results':rows},f,indent=2)


if __name__ == '__main__':
    main()

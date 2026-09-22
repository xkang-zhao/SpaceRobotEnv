"""Compare complete scripted grasp attempts on CPU and grouped Warp worlds.

This is a correctness diagnostic, not a throughput benchmark: the GPU teacher
views reconstruct CPU kinematics. No dataset or model checkpoint is written.
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TeacherView:
    """Read-only planner view of one already-reset GPU world."""

    def __init__(self, vector, local):
        import mujoco

        self.vector, self.local = vector, local
        base = vector._template
        self.info = {}
        self.unwrapped = SimpleNamespace(
            model=vector.model,
            data=mujoco.MjData(vector.model),
            controller=vector.controllers[local],
            viewer=None,
            ee_site_id=base.ee_site_id,
            pinch_site_id=base.pinch_site_id,
            target_body_id=base.target_body_id,
            target_dof_adr=base.target_dof_adr,
            is_success=lambda: bool(self.info.get("r_success", 0) > 0),
        )

    def update(self, obs, target, qvel, info):
        import mujoco

        self.obs = {key: np.array(value, copy=True) for key, value in obs.items()}
        self.obs["target_pose"] = target.copy()
        self.info = info
        data = self.unwrapped.data
        data.qpos[:] = self.vector._qpos[self.local]
        data.qvel[:] = qvel
        mujoco.mj_kinematics(self.vector.model, data)

    def reset(self, *, seed=None):
        # Group reset already assigned the seed; do not reset a sibling world.
        return self.obs, {}


def outcome(task, seed, steps, term, trunc, info, session):
    return dict(
        task=task,
        seed=seed,
        steps=steps,
        success=bool(term and info.get("is_success", False)),
        terminated=bool(term),
        truncated=bool(trunc),
        success_counter=int(info.get("success_counter", 0)),
        left_contact=bool(info.get("left_contact", False)),
        right_contact=bool(info.get("right_contact", False)),
        failure_reason=(
            session.failure_reason or "step limit"
            if not (term or trunc or session.is_done)
            else session.failure_reason
        ),
    )


def run(tasks, backend, seed, max_steps, substeps_per_graph=100):
    import gymnasium as gym
    import envs  # noqa: F401
    from planners.auto_grasp.tasks import AutoGraspSession, get_task_spec

    root = Path(__file__).resolve().parents[1]
    results = []
    start = perf_counter()
    if backend == "cpu":
        for index, task in enumerate(tasks):
            env_id = get_task_spec(task).env_id
            env = gym.make(
                env_id,
                observation_mode="state",
                scene_path=str(root / gym.spec(env_id).kwargs["scene_path"]),
                arm_path=str(root / "mjcf/arm.xml"),
            )
            try:
                session = AutoGraspSession(task, env)
                obs = session.reset(seed + index)
                obs["target_pose"] = env.unwrapped._get_target_pose()
                for step in range(1, max_steps + 1):
                    obs, reward, term, trunc, info = env.step(
                        session.compute_action(obs)
                    )
                    obs["target_pose"] = env.unwrapped._get_target_pose()
                    session.update(obs, reward, term, trunc, info)
                    if term or trunc or session.is_done:
                        break
                row = outcome(task, seed + index, step, term, trunc, info, session)
                results.append(row)
                print(json.dumps(row), flush=True)
            finally:
                env.close()
    else:
        from envs.space_ur10e_warp_group import SpaceUR10eWarpGroup

        group = SpaceUR10eWarpGroup(
            [get_task_spec(task).env_id for task in tasks],
            repo_root=str(root),
            ik_backend="gpu",
            physics_substeps_per_graph=substeps_per_graph,
        )
        try:
            obs, _ = group.reset(seed=seed)
            views = [TeacherView(*world) for world in group.worlds]
            sessions = [
                AutoGraspSession(task, view) for task, view in zip(tasks, views)
            ]
            active = np.ones(len(tasks), dtype=bool)
            info = {}

            def sync_views():
                teacher = group.get_planner_state()
                velocities = {id(env): env.d.qvel.numpy() for _, env in group.groups}
                for i in np.flatnonzero(active):
                    env, local = group.worlds[i]
                    views[i].update(
                        {k: v[i] for k, v in obs.items()},
                        teacher["target_pose"][i],
                        velocities[id(env)][local],
                        {k: v[i] for k, v in info.items()},
                    )

            sync_views()
            for index, session in enumerate(sessions):
                session.reset(seed + index)
            for step in range(1, max_steps + 1):
                actions = np.zeros((len(tasks), 7), dtype=np.float32)
                for i in np.flatnonzero(active):
                    actions[i] = sessions[i].compute_action(views[i].obs)
                obs, reward, term, trunc, info = group.step(actions, active_mask=active)
                sync_views()
                for i in np.flatnonzero(active):
                    local_info = {k: v[i] for k, v in info.items()}
                    sessions[i].update(
                        views[i].obs, reward[i], term[i], trunc[i], local_info
                    )
                    if term[i] or trunc[i] or sessions[i].is_done or step == max_steps:
                        row = outcome(
                            tasks[i],
                            seed + int(i),
                            step,
                            term[i],
                            trunc[i],
                            local_info,
                            sessions[i],
                        )
                        results.append(row)
                        active[i] = False
                        print(json.dumps(row), flush=True)
                if not active.any():
                    break
        finally:
            group.close()
    return dict(
        backend=backend,
        substeps_per_graph=substeps_per_graph,
        seconds=perf_counter() - start,
        results=results,
    )


def main():
    from planners.auto_grasp.tasks import TASK_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "warp"], required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES)
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=420)
    parser.add_argument("--substeps-per-graph", type=int, default=100)
    args = parser.parse_args()
    if args.max_steps < 1 or args.seed < 0:
        parser.error("max-steps must be positive and seed nonnegative")
    print(
        json.dumps(
            run(
                args.tasks,
                args.backend,
                args.seed,
                args.max_steps,
                args.substeps_per_graph,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

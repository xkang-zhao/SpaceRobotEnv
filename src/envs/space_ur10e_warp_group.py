"""Group heterogeneous SpaceUR10e worlds into one Warp batch per scene."""

import numpy as np

from .space_ur10e_warp_vector_env import SpaceUR10eWarpVectorEnv


class SpaceUR10eWarpGroup:
    """Preserve caller world order while dispatching each scene's CUDA batch.

    Scene batches execute sequentially on one device. Worlds within each batch
    execute in parallel. Task instructions remain the responsibility of RLinf.
    """

    def __init__(self, env_ids, **kwargs):
        if not env_ids:
            raise ValueError("env_ids must contain at least one SpaceUR10e scene")
        self.num_envs = len(env_ids)
        self.groups = []
        self.worlds = [None] * self.num_envs
        self._obs = None
        try:
            for env_id in dict.fromkeys(env_ids):
                indices = np.array(
                    [i for i, value in enumerate(env_ids) if value == env_id]
                )
                env = SpaceUR10eWarpVectorEnv(len(indices), env_id=env_id, **kwargs)
                self.groups.append((indices, env))
                for local, index in enumerate(indices):
                    self.worlds[index] = (env, local)
            self.single_action_space = self.groups[0][1].single_action_space
        except BaseException:
            self.close()
            raise

    def _merge(self, rows):
        result = {}
        for indices, values in rows:
            for key, value in values.items():
                if key not in result:
                    result[key] = np.zeros(
                        (self.num_envs, *value.shape[1:]), dtype=value.dtype
                    )
                result[key][indices] = value
        return result

    def reset(self, *, seed=None, options=None):
        mask = np.asarray(
            (options or {}).get("reset_mask", np.ones(self.num_envs, dtype=bool))
        )
        if mask.shape != (self.num_envs,) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError("reset_mask must be a nonempty per-world boolean mask")
        if self._obs is None and not mask.all():
            raise RuntimeError("First reset must initialize all scenes")
        seeds = (
            [None] * self.num_envs
            if seed is None
            else (
                [int(seed) + i for i in range(self.num_envs)]
                if isinstance(seed, (int, np.integer))
                else list(seed)
            )
        )
        if len(seeds) != self.num_envs:
            raise ValueError("seed list length must match num_envs")
        rows = []
        for indices, env in self.groups:
            if mask[indices].any():
                obs, _ = env.reset(
                    seed=[seeds[i] for i in indices],
                    options={"reset_mask": mask[indices]},
                )
            else:
                obs = {key: value[indices] for key, value in self._obs.items()}
            rows.append((indices, obs))
        self._obs = self._merge(rows)
        return self._obs, {"reset_mask": mask.copy()}

    def step(self, actions, *, active_mask=None):
        if self._obs is None:
            raise RuntimeError("reset must precede step")
        actions = np.asarray(actions, dtype=np.float32)
        active = (
            np.ones(self.num_envs, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask)
        )
        if (
            actions.shape != (self.num_envs, 7)
            or not np.isfinite(actions).all()
            or np.any(np.abs(actions) > 1)
        ):
            raise ValueError("actions must be finite [num_envs, 7] in [-1, 1]")
        if active.shape != (self.num_envs,) or active.dtype != np.bool_:
            raise ValueError("active_mask must be a per-world boolean mask")
        rewards = np.zeros(self.num_envs)
        terms = np.zeros(self.num_envs, dtype=bool)
        truncs = terms.copy()
        rows, infos = [], []
        for indices, env in self.groups:
            if active[indices].any():
                obs, reward, term, trunc, info = env.step(
                    actions[indices], active_mask=active[indices]
                )
                rewards[indices], terms[indices], truncs[indices] = reward, term, trunc
                infos.append((indices, info))
            else:
                obs = {key: value[indices] for key, value in self._obs.items()}
            rows.append((indices, obs))
        self._obs = self._merge(rows)
        info = self._merge(infos)
        info.setdefault("is_success", np.zeros(self.num_envs, dtype=bool))
        return self._obs, rewards, terms, truncs, info

    def get_planner_state(self):
        return self._merge(
            [(indices, env.get_planner_state()) for indices, env in self.groups]
        )

    def close(self):
        for _, env in self.groups:
            env.close()

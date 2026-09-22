"""Single-scene VectorEnv with CUDA physics and GPU IK / CPU fallback."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from mujoco_robot.robot_controller import RobotController

# Warp compilation options and the MuJoCo Warp function replacements below
# are process-global.  Configure them once and reject an unsafe mode switch in
# the same interpreter (separate training/evaluation processes are unaffected).
_WARP_DETERMINISM_SETTING: bool | None = None


class SpaceUR10eWarpVectorEnv(VectorEnv):
    metadata: ClassVar = {
        "render_modes": [],
        "render_fps": 20,
        "autoreset_mode": AutoresetMode.DISABLED,
    }

    def __init__(self, num_envs: int = 4, *, device: str = "cuda:0",
                 nconmax: int = 128, njmax: int = 512, ik_backend: str = "cpu",
                 repo_root: str | None = None, env_id: str = "SpaceUR10e-Cube-v0",
                 physics_substeps_per_graph: int = 100,
                 deterministic_physics: bool = False):
        global _WARP_DETERMINISM_SETTING
        if (not isinstance(physics_substeps_per_graph, (int, np.integer))
                or isinstance(physics_substeps_per_graph, bool)
                or physics_substeps_per_graph <= 0 or 100 % physics_substeps_per_graph):
            raise ValueError("physics_substeps_per_graph must be a positive divisor of 100")
        self.physics_substeps_per_graph = physics_substeps_per_graph
        if ik_backend not in {"cpu", "gpu", "shadow"}:
            raise ValueError("ik_backend must be cpu, gpu, or shadow")
        self.ik_backend = ik_backend
        self._gpu_ik = None
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0
               for value in (num_envs, nconmax, njmax)):
            raise ValueError("num_envs, nconmax and njmax must be positive integers")
        requested_determinism = bool(deterministic_physics)
        if (_WARP_DETERMINISM_SETTING is not None
                and _WARP_DETERMINISM_SETTING != requested_determinism):
            raise RuntimeError(
                "Warp deterministic_physics is process-global and cannot be "
                "changed after the first SpaceUR10eWarpVectorEnv is created; "
                "start a new Python process to switch modes"
            )
        configure_determinism = _WARP_DETERMINISM_SETTING is None
        import envs  # noqa: F401
        try:
            import warp as wp
            if deterministic_physics and configure_determinism:
                if not hasattr(wp, "DeterministicMode"):
                    raise RuntimeError(
                        "deterministic_physics requires a Warp version with "
                        "DeterministicMode support"
                    )
                # This must be selected before importing MuJoCo Warp so its
                # generated kernels inherit deterministic atomic reductions.
                wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
            import mujoco_warp as mjw
            if deterministic_physics and configure_determinism:
                # MuJoCo Warp's sensor kernel mixes atomic_max and atomic_add
                # on sensordata, which Warp's deterministic transform rejects.
                # SpaceUR10e observations and dynamics do not consume
                # sensordata, so leave only that module on its regular path.
                import mujoco_warp._src.collision_convex as mjw_collision_convex
                import mujoco_warp._src.sensor as mjw_sensor
                import mujoco_warp._src.smooth as mjw_smooth
                wp.set_module_options(
                    {"deterministic": wp.DeterministicMode.NOT_GUARANTEED},
                    module=mjw_sensor,
                )
                # smooth._M writes a distinct packed-matrix row per DOF, but
                # its dynamic indexing is conservatively lowered as a scatter
                # and overflows the inferred deterministic buffer. We replace
                # only that race-free kernel below; all real smooth reductions
                # (including tendon forces) keep deterministic ordering.
                wp.set_module_options(
                    {"deterministic": wp.DeterministicMode.RUN_TO_RUN},
                    module=mjw_smooth,
                )
        except ImportError as exc:
            raise ImportError("SpaceUR10e Warp VectorEnv requires the optional dependencies: pip install -e '.[warp]'") from exc
        from mujoco_robot import warp_state_kernels as kernels
        if deterministic_physics and configure_determinism:
            from mujoco_robot import warp_deterministic_kernels
            # These local kernels either use one thread per output row or
            # integer OR, so they need no deterministic scatter lowering.
            wp.set_module_options(
                {"deterministic": wp.DeterministicMode.NOT_GUARANTEED},
                module=kernels,
            )
            wp.set_module_options(
                {"deterministic": wp.DeterministicMode.NOT_GUARANTEED},
                module=warp_deterministic_kernels,
            )
            warp_deterministic_kernels.install(mjw_smooth)
            warp_deterministic_kernels.install_ccd_record_capacity(
                mjw_collision_convex
            )
        _WARP_DETERMINISM_SETTING = requested_determinism

        self.wp, self.mjw, self.kernels = wp, mjw, kernels
        self.deterministic_physics = deterministic_physics
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError(f"SpaceUR10eWarpVectorEnv requires CUDA, got {device!r}")
        self.num_envs = num_envs
        self.env_id = env_id
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        if not env_id.startswith("SpaceUR10e-"):
            raise ValueError(f"Unsupported SpaceUR10e environment: {env_id}")
        scene_path = Path(gym.spec(env_id).kwargs["scene_path"])
        self._template = gym.make(
            env_id, observation_mode="state",
            scene_path=str(root / scene_path),
            arm_path=str(root / "mjcf/arm.xml"),
        ).unwrapped
        base = self._template
        self.model = base.model
        self._reset_controller = base.controller
        self.controllers = [RobotController(self.model, base.data) for _ in range(num_envs)]
        self.single_action_space = base.action_space
        self.single_observation_space = base.observation_space
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self._rngs = [np.random.default_rng() for _ in range(num_envs)]
        self._needs_reset = np.ones(num_envs, dtype=bool)
        self._initialized = False
        self._graph = None
        self._obs = None
        self._teacher = np.zeros((num_envs, 10), dtype=np.float32)
        self._control = np.zeros((num_envs, self.model.nu), dtype=np.float32)
        self.closed = False
        base.reset(seed=0)
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(self.model)
            self.d = mjw.put_data(self.model, base.data, nworld=num_envs, nconmax=nconmax, njmax=njmax)
            self._mask = wp.ones(num_envs, dtype=bool)
            self._initial_qpos = wp.zeros((num_envs, self.model.nq), dtype=float)
            self._initial_ctrl = wp.zeros((num_envs, self.model.nu), dtype=float)
            self._actions = wp.zeros((num_envs, 7), dtype=float)
            self._bits = wp.zeros(num_envs, dtype=int)
            self._counts = wp.zeros(num_envs, dtype=int)
            self._state = wp.zeros((num_envs, 21), dtype=float)
            self._teacher_state = wp.zeros((num_envs, 10), dtype=float)
            self._metrics = wp.zeros((num_envs, 9), dtype=float)
            self._terminated = wp.zeros(num_envs, dtype=int)
            self._truncated = wp.zeros(num_envs, dtype=int)
            # Paused worlds retain integrator state while other worlds advance.
            self._paused = wp.zeros(num_envs, dtype=bool)
            self._paused_state = {
                name: wp.empty_like(getattr(self.d, name))
                for name in ("qpos", "qvel", "act", "time", "qacc_warmstart")
            }
            self._paused_counts = wp.empty_like(self._counts)
            self._paused_outputs = {
                name: wp.empty_like(getattr(self, name))
                for name in ("_state", "_teacher_state", "_metrics", "_terminated", "_truncated")
            }
            roles = np.zeros(self.model.ngeom, dtype=np.int32)
            roles[list(base.finger_A_pad_ids)] |= 1
            roles[list(base.finger_B_pad_ids)] |= 2
            roles[list(base.target_geom_ids)] |= 4
            self._roles = wp.array(roles, dtype=int)
            self._ids = wp.array([base.ee_site_id, base.pinch_site_id, base.target_body_id,
                                  base.gripper_qpos_adr, base.target_dof_adr,
                                  base.success_consec_threshold], dtype=int)
            weights = base.reward_weights
            self._params = wp.array([*base.gripper_joint_range, base.grasp_gripper_threshold,
                                     base.target_stable_vel_threshold, base.fail_dist_threshold,
                                     base.align_dist_threshold, weights["reach"], weights["align"],
                                     weights["contact_single"], weights["contact_dual"],
                                     weights["success_bonus"], weights["action_penalty"]], dtype=float)
            mjw.step(self.m, self.d)  # Disposable warmup; public reset restores all worlds.
            self._observe(update=False)
            wp.synchronize_device(self.device)

        if ik_backend != "cpu":
            from mujoco_robot.hybrid_gpu_ik import HybridGPUCandidates
            self._gpu_ik = HybridGPUCandidates(base.kinematics, num_envs, self.device)

    def _observe(self, *, update: bool):
        wp, d = self.wp, self.d
        self._bits.zero_()
        wp.launch(self.kernels.mark_contacts, dim=d.naconmax,
                  inputs=[d.nacon, d.contact.worldid, d.contact.geom, self._roles, self._bits])
        wp.launch(self.kernels.pack_state, dim=self.num_envs,
                  inputs=[d.qpos, d.qvel, d.ctrl, d.site_xpos, d.site_xmat, d.xpos, d.xquat,
                          self._actions, self._bits, self._counts, self._ids, self._params,
                          int(update), self._state, self._teacher_state, self._metrics,
                          self._terminated, self._truncated])

    def _check_open(self):
        if self.closed:
            raise RuntimeError("SpaceUR10eWarpVectorEnv is closed")

    def _check_state(self):
        overflow = self.d.overflow.numpy()
        if np.any(overflow):
            raise RuntimeError(f"{self.env_id} Warp capacity overflow in worlds {np.flatnonzero(overflow).tolist()}")
        self._qpos = self.d.qpos.numpy()
        if not np.isfinite(self._qpos).all() or not np.isfinite(self.d.qvel.numpy()).all():
            raise RuntimeError(f"{self.env_id} Warp produced non-finite state")

    def _read_observation(self, mask=None):
        state = self._state.numpy()
        teacher = self._teacher_state.numpy()
        obs = {"joint_pos": state[:, :6].astype(np.float64),
               "base_pose": state[:, 6:13].astype(np.float64),
               "ee_pose": state[:, 13:20].astype(np.float64),
               "gripper_pos": state[:, 20:21].copy()}
        if mask is None or self._obs is None:
            self._obs = obs
            self._teacher = teacher
        else:
            for key in obs:
                self._obs[key][mask] = obs[key][mask]
            self._teacher[mask] = teacher[mask]
        return {key: value.copy() for key, value in self._obs.items()}

    def reset(self, *, seed=None, options=None):
        self._check_open()
        options = options or {}
        mask = np.asarray(options.get("reset_mask", np.ones(self.num_envs, dtype=bool)))
        if mask.shape != (self.num_envs,) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError(f"reset_mask must be a nonempty bool mask of shape ({self.num_envs},)")
        if not self._initialized and not mask.all():
            raise RuntimeError("First reset must initialize all worlds")
        if seed is None:
            seeds = [None] * self.num_envs
        elif isinstance(seed, (int, np.integer)):
            seeds = [int(seed) + w for w in range(self.num_envs)]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError(f"seed list must contain {self.num_envs} entries")
        for s in seeds:
            if s is not None and (not isinstance(s, (int, np.integer)) or s < 0):
                raise ValueError("seeds must be nonnegative integers or None")
        qpos = np.zeros((self.num_envs, self.model.nq), dtype=np.float32)
        base = self._template
        base.controller = self._reset_controller
        for w in np.flatnonzero(mask):
            if seeds[w] is not None:
                self._rngs[w] = np.random.default_rng(int(seeds[w]))
            base._np_random = self._rngs[w]
            base.reset()
            qpos[w] = base.data.qpos
            self._control[w] = base.data.ctrl
            self.controllers[w].target_pose = base.controller.target_pose.copy()
            self.controllers[w].target_gripper = 0.0
        with self.wp.ScopedDevice(self.device):
            self._mask.assign(mask)
            self.mjw.reset_data(self.m, self.d, reset=self._mask)
            self._initial_qpos.assign(qpos)
            self._initial_ctrl.assign(self._control)
            for source, dest in ((self._initial_qpos, self.d.qpos), (self._initial_ctrl, self.d.ctrl)):
                self.wp.launch(self.kernels.masked_rows, dim=dest.shape, inputs=[self._mask, source, dest])
            self.wp.launch(self.kernels.clear_counts, dim=self.num_envs, inputs=[self._mask, self._counts])
            self.mjw.forward(self.m, self.d)
            self._observe(update=False)
            self._check_state()
            obs = self._read_observation(mask)
        self._initialized = True
        self._needs_reset[mask] = False
        return obs, {"reset_mask": mask.copy()}

    def step(self, actions, *, active_mask=None):
        """Advance selected worlds; the default still requires all worlds ready.

        An explicit mask lets RL collectors keep terminal worlds frozen until
        the end of an action chunk. It must select at least one ready world.
        """
        self._check_open()
        active = np.ones(self.num_envs, dtype=bool) if active_mask is None else np.asarray(active_mask)
        if active.shape != (self.num_envs,) or active.dtype != np.bool_ or not active.any():
            raise ValueError("active_mask must select at least one world and have bool dtype")
        if not self._initialized or (self._needs_reset & active).any():
            raise RuntimeError(f"Reset required before step for worlds {np.flatnonzero(self._needs_reset).tolist()}")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 7) or not np.isfinite(actions).all():
            raise ValueError(f"actions must be finite with shape ({self.num_envs}, 7)")
        if np.any(np.abs(actions) > 1):
            raise ValueError("actions must be in [-1, 1]")
        base = self._template
        targets = []
        for w, controller in enumerate(self.controllers):
            try:
                base.data.qpos[:] = self._qpos[w]
                base.controller = controller
                if active[w]:
                    controller.update_target_pose(actions[w])
                    controller.update_target_gripper(actions[w])
                target = controller.get_target_pose()
                if self._gpu_ik is None:
                    if active[w]:
                        base._compute_ik_control(target)
                        self._control[w] = base.data.ctrl
                else:
                    rotation = base.kinematics.euler_to_rotation_matrix(*target[3:6])
                    targets.append(base.kinematics.create_target_pose(rotation, np.array(target[:3])))
            except Exception as exc:
                self._needs_reset[:] = True
                raise RuntimeError(f"{self.env_id} Warp IK failed for world {w}; reset all worlds before retrying") from exc
        ik_info = {}
        if self._gpu_ik is not None:
            try:
                q = self._qpos[:, :13].astype(np.float64)
                q[:, 3:7] = self._qpos[:, [4, 5, 6, 3]]
                candidate, accepted, residual = self._gpu_ik.solve(q, targets)
                differences = np.full(self.num_envs, np.nan)
                for w, controller in enumerate(self.controllers):
                    if not active[w]:
                        continue
                    base.data.qpos[:] = self._qpos[w]
                    base.controller = controller
                    if accepted[w] and self.ik_backend == "gpu":
                        controller.update_control(candidate[w, 7:])
                    else:
                        base._compute_ik_control(controller.get_target_pose())
                        if self.ik_backend == "shadow":
                            # RobotController maps the six arm joints to ctrl[:6].
                            differences[w] = np.max(np.abs(base.data.ctrl[:6] - candidate[w, 7:]))
                    self._control[w] = base.data.ctrl
                ik_info = {
                    "ik_gpu_accepted": accepted & active,
                    "ik_cpu_fallback": (
                        (~accepted & active)
                        if self.ik_backend == "gpu"
                        else np.zeros(self.num_envs, dtype=bool)
                    ),
                    "ik_residual": residual,
                }
                if self.ik_backend == "shadow":
                    ik_info["ik_shadow_joint_max_abs"] = differences
            except Exception as exc:
                self._needs_reset[:] = True
                raise RuntimeError(f"{self.env_id} Warp batch IK failed; reset all worlds before retrying") from exc
        with self.wp.ScopedDevice(self.device):
            paused = not active.all()
            if paused:
                self._paused.assign(~active)
                for name, saved in self._paused_state.items():
                    self.wp.copy(saved, getattr(self.d, name))
                self.wp.copy(self._paused_counts, self._counts)
                for name, saved in self._paused_outputs.items():
                    self.wp.copy(saved, getattr(self, name))
            self.d.ctrl.assign(self._control)
            self._actions.assign(actions)
            if self.deterministic_physics:
                # Warp's deterministic scatter helpers are bit-exact when
                # launched directly, but their temporary buffers are not
                # reliably reset across CUDA Graph replays. Keep the regular
                # high-throughput path below for training and use direct
                # launches only for explicit reproducibility runs.
                for _ in range(100):
                    self.mjw.step(self.m, self.d)
            else:
                if self._graph is None:
                    # Reuse a smaller graph to limit capture-time scratch memory.
                    # Always run exactly 100 physical substeps per control action.
                    with self.wp.ScopedCapture() as capture:
                        for _ in range(self.physics_substeps_per_graph):
                            self.mjw.step(self.m, self.d)
                    self._graph = capture.graph
                for _ in range(100 // self.physics_substeps_per_graph):
                    self.wp.capture_launch(self._graph)
            # Consecutive success counts advance once per CONTROL step, not
            # once per graph replay (which would confirm a grasp too early).
            self._observe(update=True)
            if paused:
                for name, saved in self._paused_state.items():
                    dest = getattr(self.d, name)
                    if dest.size:
                        kernel = self.kernels.masked_rows if dest.ndim == 2 else self.kernels.masked_scalar
                        self.wp.launch(kernel, dim=dest.shape, inputs=[self._paused, saved, dest])
                self.wp.launch(self.kernels.masked_counts, dim=self.num_envs,
                               inputs=[self._paused, self._paused_counts, self._counts])
                # Restore only paused outputs. A full forward here also
                # recomputes active worlds after integration, changing their
                # observation phase depending on whether a neighbor paused.
                # mjw.step refreshes derived physics fields on the next step.
                for name, saved in self._paused_outputs.items():
                    dest = getattr(self, name)
                    kernel = self.kernels.masked_rows if dest.ndim == 2 else self.kernels.masked_counts
                    self.wp.launch(kernel, dim=dest.shape, inputs=[self._paused, saved, dest])
            self._check_state()
            obs = self._read_observation(active if paused else None)
            terminated = self._terminated.numpy().astype(bool)
            truncated = self._truncated.numpy().astype(bool)
            metrics = self._metrics.numpy()
            info = {key: metrics[:, j].astype(np.float64) for j, key in enumerate(
                ("distance", "r_reach", "r_align", "r_contact", "r_success", "r_action", "shaped_reward"))}
            info.update(is_success=terminated.copy(), success_counter=self._counts.numpy(),
                        left_contact=metrics[:, 7].astype(bool), right_contact=metrics[:, 8].astype(bool))
            info.update(ik_info)
        self._needs_reset[active] = (terminated | truncated)[active]
        return obs, (terminated & active).astype(np.float64), terminated, truncated, info

    def get_planner_state(self):
        """Privileged target/pinch positions for scripted teachers, outside observation_space."""
        self._check_open()
        if not self._initialized:
            raise RuntimeError("reset() must precede get_planner_state()")
        return {"target_pose": self._teacher[:, :7].copy(), "pinch_pos": self._teacher[:, 7:].copy()}

    def close_extras(self, **kwargs):
        if hasattr(self, "device"):
            self.wp.synchronize_device(self.device)
        self._graph = None
        self._gpu_ik = None
        self._paused_state = {}
        self.m = self.d = None
        if hasattr(self, "_template"):
            self._template.close()
        # Release observation, contact and reset buffers as well as physics.
        if hasattr(self, "wp"):
            for key, value in list(vars(self).items()):
                if isinstance(value, self.wp.array):
                    setattr(self, key, None)

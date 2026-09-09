"""Single-world physics backends with a CPU MjData compatibility view."""

from __future__ import annotations

import mujoco
import numpy as np


class MujocoBackend:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def reset(self) -> None:
        """The environment has already reset and forwarded the host state."""

    def step(self, steps: int) -> None:
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def close(self) -> None:
        pass


class WarpBackend:
    """Keep physics on CUDA between control steps; read back for CPU consumers.

    Only ctrl is uploaded during step. External edits to host qpos/qvel/model
    after reset are not supported. Reset replaces device state and invalidates
    captured graphs so previous episode state cannot leak into the next one.
    """

    def __init__(self, model, data, *, device: str, scene_path: str,
                 nconmax: int = 128, njmax: int = 512):
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as exc:
            raise ImportError(
                f"Warp backend for {scene_path}: install the optional dependency "
                "with `pip install -e '.[warp]'` (Python >= 3.10)."
            ) from exc
        self.wp, self.mjw = wp, mjw
        self.model, self.data = model, data
        self.scene_path = scene_path
        if nconmax <= 0 or njmax <= 0:
            raise ValueError(f"Warp backend for {scene_path}: capacities must be positive")
        self._capacities = {"nconmax": nconmax, "njmax": njmax}
        try:
            self.device = wp.get_device(device)
        except Exception as exc:
            raise RuntimeError(f"Warp backend for {scene_path}: CUDA device {device!r} unavailable") from exc
        if not self.device.is_cuda:
            raise ValueError(f"Warp backend for {scene_path} requires a CUDA device, got {device!r}")
        with wp.ScopedDevice(self.device):
            try:
                self.warp_model = mjw.put_model(model)
            except Exception as exc:
                raise RuntimeError(f"Cannot load scene {scene_path!r} into Warp: {exc}") from exc
        self.warp_data = None
        self._graphs = {}
        self._warmed = False

    def reset(self) -> None:
        self._graphs.clear()
        with self.wp.ScopedDevice(self.device):
            self.warp_data = self.mjw.put_data(self.model, self.data, nworld=1, **self._capacities)
            if not self._warmed:
                # Compile kernels on disposable state, outside graph capture.
                self.mjw.step(self.warp_model, self.warp_data)
                self.wp.synchronize_device(self.device)
                self._warmed = True
                self.warp_data = self.mjw.put_data(self.model, self.data, nworld=1, **self._capacities)

    def prepare_steps(self, steps: int) -> None:
        """Capture a substep group without advancing simulation time."""
        if self.warp_data is None:
            raise RuntimeError(f"Warp backend for {self.scene_path}: reset() must precede step().")
        with self.wp.ScopedDevice(self.device):
            if steps not in self._graphs:
                with self.wp.ScopedCapture() as capture:
                    for _ in range(steps):
                        self.mjw.step(self.warp_model, self.warp_data)
                self._graphs[steps] = capture.graph

    def step(self, steps: int) -> None:
        self.prepare_steps(steps)
        with self.wp.ScopedDevice(self.device):
            self.warp_data.ctrl.assign(np.asarray(self.data.ctrl[None, :], dtype=np.float32))
            self.wp.capture_launch(self._graphs[steps])
            # Overflow bits accumulate across substeps. Never report success
            # from a simulation that silently discarded contacts/constraints.
            overflow = int(self.warp_data.overflow.numpy()[0])
            if overflow:
                raise RuntimeError(
                    f"Warp backend for {self.scene_path}: capacity overflow {overflow:#x}; "
                    "check Warp diagnostics and increase warp_nconmax/warp_njmax as appropriate."
                )
            self.mjw.get_data_into(self.data, self.model, self.warp_data, world_id=0)

    def close(self) -> None:
        self.wp.synchronize_device(self.device)
        self._graphs.clear()
        self.warp_data = None
        self.warp_model = None


def make_physics_backend(name, model, data, *, device, scene_path, nconmax=128, njmax=512):
    if name == "mujoco":
        return MujocoBackend(model, data)
    if name == "warp":
        return WarpBackend(model, data, device=device, scene_path=scene_path,
                           nconmax=nconmax, njmax=njmax)
    raise ValueError(f"Unknown physics_backend {name!r}; choose 'mujoco' or 'warp'.")

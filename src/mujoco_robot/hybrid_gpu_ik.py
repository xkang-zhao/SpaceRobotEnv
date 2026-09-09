"""GPU candidate generation with explicit per-world CPU fallback metadata."""
import numpy as np

from .gpu_ik_prototype import GPUChainIK


class HybridGPUCandidates:
    def __init__(self, kin, nworld, device):
        self.gpu = GPUChainIK(kin, nworld, device)
        self._graph = None

    def solve(self, q, targets):
        """Return candidates only; the caller runs CPU IK for rejected rows.

        MuJoCo Warp stores FP32 quaternions. Normalize their FP64 copy for the
        prototype; preserve the original simulator configuration for fallback.
        Restrict candidates to the same small-target domain as CPU fast IK.
        """
        import warp as wp
        normalized = np.array(q, dtype=np.float64, copy=True)
        norms = np.linalg.norm(normalized[:, 3:7], axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError('GPU IK received invalid base quaternions')
        normalized[:, 3:7] /= norms[:, None]
        self.gpu.upload(normalized, targets)
        if self._graph is None:
            self._graph = self.gpu.prepare()
            self.gpu.upload(normalized, targets)
        self.gpu.evaluate()
        initial_error = self.gpu.norms.numpy()
        wp.capture_launch(self._graph)
        candidate, residual, accepted = self.gpu.read()
        accepted &= initial_error <= 0.02
        return candidate, accepted, residual

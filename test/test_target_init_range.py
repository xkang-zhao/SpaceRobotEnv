import sys
import unittest
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from envs.space_ur10e_env import SpaceUR10eEnv


class FakeModel:
    jnt_qposadr = {0: 0}


class FakeData:
    def __init__(self):
        self.qpos = np.zeros(7)


class TargetInitRangeTests(unittest.TestCase):
    def test_init_target_uses_configured_xyz_range(self):
        env = SpaceUR10eEnv.__new__(SpaceUR10eEnv)
        env.model = FakeModel()
        env.data = FakeData()
        env.target_joint_id = 0
        env.np_random = np.random.default_rng(0)
        env.target_init_range = {
            "x": (1.23, 1.23),
            "y": (-0.04, -0.04),
            "z": (2.61, 2.61),
        }

        env._init_target()

        np.testing.assert_allclose(env.data.qpos[:3], [1.23, -0.04, 2.61])
        np.testing.assert_allclose(env.data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()

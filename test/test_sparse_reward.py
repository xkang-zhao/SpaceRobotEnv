import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from envs.space_ur10e_env import SpaceUR10eEnv


class SparseRewardTests(unittest.TestCase):
    def test_only_confirmed_grasp_receives_reward(self):
        self.assertEqual(SpaceUR10eEnv._sparse_reward(False), 0.0)
        self.assertEqual(SpaceUR10eEnv._sparse_reward(True), 1.0)


if __name__ == "__main__":
    unittest.main()

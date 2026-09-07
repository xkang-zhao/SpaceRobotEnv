import sys
import unittest
from pathlib import Path

import gymnasium as gym


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class EnvsRegistrationTests(unittest.TestCase):
    def test_envs_package_registers_spaceur10e_specs(self):
        import envs  # noqa: F401

        expected_entry_point = "envs.space_ur10e_env:SpaceUR10eEnv"
        env_ids = (
            "SpaceUR10e-Cube-v0",
            "SpaceUR10e-Satellite-v0",
            "SpaceUR10e-Satellite2-v0",
            "SpaceUR10e-Satellite3-v0",
            "SpaceUR10e-DebrisAntennaPanel-v0",
            "SpaceUR10e-DebrisTruss-v0",
        )
        for env_id in env_ids:
            with self.subTest(env_id=env_id):
                self.assertEqual(gym.spec(env_id).entry_point, expected_entry_point)

        self.assertEqual(gym.spec("SpaceUR10e-DebrisAntennaPanel-v0").kwargs["scene_path"], "./mjcf/5_debris_antenna_panel_scene.xml")
        self.assertEqual(gym.spec("SpaceUR10e-DebrisTruss-v0").kwargs["scene_path"], "./mjcf/6_debris_truss_scene.xml")
        self.assertEqual(
            gym.spec("SpaceUR10e-Cube-v0").kwargs["target_init_range"]["x"],
            (1.5, 1.7),
        )
        self.assertEqual(
            gym.spec("SpaceUR10e-Satellite-v0").kwargs["target_init_range"]["x"],
            (1.8, 1.95),
        )

    def test_registered_environments_use_20_hz_render_metadata(self):
        from envs.space_ur10e_env import SpaceUR10eEnv

        self.assertEqual(SpaceUR10eEnv.metadata["render_fps"], 20)


if __name__ == "__main__":
    unittest.main()

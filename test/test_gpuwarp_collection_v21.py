"""Optional end-to-end test for CUDA Warp LeRobot v2.1 Cube collection."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(
    os.environ.get("RUN_WARP_TESTS") == "1"
    and importlib.util.find_spec("av") is not None,
    "requires CUDA Warp and PyAV; set RUN_WARP_TESTS=1",
)
class GPUWarpCollectionTests(unittest.TestCase):
    def test_two_worlds_publish_valid_cube_dataset(self) -> None:
        from data_collection.autograb_v21 import CollectionConfig
        from data_collection.gpuwarp_autograb_v21 import collect_gpuwarp_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "cube_dataset"
            summary = collect_gpuwarp_dataset(
                CollectionConfig(
                    task_name="cube",
                    output_dir=output,
                    episodes=2,
                    num_envs=2,
                    instruction="Grab the red cube",
                    seed=7,
                    max_attempts_per_episode=2,
                    camera_map={"cam1": "third_left_camera"},
                ),
                ik_backend="gpu",
            )
            self.assertEqual(summary["saved_episodes"], 2)
            self.assertEqual(summary["successful_episodes"], 2)
            self.assertGreater(summary["total_frames"], 0)
            self.assertGreater(summary["gpu_accepted_world_steps"], 0)
            self.assertEqual(summary["cpu_fallback_world_steps"], 0)

            with (output / "meta/info.json").open(encoding="utf-8") as file:
                info = json.load(file)
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(
                sorted(path.name for path in (output / "data/chunk-000").glob("*.parquet")),
                ["episode_000000.parquet", "episode_000001.parquet"],
            )
            self.assertEqual(
                sorted(path.name for path in (output / "videos/chunk-000/observation.images.cam1").glob("*.mp4")),
                ["episode_000000.mp4", "episode_000001.mp4"],
            )


if __name__ == "__main__":
    unittest.main()

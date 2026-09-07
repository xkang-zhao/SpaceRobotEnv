import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data_collection.autograb_v21 import (
    CollectionConfig,
    build_camera_map,
    collect_dataset,
    collect_episode_results,
    collect_target_episode,
    validate_collection_config,
)
from data_collection.lerobot_v21 import (
    LeRobotV21EpisodeBuffer,
    commit_staging_dir,
    finalize_dataset,
    prepare_staging_dir,
)
from planners.auto_grasp import AUTO_GRASP_TASKS, TASK_NAMES


def make_obs(image_value=0):
    image = np.full((4, 6, 3), image_value, dtype=np.uint8)
    return {
        "joint_pos": np.arange(6, dtype=np.float32),
        "ee_pose": np.arange(7, dtype=np.float32) * 0.1,
        "camera": image,
    }


def make_result(
    staging: Path,
    episode_index: int,
    *,
    frames: int = 3,
):
    buffer = LeRobotV21EpisodeBuffer(
        staging
        / ".attempts"
        / f"episode_{episode_index:06d}",
        camera_map={"cam1": "camera"},
        image_size=(6, 4),
        fps=20,
    )
    for frame_index in range(frames):
        buffer.append_pre_step(
            make_obs(20 + episode_index + frame_index),
            np.arange(7, dtype=np.float32) + frame_index,
        )
        buffer.set_last_success(frame_index == frames - 1)
    return {
        "episode_index": episode_index,
        "seed": episode_index,
        "attempts": 1,
        "episode": buffer.to_episode(),
        "is_success": True,
        "failure_reason": None,
        "worker_pid": 100 + episode_index,
    }


class TaskRegistryTests(unittest.TestCase):
    def test_all_autograb_tasks_have_one_shared_registry(self):
        self.assertEqual(
            set(TASK_NAMES),
            {
                "cube",
                "satellite_handle",
                "satellite_left_antenna_panel",
                "satellite2_left_truss_connection",
                "satellite3_left_antenna_panel",
                "satellite3_upper_rod",
                "debris_antenna_panel",
                "debris_truss",
            },
        )
        self.assertEqual(
            AUTO_GRASP_TASKS["cube"].env_id,
            "SpaceUR10e-Cube-v0",
        )
        self.assertEqual(
            AUTO_GRASP_TASKS["debris_truss"].env_id,
            "SpaceUR10e-DebrisTruss-v0",
        )


class CollectionConfigTests(unittest.TestCase):
    def make_config(self, root: Path, **overrides):
        values = {
            "task_name": "cube",
            "output_dir": root / "dataset",
            "episodes": 2,
            "num_envs": 1,
            "instruction": "Grab the cube",
            "seed": 0,
            "max_attempts_per_episode": 3,
            "camera_map": {"cam1": "camera"},
        }
        values.update(overrides)
        return CollectionConfig(**values)

    def test_camera_mapping_defaults_and_validation(self):
        self.assertEqual(
            build_camera_map(None),
            {
                "cam1": "third_left_camera",
                "cam2": "third_right_camera",
                "cam3": "left_wrist_camera",
            },
        )
        self.assertEqual(
            build_camera_map(
                [
                    "left=third_left_camera",
                    "wrist=left_wrist_camera",
                ]
            ),
            {
                "left": "third_left_camera",
                "wrist": "left_wrist_camera",
            },
        )
        with self.assertRaisesRegex(ValueError, "Duplicate camera name"):
            build_camera_map(["cam=one", "cam=two"])
        with self.assertRaisesRegex(
            ValueError,
            "Duplicate camera observation key",
        ):
            build_camera_map(["one=camera", "two=camera"])

    def test_render_and_rerun_reject_multiple_workers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(
                ValueError,
                "--render requires",
            ):
                validate_collection_config(
                    self.make_config(
                        root,
                        num_envs=2,
                        render=True,
                    )
                )
            with self.assertRaisesRegex(
                ValueError,
                "--enable-rerun requires",
            ):
                validate_collection_config(
                    self.make_config(
                        root,
                        num_envs=2,
                        enable_rerun=True,
                    )
                )


class LeRobotV21WriterTests(unittest.TestCase):
    def test_v21_layout_schema_video_and_ordering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = Path(tmpdir) / "dataset"
            staging.mkdir()
            results = [
                make_result(staging, 1),
                make_result(staging, 0),
            ]

            summary = finalize_dataset(
                staging,
                results=results,
                expected_episodes=2,
                instruction="Grab the cube",
                camera_map={"cam1": "camera"},
                fps=20,
                image_size=(6, 4),
            )

            self.assertEqual(summary["saved_episodes"], 2)
            self.assertEqual(summary["total_frames"], 6)
            first = pq.read_table(
                staging
                / "data/chunk-000/episode_000000.parquet"
            )
            second = pq.read_table(
                staging
                / "data/chunk-000/episode_000001.parquet"
            )
            self.assertEqual(
                first.column_names,
                [
                    "observation.state",
                    "action",
                    "is_success",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ],
            )
            self.assertTrue(
                pa.types.is_fixed_size_list(
                    first.schema.field("observation.state").type
                )
            )
            self.assertEqual(
                first.schema.field("observation.state").type.list_size,
                13,
            )
            self.assertEqual(
                first.schema.field("action").type.list_size,
                7,
            )
            np.testing.assert_allclose(
                first.column("timestamp").to_pylist(),
                [0.0, 0.05, 0.1],
            )
            self.assertEqual(
                first.column("is_success").to_pylist(),
                [False, False, True],
            )
            self.assertEqual(
                first.column("index").to_pylist(),
                [0, 1, 2],
            )
            self.assertEqual(
                second.column("index").to_pylist(),
                [3, 4, 5],
            )

            with open(
                staging / "meta/info.json",
                encoding="utf-8",
            ) as file:
                info = json.load(file)
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["fps"], 20)
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_videos"], 2)
            self.assertEqual(
                info["features"]["observation.state"]["shape"],
                [13],
            )
            self.assertEqual(
                info["features"]["observation.images.cam1"]["shape"],
                [4, 6, 3],
            )
            self.assertTrue(
                (
                    staging
                    / (
                        "videos/chunk-000/"
                        "observation.images.cam1/"
                        "episode_000000.mp4"
                    )
                ).exists()
            )
            self.assertTrue((staging / "meta/stats.json").exists())

    def test_transactional_output_replaces_only_after_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "dataset"
            target.mkdir()
            old_file = target / "old.txt"
            old_file.write_text("old", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_staging_dir(target, overwrite=False)
            self.assertEqual(
                old_file.read_text(encoding="utf-8"),
                "old",
            )

            staging = prepare_staging_dir(target, overwrite=True)
            (staging / "new.txt").write_text("new", encoding="utf-8")
            commit_staging_dir(staging, target, overwrite=True)
            self.assertFalse(old_file.exists())
            self.assertEqual(
                (target / "new.txt").read_text(encoding="utf-8"),
                "new",
            )

    def test_output_is_accepted_by_official_v30_converter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "hf_datasets_cache"
            dataset = root / "autograb"
            dataset.mkdir()
            result = make_result(dataset, 0)
            finalize_dataset(
                dataset,
                results=[result],
                expected_episodes=1,
                instruction="Grab the cube",
                camera_map={"cam1": "camera"},
                fps=20,
                image_size=(6, 4),
            )
            original = root / "original_v21"
            shutil.copytree(dataset, original)

            environment = os.environ.copy()
            environment.update(
                {
                    "HF_DATASETS_CACHE": str(cache_dir),
                    "KMP_DUPLICATE_LIB_OK": "TRUE",
                }
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; "
                        "import datasets; "
                        "from lerobot.datasets.v30."
                        "convert_dataset_v21_to_v30 "
                        "import convert_dataset; "
                        f"root=Path({str(root)!r}); "
                        "datasets.config.HF_DATASETS_CACHE="
                        f"{str(cache_dir)!r}; "
                        "convert_dataset("
                        "repo_id='autograb', root=root, "
                        "push_to_hub=False)"
                    ),
                ],
                check=True,
                env=environment,
            )

            with open(
                dataset / "meta/info.json",
                encoding="utf-8",
            ) as file:
                converted_info = json.load(file)
            with open(
                original / "meta/info.json",
                encoding="utf-8",
            ) as file:
                original_info = json.load(file)
            self.assertEqual(converted_info["codebase_version"], "v3.0")
            self.assertEqual(original_info["codebase_version"], "v2.1")


class CollectionLoopTests(unittest.TestCase):
    def make_config(self, root: Path, **overrides):
        values = {
            "task_name": "cube",
            "output_dir": root / "dataset",
            "episodes": 1,
            "num_envs": 1,
            "instruction": "Grab the cube",
            "seed": 10,
            "max_attempts_per_episode": 3,
            "camera_map": {"cam1": "camera"},
            "image_size": (6, 4),
        }
        values.update(overrides)
        return CollectionConfig(**values)

    def test_one_frame_causes_exactly_one_environment_step(self):
        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                self.is_done = False
                self.is_success = False
                self.failure_reason = None
                self.step_index = 0
                self.max_steps = 2

            def reset(self, seed):
                self.seed = seed
                return make_obs(seed)

            def compute_action(self, _obs):
                return np.zeros(7, dtype=np.float32)

            def update(
                self,
                _obs,
                _reward,
                terminated,
                _truncated,
                info,
            ):
                self.step_index += 1
                self.is_success = bool(
                    terminated and info.get("is_success", False)
                )
                self.is_done = self.is_success

        class FakeEnv:
            def __init__(self):
                self.steps = 0
                self.unwrapped = SimpleNamespace(viewer=None)

            def step(self, _action):
                self.steps += 1
                return (
                    make_obs(2),
                    1.0,
                    True,
                    False,
                    {"is_success": True},
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env = FakeEnv()
            with patch(
                "data_collection.autograb_v21.AutoGraspSession",
                FakeSession,
            ):
                result = collect_target_episode(
                    env,
                    self.make_config(root),
                    0,
                    root,
                )

            self.assertEqual(env.steps, 1)
            self.assertTrue(result["is_success"])
            self.assertEqual(result["seed"], 10)
            self.assertEqual(
                result["episode"]["observation.state"].shape,
                (1, 13),
            )

    def test_planner_success_without_environment_confirmation_is_rejected(self):
        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                self.is_done = False
                self.is_success = False
                self.failure_reason = None
                self.step_index = 0
                self.max_steps = 1

            def reset(self, seed):
                return make_obs(seed)

            def compute_action(self, _obs):
                return np.zeros(7, dtype=np.float32)

            def update(self, *_transition):
                self.step_index += 1
                self.is_success = True
                self.is_done = True

        class FakeEnv:
            unwrapped = SimpleNamespace(viewer=None)

            def step(self, _action):
                return (
                    make_obs(2),
                    1.0,
                    False,
                    False,
                    {"is_success": True},
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self.make_config(
                root,
                max_attempts_per_episode=1,
            )
            with patch(
                "data_collection.autograb_v21.AutoGraspSession",
                FakeSession,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "after 1 attempts",
                ):
                    collect_target_episode(
                        FakeEnv(),
                        config,
                        0,
                        root,
                    )

            self.assertEqual(list(root.rglob("*.mp4")), [])

    def test_collection_failure_removes_staging_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self.make_config(
                root,
                image_size=(640, 480),
            )
            with patch(
                "data_collection.autograb_v21."
                "collect_episode_results",
                side_effect=RuntimeError("worker failed"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "worker failed",
                ):
                    collect_dataset(config)

            self.assertFalse(config.output_dir.exists())
            self.assertEqual(
                list(root.glob(".dataset.tmp-*")),
                [],
            )

    def test_multi_worker_results_are_sorted(self):
        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

            def cancel(self):
                return True

        class FakeExecutor:
            max_workers = None

            def __init__(self, *, max_workers, **_kwargs):
                FakeExecutor.max_workers = max_workers
                self.futures = []

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, _fn, episode_index):
                future = FakeFuture(
                    {
                        "episode_index": episode_index,
                        "seed": episode_index,
                        "attempts": 1,
                        "is_success": True,
                        "worker_pid": 100 + episode_index,
                    }
                )
                self.futures.append(future)
                return future

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self.make_config(
                root,
                episodes=2,
                num_envs=4,
                image_size=(640, 480),
            )
            with (
                patch(
                    "data_collection.autograb_v21."
                    "ProcessPoolExecutor",
                    FakeExecutor,
                ),
                patch(
                    "data_collection.autograb_v21.as_completed",
                    side_effect=lambda futures: list(reversed(futures)),
                ),
            ):
                results = collect_episode_results(config, root)

            self.assertEqual(FakeExecutor.max_workers, 2)
            self.assertEqual(
                [result["episode_index"] for result in results],
                [0, 1],
            )


class CollectionCliTests(unittest.TestCase):
    def test_cli_defaults_to_v21_task_directory(self):
        script_path = (
            PROJECT_ROOT
            / "scripts"
            / "auto_collect_dataset"
            / "collect_autograb_lerobot_v21.py"
        )
        spec = importlib.util.spec_from_file_location(
            "collect_autograb_lerobot_v21_cli",
            script_path,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        config = module.parse_args(["debris_truss"])

        self.assertEqual(config.task_name, "debris_truss")
        self.assertEqual(config.num_envs, 1)
        self.assertEqual(
            config.output_dir,
            (
                PROJECT_ROOT
                / "data/local/debris_truss_lerobot_v21"
            ),
        )
        self.assertEqual(config.camera_map, build_camera_map(None))


if __name__ == "__main__":
    unittest.main()

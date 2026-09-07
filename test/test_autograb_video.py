import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from reports.record.autograb_video import (
    DEFAULT_TASK_ORDER,
    FreeCameraVideoRecorder,
    VideoRecordingConfig,
    output_path_for_task,
    record_first_successful_attempt,
    validate_recording_config,
)
from envs import CUBE_VIEWER_CONFIG
from planners.auto_grasp import TASK_NAMES


SCRIPT_PATH = (
    PROJECT_ROOT / "reports" / "record" / "record_auto_grasp_videos.py"
)


class FakeGlobalVisual:
    offwidth = 640
    offheight = 480


class FakeRenderer:
    def __init__(self, _model, *, width, height):
        self.width = width
        self.height = height
        self.camera = None
        self.closed = False

    def update_scene(self, _data, *, camera):
        self.camera = camera

    def render(self):
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self):
        self.closed = True


class FakeWriter:
    instances = []

    def __init__(self, path, *, width, height, fps, crf, gop):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"video")
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.gop = gop
        self.frame_count = 0
        self.closed = False
        self.discarded = False
        type(self).instances.append(self)

    def append(self, frame):
        self.frame_count += 1
        self.last_shape = frame.shape

    def close(self):
        self.closed = True

    def discard(self):
        self.discarded = True
        self.closed = True
        self.path.unlink(missing_ok=True)


class VideoConfigTests(unittest.TestCase):
    def test_default_task_order_covers_registry_once(self):
        self.assertEqual(set(DEFAULT_TASK_ORDER), set(TASK_NAMES))
        self.assertEqual(len(DEFAULT_TASK_ORDER), len(TASK_NAMES))

    def test_config_rejects_duplicates_and_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "must not be repeated"):
                validate_recording_config(
                    VideoRecordingConfig(
                        output_dir=root,
                        task_names=("cube", "cube"),
                    )
                )
            with self.assertRaisesRegex(ValueError, "must be even"):
                validate_recording_config(
                    VideoRecordingConfig(
                        output_dir=root,
                        task_names=("cube",),
                        width=1919,
                    )
                )

    def test_cli_supports_repeated_task_selection(self):
        spec = importlib.util.spec_from_file_location(
            "record_auto_grasp_videos_cli",
            SCRIPT_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        args = module.build_parser().parse_args(
            ["--task", "cube", "--task", "satellite_handle"]
        )
        config = module.config_from_args(args)
        self.assertEqual(
            config.task_names,
            ("cube", "satellite_handle"),
        )


class FreeCameraRecorderTests(unittest.TestCase):
    def setUp(self):
        FakeWriter.instances.clear()

    def make_env_base(self):
        global_visual = FakeGlobalVisual()
        model = SimpleNamespace(
            vis=SimpleNamespace(global_=global_visual),
        )
        return SimpleNamespace(
            model=model,
            data=object(),
            viewer_config=dict(CUBE_VIEWER_CONFIG),
        )

    def test_uses_default_viewer_camera_and_restores_buffer_size(self):
        env_base = self.make_env_base()
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = FreeCameraVideoRecorder(
                env_base,
                Path(tmpdir) / "video.mp4",
                width=1920,
                height=1080,
                renderer_factory=FakeRenderer,
                writer_factory=FakeWriter,
            )
            self.assertEqual(recorder.camera.type, 0)
            self.assertEqual(recorder.camera.distance, 5.0)
            self.assertEqual(recorder.camera.azimuth, -60.0)
            self.assertEqual(recorder.camera.elevation, -30.0)
            np.testing.assert_allclose(
                recorder.camera.lookat,
                [1.6, 0.0, 2.65],
            )
            self.assertEqual(env_base.model.vis.global_.offwidth, 1920)
            self.assertEqual(env_base.model.vis.global_.offheight, 1080)

            recorder.capture()
            self.assertEqual(recorder.frame_count, 1)
            self.assertEqual(
                FakeWriter.instances[-1].last_shape,
                (1080, 1920, 3),
            )
            recorder.close()
            self.assertEqual(env_base.model.vis.global_.offwidth, 640)
            self.assertEqual(env_base.model.vis.global_.offheight, 480)


class RecordingLoopTests(unittest.TestCase):
    def test_failed_attempt_is_discarded_and_success_has_steps_plus_one_frames(
        self,
    ):
        class FakeSession:
            attempt = 0

            def __init__(self, *_args, **_kwargs):
                type(self).attempt += 1
                self.current_attempt = type(self).attempt
                self.step_index = 0
                self.max_steps = 2
                self.is_done = False
                self.failure_reason = None

            def reset(self, seed):
                self.seed = seed
                return {"obs": seed}

            def compute_action(self, _obs):
                return np.zeros(7, dtype=np.float32)

            def update(self, _obs, _reward, terminated, truncated, _info):
                self.step_index += 1
                self.is_done = bool(terminated or truncated)
                if truncated:
                    self.failure_reason = "environment truncated"

        class FakeEnv:
            def __init__(self):
                self.unwrapped = SimpleNamespace()
                self.step_in_attempt = 0

            def step(self, _action):
                self.step_in_attempt += 1
                if FakeSession.attempt == 1:
                    self.step_in_attempt = 0
                    return {}, 0.0, False, True, {"is_success": False}
                terminated = self.step_in_attempt == 2
                return (
                    {},
                    1.0,
                    terminated,
                    False,
                    {"is_success": terminated},
                )

        class FakeRecorder:
            instances = []

            def __init__(self, _env, path, **_kwargs):
                self.path = Path(path)
                self.path.write_bytes(b"video")
                self.camera_config = {
                    "distance": 5.0,
                    "azimuth": -60.0,
                    "elevation": -30.0,
                    "lookat": [1.6, 0.0, 2.65],
                }
                self.frame_count = 0
                self.discarded = False
                type(self).instances.append(self)

            def capture(self):
                self.frame_count += 1

            def close(self):
                pass

            def discard(self):
                self.discarded = True
                self.path.unlink(missing_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = VideoRecordingConfig(
                output_dir=root,
                task_names=("cube",),
                seed=20,
                max_attempts=2,
            )
            with (
                patch(
                    "reports.record.autograb_video.AutoGraspSession",
                    FakeSession,
                ),
                patch(
                    "reports.record.autograb_video."
                    "inspect_recorded_video",
                    return_value={
                        "codec": "h264",
                        "pixel_format": "yuv420p",
                        "width": 1920,
                        "height": 1080,
                        "fps": 20,
                        "frame_count": 3,
                        "duration_seconds": 0.15,
                        "sha256": "temporary",
                    },
                ),
            ):
                result = record_first_successful_attempt(
                    FakeEnv(),
                    "cube",
                    config,
                    recorder_type=FakeRecorder,
                )

            self.assertTrue(FakeRecorder.instances[0].discarded)
            self.assertEqual(FakeRecorder.instances[0].frame_count, 2)
            self.assertEqual(FakeRecorder.instances[1].frame_count, 3)
            self.assertEqual(result["seed"], 21)
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["environment_steps"], 2)
            self.assertTrue(
                result["final_frame_after_success_confirmation"]
            )
            self.assertTrue(output_path_for_task(root, "cube").exists())


if __name__ == "__main__":
    unittest.main()

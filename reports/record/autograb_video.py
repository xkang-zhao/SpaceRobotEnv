"""Record automatic grasps from the MuJoCo viewer's default free camera."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import av
import mujoco
import numpy as np

from planners.auto_grasp import TASK_NAMES, AutoGraspSession, get_task_spec

from data_collection.lerobot_v21 import StreamingVideoWriter


VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 20
VIDEO_CRF = 18
VIDEO_GOP = 40
DEFAULT_MAX_ATTEMPTS = 10
DEFAULT_TASK_ORDER = (
    "cube",
    "debris_antenna_panel",
    "debris_truss",
    "satellite2_left_truss_connection",
    "satellite3_left_antenna_panel",
    "satellite3_upper_rod",
    "satellite_handle",
    "satellite_left_antenna_panel",
)


@dataclass(frozen=True)
class VideoRecordingConfig:
    """Configuration for a batch of presentation video recordings."""

    output_dir: Path
    task_names: tuple[str, ...] = DEFAULT_TASK_ORDER
    seed: int = 0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    overwrite: bool = False
    width: int = VIDEO_WIDTH
    height: int = VIDEO_HEIGHT
    fps: int = VIDEO_FPS
    crf: int = VIDEO_CRF
    gop: int = VIDEO_GOP


def validate_recording_config(config: VideoRecordingConfig) -> None:
    if not config.task_names:
        raise ValueError("At least one automatic grasp task is required.")
    if len(set(config.task_names)) != len(config.task_names):
        raise ValueError("Automatic grasp tasks must not be repeated.")
    unknown = set(config.task_names).difference(TASK_NAMES)
    if unknown:
        raise ValueError(f"Unknown automatic grasp tasks: {sorted(unknown)}")
    if config.max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    if config.width <= 0 or config.height <= 0:
        raise ValueError("Video dimensions must be positive.")
    if config.width % 2 or config.height % 2:
        raise ValueError("H.264 yuv420p video dimensions must be even.")
    if config.fps != VIDEO_FPS:
        raise ValueError(
            f"Automatic grasp videos must remain {VIDEO_FPS} FPS."
        )
    if not 0 <= config.crf <= 51:
        raise ValueError("H.264 CRF must be between 0 and 51.")
    if config.gop <= 0:
        raise ValueError("H.264 GOP must be positive.")


def output_path_for_task(output_dir: Path, task_name: str) -> Path:
    get_task_spec(task_name)
    return Path(output_dir) / f"{task_name}.mp4"


class FreeCameraVideoRecorder:
    """Stream frames rendered from the environment's viewer camera config."""

    def __init__(
        self,
        env_base,
        path: Path,
        *,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        fps: int = VIDEO_FPS,
        crf: int = VIDEO_CRF,
        gop: int = VIDEO_GOP,
        renderer_factory: Callable[..., object] = mujoco.Renderer,
        writer_factory: Callable[..., object] = StreamingVideoWriter,
    ):
        self.env_base = env_base
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._closed = False
        self._renderer = None
        self._writer = None
        self._previous_offscreen_size = (
            int(env_base.model.vis.global_.offwidth),
            int(env_base.model.vis.global_.offheight),
        )

        viewer_config = dict(env_base.viewer_config)
        self.camera_config = {
            "distance": float(viewer_config["distance"]),
            "azimuth": float(viewer_config["azimuth"]),
            "elevation": float(viewer_config["elevation"]),
            "lookat": [
                float(value) for value in viewer_config["lookat"]
            ],
        }
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.fixedcamid = -1
        self.camera.trackbodyid = -1
        self.camera.distance = self.camera_config["distance"]
        self.camera.azimuth = self.camera_config["azimuth"]
        self.camera.elevation = self.camera_config["elevation"]
        self.camera.lookat[:] = self.camera_config["lookat"]

        env_base.model.vis.global_.offwidth = max(
            self._previous_offscreen_size[0],
            self.width,
        )
        env_base.model.vis.global_.offheight = max(
            self._previous_offscreen_size[1],
            self.height,
        )
        try:
            self._renderer = renderer_factory(
                env_base.model,
                width=self.width,
                height=self.height,
            )
            self._writer = writer_factory(
                self.path,
                width=self.width,
                height=self.height,
                fps=self.fps,
                crf=crf,
                gop=gop,
            )
        except BaseException:
            if self._renderer is not None:
                self._renderer.close()
            self._restore_offscreen_size()
            raise

    @property
    def frame_count(self) -> int:
        if self._writer is None:
            return 0
        return int(self._writer.frame_count)

    def capture(self) -> None:
        if self._closed:
            raise RuntimeError("Cannot capture with a closed video recorder.")
        self._renderer.update_scene(
            self.env_base.data,
            camera=self.camera,
        )
        frame = np.asarray(self._renderer.render())
        self._writer.append(frame)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._writer is not None:
                self._writer.close()
        finally:
            try:
                if self._renderer is not None:
                    self._renderer.close()
            finally:
                self._restore_offscreen_size()
                self._closed = True

    def discard(self) -> None:
        if self._closed:
            self.path.unlink(missing_ok=True)
            return
        try:
            if self._writer is not None:
                self._writer.discard()
        finally:
            try:
                if self._renderer is not None:
                    self._renderer.close()
            finally:
                self._restore_offscreen_size()
                self._closed = True
                self.path.unlink(missing_ok=True)

    def _restore_offscreen_size(self) -> None:
        width, height = self._previous_offscreen_size
        self.env_base.model.vis.global_.offwidth = width
        self.env_base.model.vis.global_.offheight = height


def _temporary_video_path(output_dir: Path, task_name: str) -> Path:
    token = uuid.uuid4().hex
    return Path(output_dir) / f".{task_name}.{token}.mp4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_recorded_video(
    path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    expected_frames: int,
) -> dict:
    path = Path(path)
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path}.")
        stream = container.streams.video[0]
        actual_fps = float(stream.average_rate or stream.base_rate)
        decoded_frames = sum(1 for _ in container.decode(video=0))
        codec = str(stream.codec_context.name)
        pixel_format = str(stream.codec_context.format.name)
        actual_width = int(stream.width)
        actual_height = int(stream.height)

    if codec != "h264":
        raise ValueError(f"Expected H.264 video, got {codec!r}.")
    if pixel_format != "yuv420p":
        raise ValueError(
            f"Expected yuv420p video, got {pixel_format!r}."
        )
    if (actual_width, actual_height) != (width, height):
        raise ValueError(
            "Unexpected video dimensions: "
            f"{actual_width}x{actual_height}, expected {width}x{height}."
        )
    if not np.isclose(actual_fps, fps, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Unexpected video FPS: {actual_fps}, expected {fps}."
        )
    if decoded_frames != expected_frames:
        raise ValueError(
            f"Decoded {decoded_frames} frames, expected {expected_frames}."
        )
    return {
        "codec": codec,
        "pixel_format": pixel_format,
        "width": actual_width,
        "height": actual_height,
        "fps": int(round(actual_fps)),
        "frame_count": decoded_frames,
        "duration_seconds": decoded_frames / fps,
        "sha256": _sha256(path),
    }


def record_first_successful_attempt(
    env,
    task_name: str,
    config: VideoRecordingConfig,
    *,
    recorder_type=FreeCameraVideoRecorder,
) -> dict:
    task = get_task_spec(task_name)
    final_path = output_path_for_task(config.output_dir, task_name)

    for attempt_index in range(config.max_attempts):
        attempt_number = attempt_index + 1
        attempt_seed = config.seed + attempt_index
        temporary_path = _temporary_video_path(
            config.output_dir,
            task_name,
        )
        session = AutoGraspSession(task_name, env)
        recorder = None
        try:
            obs = session.reset(attempt_seed)
            recorder = recorder_type(
                env.unwrapped,
                temporary_path,
                width=config.width,
                height=config.height,
                fps=config.fps,
                crf=config.crf,
                gop=config.gop,
            )
            recorder.capture()
            confirmed = False

            while (
                not session.is_done
                and session.step_index < session.max_steps
            ):
                action = session.compute_action(obs)
                transition = env.step(action)
                obs, reward, terminated, truncated, info = transition
                session.update(
                    obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                )
                recorder.capture()
                confirmed = bool(
                    terminated and info.get("is_success", False)
                )
                if confirmed or truncated or session.is_done:
                    break

            if confirmed:
                expected_frames = session.step_index + 1
                if recorder.frame_count != expected_frames:
                    raise RuntimeError(
                        f"Recorded {recorder.frame_count} frames for "
                        f"{session.step_index} environment steps."
                    )
                recorder.close()
                video_info = inspect_recorded_video(
                    temporary_path,
                    width=config.width,
                    height=config.height,
                    fps=config.fps,
                    expected_frames=expected_frames,
                )
                os.replace(temporary_path, final_path)
                video_info["sha256"] = _sha256(final_path)
                return {
                    "task_name": task.name,
                    "env_id": task.env_id,
                    "output": str(final_path),
                    "camera": dict(recorder.camera_config),
                    "seed": attempt_seed,
                    "attempts": attempt_number,
                    "environment_steps": int(session.step_index),
                    "is_success": True,
                    "selection": "first successful attempt",
                    "final_frame_after_success_confirmation": True,
                    **video_info,
                }

            failure_reason = (
                session.failure_reason
                or (
                    f"max_steps={session.max_steps} reached"
                    if session.step_index >= session.max_steps
                    else "environment did not confirm success"
                )
            )
            recorder.discard()
            print(
                f"[VIDEO] {task_name}: attempt {attempt_number}/"
                f"{config.max_attempts} failed ({failure_reason}).",
                flush=True,
            )
        except BaseException:
            if recorder is not None:
                recorder.discard()
            temporary_path.unlink(missing_ok=True)
            raise

    raise RuntimeError(
        f"Task {task_name!r} did not succeed after "
        f"{config.max_attempts} attempts."
    )


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def record_video_batch(config: VideoRecordingConfig) -> dict:
    """Record one first-successful-attempt video for every selected task."""
    validate_recording_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_path_for_task(config.output_dir, task_name)
        for task_name in config.task_names
    ]
    existing = [path for path in output_paths if path.exists()]
    manifest_path = config.output_dir / "manifest.json"
    if not config.overwrite and (existing or manifest_path.exists()):
        conflicts = [str(path) for path in existing]
        if manifest_path.exists():
            conflicts.append(str(manifest_path))
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(conflicts)
        )

    import gymnasium as gym
    import envs  # noqa: F401  # Register Gymnasium environments.

    results = []
    for task_name in config.task_names:
        task = get_task_spec(task_name)
        env = gym.make(task.env_id, render_mode=None)
        try:
            result = record_first_successful_attempt(
                env,
                task_name,
                config,
            )
        finally:
            env.close()
        results.append(result)
        print(
            f"[VIDEO] {task_name}: success, "
            f"frames={result['frame_count']}, "
            f"duration={result['duration_seconds']:.2f}s, "
            f"output={result['output']}",
            flush=True,
        )

    camera_configs = [result["camera"] for result in results]
    if any(camera != camera_configs[0] for camera in camera_configs[1:]):
        raise RuntimeError(
            "Selected environments do not share one default viewer camera."
        )
    first_env_config = {
        "type": "free",
        "source": "env.unwrapped.viewer_config",
        **camera_configs[0],
    }
    manifest = {
        "description": (
            "First successful automatic-grasp attempts rendered from "
            "the MuJoCo default viewer camera. This is not a success rate."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "first successful attempt",
        "camera": first_env_config,
        "video": {
            "codec": "h264",
            "encoder": "libx264",
            "pixel_format": "yuv420p",
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "crf": config.crf,
            "gop": config.gop,
        },
        "tasks": results,
    }
    _write_manifest(manifest_path, manifest)
    return manifest

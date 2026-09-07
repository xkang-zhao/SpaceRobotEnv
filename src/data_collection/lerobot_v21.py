"""Direct LeRobot v2.1 dataset writer with streaming H.264 videos."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import uuid
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
VIDEO_CODEC = "libx264"
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_CRF = 23
VIDEO_GOP = 2
ROBOT_TYPE = "spaceur10e_2f85_mujoco"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATE_NAMES = [
    "j1",
    "j2",
    "j3",
    "j4",
    "j5",
    "j6",
    "x",
    "y",
    "z",
    "qw",
    "qx",
    "qy",
    "qz",
]
ACTION_NAMES = [
    "dx",
    "dy",
    "dz",
    "wx",
    "wy",
    "wz",
    "gripper",
]
STANDARD_FEATURES = {
    "is_success": {"dtype": "bool", "shape": [1], "names": None},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


class StreamingVideoWriter:
    """Encode one RGB stream without retaining previous frames."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        crf: int = VIDEO_CRF,
        gop: int = VIDEO_GOP,
    ):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.crf = int(crf)
        self.gop = int(gop)
        self.frame_count = 0
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), mode="w")
        self._stream = self._container.add_stream(
            VIDEO_CODEC,
            rate=self.fps,
            options={
                "crf": str(self.crf),
                "g": str(self.gop),
            },
        )
        self._stream.width = self.width
        self._stream.height = self.height
        self._stream.pix_fmt = VIDEO_PIXEL_FORMAT
        self._stream.thread_count = 1

    def append(self, rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed video writer.")
        image = np.asarray(rgb)
        expected = (self.height, self.width, 3)
        if image.shape != expected:
            raise ValueError(
                f"Expected RGB frame {expected}, got {image.shape}."
            )
        if image.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 RGB frame, got {image.dtype}."
            )
        frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(image),
            format="rgb24",
        )
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()
            self._closed = True

    def discard(self) -> None:
        try:
            self.close()
        finally:
            self.path.unlink(missing_ok=True)


class ImageChannelStats:
    """Compute exact normalized per-channel statistics online."""

    def __init__(self):
        self.frame_count = 0
        self.pixel_count = 0
        self.sum = np.zeros(3, dtype=np.float64)
        self.sum_sq = np.zeros(3, dtype=np.float64)
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)

    def update(self, rgb: np.ndarray) -> None:
        image = np.asarray(rgb, dtype=np.uint8)
        pixels = image.reshape(-1, 3).astype(np.float64) / 255.0
        self.frame_count += 1
        self.pixel_count += int(pixels.shape[0])
        self.sum += pixels.sum(axis=0)
        self.sum_sq += np.square(pixels).sum(axis=0)
        self.minimum = np.minimum(self.minimum, pixels.min(axis=0))
        self.maximum = np.maximum(self.maximum, pixels.max(axis=0))

    def finalize(self) -> dict[str, np.ndarray]:
        if self.frame_count == 0 or self.pixel_count == 0:
            raise ValueError("Cannot compute image stats without frames.")
        mean = self.sum / self.pixel_count
        variance = np.maximum(
            self.sum_sq / self.pixel_count - np.square(mean),
            0.0,
        )
        channel_shape = (3, 1, 1)
        return {
            "min": self.minimum.reshape(channel_shape),
            "max": self.maximum.reshape(channel_shape),
            "mean": mean.reshape(channel_shape),
            "std": np.sqrt(variance).reshape(channel_shape),
            "count": np.array([self.frame_count], dtype=np.int64),
        }


class LeRobotV21EpisodeBuffer:
    """Numeric episode buffer plus streaming encoders for one attempt."""

    def __init__(
        self,
        attempt_dir: Path,
        *,
        camera_map: dict[str, str],
        image_size: tuple[int, int],
        fps: int,
    ):
        self.attempt_dir = Path(attempt_dir)
        self.camera_map = dict(camera_map)
        self.image_size = tuple(image_size)
        self.fps = int(fps)
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.successes: list[bool] = []
        self.timestamps: list[float] = []
        width, height = self.image_size
        self.video_paths = {
            name: (
                self.attempt_dir
                / f"observation.images.{name}.mp4"
            )
            for name in self.camera_map
        }
        self.video_writers = {
            name: StreamingVideoWriter(
                path,
                width=width,
                height=height,
                fps=self.fps,
            )
            for name, path in self.video_paths.items()
        }
        self.video_stats = {
            name: ImageChannelStats() for name in self.camera_map
        }
        self._finalized = False

    def append_pre_step(self, obs: dict, action) -> None:
        state = np.concatenate(
            [
                np.asarray(obs["joint_pos"], dtype=np.float32).reshape(-1),
                np.asarray(obs["ee_pose"], dtype=np.float32).reshape(-1),
            ]
        )
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.shape != (len(STATE_NAMES),):
            raise ValueError(
                f"Expected state shape {(len(STATE_NAMES),)}, "
                f"got {state.shape}."
            )
        if action_array.shape != (len(ACTION_NAMES),):
            raise ValueError(
                f"Expected action shape {(len(ACTION_NAMES),)}, "
                f"got {action_array.shape}."
            )
        self.states.append(state)
        self.actions.append(action_array)
        self.successes.append(False)
        self.timestamps.append(len(self.timestamps) / self.fps)

        for name, obs_key in self.camera_map.items():
            if obs_key not in obs:
                available = ", ".join(sorted(obs))
                raise KeyError(
                    f"Missing camera observation {obs_key!r}. "
                    f"Available keys: {available}"
                )
            image = np.asarray(obs[obs_key])
            self.video_writers[name].append(image)
            self.video_stats[name].update(image)

    def set_last_success(self, success: bool) -> None:
        if not self.successes:
            raise RuntimeError("Cannot update success before appending a frame.")
        self.successes[-1] = bool(success)

    def to_episode(self) -> dict:
        if self._finalized:
            raise RuntimeError("Episode buffer has already been finalized.")
        self._finalized = True
        try:
            for writer in self.video_writers.values():
                writer.close()
        except BaseException:
            self.discard()
            raise

        state = np.asarray(self.states, dtype=np.float32)
        action = np.asarray(self.actions, dtype=np.float32)
        success = np.asarray(self.successes, dtype=np.bool_)
        timestamp = np.asarray(self.timestamps, dtype=np.float32)
        if state.ndim != 2 or state.shape[0] == 0:
            raise ValueError("Expected a non-empty [T, 13] state array.")
        if action.ndim != 2:
            raise ValueError("Expected a [T, 7] action array.")
        length = state.shape[0]
        if not (
            action.shape[0]
            == success.shape[0]
            == timestamp.shape[0]
            == length
        ):
            raise ValueError("Episode feature frame counts do not match.")
        for name, writer in self.video_writers.items():
            if writer.frame_count != length:
                raise ValueError(
                    f"Camera {name!r} has {writer.frame_count} frames, "
                    f"expected {length}."
                )
        return {
            "observation.state": state,
            "action": action,
            "is_success": success,
            "timestamp": timestamp,
            "video_paths": {
                name: str(path) for name, path in self.video_paths.items()
            },
            "video_stats": {
                f"observation.images.{name}": stats.finalize()
                for name, stats in self.video_stats.items()
            },
        }

    def discard(self) -> None:
        for writer in self.video_writers.values():
            if not writer._closed:
                writer.discard()
        if self.attempt_dir.exists():
            shutil.rmtree(self.attempt_dir)


def _numeric_stats(array: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(array)
    if values.shape[0] == 0:
        raise ValueError("Cannot compute stats for an empty array.")
    keepdims = values.ndim == 1
    return {
        "min": np.min(values, axis=0, keepdims=keepdims),
        "max": np.max(values, axis=0, keepdims=keepdims),
        "mean": np.mean(values, axis=0, keepdims=keepdims),
        "std": np.std(values, axis=0, keepdims=keepdims),
        "count": np.array([values.shape[0]], dtype=np.int64),
    }


def aggregate_feature_stats(
    stats_list: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    means = np.stack([item["mean"] for item in stats_list])
    variances = np.stack(
        [np.square(item["std"]) for item in stats_list]
    )
    counts = np.stack([item["count"] for item in stats_list])
    total_count = counts.sum(axis=0)
    broadcast_counts = counts
    while broadcast_counts.ndim < means.ndim:
        broadcast_counts = np.expand_dims(broadcast_counts, axis=-1)
    total_mean = (
        means * broadcast_counts
    ).sum(axis=0) / total_count
    total_variance = (
        (
            variances
            + np.square(means - total_mean)
        )
        * broadcast_counts
    ).sum(axis=0) / total_count
    return {
        "min": np.min(
            np.stack([item["min"] for item in stats_list]),
            axis=0,
        ),
        "max": np.max(
            np.stack([item["max"] for item in stats_list]),
            axis=0,
        ),
        "mean": total_mean,
        "std": np.sqrt(np.maximum(total_variance, 0.0)),
        "count": total_count,
    }


def aggregate_stats(
    stats_list: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    keys = {key for stats in stats_list for key in stats}
    return {
        key: aggregate_feature_stats(
            [stats[key] for stats in stats_list if key in stats]
        )
        for key in sorted(keys)
    }


def serialize_numpy(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            key: serialize_numpy(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [serialize_numpy(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fixed_size_float_array(array: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D float array, got {values.shape}.")
    flattened = pa.array(
        np.ascontiguousarray(values).reshape(-1),
        type=pa.float32(),
    )
    return pa.FixedSizeListArray.from_arrays(
        flattened,
        list_size=int(values.shape[1]),
    )


def write_episode_parquet(
    path: Path,
    episode: dict,
    *,
    episode_index: int,
    start_global_index: int,
) -> dict[str, dict[str, np.ndarray]]:
    state = np.asarray(episode["observation.state"], dtype=np.float32)
    action = np.asarray(episode["action"], dtype=np.float32)
    success = np.asarray(episode["is_success"], dtype=np.bool_)
    timestamp = np.asarray(episode["timestamp"], dtype=np.float32)
    length = int(action.shape[0])
    if not (
        state.shape[0]
        == success.shape[0]
        == timestamp.shape[0]
        == length
    ):
        raise ValueError("Episode feature frame counts do not match.")

    frame_index = np.arange(length, dtype=np.int64)
    episode_indices = np.full(
        length,
        episode_index,
        dtype=np.int64,
    )
    global_indices = np.arange(
        start_global_index,
        start_global_index + length,
        dtype=np.int64,
    )
    task_indices = np.zeros(length, dtype=np.int64)
    table = pa.table(
        {
            "observation.state": _fixed_size_float_array(state),
            "action": _fixed_size_float_array(action),
            "is_success": pa.array(success, type=pa.bool_()),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(
                episode_indices,
                type=pa.int64(),
            ),
            "index": pa.array(global_indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    numeric = {
        "observation.state": state,
        "action": action,
        "is_success": success,
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_indices,
        "index": global_indices,
        "task_index": task_indices,
    }
    stats = {
        key: _numeric_stats(value) for key, value in numeric.items()
    }
    stats.update(episode["video_stats"])
    return stats


def inspect_video(
    path: Path,
    *,
    expected_frames: int,
    fps: int,
    image_size: tuple[int, int],
) -> dict:
    width, height = image_size
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path}.")
        stream = container.streams.video[0]
        actual_fps = float(stream.average_rate or stream.base_rate)
        codec = stream.codec_context.codec.canonical_name
        pixel_format = stream.codec_context.pix_fmt
        actual_frames = sum(1 for _ in container.decode(video=0))
        if (stream.width, stream.height) != (width, height):
            raise ValueError(
                f"Video size mismatch in {path}: "
                f"{stream.width}x{stream.height}, "
                f"expected {width}x{height}."
            )
        if not np.isclose(actual_fps, fps, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Video FPS mismatch in {path}: "
                f"{actual_fps}, expected {fps}."
            )
        if actual_frames != expected_frames:
            raise ValueError(
                f"Video frame count mismatch in {path}: "
                f"{actual_frames}, expected {expected_frames}."
            )
        if codec != "h264":
            raise ValueError(
                f"Video codec mismatch in {path}: {codec}."
            )
        if pixel_format not in {"yuv420p", "yuvj420p"}:
            raise ValueError(
                f"Video pixel format mismatch in {path}: {pixel_format}."
            )
        return {
            "video.height": int(stream.height),
            "video.width": int(stream.width),
            "video.codec": codec,
            "video.pix_fmt": pixel_format,
            "video.is_depth_map": False,
            "video.fps": int(round(actual_fps)),
            "video.channels": 3,
            "has_audio": False,
        }


def build_features(
    camera_map: dict[str, str],
    image_size: tuple[int, int],
    video_info: dict[str, dict],
) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(STATE_NAMES)],
            "names": list(STATE_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": [len(ACTION_NAMES)],
            "names": list(ACTION_NAMES),
        },
    }
    width, height = image_size
    for name in camera_map:
        video_key = f"observation.images.{name}"
        features[video_key] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": video_info[video_key],
        }
    features.update(STANDARD_FEATURES)
    return features


def _write_json(path: Path, value) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for value in values:
            file.write(
                json.dumps(
                    serialize_numpy(value),
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_metadata(
    root: Path,
    *,
    instruction: str,
    fps: int,
    features: dict,
    camera_count: int,
    episode_metadata: list[dict],
    episode_stats: list[dict],
    total_frames: int,
) -> None:
    total_episodes = len(episode_metadata)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ROBOT_TYPE,
        "total_episodes": total_episodes,
        "total_frames": int(total_frames),
        "total_tasks": 1,
        "total_videos": total_episodes * camera_count,
        "total_chunks": math.ceil(total_episodes / CHUNK_SIZE),
        "chunks_size": CHUNK_SIZE,
        "fps": int(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
            if camera_count
            else None
        ),
        "features": features,
    }
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    _write_json(meta_dir / "info.json", info)
    _write_jsonl(meta_dir / "episodes.jsonl", episode_metadata)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats)
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [{"task_index": 0, "task": instruction}],
    )
    _write_json(
        meta_dir / "stats.json",
        serialize_numpy(
            aggregate_stats(
                [item["stats"] for item in episode_stats]
            )
        ),
    )


def finalize_dataset(
    staging_dir: Path,
    *,
    results: list[dict],
    expected_episodes: int,
    instruction: str,
    camera_map: dict[str, str],
    fps: int,
    image_size: tuple[int, int],
) -> dict:
    ordered = sorted(results, key=lambda item: item["episode_index"])
    expected_indices = list(range(expected_episodes))
    actual_indices = [item["episode_index"] for item in ordered]
    if actual_indices != expected_indices:
        raise ValueError(
            "Collected episode indices are incomplete or duplicated: "
            f"{actual_indices}, expected {expected_indices}."
        )

    episode_metadata = []
    episode_stats = []
    video_info: dict[str, dict] = {}
    total_frames = 0
    for result in ordered:
        episode_index = int(result["episode_index"])
        episode = result["episode"]
        length = int(episode["action"].shape[0])
        chunk = episode_index // CHUNK_SIZE
        parquet_path = (
            staging_dir
            / f"data/chunk-{chunk:03d}"
            / f"episode_{episode_index:06d}.parquet"
        )
        stats = write_episode_parquet(
            parquet_path,
            episode,
            episode_index=episode_index,
            start_global_index=total_frames,
        )
        for name in camera_map:
            video_key = f"observation.images.{name}"
            source = Path(episode["video_paths"][name])
            target = (
                staging_dir
                / f"videos/chunk-{chunk:03d}"
                / video_key
                / f"episode_{episode_index:06d}.mp4"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            current_info = inspect_video(
                target,
                expected_frames=length,
                fps=fps,
                image_size=image_size,
            )
            previous = video_info.get(video_key)
            if previous is not None and previous != current_info:
                raise ValueError(
                    f"Inconsistent video schema for {video_key}."
                )
            video_info[video_key] = current_info
        episode_metadata.append(
            {
                "episode_index": episode_index,
                "tasks": [instruction],
                "length": length,
            }
        )
        episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": stats,
            }
        )
        total_frames += length

    attempts_dir = staging_dir / ".attempts"
    if attempts_dir.exists():
        shutil.rmtree(attempts_dir)
    features = build_features(camera_map, image_size, video_info)
    write_metadata(
        staging_dir,
        instruction=instruction,
        fps=fps,
        features=features,
        camera_count=len(camera_map),
        episode_metadata=episode_metadata,
        episode_stats=episode_stats,
        total_frames=total_frames,
    )
    return {
        "saved_episodes": len(ordered),
        "successful_episodes": sum(
            bool(item["is_success"]) for item in ordered
        ),
        "attempts": sum(int(item["attempts"]) for item in ordered),
        "total_frames": total_frames,
        "worker_pids": sorted(
            {int(item["worker_pid"]) for item in ordered}
        ),
    }


def validate_output_target(path: Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_symlink():
        raise ValueError(
            f"Output directory must not be a symlink: {expanded}"
        )
    target = expanded.resolve()
    forbidden = {
        Path("/").resolve(),
        Path.home().resolve(),
        PROJECT_ROOT.resolve(),
    }
    if target in forbidden:
        raise ValueError(f"Refusing unsafe output directory: {target}")
    if target.exists() and not target.is_dir():
        raise ValueError(
            f"Output path exists and is not a directory: {target}"
        )
    return target


def prepare_staging_dir(target: Path, *, overwrite: bool) -> Path:
    target = validate_output_target(target)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {target}. "
            "Pass --overwrite to replace it after successful collection."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
    )


def commit_staging_dir(
    staging_dir: Path,
    target: Path,
    *,
    overwrite: bool,
) -> None:
    staging_dir = Path(staging_dir)
    target = validate_output_target(target)
    backup = None
    if target.exists():
        if any(target.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory became non-empty: {target}"
            )
        backup = (
            target.parent
            / f".{target.name}.backup-{uuid.uuid4().hex}"
        )
        target.replace(backup)
    try:
        staging_dir.replace(target)
    except BaseException:
        if (
            backup is not None
            and backup.exists()
            and not target.exists()
        ):
            backup.replace(target)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)

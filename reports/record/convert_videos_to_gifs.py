#!/usr/bin/env python3
"""Convert automatic-grasp MP4 videos into PowerPoint-friendly GIFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "assets/videos/auto_grasp"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets/gifs/auto_grasp"
DEFAULT_WIDTH = 960
DEFAULT_FPS = 10
DEFAULT_COLORS = 128


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg was not found and imageio-ffmpeg is unavailable."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_video_paths(input_dir: Path) -> list[Path]:
    input_dir = Path(input_dir)
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        paths = [
            input_dir / f"{item['task_name']}.mp4"
            for item in manifest["tasks"]
        ]
    else:
        paths = sorted(input_dir.glob("*.mp4"))
    if not paths:
        raise FileNotFoundError(f"No MP4 videos found in {input_dir}.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing source videos: " + ", ".join(map(str, missing))
        )
    return paths


def validate_options(*, width: int, fps: int, colors: int) -> None:
    if width <= 0 or width % 2:
        raise ValueError("GIF width must be a positive even integer.")
    if fps <= 0:
        raise ValueError("GIF FPS must be positive.")
    if not 2 <= colors <= 256:
        raise ValueError("GIF colors must be between 2 and 256.")


def build_filter(*, width: int, fps: int, colors: int) -> str:
    return (
        f"fps={fps},"
        f"scale={width}:-2:flags=lanczos,"
        "split[original][palette_source];"
        f"[palette_source]palettegen=max_colors={colors}:"
        "stats_mode=diff[palette];"
        "[original][palette]paletteuse="
        "dither=sierra2_4a:diff_mode=rectangle"
    )


def inspect_gif(path: Path, *, expected_width: int, expected_fps: int) -> dict:
    durations = []
    with Image.open(path) as image:
        width, height = image.size
        frame_count = int(getattr(image, "n_frames", 1))
        loop = image.info.get("loop")
        for frame_index in range(frame_count):
            image.seek(frame_index)
            durations.append(int(image.info.get("duration", 0)))

    if width != expected_width:
        raise ValueError(
            f"Unexpected GIF width {width}, expected {expected_width}."
        )
    if height <= 0:
        raise ValueError("GIF height must be positive.")
    if frame_count <= 1:
        raise ValueError(f"GIF is not animated: {path}")
    if loop != 0:
        raise ValueError(f"GIF does not loop continuously: {path}")

    expected_duration_ms = round(1000 / expected_fps)
    if any(duration != expected_duration_ms for duration in durations):
        raise ValueError(
            f"GIF frame duration is not {expected_duration_ms} ms: {path}"
        )
    return {
        "width": width,
        "height": height,
        "fps": expected_fps,
        "frame_count": frame_count,
        "duration_seconds": sum(durations) / 1000.0,
        "loop": "continuous",
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def convert_video(
    source: Path,
    output: Path,
    *,
    ffmpeg: str,
    width: int,
    fps: int,
    colors: int,
) -> dict:
    temporary = output.with_name(
        f".{output.stem}.{uuid.uuid4().hex}.gif"
    )
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-filter_complex",
                build_filter(width=width, fps=fps, colors=colors),
                "-loop",
                "0",
                str(temporary),
            ],
            check=True,
        )
        info = inspect_gif(
            temporary,
            expected_width=width,
            expected_fps=fps,
        )
        os.replace(temporary, output)
        info["sha256"] = sha256(output)
        return {
            "task_name": source.stem,
            "source": str(source),
            "output": str(output),
            **info,
        }
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def convert_all(
    input_dir: Path,
    output_dir: Path,
    *,
    width: int = DEFAULT_WIDTH,
    fps: int = DEFAULT_FPS,
    colors: int = DEFAULT_COLORS,
    overwrite: bool = False,
) -> dict:
    validate_options(width=width, fps=fps, colors=colors)
    sources = load_video_paths(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{source.stem}.gif" for source in sources]
    manifest_path = output_dir / "manifest.json"

    conflicts = [path for path in outputs if path.exists()]
    if manifest_path.exists():
        conflicts.append(manifest_path)
    if conflicts and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(map(str, conflicts))
        )

    ffmpeg = find_ffmpeg()
    results = []
    for source, output in zip(sources, outputs):
        result = convert_video(
            source,
            output,
            ffmpeg=ffmpeg,
            width=width,
            fps=fps,
            colors=colors,
        )
        results.append(result)
        print(
            f"[GIF] {source.stem}: frames={result['frame_count']}, "
            f"duration={result['duration_seconds']:.2f}s, "
            f"size={result['file_size_bytes'] / 1024 / 1024:.2f} MiB",
            flush=True,
        )

    manifest = {
        "description": (
            "PowerPoint-friendly looping GIFs converted from the "
            "MuJoCo automatic-grasp videos."
        ),
        "conversion": {
            "width": width,
            "fps": fps,
            "colors": colors,
            "scaler": "lanczos",
            "dither": "sierra2_4a",
            "loop": "continuous",
        },
        "tasks": results,
    }
    write_manifest(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--colors", type=int, default=DEFAULT_COLORS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = convert_all(
            args.input_dir.resolve(),
            args.output_dir.resolve(),
            width=args.width,
            fps=args.fps,
            colors=args.colors,
            overwrite=args.overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        parser.error(str(exc))
    print(
        f"[GIF] finished: tasks={len(manifest['tasks'])}, "
        f"output={args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

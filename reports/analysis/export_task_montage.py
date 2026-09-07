#!/usr/bin/env python3
"""Export a lossless high-resolution montage from the eight task videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS = [
    ("cube", "立方体"),
    ("debris_antenna_panel", "碎片天线板"),
    ("debris_truss", "碎片桁架"),
    ("satellite2_left_truss_connection", "卫星2左桁架"),
    ("satellite3_left_antenna_panel", "卫星3左天线板"),
    ("satellite3_upper_rod", "卫星3上杆"),
    ("satellite_handle", "卫星把手"),
    ("satellite_left_antenna_panel", "卫星左天线板"),
]


def read_middle_frame(video_path: Path) -> Image.Image:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Unable to read middle frame from {video_path}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/STHeiti Medium.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def build_montage(dataset_root: Path) -> Image.Image:
    frames: list[tuple[Image.Image, str]] = []
    for task_name, label in TASKS:
        video_path = (
            dataset_root
            / f"{task_name}_lerobot_v21"
            / "videos"
            / "chunk-000"
            / "observation.images.cam1"
            / "episode_000000.mp4"
        )
        frames.append((read_middle_frame(video_path), label))

    tile_width, tile_height = frames[0][0].size
    if any(frame.size != (tile_width, tile_height) for frame, _ in frames):
        raise ValueError("All source videos must have the same frame dimensions.")

    columns, rows = 4, 2
    margin_x, margin_y = 28, 28
    gap_x, gap_y = 24, 36
    canvas_width = columns * tile_width + (columns - 1) * gap_x + 2 * margin_x
    canvas_height = rows * tile_height + (rows - 1) * gap_y + 2 * margin_y
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    font = load_font(29)
    label_height = 64

    for index, (frame, label) in enumerate(frames):
        column = index % columns
        row = index // columns
        x = margin_x + column * (tile_width + gap_x)
        y = margin_y + row * (tile_height + gap_y)
        canvas.paste(frame, (x, y))

        overlay = Image.new("RGBA", (tile_width, label_height), (7, 29, 40, 220))
        canvas.paste(
            overlay,
            (x, y + tile_height - label_height),
            overlay,
        )
        draw = ImageDraw.Draw(canvas)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        draw.text(
            (x + 18, y + tile_height - label_height + (label_height - text_height) / 2 - 2),
            label,
            fill="white",
            font=font,
        )
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--output", default="assets/task_montage.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = (REPO_ROOT / args.dataset_root).resolve()
    output_path = (REPO_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    montage = build_montage(dataset_root)
    montage.save(output_path, format="PNG", compress_level=4, optimize=False)
    print(f"montage: {output_path}")
    print(f"size: {montage.width}x{montage.height} px")


if __name__ == "__main__":
    main()

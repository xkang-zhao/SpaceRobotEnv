#!/usr/bin/env python3
"""Decompose a mesh with CoACD and emit MuJoCo-ready collision parts."""

from __future__ import annotations

import argparse
from pathlib import Path

import coacd
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CoACD on an OBJ/STL mesh and generate individual convex mesh files plus an MJCF snippet."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input OBJ/STL mesh.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for convex-part mesh files.")
    parser.add_argument("--name-prefix", required=True, help="Prefix for mesh and geom names, e.g. satellite_2_col.")
    parser.add_argument(
        "--mjcf-mesh-prefix",
        required=True,
        help="File prefix used in MJCF mesh paths, relative to MuJoCo meshdir, e.g. satellite_2/collision_parts.",
    )
    parser.add_argument("--threshold", type=float, default=0.05, help="CoACD concavity threshold.")
    parser.add_argument("--max-convex-hull", type=int, default=16, help="Maximum convex hull count.")
    parser.add_argument("--scale", default="1 1 1", help='MJCF mesh scale string, e.g. "0.001 0.001 0.001".')
    parser.add_argument("--geom-euler", default=None, help='Optional MJCF geom euler string, e.g. "0 0 -1.57".')
    parser.add_argument("--friction", default="1.5 0.5 0.005", help="MJCF collision geom friction string.")
    parser.add_argument("--rgba", default="1 0.2 0.1 0.25", help="MJCF collision geom rgba string.")
    parser.add_argument(
        "--output-format",
        choices=("obj", "stl"),
        default="obj",
        help="Convex part mesh format to export.",
    )
    parser.add_argument("--real-metric", action="store_true", help="Enable CoACD real metric mode.")
    parser.add_argument("--seed", type=int, default=0, help="CoACD random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(args.input, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a single mesh from {args.input}, got {type(mesh)!r}")

    if not mesh.is_watertight:
        print(f"[Warning] {args.input} is not watertight; decomposition may be imperfect.")

    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
        real_metric=args.real_metric,
        seed=args.seed,
    )

    print(f"Generated {len(parts)} convex parts.")
    part_paths: list[Path] = []
    for i, (vertices, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        out_path = args.output_dir / f"{args.name_prefix}_{i:03d}.{args.output_format}"
        part_mesh.export(out_path)
        part_paths.append(out_path)
        print(f"Saved: {out_path}")

    snippet_path = args.output_dir / f"{args.name_prefix}_mjcf_snippet.xml"
    euler_attr = f' euler="{args.geom_euler}"' if args.geom_euler else ""
    mesh_prefix = args.mjcf_mesh_prefix.rstrip("/")

    lines = ["<!-- Add these mesh assets inside <asset> -->"]
    for i, _ in enumerate(part_paths):
        mesh_name = f"{args.name_prefix}_{i:03d}"
        mesh_file = f"{mesh_prefix}/{mesh_name}.{args.output_format}"
        lines.append(f'<mesh name="{mesh_name}" file="{mesh_file}" scale="{args.scale}"/>')

    lines.extend(["", "<!-- Add these collision geoms inside the same body as the visual mesh -->"])
    for i, _ in enumerate(part_paths):
        mesh_name = f"{args.name_prefix}_{i:03d}"
        geom_name = f"{args.name_prefix}_geom_{i:03d}"
        lines.append(
            f'<geom name="{geom_name}" type="mesh" mesh="{mesh_name}"{euler_attr} '
            f'group="3" rgba="{args.rgba}" friction="{args.friction}" '
            f'contype="1" conaffinity="1"/>'
        )

    snippet_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved MJCF snippet: {snippet_path}")


if __name__ == "__main__":
    main()

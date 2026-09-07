from pathlib import Path

import coacd
import trimesh


INPUT_FILE = Path("satellite_2.obj")
OUTPUT_DIR = Path("collision_parts")
SNIPPET_FILE = OUTPUT_DIR / "satellite_2_collision_snippet.xml"
MJCF_MESHDIR_PREFIX = Path("satellite_2")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    mesh = trimesh.load(INPUT_FILE, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a single mesh from {INPUT_FILE}, got {type(mesh)!r}")

    if not mesh.is_watertight:
        print("[Warning] satellite_2.obj is not watertight; decomposition may be imperfect.")

    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=0.05,
        max_convex_hull=16,
        seed=0,
    )

    print(f"Generated {len(parts)} convex parts.")
    part_paths = []
    for i, (vertices, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        out_path = OUTPUT_DIR / f"satellite_2_col_{i:03d}.obj"
        part_mesh.export(out_path)
        part_paths.append(out_path)
        print(f"Saved: {out_path}")

    lines = ["<!-- Add these mesh assets inside <asset> -->"]
    for i, out_path in enumerate(part_paths):
        mesh_file = MJCF_MESHDIR_PREFIX / out_path
        lines.append(
            f'<mesh name="satellite_2_col_{i:03d}" '
            f'file="{mesh_file.as_posix()}" scale="0.001 0.001 0.001"/>'
        )

    lines.extend(["", "<!-- Add these collision geoms inside body name=\"cube\" -->"])
    for i, _ in enumerate(part_paths):
        lines.append(
            f'<geom name="satellite_2_col_geom_{i:03d}" type="mesh" '
            f'mesh="satellite_2_col_{i:03d}" euler="0 0 -1.57" '
            f'group="3" rgba="1 0.2 0.1 0.25" '
            f'friction="1.5 0.5 0.005" contype="1" conaffinity="1"/>'
        )

    SNIPPET_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved MJCF snippet: {SNIPPET_FILE}")


if __name__ == "__main__":
    main()

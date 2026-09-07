---
name: coacd-convex-decomposition
description: Use when converting OBJ/STL meshes into multiple convex collision parts for MuJoCo/MJCF, especially when a complex mesh is currently used as one collision geom and needs CoACD hulls, visual/collision separation, or generated MJCF mesh and geom snippets.
---

# CoACD Convex Decomposition

## Overview

Use CoACD to turn one complex OBJ/STL into several convex OBJ or STL files for MuJoCo collision. Keep the original mesh as visual-only; add each convex part as its own collision geom on the same body.

## Workflow

1. Locate the source mesh and the MJCF that references it.
2. Check the active Python/conda environment has `coacd` and `trimesh`; install them only with user approval if missing.
3. Run the bundled script to export independent collision mesh files and an MJCF snippet.
4. Edit the MJCF so:
   - original mesh geom is visual-only: `contype="0" conaffinity="0"`
   - every convex part is declared as a separate `<mesh>`
   - every convex part is added as a separate `<geom type="mesh">`
   - collision geoms preserve the original visual geom's pose attributes such as `pos`, `quat`, `euler`, and mesh `scale`
5. Verify with `mujoco.MjModel.from_xml_path(...)` and count the expected collision geoms.

Never combine the CoACD output back into one mesh geom. MuJoCo may treat that as one mesh and build one broad convex hull again.

## Script

Use `scripts/decompose_for_mujoco.py` from this skill:

```bash
python .agent/skills/coacd-convex-decomposition/scripts/decompose_for_mujoco.py \
  --input mjcf/assets/satellite_2/satellite_2.obj \
  --output-dir mjcf/assets/satellite_2/collision_parts \
  --name-prefix satellite_2_col \
  --mjcf-mesh-prefix satellite_2/collision_parts \
  --scale "0.001 0.001 0.001" \
  --geom-euler "0 0 -1.57" \
  --threshold 0.05 \
  --max-convex-hull 16
```

If the source asset should stay STL-only, add:

```bash
--output-format stl
```

Choose `--mjcf-mesh-prefix` relative to the active MuJoCo `<compiler meshdir="...">`. In this repository, included robot XML sets `meshdir="assets"`, so paths usually start below `mjcf/assets`.

Start with:

```text
max-convex-hull: 8-16
threshold: 0.03-0.05
```

If collision is too coarse, try `--max-convex-hull 32 --threshold 0.02`. If the model uses real meter units and you want threshold in physical units, add `--real-metric`.

## MJCF Pattern

Visual mesh:

```xml
<mesh name="object_visual" file="object/object.stl" scale="0.001 0.001 0.001"/>
<geom name="object_visual_geom" type="mesh" mesh="object_visual"
      contype="0" conaffinity="0" group="1"/>
```

Collision parts:

```xml
<mesh name="object_col_000" file="object/collision_parts/object_col_000.obj" scale="0.001 0.001 0.001"/>
<mesh name="object_col_001" file="object/collision_parts/object_col_001.obj" scale="0.001 0.001 0.001"/>

<geom name="object_col_geom_000" type="mesh" mesh="object_col_000"
      group="3" contype="1" conaffinity="1"/>
<geom name="object_col_geom_001" type="mesh" mesh="object_col_001"
      group="3" contype="1" conaffinity="1"/>
```

Copy any orientation or offset from the original geom onto all collision geoms, for example `euler="0 0 -1.57"`.

## Verification

Run a non-interactive MuJoCo load check:

```bash
python -c "import mujoco; m=mujoco.MjModel.from_xml_path('mjcf/your_scene.xml'); print(m.nmesh, m.ngeom)"
```

For a stronger check, assert the generated files and named geoms:

```bash
python -c "from pathlib import Path; import mujoco; files=list(Path('mjcf/assets/object/collision_parts').glob('object_col_*.obj')); assert files; m=mujoco.MjModel.from_xml_path('mjcf/your_scene.xml'); print('collision_objs', len(files)); print('loaded')"
```

If a viewer is available, inspect collision group 3/hull display. You should see multiple smaller hulls, not one large shell around the whole object.

## Common Mistakes

- Using one combined decomposed file once and one `<geom>`: still risks one broad collision mesh.
- Forgetting `contype="0" conaffinity="0"` on the visual geom.
- Losing the original geom's `scale`, `euler`, `quat`, or `pos`.
- Writing mesh file paths relative to the scene XML instead of MuJoCo `meshdir`.
- Ignoring a non-watertight warning: CoACD can still run, but contact quality may need manual primitives in important grasp regions.

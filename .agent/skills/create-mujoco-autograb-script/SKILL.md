---
name: create-mujoco-autograb-script
description: Use when adding or updating a MuJoCo/Gymnasium automatic grasp task from calibrated object/body offsets, end-effector coordinates, or README calibration notes. Register tasks in the reusable planner package and expose them through the unified auto_grasp.py CLI.
---

# Create MuJoCo Auto-Grasp Task

## Overview

Add calibrated automatic grasp tasks to `src/planners/auto_grasp/` and expose
them through the single `scripts/auto_grasp.py` command-line entry point. Do
not create task-specific scripts under the removed `scripts/autograd/`
directory.

Prefer adding a declarative lateral-grasp profile over copying or subclassing
planner logic. Add a new reusable planner module only when the motion state
machine is genuinely different from the existing cube, handle, and lateral
planners.

## Inputs To Extract

From the user's calibration note, identify:

- `env_id`, for example `SpaceUR10e-Satellite2-v0`
- task name, for example `satellite2_left_truss_connection`
- body-frame grasp offset; prefer it over a world-frame offset
- target frame: default to `ee` when calibration used the end-effector
  coordinate, and use `pinch` only for pinch/contact-site calibration
- rotation mode:
  - `skip` when no tool rotation is required
  - `absolute` for the configured base orientation plus local-Z rotation
  - `relative_local_z` to rotate from the current end-effector orientation
- task-specific motion tolerances, when calibration requires them

If orientation is ambiguous and affects grasp safety, ask one short
clarification instead of copying a rotation from an unrelated task.

## Workflow

1. Inspect `src/planners/auto_grasp/`, `src/envs/__init__.py`, and the README
   calibration notes.
2. Decide whether the task fits the existing `cube`, `handle`, or `lateral`
   planner.
3. For a lateral task, add or update a `LateralGraspTaskProfile` in
   `src/planners/auto_grasp/profiles.py`.
4. If a new state machine is required, implement it as a reusable module under
   `src/planners/auto_grasp/` and add a dispatch branch to
   `scripts/auto_grasp.py`.
5. Add focused tests before implementation:
   - the profile or config uses the requested `env_id`
   - the calibrated body offset is preserved
   - CLI overrides are mapped into the config
   - the rotation branch is correct
   - the unified CLI exposes the task name
6. Update the README task table and runnable examples.
7. Verify focused tests, all automatic-grasp tests, every affected `--help`
   command, and Python syntax.

## Lateral Profile Pattern

Add a descriptive key to `LATERAL_GRASP_PROFILES`:

```python
"satellite2_left_truss_connection": LateralGraspTaskProfile(
    name="satellite2_left_truss_connection",
    env_id="SpaceUR10e-Satellite2-v0",
    description="Auto-grasp the satellite2 left truss connection point.",
    success_label="satellite2 left truss connection grasp confirmed",
    grasp_offset_body=(-1.26778, 0.476677, 0.426568),
    tool_z_rotation=0.0,
    rotation_mode="skip",
    object_label="satellite",
    expose_tool_z_rotation=False,
),
```

The unified CLI builds its lateral task list from
`LATERAL_GRASP_PROFILES`, so a valid profile is automatically available as:

```bash
python scripts/auto_grasp.py satellite2_left_truss_connection
```

## Rotation Decision

### No Rotation

Use `rotation_mode="skip"`, set `tool_z_rotation=0.0`, and set
`expose_tool_z_rotation=False`. The shared planner will move directly from the
pre-grasp translation to `move_to_grasp`.

### Absolute Rotation

Use `rotation_mode="absolute"` when the task should use
`base_grasp_orientation_rpy` followed by the configured local-Z tool rotation.
Expose the base orientation only when developers need to tune it.

### Relative Local-Z Rotation

Use `rotation_mode="relative_local_z"` when the calibrated motion rotates the
tool from its current end-effector orientation. Set the signed local-Z angle
explicitly and expose `--tool_z_rotation` only when tuning is expected.

## Test Pattern

Add profile calibration and rotation cases to
`test/test_auto_grasp_lateral.py`. Add a new task to the expected task set in
`test/test_auto_grasp_cli.py`.

Minimum profile assertions:

```python
profile = LATERAL_GRASP_PROFILES["satellite2_left_truss_connection"]
config = config_from_profile(profile, LeftAntennaPanelGrabConfig)

self.assertEqual(config.env_id, "SpaceUR10e-Satellite2-v0")
np.testing.assert_allclose(
    config.panel_offset,
    [-1.26778, 0.476677, 0.426568],
)
self.assertEqual(config.target_frame, "ee")
self.assertEqual(config.rotation_mode, "skip")
```

Tests must import from `planners.auto_grasp`, not load a script by file path.

## Verification

Run these in `myrobot` when available:

```bash
conda run -n myrobot python -m unittest discover \
  -s test -p 'test_auto_grasp*.py' -v

conda run -n myrobot python scripts/auto_grasp.py TASK --help

conda run -n myrobot python -m compileall -q \
  scripts/auto_grasp.py src/planners/auto_grasp test
```

Only run a full MuJoCo GUI trajectory when the user asks or a display is
clearly available.

## Common Mistakes

- Recreating `scripts/autograd/` or adding another task-specific CLI.
- Copying a rotation from another grasp without checking calibration intent.
- Using a world offset when a body-frame offset is available.
- Subclassing the lateral planner for behavior already represented by
  `rotation_mode`.
- Duplicating a motion state machine instead of extending a shared planner.
- Updating a profile without updating the unified CLI tests and README.

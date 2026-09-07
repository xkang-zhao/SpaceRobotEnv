"""Calibrated task data for the lateral automatic grasp planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RotationMode = Literal["absolute", "relative_local_z", "skip"]


@dataclass(frozen=True)
class LateralGraspTaskProfile:
    """Task-specific calibration separated from planner behavior."""

    name: str
    env_id: str
    description: str
    success_label: str
    grasp_offset_body: tuple[float, float, float]
    tool_z_rotation: float
    rotation_mode: RotationMode = "absolute"
    position_tolerance: float = 0.03
    grasp_position_tolerance: float = 0.04
    object_label: str = "target"
    expose_tool_z_rotation: bool = True
    expose_base_orientation: bool = False


LATERAL_GRASP_PROFILES = {
    "satellite_left_antenna_panel": LateralGraspTaskProfile(
        name="satellite_left_antenna_panel",
        env_id="SpaceUR10e-Satellite-v0",
        description="Auto-grasp the satellite left antenna panel.",
        success_label="left antenna panel grasp confirmed",
        grasp_offset_body=(-0.265751, 0.544939, 0.0),
        tool_z_rotation=1.5707963267948966,
        object_label="satellite",
        expose_base_orientation=True,
    ),
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
    "satellite3_left_antenna_panel": LateralGraspTaskProfile(
        name="satellite3_left_antenna_panel",
        env_id="SpaceUR10e-Satellite3-v0",
        description=(
            "Auto-grasp the satellite3 left antenna panel calibrated grasp point."
        ),
        success_label="satellite3 left antenna panel grasp confirmed",
        grasp_offset_body=(-0.776456, 0.576061, -0.007349),
        tool_z_rotation=1.5707963267948966,
        object_label="satellite",
    ),
    "satellite3_upper_rod": LateralGraspTaskProfile(
        name="satellite3_upper_rod",
        env_id="SpaceUR10e-Satellite3-v0",
        description="Auto-grasp the satellite3 upper rod calibrated grasp point.",
        success_label="satellite3 upper rod grasp confirmed",
        grasp_offset_body=(-0.678392, 0.00316, 0.218),
        tool_z_rotation=0.0,
        rotation_mode="skip",
        object_label="satellite",
        expose_tool_z_rotation=False,
    ),
    "debris_antenna_panel": LateralGraspTaskProfile(
        name="debris_antenna_panel",
        env_id="SpaceUR10e-DebrisAntennaPanel-v0",
        description="Auto-grasp the DebrisAntennaPanel calibrated grasp point.",
        success_label="DebrisAntennaPanel grasp confirmed",
        grasp_offset_body=(-0.46839, 0.127419, 0.132495),
        tool_z_rotation=-0.61,
        rotation_mode="relative_local_z",
        object_label="debris",
    ),
    "debris_truss": LateralGraspTaskProfile(
        name="debris_truss",
        env_id="SpaceUR10e-DebrisTruss-v0",
        description="Auto-grasp the DebrisTruss calibrated grasp point.",
        success_label="DebrisTruss grasp confirmed",
        grasp_offset_body=(-0.136453, 0.075573, 0.118878),
        tool_z_rotation=-0.8,
        rotation_mode="relative_local_z",
        position_tolerance=0.015,
        grasp_position_tolerance=0.012,
        object_label="debris",
    ),
}

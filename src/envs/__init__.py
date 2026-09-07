"""Gymnasium registrations for the SpaceUR10e grasping environments."""

from gymnasium import register


ENTRY_POINT = "envs.space_ur10e_env:SpaceUR10eEnv"

COMMON_KWARGS = {
    "render_mode": None,
    "arm_path": "./mjcf/arm.xml",
    "frame_name": "left_attachment",
    "use_depth": False,
}

CUBE_VIEWER_CONFIG = {
    "distance": 5.0,
    "azimuth": -60,
    "elevation": -30,
    "lookat": (1.6, 0.0, 2.65),
}

SATELLITE_VIEWER_CONFIG = {
    "distance": 5.0,
    "azimuth": -60,
    "elevation": -30,
    "lookat": (1.6, 0.0, 2.65),
}


def _register_space_env(
    env_id,
    scene_path,
    x_range,
    y_range=(-0.1, 0.1),
    z_range=(2.65, 2.70),
    viewer_config=SATELLITE_VIEWER_CONFIG,
):
    register(
        id=env_id,
        entry_point=ENTRY_POINT,
        kwargs={
            **COMMON_KWARGS,
            "scene_path": scene_path,
            "target_init_range": {
                "x": x_range,
                "y": y_range,
                "z": z_range,
            },
            "viewer_config": viewer_config,
        },
    )


_register_space_env(
    "SpaceUR10e-Cube-v0",
    "./mjcf/1_cube_scene.xml",
    x_range=(1.5, 1.7),
    y_range=(-0.2, 0.2),
    z_range=(2.6, 2.7),
    viewer_config=CUBE_VIEWER_CONFIG,
)
_register_space_env(
    "SpaceUR10e-Satellite-v0",
    "./mjcf/2_satellite_scene.xml",
    x_range=(1.8, 1.95),
)
_register_space_env(
    "SpaceUR10e-Satellite2-v0",
    "./mjcf/3_satellite_2_scene.xml",
    x_range=(2.65, 2.8),
)
_register_space_env(
    "SpaceUR10e-Satellite3-v0",
    "./mjcf/4_satellite3_scene.xml",
    x_range=(2.15, 2.3),
)
_register_space_env(
    "SpaceUR10e-DebrisAntennaPanel-v0",
    "./mjcf/5_debris_antenna_panel_scene.xml",
    x_range=(1.85, 2.0),
)
_register_space_env(
    "SpaceUR10e-DebrisTruss-v0",
    "./mjcf/6_debris_truss_scene.xml",
    x_range=(1.5, 1.65),
)

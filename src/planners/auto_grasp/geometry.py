"""Shared geometry helpers for automatic grasp planners."""

from __future__ import annotations

import argparse
import ast

import numpy as np


def as_vector3(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values, got {value!r}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values, got {value!r}")
    return vector


def parse_float_list(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise argparse.ArgumentTypeError(f"Expected a list, got {value!r}")
    return [float(item) for item in parsed]


def normalize(vector, fallback):
    norm = np.linalg.norm(vector)
    return fallback.astype(float) if norm < 1e-9 else vector / norm


def rotation_vector(rotation):
    trace_term = np.clip((np.trace(rotation) - 1) / 2, -1, 1)
    angle = float(np.arccos(trace_term))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    return axis / (2 * np.sin(angle)) * angle


def local_z_rotation(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * value * (
        10.0 - 15.0 * value + 6.0 * value * value
    )

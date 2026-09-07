import ast
import importlib.util
import io
import math
import sys
from contextlib import redirect_stdout
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "get_offet_mac.py"


class Vector(list):
    def __setitem__(self, key, value):
        if isinstance(key, slice) and isinstance(value, (int, float)):
            start, stop, step = key.indices(len(self))
            for index in range(start, stop, step):
                super().__setitem__(index, value)
            return
        super().__setitem__(key, value)

    def __sub__(self, other):
        return Vector(a - b for a, b in zip(self, other))

    def __add__(self, other):
        return Vector(a + b for a, b in zip(self, other))

    def reshape(self, *shape):
        if shape == (-1,):
            return Vector(self)
        if shape == (3, 3):
            return Matrix([self[0:3], self[3:6], self[6:9]])
        raise NotImplementedError(shape)


class Matrix(list):
    @property
    def T(self):
        return Matrix([list(row) for row in zip(*self)])

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return Matrix([
                [sum(self[i][k] * other[k][j] for k in range(len(other))) for j in range(len(other[0]))]
                for i in range(len(self))
            ])
        return Vector(sum(row[k] * other[k] for k in range(len(other))) for row in self)

    def reshape(self, *shape):
        flat = Vector(item for row in self for item in row)
        if shape == (-1,):
            return flat
        return flat.reshape(*shape)


def install_dependency_stubs():
    numpy_stub = ModuleType("numpy")
    numpy_stub.float32 = float
    numpy_stub.pi = math.pi
    numpy_stub.cos = math.cos
    numpy_stub.sin = math.sin
    numpy_stub.asarray = lambda values, dtype=None: array(values)
    numpy_stub.array = lambda values, dtype=None: array(values)
    numpy_stub.zeros = lambda size, dtype=None: Vector([0.0] * size)
    numpy_stub.eye = lambda size: Matrix([[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)])
    numpy_stub.array2string = lambda values, precision=6, suppress_small=False: str(list(values))
    numpy_stub.testing = SimpleNamespace(assert_allclose=assert_allclose)
    sys.modules["numpy"] = numpy_stub

    sys.modules.setdefault("gymnasium", ModuleType("gymnasium"))
    sys.modules.setdefault("glfw", SimpleNamespace(PRESS=1, REPEAT=2, get_key=lambda *_args: 0))
    mujoco_stub = ModuleType("mujoco")
    mujoco_stub.mjtCamera = SimpleNamespace(mjCAMERA_FREE=0)
    mujoco_viewer_stub = ModuleType("mujoco.viewer")
    mujoco_stub.viewer = mujoco_viewer_stub
    sys.modules["mujoco"] = mujoco_stub
    sys.modules["mujoco.viewer"] = mujoco_viewer_stub
    sys.modules.setdefault("envs", ModuleType("envs"))


def array(values):
    if isinstance(values, (Vector, Matrix)):
        return values
    values = list(values)
    if values and isinstance(values[0], (list, tuple, Vector)):
        return Matrix([list(row) for row in values])
    return Vector(values)


def assert_allclose(actual, expected, atol=1e-7):
    actual_flat = flatten(actual)
    expected_flat = flatten(expected)
    assert len(actual_flat) == len(expected_flat)
    for actual_item, expected_item in zip(actual_flat, expected_flat):
        assert abs(actual_item - expected_item) <= atol


def flatten(values):
    if isinstance(values, Matrix):
        return [item for row in values for item in row]
    return list(values)


def load_module():
    install_dependency_stubs()
    spec = importlib.util.spec_from_file_location("get_offet_mac", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GetOffetMacTests(unittest.TestCase):
    def test_no_pynput_import_and_held_key_moves_continuously(self):
        module = load_module()
        state = module.OffsetCalibrationState()

        state.set_pressed_keys({"8"})
        first_action = state.consume_action()
        second_action = state.consume_action()

        assert_allclose(first_action[:3], [module.TRAN_STEP, 0.0, 0.0])
        assert_allclose(second_action[:3], [module.TRAN_STEP, 0.0, 0.0])
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        imported_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.append(node.module)
        self.assertNotIn("pynput", imported_names)

    def test_minus_key_toggles_rotation_mode_for_movement_keys(self):
        module = load_module()
        state = module.OffsetCalibrationState()

        with redirect_stdout(io.StringIO()):
            state.handle_key(ord("-"))
        state.set_pressed_keys({"8"})
        rotation_action = state.consume_action()

        assert_allclose(rotation_action[:3], [0.0, 0.0, 0.0])
        assert_allclose(rotation_action[3:6], [module.ROT_STEP, 0.0, 0.0])

        with redirect_stdout(io.StringIO()):
            state.handle_key(ord("-"))
        translation_action = state.consume_action()

        assert_allclose(translation_action[:3], [module.TRAN_STEP, 0.0, 0.0])
        assert_allclose(translation_action[3:6], [0.0, 0.0, 0.0])

    def test_compute_offsets_returns_world_and_body_offsets(self):
        module = load_module()
        satellite_xyz = Vector([1.0, 2.0, 3.0])
        ee_xyz = Vector([1.2, 1.9, 3.4])
        sat_rot = Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        world_offset, body_offset = module.compute_offsets(satellite_xyz, ee_xyz, sat_rot)

        assert_allclose(world_offset, [0.2, -0.1, 0.4])
        assert_allclose(body_offset, [0.2, -0.1, 0.4])

    def test_compute_relative_rotation_returns_body_to_ee_rotation(self):
        module = load_module()
        target_rot = Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        angle = math.pi / 2
        ee_rot = Matrix([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])

        relative_rot = module.compute_relative_rotation(target_rot, ee_rot)

        assert_allclose(relative_rot, ee_rot, atol=1e-12)

    def test_print_offsets_includes_orientation_information(self):
        module = load_module()
        target_rot = Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ee_rot = Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        obs = {
            "target_pose": Vector([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]),
            "ee_pose": Vector([1.2, 1.9, 3.4, 1.0, 0.0, 0.0, 0.0]),
        }
        data = SimpleNamespace(
            xmat=Matrix([target_rot.reshape(-1)]),
            site_xmat=Matrix([ee_rot.reshape(-1)]),
        )
        unwrapped = SimpleNamespace(data=data, target_body_id=0, ee_site_id=0)
        env = SimpleNamespace(unwrapped=unwrapped)
        output = io.StringIO()

        with redirect_stdout(output):
            module.print_offsets(env, obs)

        text = output.getvalue()
        self.assertIn("卫星姿态 quat", text)
        self.assertIn("机器人末端姿态 quat", text)
        self.assertIn("相对姿态 RPY", text)


if __name__ == "__main__":
    unittest.main()

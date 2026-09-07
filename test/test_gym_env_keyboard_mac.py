import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "keyboard" / "gym_env_keyboard_mac.py"


def load_mac_keyboard_module():
    spec = importlib.util.spec_from_file_location("gym_env_keyboard_mac", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MacKeyboardTeleopTests(unittest.TestCase):
    def test_key_callback_updates_pending_action_without_pynput(self):
        module = load_mac_keyboard_module()
        state = module.KeyboardTeleopState()

        state.handle_key(ord("8"))
        action = state.consume_action()

        np.testing.assert_allclose(action[:3], [module.TRAN_STEP, 0.0, 0.0])
        self.assertEqual(action[6], 0.0)
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        imported_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.append(node.module)
        self.assertNotIn("pynput", imported_names)

    def test_held_key_continues_action_until_released(self):
        module = load_mac_keyboard_module()
        state = module.KeyboardTeleopState()

        state.set_pressed_keys({"8"})

        first_action = state.consume_action()
        second_action = state.consume_action()
        np.testing.assert_allclose(first_action[:3], [module.TRAN_STEP, 0.0, 0.0])
        np.testing.assert_allclose(second_action[:3], [module.TRAN_STEP, 0.0, 0.0])

        state.set_pressed_keys(set())
        released_action = state.consume_action()
        np.testing.assert_allclose(released_action[:3], [0.0, 0.0, 0.0])

    def test_quartz_poll_updates_held_movement_keys_without_glfw_window(self):
        module = load_mac_keyboard_module()
        state = module.KeyboardTeleopState()

        def key_state(_source, key_code):
            return key_code == module.MAC_KEY_8

        self.assertTrue(module.poll_quartz_pressed_keys(state, key_state_getter=key_state))
        action = state.consume_action()
        np.testing.assert_allclose(action[:3], [module.TRAN_STEP, 0.0, 0.0])

    def test_gripper_toggle_and_escape_state(self):
        module = load_mac_keyboard_module()
        state = module.KeyboardTeleopState()

        state.handle_key(ord("1"))
        self.assertEqual(state.gripper_signal, 1.0)

        state.handle_key(ord("1"))
        self.assertEqual(state.gripper_signal, -1.0)

        state.handle_key(module.KEY_ESCAPE)
        self.assertFalse(state.running)


if __name__ == "__main__":
    unittest.main()

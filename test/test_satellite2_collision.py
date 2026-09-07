import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT_ROOT / "mjcf/3_satellite_2_scene.xml"
CONNECTION_POINT = np.array([-1.1002, 0.4727, 0.3262])


class Satellite2CollisionTests(unittest.TestCase):
    def test_left_truss_connection_has_collidable_geometry(self):
        root = ET.parse(SCENE_PATH).getroot()
        geom = root.find(
            ".//geom[@name='satellite_2_left_truss_connection_collision']"
        )

        self.assertIsNotNone(geom)
        self.assertEqual(geom.get("type"), "box")
        self.assertEqual(geom.get("contype"), "1")
        self.assertEqual(geom.get("conaffinity"), "1")
        self.assertEqual(geom.get("mass"), "0")

        position = np.fromstring(geom.get("pos"), sep=" ")
        half_size = np.fromstring(geom.get("size"), sep=" ")
        self.assertTrue(
            np.all(CONNECTION_POINT >= position - half_size)
        )
        self.assertTrue(
            np.all(CONNECTION_POINT <= position + half_size)
        )


if __name__ == "__main__":
    unittest.main()

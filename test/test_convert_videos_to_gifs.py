import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.record.convert_videos_to_gifs import (
    build_filter,
    load_video_paths,
    validate_options,
)


class GifConversionTests(unittest.TestCase):
    def test_filter_uses_ppt_defaults_and_palette_optimization(self):
        value = build_filter(width=960, fps=10, colors=128)
        self.assertIn("fps=10", value)
        self.assertIn("scale=960:-2:flags=lanczos", value)
        self.assertIn("palettegen=max_colors=128", value)
        self.assertIn("dither=sierra2_4a", value)
        self.assertIn("diff_mode=rectangle", value)

    def test_options_require_valid_width_fps_and_palette(self):
        validate_options(width=960, fps=10, colors=128)
        with self.assertRaisesRegex(ValueError, "positive even"):
            validate_options(width=959, fps=10, colors=128)
        with self.assertRaisesRegex(ValueError, "FPS"):
            validate_options(width=960, fps=0, colors=128)
        with self.assertRaisesRegex(ValueError, "between 2 and 256"):
            validate_options(width=960, fps=10, colors=1)

    def test_manifest_controls_source_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "second.mp4").touch()
            (root / "first.mp4").touch()
            (root / "manifest.json").write_text(
                '{"tasks": ['
                '{"task_name": "first"}, '
                '{"task_name": "second"}]}',
                encoding="utf-8",
            )
            self.assertEqual(
                [path.name for path in load_video_paths(root)],
                ["first.mp4", "second.mp4"],
            )


if __name__ == "__main__":
    unittest.main()

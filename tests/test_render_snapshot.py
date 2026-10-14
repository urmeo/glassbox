"""Rendering uses one detached presentation for text, HTML, and structured input."""

import copy
import tempfile
import unittest
from unittest.mock import patch

from glassbox import interfaces, render, schema


class TestRenderSnapshot(unittest.TestCase):
    def test_lossy_numeric_display_is_rejected_before_capture(self):
        presentation = interfaces.variant(schema.load_all_scenarios()["loans"], "table")
        for skip in (True, False):
            with (
                self.subTest(skip=skip),
                patch("glassbox.interfaces._num", return_value="0"),
                patch("glassbox.render.chrome_available", return_value=True),
                patch("glassbox.render._screenshot") as capture,
            ):
                with self.assertRaisesRegex(ValueError, "numeric display"):
                    render.render(presentation, skip=skip)
                capture.assert_not_called()

    def test_caller_mutation_during_capture_does_not_change_recorded_input(self):
        scenario = schema.load_all_scenarios()["loans"]
        presentation = interfaces.variant(scenario, "table")
        expected = copy.deepcopy(presentation)

        def screenshot(document, destination):
            self.assertEqual(document, expected.to_html())
            presentation.shown["A"]["total_cost"] = -1
            presentation.order.reverse()
            presentation.labels["A"] = "Changed after capture started"
            return {"width": 820, "height": 620, "sha256": "a" * 64,
                    "completion": "natural_exit"}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("glassbox.render.chrome_available", return_value=True),
            patch("glassbox.render.renderer_version", return_value="test-browser"),
            patch("glassbox.render._screenshot", side_effect=screenshot) as capture,
        ):
            stimulus = render.render(presentation, out_dir=directory)
        capture.assert_called_once()
        self.assertEqual(stimulus.presentation, expected)
        self.assertEqual(stimulus.text, expected.to_text())
        self.assertEqual(stimulus.text, stimulus.presentation.to_text())
        self.assertNotEqual(stimulus.presentation.shown, presentation.shown)


if __name__ == "__main__":
    unittest.main()

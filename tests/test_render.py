"""Rendering — skip-render degradation and clean RenderError on Chrome failure."""

import os
import tempfile
import unittest

from glassbox import interfaces, render, schema


class TestRender(unittest.TestCase):
    def setUp(self):
        self.loans = schema.load_all_scenarios()["loans"]

    def test_skip_render_needs_no_chrome(self):
        st = render.render(interfaces.variant(self.loans, "table"), skip=True)
        self.assertEqual(st.kind, "spec")
        self.assertIsNone(st.image_path)

    def test_screenshot_wraps_exec_failure_as_render_error(self):
        # A bad Chrome path makes subprocess.run raise OSError; it must surface as a
        # RenderError (which the CLI catches), not an unwrapped traceback.
        p = interfaces.variant(self.loans, "table")
        saved = os.environ.get("GLASSBOX_CHROME")
        os.environ["GLASSBOX_CHROME"] = "/nonexistent/chrome-binary-xyz"
        try:
            with self.assertRaises(render.RenderError):
                render._screenshot(p.to_html(), os.path.join(tempfile.mkdtemp(), "x.png"))
        finally:
            if saved is None:
                os.environ.pop("GLASSBOX_CHROME", None)
            else:
                os.environ["GLASSBOX_CHROME"] = saved


if __name__ == "__main__":
    unittest.main()

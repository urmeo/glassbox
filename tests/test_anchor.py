"""H3 anchoring — anchor sets, model-vs-human correlation, fixture honesty."""

import copy
import os
import tempfile
import unittest

import glassbox
from glassbox import anchor
from glassbox.cli import main
from glassbox.readers import build_readers

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(glassbox.__file__))))
FIXTURE = os.path.join(ROOT, "data", "anchors", "fixture.json")

VALID = {
    "id": "t", "source": "fixture test", "is_fixture": True, "condition": "raw",
    "items": [{"scenario": "loans", "question": "cheapest_total", "human_accuracy": 0.5}],
}


class TestAnchorSet(unittest.TestCase):
    def test_load_fixture(self):
        aset = anchor.load_anchor_set(FIXTURE)
        self.assertTrue(aset.is_fixture)
        self.assertEqual(len(aset.items), 9)
        self.assertEqual(aset.condition, "raw")

    def test_unknown_scenario_rejected(self):
        bad = copy.deepcopy(VALID)
        bad["items"][0]["scenario"] = "nope"
        with self.assertRaises(anchor.AnchorError):
            anchor.parse_anchor_set(bad)

    def test_unknown_question_rejected(self):
        bad = copy.deepcopy(VALID)
        bad["items"][0]["question"] = "nope"
        with self.assertRaises(anchor.AnchorError):
            anchor.parse_anchor_set(bad)

    def test_out_of_range_human_accuracy_rejected(self):
        bad = copy.deepcopy(VALID)
        bad["items"][0]["human_accuracy"] = 1.5
        with self.assertRaises(anchor.AnchorError):
            anchor.parse_anchor_set(bad)


class TestRunAnchor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        aset = anchor.load_anchor_set(FIXTURE)
        cls.result = anchor.run_anchor(aset, build_readers(["simulated"]), skip_render=True)

    def test_correlations_computed(self):
        self.assertEqual(self.result.n_items, 9)
        for c in (self.result.spearman, self.result.pearson, self.result.kendall):
            self.assertEqual(c, c)  # not nan
            self.assertGreaterEqual(c, -1.0)
            self.assertLessEqual(c, 1.0)

    def test_breaks_identifies_largest_gap(self):
        top = self.result.breaks(1)[0]
        self.assertEqual((top.scenario, top.question), ("growth", "biggest_percent_gain"))

    def test_cli_anchor_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["anchor", "--set", FIXTURE, "--skip-render", "--out", tmp]), 0)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "anchor.md")))


if __name__ == "__main__":
    unittest.main()

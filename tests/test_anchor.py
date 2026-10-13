"""H3 anchoring; anchor sets, model-vs-human correlation, fixture honesty."""

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

from glassbox import anchor
from glassbox.cli import main
from glassbox.readers import build_readers

FIXTURE = "fixture"

VALID = {
    "id": "t",
    "source": "fixture test",
    "is_fixture": True,
    "condition": "raw",
    "items": [
        {"scenario": "loans", "question": "cheapest_total", "human_accuracy": 0.5}
    ],
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

    def test_is_fixture_defaults_true(self):
        raw = copy.deepcopy(VALID)
        del raw["is_fixture"]
        self.assertTrue(anchor.parse_anchor_set(raw).is_fixture)


class TestRunAnchor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        aset = anchor.load_anchor_set(FIXTURE)
        cls.result = anchor.run_anchor(
            aset, build_readers(["simulated"]), skip_render=True
        )

    def test_correlations_computed(self):
        self.assertEqual(self.result.n_items, 9)
        for c in (self.result.spearman, self.result.pearson, self.result.kendall):
            self.assertEqual(c, c)
            self.assertGreaterEqual(c, -1.0)
            self.assertLessEqual(c, 1.0)

    def test_breaks_identifies_largest_gap(self):
        top = self.result.breaks(1)[0]
        self.assertEqual(
            (top.scenario, top.question), ("growth", "biggest_percent_gain")
        )

    def test_subset_asks_only_declared_questions_and_replicates(self):
        anchor_set = anchor.parse_anchor_set(VALID)
        reader = build_readers(["simulated:diligent"])[0]
        with patch.object(reader, "answer", wraps=reader.answer) as answer:
            result = anchor.run_anchor(anchor_set, [reader], skip_render=True, replicates=2)
        self.assertEqual(answer.call_count, 2)
        self.assertEqual(
            [call.args[1].id for call in answer.call_args_list],
            ["cheapest_total", "cheapest_total"],
        )
        self.assertEqual(result.n_items, 1)
        self.assertEqual(result.points[0].model_accuracy, 1.0)

    def test_missing_reader_answers_do_not_become_zero_accuracy(self):
        anchor_set = anchor.parse_anchor_set(VALID)
        with (
            patch("glassbox.anchor.scoring.read_all", return_value=[]),
            self.assertRaisesRegex(anchor.AnchorError, "no reader answers"),
        ):
            anchor.run_anchor(anchor_set, build_readers(["simulated"]), skip_render=True)

    def test_cli_anchor_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                main(["anchor", "--set", FIXTURE, "--skip-render", "--out", tmp]), 0
            )
            self.assertTrue(os.path.isfile(os.path.join(tmp, "anchor.md")))


if __name__ == "__main__":
    unittest.main()

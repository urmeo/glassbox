"""Strict result files and provenance."""

import json
import io
import os
import tempfile
import unittest

from glassbox import analysis, report, schema, scoring
from glassbox.judge import (
    PairwiseJudge,
    PairwiseJudgment,
    PairwiseRatingJudge,
    SimulatedJudge,
)
from glassbox.readers import build_readers


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(schema.load_all_scenarios().values())
        cls.results = scoring.read_all(
            cls.scenarios, build_readers(["simulated"]), skip_render=True
        )
        cls.book = scoring.ScoreBook(cls.results)
        cls.h1 = analysis.analyze_h1(cls.book, cls.scenarios, SimulatedJudge())
        cls.cross = analysis.analyze_cross_family(
            cls.book, cls.scenarios, SimulatedJudge()
        )

    def test_results_json_is_serializable_and_shaped(self):
        payload = report.build_results(
            self.scenarios,
            self.results,
            self.book,
            self.h1,
            self.cross,
            skip_render=True,
        )
        restored = json.loads(json.dumps(payload))
        self.assertIn("meta", restored)
        self.assertIn("metrics", restored)
        self.assertIn("h1_summary", restored)
        self.assertIn("cross_family", restored)
        self.assertTrue(restored["h1_summary"]["divergence_found"])
        self.assertEqual(restored["meta"]["renderer"], "skip-render")
        for s in self.scenarios:
            self.assertIn(s.id, restored["meta"]["scenarios"])

    def test_report_md_has_caveat_and_reversal(self):
        md = report.render_report_md(self.book, self.h1, self.cross)
        self.assertIn("Simulated readers only", md)
        self.assertIn("Reversal", md)
        self.assertIn("MCQ lift vs synthetic polish", md)
        self.assertIn("Synthetic judge", md)
        self.assertIn("Cross-family agreement", md)

    def test_results_json_has_no_bare_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            report.write_run(
                tmp, self.scenarios, self.results, self.book, self.h1, self.cross, True
            )
            with open(os.path.join(tmp, "results.json"), encoding="utf-8") as fh:
                raw = fh.read()
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)

        def _reject(token):
            raise ValueError("non-standard JSON constant: " + token)

        json.loads(raw, parse_constant=_reject)

    def test_scenario_hash_stable(self):
        loans = schema.load_all_scenarios()["loans"]
        h1 = report._scenario_hash(loans)
        h2 = report._scenario_hash(schema.load_all_scenarios()["loans"])
        self.assertEqual(h1, h2)

    def test_missing_spread_serializes_as_null(self):
        stream = io.StringIO()
        report._dump({"spread": self.book.accuracy_std("missing-reader", "loans", "cards")}, stream)
        self.assertEqual(json.loads(stream.getvalue()), {"spread": None})
        self.assertNotIn("NaN", stream.getvalue())

    def test_undefined_lift_is_labeled_in_markdown(self):
        from dataclasses import replace

        scenario = self.h1.scenarios[0]
        row = replace(scenario.rows[0], comp_lift=float("nan"))
        summary = replace(self.h1, scenarios=[replace(scenario, rows=[row])])
        text = report.render_report_md(self.book, summary)
        self.assertIn("undefined", text)
        self.assertNotIn("nan pts", text)

    def test_all_invalid_judgments_stay_missing_through_analysis_and_reports(self):
        class InvalidJudge(PairwiseJudge):
            name = "simulated:invalid"
            simulated = True
            deterministic = True

            def compare(self, scenario, a, b):
                return PairwiseJudgment("invalid", raw="A or B")

        judge = PairwiseRatingJudge(InvalidJudge())
        h1 = analysis.analyze_h1(self.book, self.scenarios, judge)
        cross = analysis.analyze_cross_family(self.book, self.scenarios, judge)
        self.assertFalse(h1.divergence_found)
        self.assertFalse(cross.survives_across_families)

        with tempfile.TemporaryDirectory() as directory:
            paths = report.write_run(
                directory, self.scenarios, self.results, self.book, h1, cross, True
            )
            with open(paths["results"], encoding="utf-8") as stream:
                raw = stream.read()
            with open(paths["report"], encoding="utf-8") as stream:
                markdown = stream.read()

        def reject_constant(token):
            raise ValueError("nonstandard JSON constant: " + token)

        results = json.loads(raw, parse_constant=reject_constant)
        self.assertFalse(results["cross_family"]["reversal_comparison_available"])
        self.assertIsNone(results["h1_summary"]["mean_spearman"])
        self.assertFalse(results["h1_summary"]["divergence_found"])
        self.assertTrue(all(
            value is None for value in results["cross_family"]["preference"].values()
        ))
        for scenario in results["metrics"].values():
            self.assertEqual(scenario["h1"]["reversals"], [])
            for row in scenario["h1"]["rows"]:
                self.assertIsNone(row["pref_score"])
                self.assertIsNone(row["pref_rank"])
                self.assertIsNotNone(row["comp_rank"])

        metadata = results["h1_summary"]["judge"]
        self.assertEqual(metadata["comparison_count"], 18)
        self.assertEqual(metadata["invalid_count"], 18)
        self.assertEqual(metadata["valid_count"], 0)
        self.assertEqual(metadata["tie_count"], 0)
        self.assertEqual(metadata["abstained_count"], 0)
        self.assertEqual(len(metadata["evidence"]), 3)
        for evidence in metadata["evidence"]:
            for score in evidence["variant_scores"]:
                self.assertEqual(score["credit"], 0)
                self.assertEqual(score["denominator"], 0)
                self.assertEqual(score["planned_count"], 4)
                self.assertIsNone(score["score"])
            for comparison in evidence["comparisons"]:
                self.assertEqual(comparison["raw"], "A or B")
                self.assertEqual(comparison["status"], "invalid")
        self.assertNotIn("nan", markdown.lower())
        self.assertIn("judge scores unavailable", markdown)
        self.assertIn("Too few scored interfaces to compare rankings.", markdown)
        self.assertIn("Any reversal in every resolved family: **unavailable**", markdown)
        self.assertNotIn("**Reversal:**", markdown)


if __name__ == "__main__":
    unittest.main()

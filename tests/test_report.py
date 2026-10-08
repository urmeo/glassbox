"""Strict result files and provenance."""

import json
import io
import os
import tempfile
import unittest

from glassbox import analysis, report, schema, scoring
from glassbox.judge import SimulatedJudge
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


if __name__ == "__main__":
    unittest.main()

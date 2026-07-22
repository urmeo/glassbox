"""Report writers — results.json shape, report.md content, provenance."""

import json
import unittest

from glassbox import analysis, report, schema, scoring
from glassbox.judge import SimulatedJudge
from glassbox.readers import build_readers


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(schema.load_all_scenarios().values())
        cls.results = scoring.read_all(cls.scenarios, build_readers(["simulated"]),
                                       skip_render=True)
        cls.book = scoring.ScoreBook(cls.results)
        cls.h1 = analysis.analyze_h1(cls.book, cls.scenarios, SimulatedJudge())

    def test_results_json_is_serializable_and_shaped(self):
        payload = report.build_results(self.scenarios, self.results, self.book,
                                       self.h1, skip_render=True)
        # round-trips through JSON
        restored = json.loads(json.dumps(payload))
        self.assertIn("meta", restored)
        self.assertIn("metrics", restored)
        self.assertIn("h1_summary", restored)
        self.assertTrue(restored["h1_summary"]["divergence_found"])
        self.assertEqual(restored["meta"]["renderer"], "skip-render")
        # every scenario has a provenance hash
        for s in self.scenarios:
            self.assertIn(s.id, restored["meta"]["scenarios"])

    def test_report_md_has_caveat_and_reversal(self):
        md = report.render_report_md(self.book, self.h1)
        self.assertIn("Simulated readers only", md)
        self.assertIn("Reversal", md)
        self.assertIn("comprehension vs preference", md)

    def test_scenario_hash_stable(self):
        loans = schema.load_all_scenarios()["loans"]
        h1 = report._scenario_hash(loans)
        h2 = report._scenario_hash(schema.load_all_scenarios()["loans"])
        self.assertEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()

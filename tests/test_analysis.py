"""H1 analysis — Spearman correctness and the divergence finding."""

import unittest

from glassbox import analysis, schema, scoring
from glassbox.judge import SimulatedJudge
from glassbox.readers import build_readers

try:
    from scipy.stats import spearmanr as _scipy_spearman
    HAVE_SCIPY = True
except Exception:  # scipy is an optional cross-check only (D-0001)
    HAVE_SCIPY = False


class TestSpearman(unittest.TestCase):
    def test_perfect_and_inverse(self):
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_with_ties(self):
        # constant second vector -> undefined correlation (nan), must not crash
        self.assertNotEqual(analysis.spearman([1, 2, 2, 3], [1, 1, 1, 1]),
                            analysis.spearman([1, 2, 2, 3], [1, 1, 1, 1]))  # nan != nan

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed (optional cross-check)")
    def test_matches_scipy(self):
        cases = [
            ([0.1, 0.5, 0.9, 0.3], [3.0, 2.0, 1.0, 4.0]),
            ([-0.44, 0.33, 0.44], [0.605, 0.290, 0.760]),  # loans comp vs pref
            ([1.0, 2.0, 2.0, 4.0, 5.0], [5.0, 4.0, 4.0, 2.0, 1.0]),  # ties both sides
        ]
        for xs, ys in cases:
            mine = analysis.spearman(xs, ys)
            theirs = float(_scipy_spearman(xs, ys).statistic)
            self.assertAlmostEqual(mine, theirs, places=9, msg="xs=%s ys=%s" % (xs, ys))


class TestH1Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(schema.load_all_scenarios().values())
        book = scoring.ScoreBook(
            scoring.read_all(cls.scenarios, build_readers(["simulated"]), skip_render=True))
        cls.report = analysis.analyze_h1(book, cls.scenarios, SimulatedJudge())

    def test_divergence_found(self):
        self.assertTrue(self.report.divergence_found)

    def test_labeled_simulated(self):
        self.assertTrue(self.report.simulated)

    def test_cards_over_table_reversal_in_every_scenario(self):
        for s in self.report.scenarios:
            pairs = {(r.preferred, r.understood) for r in s.reversals}
            self.assertIn(("cards", "table"), pairs,
                          "%s: expected cards-preferred-but-table-understood reversal" % s.scenario_id)

    def test_preference_and_comprehension_disagree(self):
        # Weak-to-moderate correlation, never a perfect match.
        self.assertLess(self.report.mean_spearman, 0.99)


if __name__ == "__main__":
    unittest.main()

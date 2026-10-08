"""Correlation and reversal fixtures."""

import unittest
from dataclasses import replace

from glassbox import analysis, schema, scoring
from glassbox.judge import SimulatedJudge
from glassbox.readers import build_readers

try:
    from scipy.stats import spearmanr as _scipy_spearman
    from scipy.stats import kendalltau as _scipy_kendall
    from scipy.stats import pearsonr as _scipy_pearson

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


class _TruthValue:
    def __init__(self, value):
        self.value = bool(value)

    def __bool__(self):
        return self.value

    def __int__(self):
        return int(self.value)


class _ScalarFloat(float):
    def __gt__(self, other):
        return _TruthValue(super().__gt__(other))

    def __lt__(self, other):
        return _TruthValue(super().__lt__(other))


class TestSpearman(unittest.TestCase):
    def test_perfect_and_inverse(self):
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(analysis.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_with_ties(self):
        self.assertNotEqual(
            analysis.spearman([1, 2, 2, 3], [1, 1, 1, 1]),
            analysis.spearman([1, 2, 2, 3], [1, 1, 1, 1]),
        )

    def test_kendall_accepts_scalar_truth_values(self):
        xs = [_ScalarFloat(value) for value in [1, 2, 3]]
        ys = [_ScalarFloat(value) for value in [2, 1, 3]]
        self.assertAlmostEqual(analysis.kendall_tau(xs, ys), 1 / 3)

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed (optional cross-check)")
    def test_matches_scipy(self):
        cases = [
            ([0.1, 0.5, 0.9, 0.3], [3.0, 2.0, 1.0, 4.0]),
            ([-0.44, 0.33, 0.44], [0.605, 0.290, 0.760]),
            ([1.0, 2.0, 2.0, 4.0, 5.0], [5.0, 4.0, 4.0, 2.0, 1.0]),
        ]
        for xs, ys in cases:
            mine = analysis.spearman(xs, ys)
            theirs = float(_scipy_spearman(xs, ys).statistic)
            self.assertAlmostEqual(mine, theirs, places=9, msg="xs=%s ys=%s" % (xs, ys))

    @unittest.skipUnless(HAVE_SCIPY, "scipy not installed (optional cross-check)")
    def test_kendall_and_pearson_match_scipy(self):
        cases = [
            ([0.1, 0.5, 0.9, 0.3, 0.6], [3.0, 2.0, 1.0, 4.0, 2.5]),
            (
                [0.33, 0.67, 0.67, 0.67, 0.4],
                [0.45, 0.70, 0.62, 0.68, 0.40],
            ),
        ]
        for xs, ys in cases:
            self.assertAlmostEqual(
                analysis.kendall_tau(xs, ys),
                float(_scipy_kendall(xs, ys).statistic),
                places=9,
            )
            self.assertAlmostEqual(
                analysis.pearson(xs, ys), float(_scipy_pearson(xs, ys)[0]), places=9
            )


class TestH1Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(schema.load_all_scenarios().values())
        book = scoring.ScoreBook(
            scoring.read_all(
                cls.scenarios, build_readers(["simulated"]), skip_render=True
            )
        )
        cls.report = analysis.analyze_h1(book, cls.scenarios, SimulatedJudge())

    def test_divergence_found(self):
        self.assertTrue(self.report.divergence_found)

    def test_labeled_simulated(self):
        self.assertTrue(self.report.simulated)

    def test_cards_over_table_reversal_in_every_scenario(self):
        for s in self.report.scenarios:
            pairs = {(r.preferred, r.understood) for r in s.reversals}
            self.assertIn(
                ("cards", "table"),
                pairs,
                "%s: expected cards-preferred-but-table-understood reversal"
                % s.scenario_id,
            )

    def test_preference_and_comprehension_disagree(self):
        self.assertLess(self.report.mean_spearman, 0.99)


class TestCrossFamily(unittest.TestCase):
    """Cross-family logic, exercised by relabeling personas into two families."""

    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(schema.load_all_scenarios().values())
        results = scoring.read_all(
            cls.scenarios, build_readers(["simulated"]), skip_render=True
        )
        mapping = {
            "simulated:literal": "famA",
            "simulated:diligent": "famA",
            "simulated:careless": "famB",
        }
        relabeled = [
            replace(
                qr,
                reader_model_family=mapping[qr.reader],
                reader_family_resolved=True,
                reader_family_basis="synthetic-fixture",
            )
            for qr in results
        ]
        cls.book = scoring.ScoreBook(relabeled)
        cls.cross = analysis.analyze_cross_family(
            cls.book, cls.scenarios, SimulatedJudge()
        )

    def test_two_families_detected(self):
        self.assertEqual(self.cross.n_families, 2)
        self.assertEqual(
            sorted(f.family for f in self.cross.families), ["famA", "famB"]
        )

    def test_reversal_holds_in_each_family(self):
        for f in self.cross.families:
            self.assertTrue(
                f.reversal_holds, "reversal missing in family %s" % f.family
            )

    def test_divergence_survives_and_families_agree(self):
        self.assertTrue(self.cross.survives_across_families)
        self.assertGreater(self.cross.cross_family_agreement, 0.0)

    def test_single_family_cannot_survive(self):
        book = scoring.ScoreBook(
            scoring.read_all(
                self.scenarios, build_readers(["simulated"]), skip_render=True
            )
        )
        cross = analysis.analyze_cross_family(book, self.scenarios, SimulatedJudge())
        self.assertEqual(cross.n_families, 1)
        self.assertFalse(cross.survives_across_families)


if __name__ == "__main__":
    unittest.main()

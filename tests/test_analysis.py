"""Correlation and reversal fixtures."""

import math
import unittest
from dataclasses import replace

from glassbox import analysis, schema, scoring
from glassbox.judge import Judge, SimulatedJudge
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


class _FixedWidthInt(int):
    """Model the wrapping subtraction of a signed 64-bit integer scalar."""

    def __sub__(self, other):
        difference = int(self) - int(other)
        return (difference + 2**63) % 2**64 - 2**63


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

    def test_pearson_centers_fixed_width_integers_without_overflow(self):
        values = [-9_000_000_000_000_000_000, 0, 9_000_000_000_000_000_000]
        scalars = [_FixedWidthInt(value) for value in values]
        target = [-2.0, 0.0, 1.0]
        self.assertAlmostEqual(
            analysis.pearson(scalars, target), analysis.pearson(values, target)
        )

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
        self.assertTrue(self.cross.reversal_comparison_available)
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
        self.assertFalse(cross.reversal_comparison_available)
        self.assertFalse(cross.survives_across_families)

    def test_missing_judge_scores_make_comparison_unavailable(self):
        for scores in ({}, {"cards": 0.5}):
            with self.subTest(scores=scores):
                cross = analysis.analyze_cross_family(
                    self.book, self.scenarios,
                    _ScoreJudge({scenario.id: scores for scenario in self.scenarios}),
                )
                self.assertFalse(cross.reversal_comparison_available)
                self.assertFalse(cross.survives_across_families)

    def test_missing_family_lifts_make_comparison_unavailable(self):
        book = scoring.ScoreBook([
            row for row in self.book.results
            if not (row.reader_model_family == "famB" and row.variant == "baseline")
        ])
        cross = analysis.analyze_cross_family(book, self.scenarios, SimulatedJudge())
        self.assertEqual(cross.n_families, 2)
        self.assertFalse(cross.reversal_comparison_available)
        self.assertFalse(cross.survives_across_families)

    def test_comparison_needs_finite_scores_on_the_same_views(self):
        book = scoring.ScoreBook([
            row for row in self.book.results
            if not (row.reader_model_family == "famB" and row.variant == "annotated")
        ])
        judge = _ScoreJudge({
            scenario.id: {"table": 0.5, "annotated": 0.5}
            for scenario in self.scenarios
        })
        cross = analysis.analyze_cross_family(book, self.scenarios, judge)
        self.assertFalse(cross.reversal_comparison_available)

    def test_finite_ties_allow_comparison_without_a_reversal(self):
        judge = _ScoreJudge({
            scenario.id: {"cards": 0.5, "annotated": 0.5}
            for scenario in self.scenarios
        })
        cross = analysis.analyze_cross_family(self.book, self.scenarios, judge)
        self.assertTrue(cross.reversal_comparison_available)
        self.assertFalse(cross.survives_across_families)
        self.assertEqual(cross.common_reversals, [])


class _ScoreJudge(Judge):
    simulated = True
    deterministic = True

    def __init__(self, scores):
        self.scores = scores

    def preference(self, scenario, presentation):
        return self.scores.get(scenario.id, {}).get(presentation.variant, float("nan"))


class TestMissingJudgeScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = schema.load_all_scenarios()
        cls.book = scoring.ScoreBook(
            scoring.read_all(
                list(cls.scenarios.values()), build_readers(["simulated"]), skip_render=True
            )
        )

    def test_invalid_only_judge_has_no_judge_ranks_or_reversals(self):
        scenarios = list(self.scenarios.values())
        judge = _ScoreJudge({})
        result = analysis.analyze_h1(self.book, scenarios, judge)
        self.assertTrue(math.isnan(result.mean_spearman))
        self.assertFalse(result.divergence_found)
        for scenario in result.scenarios:
            self.assertTrue(math.isnan(scenario.spearman))
            self.assertEqual(scenario.reversals, [])
            for row in scenario.rows:
                self.assertTrue(math.isnan(row.pref_score))
                self.assertIsNone(row.pref_rank)
                self.assertIsNotNone(row.comp_rank)
        cross = analysis.analyze_cross_family(self.book, scenarios, judge)
        self.assertTrue(all(math.isnan(score) for score in cross.preference.values()))
        self.assertFalse(cross.survives_across_families)
        self.assertEqual(cross.common_reversals, [])

    def test_partial_scores_rank_and_correlate_only_finite_pairs(self):
        judge = _ScoreJudge({"loans": {"cards": 0.9, "annotated": 0.1}})
        result = analysis.analyze_h1(self.book, [self.scenarios["loans"]], judge)
        scenario = result.scenarios[0]
        rows = {row.variant: row for row in scenario.rows}
        self.assertEqual(rows["cards"].pref_rank, 1)
        self.assertIsNone(rows["table"].pref_rank)
        self.assertEqual(rows["annotated"].pref_rank, 2)
        self.assertAlmostEqual(scenario.spearman, -1.0)
        self.assertEqual(
            [(reversal.preferred, reversal.understood) for reversal in scenario.reversals],
            [("cards", "annotated")],
        )

    def test_constant_finite_scores_stay_tied_with_undefined_correlation(self):
        judge = _ScoreJudge({"loans": {"cards": 0.5, "annotated": 0.5}})
        result = analysis.analyze_h1(self.book, [self.scenarios["loans"]], judge)
        scenario = result.scenarios[0]
        self.assertEqual([row.pref_rank for row in scenario.rows], [1, None, 1])
        self.assertTrue(math.isnan(scenario.spearman))
        self.assertEqual(scenario.reversals, [])

    def test_cross_family_does_not_average_over_missing_scenarios(self):
        judge = _ScoreJudge({
            "loans": {"cards": 0.9, "table": 0.2, "annotated": 0.6},
            "plans": {"table": 0.4, "annotated": 0.8},
        })
        scenarios = [self.scenarios["loans"], self.scenarios["plans"]]
        cross = analysis.analyze_cross_family(self.book, scenarios, judge)
        self.assertTrue(math.isnan(cross.preference["cards"]))
        self.assertAlmostEqual(cross.preference["table"], 0.3)
        self.assertAlmostEqual(cross.preference["annotated"], 0.7)
        self.assertTrue(
            all("cards" not in pair for family in cross.families for pair in family.reversals)
        )

    def test_missing_baseline_has_no_lift_rank_or_zero_family_lift(self):
        book = scoring.ScoreBook([
            row for row in self.book.results if row.variant != "baseline"
        ])
        scenarios = list(self.scenarios.values())
        result = analysis.analyze_h1(book, scenarios, SimulatedJudge())
        for scenario in result.scenarios:
            self.assertTrue(math.isnan(scenario.spearman))
            self.assertEqual(scenario.reversals, [])
            self.assertTrue(all(row.comp_rank is None for row in scenario.rows))
        cross = analysis.analyze_cross_family(book, scenarios, SimulatedJudge())
        self.assertTrue(
            all(math.isnan(lift) for family in cross.families for lift in family.comp_lift.values())
        )

    def test_missing_reader_cell_does_not_break_agreement(self):
        book = scoring.ScoreBook([
            row for row in self.book.results
            if not (row.scenario_id == "loans"
                    and row.reader == "simulated:literal"
                    and row.variant == "cards")
        ])
        result = analysis.analyze_h1(book, [self.scenarios["loans"]], SimulatedJudge())
        self.assertTrue(math.isfinite(result.scenarios[0].reader_agreement))


if __name__ == "__main__":
    unittest.main()

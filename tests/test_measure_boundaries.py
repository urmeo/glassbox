"""Offline boundaries for measures, judge selection and provenance."""

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from glassbox import analysis, interfaces, report, reward, schema, scoring
from glassbox.judge import (
    ApiPairwiseJudge,
    PairwiseJudge,
    PairwiseRatingJudge,
    SimulatedJudge,
    build_judge,
    parse_ab,
    validate_judge_spec,
)
from glassbox.readers import build_readers
from glassbox.readers.base import Answer
from glassbox.stimuli import Stimulus


class CountingJudge(PairwiseJudge):
    target = "perceived_comprehension"
    modality = "text"

    def __init__(self):
        self.pairs = []
        self.template = "template1"

    @property
    def prompt_sha256(self):
        return hashlib.sha256(self.template.encode()).hexdigest()

    def compare(self, scenario, a, b):
        self.pairs.append((a.variant, b.variant))
        return "A"


class TestJudgeBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loans = schema.load_all_scenarios()["loans"]

    def test_exact_selected_pair_and_shared_analysis_cache(self):
        pairwise = CountingJudge()
        judge = PairwiseRatingJudge(pairwise, ["cards", "table"])
        book = scoring.ScoreBook(
            scoring.read_all(
                [self.loans],
                build_readers(["simulated"]),
                variants=["raw", "baseline", "cards", "table"],
                skip_render=True,
            )
        )
        h1 = analysis.analyze_h1(book, [self.loans], judge)
        analysis.analyze_cross_family(book, [self.loans], judge)
        self.assertEqual(pairwise.pairs, [("cards", "table")])
        self.assertEqual(h1.judge.comparison_count, 1)
        self.assertEqual(h1.judge.variants, ("cards", "table"))

    def test_three_variants_compare_three_pairs(self):
        pairwise = CountingJudge()
        judge = PairwiseRatingJudge(pairwise)
        judge.preference(self.loans, interfaces.variant(self.loans, "cards"))
        self.assertEqual(len(pairwise.pairs), 3)
        self.assertEqual(judge.comparison_count, 3)

    def test_content_prompt_and_protocol_invalidate_cache(self):
        pairwise = CountingJudge()
        judge = PairwiseRatingJudge(pairwise, ["cards", "table"])
        presentation = interfaces.variant(self.loans, "cards")
        judge.preference(self.loans, presentation)
        judge.preference(copy.deepcopy(self.loans), presentation)
        self.assertEqual(len(pairwise.pairs), 1)
        for mutated in (
            replace(self.loans, title="changed"),
            replace(
                self.loans,
                questions=[replace(self.loans.questions[0], stem="changed")]
                + self.loans.questions[1:],
            ),
        ):
            judge.preference(mutated, interfaces.variant(mutated, "cards"))
        self.assertEqual(len(pairwise.pairs), 3)
        pairwise.template = "template2"
        judge.preference(self.loans, presentation)
        judge.protocol = "changed_protocol"
        judge.preference(self.loans, presentation)
        self.assertEqual(len(pairwise.pairs), 5)

    def test_empty_duplicate_unknown_and_scalar_variants_reject(self):
        for variants in ([], ["cards", "cards"], ["unknown"], "cards", [True]):
            with self.subTest(variants=variants), self.assertRaises(ValueError):
                build_judge("pairwise", variants=variants)

    def test_unselected_presentation_rejects_without_comparing(self):
        pairwise = CountingJudge()
        judge = PairwiseRatingJudge(pairwise, ["cards", "table"])
        with self.assertRaises(ValueError):
            judge.preference(self.loans, interfaces.variant(self.loans, "annotated"))
        self.assertEqual(pairwise.pairs, [])

    def test_single_variant_has_no_comparisons(self):
        judge = PairwiseRatingJudge(CountingJudge(), ["cards"])
        self.assertEqual(
            judge.preference(self.loans, interfaces.variant(self.loans, "cards")), 0
        )
        self.assertEqual(judge.comparison_count, 0)

    def test_selector_validation_is_pure_and_strict(self):
        for spec in (
            None,
            True,
            2,
            "",
            "pairwise:",
            "pairwise:simulated:literal",
            "pairwise:openai: ",
            "pairwise:openai:gpt 4",
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_judge_spec(spec)
        self.assertEqual(validate_judge_spec(" simulated:polish "), "polish")
        for value in (0, True, 1.5):
            with self.assertRaises(ValueError):
                ApiPairwiseJudge("openai:gpt-4o", max_tokens=value)

    def test_shared_parser_has_no_incidental_default_win(self):
        self.assertEqual(parse_ab("A has issues; I choose B."), "B")
        self.assertEqual(parse_ab("Answer: B"), "B")
        for reply in ("A or B", "A comparison is helpful", "Answer A. Answer B.", None):
            self.assertIsNone(parse_ab(reply), reply)

    def test_api_metadata_is_text_comprehension_opinion(self):
        judge = build_judge("pairwise:anthropic:claude-sonnet-5", ["cards", "table"])
        metadata = judge.metadata()
        self.assertEqual(metadata.target, "perceived_comprehension")
        self.assertEqual(metadata.modality, "text")
        self.assertFalse(metadata.simulated)
        self.assertFalse(metadata.deterministic)
        self.assertEqual(len(metadata.prompt_sha256), 64)


class TestProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(schema.load_all_scenarios().values())
        cls.rows = scoring.read_all(
            cls.scenarios, build_readers(["simulated"]), skip_render=True
        )

    def analyze(self, rows, judge=None):
        book = scoring.ScoreBook(rows)
        h1 = analysis.analyze_h1(book, self.scenarios, judge or SimulatedJudge())
        cross = analysis.analyze_cross_family(
            book, self.scenarios, judge or SimulatedJudge()
        )
        return book, h1, cross

    def test_real_readers_cannot_hide_synthetic_judge(self):
        rows = [
            replace(r, reader_simulated=False, reader_deterministic=False)
            for r in self.rows
        ]
        book, h1, cross = self.analyze(rows)
        payload = report.build_results(self.scenarios, rows, book, h1, cross, True)
        self.assertFalse(h1.simulated)
        self.assertTrue(payload["meta"]["judge"]["simulated"])
        self.assertFalse(payload["meta"]["deterministic"])
        self.assertIn("Synthetic judge", report.render_report_md(book, h1))

    def test_mixed_and_real_judge_status_are_independent(self):
        rows = [
            replace(
                r,
                reader_simulated=r.reader.endswith("literal"),
                reader_deterministic=r.reader.endswith("literal"),
            )
            for r in self.rows
        ]
        book, h1, _ = self.analyze(rows)
        self.assertTrue(h1.any_reader_simulated)
        self.assertTrue(h1.mixed_readers)
        self.assertFalse(h1.simulated)
        self.assertIn("Mixed readers", report.render_report_md(book, h1))
        judge = PairwiseRatingJudge(CountingJudge())
        book, h1, cross = self.analyze(self.rows, judge)
        self.assertTrue(h1.simulated)
        self.assertFalse(h1.deterministic)
        self.assertFalse(h1.judge.simulated)
        md = report.render_report_md(book, h1, cross)
        self.assertIn("perceived comprehension", md)
        self.assertNotIn("aesthetic preference", md)

    def test_task_snapshots_resource_and_stimulus_hashes_are_distinct(self):
        book, h1, cross = self.analyze(self.rows)
        payload = report.build_results(self.scenarios, self.rows, book, h1, cross, True)
        meta = payload["meta"]
        self.assertEqual(meta["scenario_hash_format"], "glassbox.scenario.v1")
        for scenario in self.scenarios:
            self.assertEqual(
                meta["scenarios"][scenario.id], schema.scenario_sha256(scenario)
            )
            self.assertEqual(
                meta["scenario_snapshots"][scenario.id],
                schema.scenario_payload(scenario),
            )
        self.assertEqual(len(meta["prompt_resource_sha256"]["reader_mcq"]), 64)
        self.assertEqual(len(self.rows[0].stimulus_text_sha256), 64)
        self.assertIsNone(self.rows[0].stimulus_image_sha256)
        self.assertIsNone(self.rows[0].reader_prompt_sha256)
        self.assertTrue(meta["deterministic"])

    def test_prompt_edit_changes_template_provenance(self):
        book, h1, cross = self.analyze(self.rows)
        original = report.build_results(
            self.scenarios, self.rows, book, h1, cross, True
        )
        with patch("glassbox.prompts.load_prompt", return_value="edited template"):
            edited = report.build_results(
                self.scenarios, self.rows, book, h1, cross, True
            )
        self.assertNotEqual(
            original["meta"]["prompt_template_sha256"],
            edited["meta"]["prompt_template_sha256"],
        )
        self.assertEqual(
            original["meta"]["prompt_resource_sha256"],
            edited["meta"]["prompt_resource_sha256"],
        )

    def test_api_raw_reply_actual_prompt_and_image_metadata_retained(self):
        scenario = self.scenarios[0]
        reader = build_readers(["openai:gpt-4o"])[0]
        presentation = interfaces.variant(scenario, "table")
        stimulus = Stimulus(
            presentation,
            "image",
            presentation.to_text(),
            "synthetic.png",
            image_sha256="8" * 64,
            render_metadata={"renderer": "synthetic-test", "scale": 2},
        )
        prompt_hash = "6" * 64
        with (
            patch("glassbox.render.render", return_value=stimulus),
            patch.object(
                reader,
                "answer",
                return_value=Answer(
                    None, "api", "unparseable raw reply", prompt_sha256=prompt_hash
                ),
            ),
        ):
            rows = scoring.read_all(
                [scenario], [reader], variants=["table"], skip_render=True
            )
        self.assertEqual(rows[0].raw_answer, "unparseable raw reply")
        self.assertEqual(rows[0].reader_provider, "openai")
        self.assertEqual(rows[0].reader_model_family, "openai")
        self.assertEqual(rows[0].stimulus_image_sha256, "8" * 64)
        self.assertEqual(rows[0].render_metadata, stimulus.render_metadata)
        self.assertEqual(rows[0].reader_prompt_sha256, prompt_hash)
        self.assertFalse(rows[0].correct)

    def test_simulation_does_not_establish_determinism(self):
        rows = [replace(r, reader_deterministic=None) for r in self.rows]
        _, h1, _ = self.analyze(rows)
        self.assertTrue(h1.simulated)
        self.assertFalse(h1.all_reader_deterministic)
        self.assertFalse(h1.deterministic)

    def test_strict_json_written_and_unknown_correlation_displayed(self):
        book, h1, cross = self.analyze(self.rows)
        with tempfile.TemporaryDirectory() as directory:
            paths = report.write_run(
                directory, self.scenarios, self.rows, book, h1, cross, True
            )
            with open(paths["results"], encoding="utf-8") as stream:
                raw = stream.read()

        def reject(value):
            raise ValueError(value)

        payload = json.loads(raw, parse_constant=reject)
        self.assertIsNone(payload["cross_family"]["cross_family_agreement"])
        self.assertEqual(report._corr(float("nan")), "undefined")

    def test_writer_prevalidates_every_output_and_declared_input(self):
        book, h1, cross = self.analyze(self.rows)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text('{"id":"source-config"}')
            before = source.read_bytes()
            with self.assertRaises(ValueError):
                report.write_run(
                    directory,
                    self.scenarios,
                    self.rows,
                    book,
                    h1,
                    cross,
                    True,
                    protected_paths=[str(source)],
                )
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse((Path(directory) / "report.md").exists())
            source.unlink()
            source = Path(directory) / "source.json"
            source.write_text("source")
            (Path(directory) / "report.md").symlink_to(source)
            with self.assertRaises(ValueError):
                report.write_run(
                    directory,
                    self.scenarios,
                    self.rows,
                    book,
                    h1,
                    cross,
                    True,
                    protected_paths=[str(source)],
                )
            self.assertFalse((Path(directory) / "results.json").exists())
            self.assertEqual(source.read_text(), "source")

    def test_conflicting_reader_identity_rejected(self):
        rows = list(self.rows)
        rows[0] = replace(rows[0], reader_simulated=False)
        with self.assertRaises(ValueError):
            scoring.ScoreBook(rows)

    def test_unresolved_provider_cannot_create_independence(self):
        rows = [
            replace(
                r,
                reader_family="openrouter",
                reader_provider="openrouter",
                reader_model_family="unknown",
                reader_family_resolved=False,
            )
            for r in self.rows
        ]
        book, _, cross = self.analyze(rows)
        self.assertEqual(book.families(), [])
        self.assertEqual(cross.n_families, 0)
        self.assertFalse(cross.survives_across_families)
        self.assertEqual(cross.unresolved_readers, book.readers)

    def test_same_family_routes_group_together(self):
        rows = [
            replace(
                r,
                reader_family=(
                    "anthropic" if r.reader.endswith("literal") else "openrouter"
                ),
                reader_model_family="anthropic",
                reader_family_resolved=True,
            )
            for r in self.rows
        ]
        book, _, cross = self.analyze(rows)
        self.assertEqual(book.families(), ["anthropic"])
        self.assertEqual(cross.n_families, 1)

    def test_any_reversal_per_family_need_not_share_one_pair(self):
        scenario = self.scenarios[0]
        rows = []
        for family, correct in (
            ("fixtureA", (False, True, True)),
            ("fixtureB", (True, False, True)),
        ):
            for variant, value in zip(("cards", "table", "annotated"), correct):
                rows.append(
                    scoring.QuestionResult(
                        scenario.id,
                        family,
                        "synthetic",
                        True,
                        variant,
                        "q",
                        None,
                        "answer",
                        value,
                        "fixture",
                        reader_model_family=family,
                        reader_family_resolved=True,
                        reader_family_basis="synthetic-fixture",
                    )
                )
            rows.append(
                scoring.QuestionResult(
                    scenario.id,
                    family,
                    "synthetic",
                    True,
                    "baseline",
                    "q",
                    None,
                    "answer",
                    False,
                    "fixture",
                    reader_model_family=family,
                    reader_family_resolved=True,
                    reader_family_basis="synthetic-fixture",
                )
            )
        cross = analysis.analyze_cross_family(
            scoring.ScoreBook(rows), [scenario], PairwiseRatingJudge(CountingJudge())
        )
        self.assertTrue(cross.survives_across_families)
        self.assertEqual(cross.common_reversals, [])
        self.assertTrue(all(f.reversals for f in cross.families))
        self.assertIn(
            "Common reversed pairs: none", report.render_cross_family_md(cross)
        )


class TestMathAndSearch(unittest.TestCase):
    def test_finite_extreme_correlations(self):
        vector = [-1e308, 0.0, 1e308]
        self.assertAlmostEqual(analysis.pearson(vector, vector), 1)
        self.assertAlmostEqual(analysis.pearson(vector, vector[::-1]), -1)
        self.assertAlmostEqual(analysis.kendall_tau(vector, vector), 1)
        self.assertAlmostEqual(analysis.spearman(vector, vector), 1)
        close_integers = [2**60 + 1, 2**60 + 2, 2**60 + 3]
        for function in (analysis.pearson, analysis.spearman, analysis.kendall_tau):
            self.assertAlmostEqual(function(close_integers, close_integers), 1)

    def test_invalid_correlations_reject_and_constant_is_undefined(self):
        for function in (analysis.pearson, analysis.spearman, analysis.kendall_tau):
            for invalid in (True, "1", 1j, float("nan"), float("inf"), 10**1000):
                with (
                    self.subTest(function=function.__name__, invalid=type(invalid)),
                    self.assertRaises(ValueError),
                ):
                    function([invalid, 2], [1, 2])
            with self.assertRaises(ValueError):
                function([1], [1, 2])
            self.assertTrue(math.isnan(function([1, 1], [2, 3])))
            self.assertTrue(math.isnan(function([], [])))

    def test_missing_cells_do_not_invent_accuracy_or_lift(self):
        scenario = schema.load_all_scenarios()["loans"]
        book = scoring.ScoreBook(
            scoring.read_all(
                [scenario],
                build_readers(["simulated:literal"]),
                variants=["table"],
                skip_render=True,
            )
        )
        self.assertTrue(math.isnan(book.baseline("simulated:literal", "loans")))
        self.assertTrue(math.isnan(book.lift("simulated:literal", "loans", "table")))
        empty = scoring.ScoreBook([])
        self.assertFalse(empty.all_simulated())
        self.assertTrue(math.isnan(empty.mean_accuracy("missing", "table")))

    def test_read_all_bad_selections_never_render(self):
        scenario = schema.load_all_scenarios()["loans"]
        readers = build_readers(["simulated:literal"])
        for options in (
            {"replicates": True},
            {"replicates": 1.5},
            {"skip_render": "false"},
            {"variants": []},
            {"variants": ["table", "table"]},
            {"variants": "table"},
        ):
            with patch("glassbox.render.render") as render:
                with self.assertRaises(ValueError):
                    scoring.read_all([scenario], readers, **options)
                render.assert_not_called()

    def test_search_declares_in_sample_and_fresh_baselines(self):
        scenarios = list(schema.load_all_scenarios().values())
        readers = build_readers(["simulated"])
        result = reward.evaluate_generator(
            scenarios, readers, pool=["derived", "detail"]
        )
        self.assertEqual(result.candidate_count, 4)
        self.assertEqual(result.scenario_count, 3)
        self.assertEqual(result.baseline_sampling, "fresh_per_candidate_and_cards")
        self.assertTrue(result.all_readers_simulated)
        md = report.render_optimize_md(result, [r.name for r in readers])
        self.assertIn("In-sample feature search", md)
        self.assertIn("no held-out evaluation", md)
        self.assertEqual(
            len(reward.search_best_interface(scenarios, readers, pool=[])), 1
        )

    def test_bad_search_pool_fails_before_reading(self):
        scenario = schema.load_all_scenarios()["loans"]
        readers = build_readers(["simulated"])
        for pool in (["derived", "derived"], [True], ["unknown"], "derived"):
            with patch("glassbox.render.render") as render:
                with self.assertRaises(ValueError):
                    reward.search_best_interface([scenario], readers, pool=pool)
                render.assert_not_called()

    def test_unknown_family_cannot_pass_exclusion(self):
        self.assertFalse(
            reward.pool_excludes_family(["openrouter:unknown/model"], "qwen")
        )
        self.assertFalse(reward.pool_excludes_family(["anthropic:claude-3"], "unknown"))
        self.assertFalse(reward.pool_excludes_family([], "qwen"))
        self.assertFalse(
            reward.pool_excludes_family(["openrouter:anthropic/claude-3"], "claude")
        )


if __name__ == "__main__":
    unittest.main()

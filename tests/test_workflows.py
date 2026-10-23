import copy
import io
import json
import math
import os
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml

from glassbox import core, evaluation
from glassbox.__main__ import build_parser, main
from glassbox.core import (
    Answer,
    Reader,
    RenderError,
    Stimulus,
    build_reader,
    build_readers,
    render_stimulus as render_presentation,
    validate_output_dir,
)
from glassbox.evaluation import (
    PairwiseJudge,
    PairwiseJudgment,
    PairwiseRatingJudge,
    SimulatedJudge,
    SimulatedPairwiseJudge,
    build_judge,
    validate_judge_spec,
)


def run_cli(args):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = main(args)
    return status, stdout.getvalue(), stderr.getvalue()


class ContractCase(unittest.TestCase):
    def assert_fields(self, actual, **expected):
        value = actual.__getitem__ if isinstance(actual, dict) else lambda name: getattr(actual, name)
        self.assertEqual({name: value(name) for name in expected}, expected)


class WorkflowCase(ContractCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def cli(self, args, expected=0):
        status, output, error = run_cli([str(arg) for arg in args])
        self.assertEqual(status, expected, error)
        self.assertNotIn("Traceback", output + error)
        return output, error


@contextmanager
def untouched(*targets):
    with ExitStack() as stack:
        mocks = [stack.enter_context(patch(target)) for target in targets]
        yield
        for mock in mocks:
            mock.assert_not_called()


STUDY_CONFIG = {"id": "test", "description": "test", "readers": ["simulated"]}


MINIMAL_ANCHOR_CONFIG = {
    "id": "test",
    "items": [
        {"scenario": "loans", "question": "cheapest_total", "human_accuracy": 0.5}
    ],
}


class TestConfigurations(unittest.TestCase):
    def test_invalid_shapes_fields_and_runtime_types(self):
        invalid_items = [{"items": [dict(MINIMAL_ANCHOR_CONFIG["items"][0], **change)]}
                         for change in ({"scenario": "unknown"}, {"question": "unknown"},
                                        {"human_accuracy": True}, {"human_accuracy": float("nan")},
                                        {"human_accuracy": 1.5}, {"human_accuracy": 10**1000})]
        for parser, error, valid, changes in (
            (evaluation.parse_study, evaluation.StudyError, STUDY_CONFIG,
             [{"readers": []}, {"readers": ["unknown:x"]}, {"replicates": True},
              {"interfaces": ["raw"]}, {"skip_render": "false"}, {"judge": "invalid"}]),
            (evaluation.parse_anchor_set, evaluation.AnchorError, MINIMAL_ANCHOR_CONFIG,
             invalid_items + [{"items": []}, {"is_fixture": "false"}, {"condition": "unknown"}]),
        ):
            for invalid in [None, [], {}, dict(valid, unexpected=True)] + [dict(valid, **change) for change in changes]:
                with self.subTest(parser=parser.__name__, invalid=invalid):
                    self.assertRaises(error, parser, invalid)


class TestPreflight(WorkflowCase):
    def test_removed_runtime_options_fail_before_any_work(self):
        for command, option, value in (
            ("run", "--readers", "remote:model"),
            ("run", "--judge", "pairwise:remote:model"),
            ("repair", "--reader", "proxy:model"),
            ("anchor", "--readers", "remote:model"),
            ("optimize", "--readers", "remote:model"),
            ("optimize", "--training", "removed-plan.yaml"),
        ):
            args = [command, option, value, "--skip-render", "--out", self.root]
            if command == "anchor":
                args.extend(["--set", "fixture"])
            with self.subTest(command=command, option=option), untouched(
                "glassbox.evaluation.read_all", "glassbox.evaluation.evaluate_generator"
            ):
                self.cli(args, 2)
            self.assertEqual(list(self.root.iterdir()), [])

    def test_invalid_study_rejected_before_work(self):
        mutated = evaluation.StudyConfig(**copy.deepcopy(STUDY_CONFIG))
        mutated.readers.append("simulated:literal")
        unknown = evaluation.StudyConfig(**dict(STUDY_CONFIG, scenarios=["unknown"]))
        for config in (mutated, unknown):
            with self.subTest(config=config), untouched(
                "glassbox.evaluation.build_judge", "glassbox.core.build_readers",
                "glassbox.evaluation.read_all",
            ), self.assertRaises(evaluation.StudyError):
                evaluation.run_study(config, self.root)
        path = self.root / "invalid.yaml"
        path.write_text("{")
        self.assertRaises(evaluation.StudyError, evaluation.load_study, path)

    def test_mutated_anchor_and_invalid_runtime_flags_rejected_before_reading(self):
        config = evaluation.parse_anchor_set(MINIMAL_ANCHOR_CONFIG)
        readers = [build_reader("simulated:literal")]
        for fields in ({"replicates": True}, {"skip_render": "false"}, {"readers": []}):
            kwargs = dict(readers=readers, skip_render=True)
            kwargs.update(fields)
            with self.subTest(fields=fields), untouched("glassbox.evaluation.read_all"):
                with self.assertRaises(evaluation.AnchorError):
                    evaluation.run_anchor(config, **kwargs)
        config.items.append(config.items[0])
        with untouched("glassbox.evaluation.read_all"), self.assertRaises(evaluation.AnchorError):
            evaluation.run_anchor(config, readers, skip_render=True)


    def test_cli_judge_and_selected_variants_are_forwarded(self):
        self.cli([
            "run", "--judge", "pairwise", "--interfaces", "cards,table",
            "--scenarios", "loans", "--skip-render", "--out", self.root,
        ])
        raw = core.decode_yaml((self.root / "results.yaml").read_text())
        self.assertEqual(raw["meta"]["study"]["judge"], "pairwise")
        self.assertEqual(raw["meta"]["study"]["interfaces"], ["cards", "table"])
        self.assertEqual(raw["h1_summary"]["judge"]["variants"], ["cards", "table"])
        self.assertEqual(raw["h1_summary"]["judge"]["comparison_count"], 2)
        self.assertEqual(list(raw["metrics"]), ["loans"])

    def test_cli_runtime_failures_return_one(self):
        for error in (RenderError("render failure"), OSError("write failure")):
            with self.subTest(error=error), patch("glassbox.evaluation.run_study", side_effect=error):
                self.cli(["run", "--skip-render"], 1)

    def test_cli_status_contract_and_no_tracebacks(self):
        for args, expected in (
            (["--help"], 0), (["run", "--unknown"], 2),
            (["run", "--replicates", "0", "--skip-render"], 2),
            (["run", "--readers", "simulated,", "--skip-render"], 2),
            (["run", "--scenarios", "loans,loans", "--skip-render"], 2),
            (["run", "--judge", "unknown", "--skip-render"], 2),
            (["run", "--scenarios", "nonexistent", "--skip-render"], 2),
            (["run", "--interfaces", "", "--skip-render"], 2),
            (["run", "--readers", "hosted:model", "--skip-render"], 2),
            (["run", "--judge", "pairwise:hosted:model", "--skip-render"], 2),
            (["run", "--rep", "2"], 2),
        ):
            with self.subTest(args=args):
                self.cli(args, expected)


class TestOutputPreflight(WorkflowCase):
    def test_command_outputs_are_safe_from_source_directory_and_keep_overrides(self):
        commands = (
            ["run"], ["study", "--config", "offline_demo"], ["optimize"],
            ["evaluate", "--selection", "selection.yaml", "--scenarios", "growth"],
            ["anchor", "--set", "fixture"], ["repair"],
        )
        source = Path(core.__file__).resolve().parent.parent
        original = Path.cwd()
        try:
            os.chdir(source)
            for command in commands:
                with self.subTest(command=command[0]):
                    parser = build_parser()
                    destination = validate_output_dir(parser.parse_args(command).out)
                    self.assertNotEqual(destination, source)
                    self.assertNotIn(source, destination.parents)
                    self.assertNotIn(destination, source.parents)
                    for override in (str(self.root / "custom"), "../custom-results"):
                        parsed = parser.parse_args([*command, "--out", override])
                        self.assertEqual(parsed.out, override)
        finally:
            os.chdir(original)

    def test_source_checkout_paths_are_protected_without_packaging_metadata(self):
        root = self.root.resolve()
        package = root / "src/glassbox"
        package.mkdir(parents=True)
        with patch("glassbox.core.__file__", str(package / "core.py")):
            for folder in ("src", "data", "docs", "tests", "scripts", ".github"):
                with self.subTest(folder=folder), self.assertRaises(ValueError):
                    validate_output_dir(root / folder / "results")
            self.assertEqual(validate_output_dir(root / "outputs"), root / "outputs")

    def test_source_directory_rejected_before_reader_build(self):
        directory = Path(core.__file__).parent / "_data/scenarios"
        with untouched("glassbox.core.build_readers", "glassbox.evaluation.read_all"):
            self.cli(["run", "--skip-render", "--out", directory], 2)
        self.assertFalse((directory / "results.yaml").exists())

    def test_external_config_artifact_collisions_fail_before_work(self):
        for command, filename, raw, flag in (
            ("study", "results.yaml", STUDY_CONFIG, "--config"),
            ("anchor", "anchor.yaml", MINIMAL_ANCHOR_CONFIG, "--set"),
        ):
            with self.subTest(command=command):
                config = self.root / filename
                config.write_text(yaml.safe_dump(raw))
                before = config.read_bytes()
                args = [command, flag, config, "--out", self.root]
                if command != "study":
                    args.append("--skip-render")
                with untouched("glassbox.evaluation.read_all", "glassbox.evaluation.evaluate_generator"):
                    self.cli(args, 2)
                self.assertEqual(config.read_bytes(), before)

    def test_load_then_direct_run_preserves_external_config(self):
        path = self.root / "results.yaml"
        path.write_text(yaml.safe_dump(STUDY_CONFIG))
        config = evaluation.load_study(path)
        with untouched("glassbox.core.build_readers"), self.assertRaises(ValueError):
            evaluation.run_study(config, self.root)

    def test_report_symlink_and_hardlink_rejected_before_work(self):
        source = self.root / "input.yaml"
        source.write_text("unchanged")
        for link in ("symbolic", "hard"):
            with self.subTest(link=link):
                out = self.root / link
                out.mkdir()
                target = out / "results.yaml"
                if link == "symbolic":
                    target.symlink_to(source)
                else:
                    os.link(source, target)
                with untouched("glassbox.core.build_readers", "glassbox.evaluation.read_all"):
                    self.cli(["run", "--skip-render", "--out", out], 2)
                self.assertEqual(source.read_text(), "unchanged")


class TestCLI(WorkflowCase):
    def test_basic_commands(self):
        for command, options, artifact in (
            ("validate", [], None),
            ("study", ["--config", "offline_demo"], "report.md"),
            ("anchor", ["--set", "fixture", "--skip-render"], "anchor.md"),
            ("repair", ["--scenario", "loans", "--reader", "simulated:literal", "--skip-render"], "transcript.md"),
        ):
            with self.subTest(command=command):
                output = self.root / command
                args = [command, *options, "--out", output] if artifact else [command]
                self.cli(args)
                if artifact:
                    self.assertTrue((output / artifact).is_file())
                    self.assertEqual(list(output.rglob("*.json")), [])


class CountingJudge(PairwiseJudge):
    target = "fixture"
    modality = "text"

    def __init__(self, outcome="A"):
        self.pairs = []
        self.outcome = outcome

    def compare(self, scenario, a, b):
        self.pairs.append((a.variant, b.variant))
        return self.outcome


class JudgeCase(ContractCase):
    @classmethod
    def setUpClass(cls):
        cls.loans = core.load_all_scenarios()["loans"]


class TestPairwise(JudgeCase):
    def test_winrate_ranking(self):
        j = PairwiseRatingJudge(SimulatedPairwiseJudge())
        wr = {
            v: j.preference(self.loans, core.build_variant(self.loans, v))
            for v in ("cards", "table", "annotated")
        }
        self.assertEqual(wr["annotated"], 1.0)
        self.assertEqual(wr["table"], 0.0)
        self.assertAlmostEqual(wr["cards"], 0.5)

    def test_always_first_choice_does_not_favor_a_variant(self):
        for variants in (("cards", "table"), ("table", "cards")):
            judge = PairwiseRatingJudge(CountingJudge(), variants)
            for variant in variants:
                view = core.build_variant(self.loans, variant)
                self.assertEqual(judge.preference(self.loans, view), 0.5)
            metadata = judge.metadata()
            self.assert_fields(metadata, comparison_count=2, protocol="counterbalanced_pairs_winrate.v2")
            evidence = metadata.evidence[0]
            self.assertEqual([(item.variant_a, item.variant_b) for item in evidence.comparisons],
                             [variants, tuple(reversed(variants))])
            self.assertTrue(all(item.denominator == 2 for item in evidence.variant_scores))

    def test_mixed_outcomes_record_credit_and_coverage_without_cache_duplicates(self):
        class MixedJudge(PairwiseJudge):
            def __init__(self):
                self.replies = iter(("A", PairwiseJudgment("invalid", "A or B"),
                                     PairwiseJudgment("abstained", "ABSTAIN"), "tie", "B",
                                     PairwiseJudgment("invalid", "unclear")))
                self.calls = 0

            def compare(self, scenario, a, b):
                self.calls += 1
                return next(self.replies)

        pairwise = MixedJudge()
        judge = PairwiseRatingJudge(pairwise)
        for _ in range(2):
            for variant, score in {"cards": 0.75, "table": 0.0, "annotated": 0.75}.items():
                view = core.build_variant(self.loans, variant)
                self.assertEqual(judge.preference(self.loans, view), score)
        metadata = judge.metadata()
        self.assertEqual(pairwise.calls, 6)
        self.assert_fields(metadata, comparison_count=6, valid_count=3,
                           invalid_count=2, abstained_count=1, tie_count=1)
        evidence = metadata.evidence[0]
        self.assertEqual({item.variant: item.credit for item in evidence.variant_scores},
                         {"cards": 1.5, "table": 0.0, "annotated": 1.5})
        for score in evidence.variant_scores:
            self.assert_fields(score, denominator=2, planned_count=4)
        self.assertEqual(evidence.comparisons[1].raw, "A or B")
        self.assertEqual(evidence.comparisons[2].status, "abstained")


class TestJudgeBoundaries(JudgeCase):
    def test_exact_selected_pair_and_repeated_analysis_cache(self):
        pairwise = CountingJudge()
        judge = PairwiseRatingJudge(pairwise, ["cards", "table"])
        book = evaluation.ScoreBook(
            evaluation.read_all(
                [self.loans],
                build_readers(["simulated"]),
                variants=["raw", "baseline", "cards", "table"],
                skip_render=True,
            )
        )
        h1 = evaluation.analyze_h1(book, [self.loans], judge)
        evaluation.analyze_h1(book, [self.loans], judge)
        self.assertEqual(pairwise.pairs, [("cards", "table"), ("table", "cards")])
        self.assertEqual(h1.judge.comparison_count, 2)
        self.assertEqual(h1.judge.variants, ("cards", "table"))

    def test_cache_tracks_scenario_protocol_text_and_features(self):
        pairwise = CountingJudge()
        judge = PairwiseRatingJudge(pairwise, ["cards", "table"])
        original = core.build_variant(self.loans, "cards")
        judge.preference(self.loans, original)
        judge.preference(copy.deepcopy(self.loans), original)
        self.assertEqual(len(pairwise.pairs), 2)
        render_text = core.render_text
        changes = [
            (replace(self.loans, title="changed"), nullcontext()),
            (replace(self.loans, questions=[replace(self.loans.questions[0], stem="changed")]
                     + self.loans.questions[1:]), nullcontext()),
            (self.loans, patch.object(judge, "protocol", "changed_protocol")),
            (self.loans, patch("glassbox.core.render_text", side_effect=lambda *args, **kwargs:
                              render_text(*args, **kwargs) + "\nChanged display text")),
            (self.loans, patch.dict(core.VARIANT_FEATURES,
                                  {"cards": core.VARIANT_FEATURES["cards"] - {"polished"}})),
        ]
        for index, (scenario, context) in enumerate(changes, 2):
            with self.subTest(change=index), context:
                judge.preference(scenario, core.build_variant(scenario, "cards"))
            self.assertEqual(len(pairwise.pairs), index * 2)

    def test_empty_duplicate_unknown_and_scalar_variants_reject(self):
        for variants in ([], ["cards", "cards"], ["unknown"], "cards", [True], 1, True, 1.5):
            with self.subTest(variants=variants), self.assertRaises(ValueError):
                build_judge("pairwise", variants=variants)

    def test_unselected_presentation_rejects_without_comparing(self):
        pairwise = CountingJudge()
        judge = PairwiseRatingJudge(pairwise, ["cards", "table"])
        with self.assertRaises(ValueError):
            judge.preference(self.loans, core.build_variant(self.loans, "annotated"))
        self.assertEqual(pairwise.pairs, [])

    def test_single_variant_has_no_comparisons(self):
        judge = PairwiseRatingJudge(CountingJudge(), ["cards"])
        self.assertTrue(math.isnan(
            judge.preference(self.loans, core.build_variant(self.loans, "cards"))
        ))
        self.assertEqual(judge.comparison_count, 0)

    def test_selector_validation_is_pure_and_strict(self):
        for spec in (
            None,
            "",
            "pairwise:",
            "pairwise:simulated:literal",
            "pairwise:hosted:model",
            "pairwise:proxy:third-party-model",
            "pairwise:hosted: ",
            "pairwise:hosted:model with spaces",
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                validate_judge_spec(spec)
        self.assertEqual(validate_judge_spec(" simulated:polish "), "polish")


class TestRunAnchor(ContractCase):

    def test_subset_asks_only_declared_questions_and_replicates(self):
        anchor_set = evaluation.parse_anchor_set(MINIMAL_ANCHOR_CONFIG)
        self.assertTrue(anchor_set.is_fixture)
        reader = build_readers(["simulated:diligent"])[0]
        with patch.object(reader, "answer", wraps=reader.answer) as answer:
            result = evaluation.run_anchor(anchor_set, [reader], skip_render=True, replicates=2)
        self.assertEqual(answer.call_count, 2)
        self.assertEqual(
            [call.args[1].id for call in answer.call_args_list],
            ["cheapest_total", "cheapest_total"],
        )
        self.assertEqual(result.n_items, 1)
        self.assertEqual(result.points[0].model_accuracy, 1.0)

    def test_missing_reader_answers_do_not_become_zero_accuracy(self):
        anchor_set = evaluation.parse_anchor_set(MINIMAL_ANCHOR_CONFIG)
        with (
            patch("glassbox.evaluation.read_all", return_value=[]),
            self.assertRaisesRegex(evaluation.AnchorError, "no reader answers"),
        ):
            evaluation.run_anchor(anchor_set, build_readers(["simulated"]), skip_render=True)


class TestReward(ContractCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(core.load_all_scenarios().values())
        cls.readers = build_readers(["simulated"])

    def test_selected_interface_beats_fixture_baselines(self):
        ev = evaluation.evaluate_generator(self.scenarios, self.readers)
        self.assertTrue(ev.beats_preference_tuned)
        self.assertTrue(ev.beats_plaintext)
        self.assertIn("derived", ev.generator_features)
        self.assertGreater(ev.generator_reward, 0)
        self.assertLess(ev.cards_reward, 0)
        self.assert_fields(ev, candidate_count=32, scenario_count=3,
                           baseline_sampling="fresh_per_candidate_and_cards", all_readers_simulated=True)
        markdown = evaluation.render_optimize_md(ev, [reader.name for reader in self.readers])
        for label in ("In-sample feature search", "no held-out evaluation", "Simulated readers only"):
            self.assertIn(label, markdown)

    def test_duplicate_inputs_fail_before_reading(self):
        scenario = self.scenarios[0]
        reader = self.readers[0]
        for scenarios, readers in (
            ([scenario, scenario], [reader]),
            ([scenario], [reader, reader]),
            (True, [reader]),
            ([scenario], True),
        ):
            with (
                self.subTest(scenarios=scenarios, readers=readers),
                patch("glassbox.core.render_stimulus") as render,
                self.assertRaises(ValueError),
            ):
                evaluation.search_best_interface(scenarios, readers)
            render.assert_not_called()


class TestSearchBoundaries(unittest.TestCase):

    def test_bad_search_pool_fails_before_reading(self):
        scenario = core.load_all_scenarios()["loans"]
        readers = build_readers(["simulated"])
        for pool in (["derived", "derived"], [True], ["unknown"], "derived", True):
            with self.subTest(pool=pool), untouched("glassbox.core.render_stimulus"), self.assertRaises(ValueError):
                evaluation.search_best_interface([scenario], readers, pool=pool)


class RecordingReader(Reader):
    name = "recording"
    simulated = True
    deterministic = True

    def __init__(self):
        self.calls = []

    def answer(self, scenario, question, stimulus):
        self.calls.append((scenario.id, question.id, stimulus.presentation.variant))
        chosen = question.answer if stimulus.presentation.variant == "selected" else None
        return Answer(chosen, "fixture")


class SelectionCase(WorkflowCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = core.load_all_scenarios()
        cls.sources = [cls.scenarios["loans"], cls.scenarios["plans"]]
        cls.readers = build_readers(["simulated"])
        cls.search = evaluation.evaluate_generator(cls.sources, cls.readers, pool=["detail", "derived"])
        cls.artifact = evaluation.freeze_selection(cls.search, cls.sources, cls.readers, True)


class TestSelection(SelectionCase):

    def test_frozen_inputs_round_trip_and_reject_caller_mutation(self):
        payload = evaluation.selection_payload(self.artifact)
        parsed = evaluation.parse_selection(payload)
        self.assertEqual(parsed, self.artifact)
        self.assert_fields(payload, candidate_count=4, baseline_sampling="fresh_per_candidate_and_cards")
        self.assertEqual(payload["format"], "glassbox.selection.v2")
        for source, recorded in zip(self.sources, payload["scenarios"]):
            self.assert_fields(recorded, snapshot=core.scenario_payload(source), sha256=core.scenario_sha256(source))
        payload["features"].clear()
        payload["scenarios"][0]["snapshot"]["data"]["items"][0]["fees"] = 999
        self.assertEqual(evaluation.selection_payload(parsed), evaluation.selection_payload(self.artifact))

    def test_exclusive_writer_and_explicit_loader(self):
        path = self.root / "selection.yaml"
        self.assertEqual(evaluation.write_selection(path, self.artifact), str(path.resolve()))
        before = path.read_bytes()
        loaded = evaluation.load_selection(path)
        self.assert_fields(loaded, sha256=self.artifact.sha256, source_path=str(path.resolve()))
        self.assertRaises(FileExistsError, evaluation.write_selection, path, self.artifact)
        self.assertEqual(path.read_bytes(), before)
        self.assertRaises(evaluation.SelectionError, evaluation.load_selection, self.root / "missing.yaml")

    def test_strict_artifact_schema_features_protocol_and_digests(self):
        valid = evaluation.selection_payload(self.artifact)
        mutations = (
            ({"extra": True}, "selection has missing or unknown fields"),
            ({"format": "glassbox.selection.v1"}, "unsupported selection artifact format"),
            ({"features": ["headline"]}, "selected features must be distinct supported features"),
            ({"features": [True]}, "selected features must be distinct supported features"),
            ({"features": ["detail", "detail"]}, "selected features must be distinct supported features"),
            ({"features": "detail"}, "selected features must be distinct supported features"),
            ({"candidate_count": True}, "candidate_count must match an exhaustive feature search"),
            ({"candidate_count": 3}, "candidate_count must match an exhaustive feature search"),
            ({"selection_reward": float("nan")}, "configuration must contain finite JSON values"),
            ({"selection_reward": 2}, "selection_reward must be finite and between -1 and 1"),
            ({"selection_reward": True}, "selection_reward must be finite and between -1 and 1"),
            ({"skip_render": "false"}, "skip_render must be boolean"),
            ({"selection_protocol": "other"}, "unsupported selection protocol"),
            ({"baseline_sampling": "other"}, "unsupported selection baseline protocol"),
            ({"scenarios": []}, "selection requires source scenarios"),
            ({"readers": []}, "selection requires source readers"),
            ({"sha256": "0" * 64}, "selection artifact digest does not match its content"),
        )
        for mutation, message in mutations:
            raw = dict(valid, **mutation)
            with self.subTest(mutation=mutation):
                self.assertRaisesRegex(evaluation.SelectionError, message, evaluation.parse_selection, raw)
        for raw in (None, [], "selection", {key: value for key, value in valid.items() if key != "features"}):
            with self.subTest(raw=type(raw)):
                self.assertRaisesRegex(evaluation.SelectionError, "selection has missing or unknown fields",
                                       evaluation.parse_selection, raw)

    def test_nested_snapshot_and_reader_edits_reject(self):
        valid = evaluation.selection_payload(self.artifact)
        invalid_snapshot = "selection scenario identity, snapshot or digest is invalid"
        mutations = (
            ("scenario id", lambda raw: raw["scenarios"][0].update(id="changed"), invalid_snapshot),
            ("scenario hash", lambda raw: raw["scenarios"][0].update(sha256="0" * 64), invalid_snapshot),
            ("content hash", lambda raw: raw["scenarios"][0].update(content_sha256="0" * 64), invalid_snapshot),
            ("source hash", lambda raw: raw["scenarios"][0].update(source_sha256="0" * 64), invalid_snapshot),
            ("title", lambda raw: raw["scenarios"][0]["snapshot"].update(title="changed"), invalid_snapshot),
            ("answer", lambda raw: raw["scenarios"][0]["snapshot"]["questions"][0].update(answer="a"),
             "invalid selection scenario: loans/cheapest_total: authored answer"),
            ("reader field", lambda raw: raw["readers"][0].update(extra=True),
             "source reader has missing or unknown fields"),
            ("reader simulated", lambda raw: raw["readers"][0].update(simulated="false"),
             "source reader flags must be boolean"),
            ("reader determinism", lambda raw: raw["readers"][0].update(deterministic="true"),
             "source reader determinism must be boolean or null"),
            ("reader name", lambda raw: raw["readers"][1].update(name=raw["readers"][0]["name"]),
             "source reader models must be text and names distinct"),
        )
        for field, mutate, message in mutations:
            raw = copy.deepcopy(valid)
            mutate(raw)
            with self.subTest(field=field):
                self.assertRaisesRegex(evaluation.SelectionError, message, evaluation.parse_selection, raw)

    def test_invalid_selection_yaml_rejects(self):
        path = self.root / "selection.yaml"
        valid = yaml.safe_dump(evaluation.selection_payload(self.artifact), sort_keys=False)
        duplicate = valid + "sha256: " + self.artifact.sha256 + "\n"
        for text in (duplicate, "selection_reward: .nan", "selection_reward: 1.0e+999",
                     "!!python/object:builtins.object {}", "x: &ref [*ref]", "[]"):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(evaluation.SelectionError):
                evaluation.load_selection(path)

    def freeze(self, *, search=None, sources=None, readers=None, skip_render=True):
        return evaluation.freeze_selection(
            self.search if search is None else search,
            self.sources if sources is None else sources,
            self.readers if readers is None else readers, skip_render,
        )

    def test_freeze_requires_the_original_search_inputs(self):
        mutated = copy.deepcopy(self.sources)
        mutated[0].data["items"][0]["fees"] += 1
        cases = [
            ("scenario count", {"sources": self.sources[:1]}),
            ("reader count", {"readers": self.readers[:1]}),
            ("substitution", {"sources": [self.sources[0], self.scenarios["growth"]]}),
            ("order", {"sources": list(reversed(self.sources))}),
            ("data mutation", {"sources": mutated}),
            ("render mode", {"skip_render": False}),
            ("missing provenance", {"search": replace(self.search, selection_inputs_sha256="")}),
        ]
        for field, value in (("name", "renamed"), ("model", "another-model"),
                             ("model_family", "another-family"), ("deterministic", False)):
            readers = copy.deepcopy(self.readers)
            setattr(readers[0], field, value)
            cases.append((field, {"readers": readers}))
        for name, changes in cases:
            with self.subTest(name=name), self.assertRaisesRegex(evaluation.SelectionError, "inputs do not match"):
                self.freeze(**changes)
        self.assertEqual(self.freeze(sources=copy.deepcopy(self.sources),
                                     readers=build_readers(["simulated"])), self.artifact)

    def test_search_records_inputs_before_readers_run(self):
        class MutatingReader(RecordingReader):
            def answer(self, scenario, question, stimulus):
                scenario.data["value_label"] = "changed during search"
                return super().answer(scenario, question, stimulus)

        sources = copy.deepcopy(self.sources)
        readers = [MutatingReader()]
        result = evaluation.evaluate_generator(sources, readers, pool=[])
        with self.assertRaisesRegex(evaluation.SelectionError, "inputs do not match"):
            evaluation.freeze_selection(result, sources, readers, True)

    def test_empty_feature_selection_is_valid(self):
        result = evaluation.evaluate_generator(self.sources, self.readers, pool=[])
        artifact = evaluation.freeze_selection(result, self.sources, self.readers, True)
        self.assertEqual(artifact.features, frozenset())
        self.assertEqual(evaluation.selection_payload(artifact)["candidate_count"], 1)

    def test_overlap_and_renamed_source_reject_before_rendering_or_answers(self):
        renamed = replace(self.sources[0], id="renamed", title="Another title",
                          description="Another description", domain="Another domain")
        changed_same_id = replace(self.sources[0], questions=[
            replace(self.sources[0].questions[0], stem="Another wording")
        ] + self.sources[0].questions[1:])
        renamed_questions = replace(renamed, questions=[
            replace(question, id=question.id + "_copy", stem=question.stem + " Again.")
            for question in renamed.questions
        ])
        growth = self.scenarios["growth"]
        for scenarios in (
            [self.sources[0]], [renamed], [changed_same_id], [renamed_questions],
            [growth, replace(growth, id="growth-copy")], [],
        ):
            reader = RecordingReader()
            with untouched("glassbox.core.render_stimulus"), self.assertRaises(ValueError):
                evaluation.evaluate_selection(self.artifact, scenarios, [reader])
            self.assertEqual(reader.calls, [])

    def test_artifact_edits_fail_before_rendering_or_answers(self):
        payload = evaluation.selection_payload(self.artifact)
        payload["features"] = ["polished"]
        artifact = replace(self.artifact, payload_json=json.dumps(payload))
        reader = RecordingReader()
        with untouched("glassbox.core.render_stimulus"), self.assertRaisesRegex(ValueError, "digest"):
            evaluation.evaluate_selection(artifact, [self.scenarios["growth"]], [reader])
        self.assertEqual(reader.calls, [])

    def test_numeric_formatting_and_item_order_do_not_create_new_source(self):
        source = self.sources[0]
        snapshot = core.scenario_payload(source)
        for reorder in (False, True):
            data = copy.deepcopy(source.data)
            data["items"][0]["monthly_payment"] = int(data["items"][0]["monthly_payment"])
            if reorder:
                data["items"].reverse()
            clone = replace(source, id="cosmetic-copy", title="A renamed task", data=data,
                            questions=[replace(q, id=q.id + "_copy", stem=q.stem + " Again.")
                                       for q in source.questions])
            self.assertNotEqual(core.scenario_sha256(source), core.scenario_sha256(clone))
            reader = RecordingReader()
            with self.subTest(reorder=reorder), untouched("glassbox.core.render_stimulus"):
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    evaluation.evaluate_selection(self.artifact, [clone], [reader])
            self.assertEqual(reader.calls, [])
        self.assertEqual(core.scenario_payload(source), snapshot)

    def test_held_out_evaluation_does_not_search_and_shares_one_baseline(self):
        reader = RecordingReader()
        with untouched("glassbox.evaluation.search_best_interface"):
            result = evaluation.evaluate_selection(self.artifact, [self.scenarios["growth"]], [reader])
        self.assert_fields(result, selected_features=self.artifact.features, selection_sha256=self.artifact.sha256,
                           selection_scenario_ids=("loans", "plans"), selection_reward=self.search.generator_reward,
                           reader_names=("recording",), skip_render=True, all_readers_simulated=True,
                           baseline_sampling="shared_per_scenario")
        self.assert_fields(result.aggregate, selected_accuracy=1.0, cards_accuracy=0.0,
                           baseline_accuracy=0.0, selected_lift=1.0, cards_lift=0.0)
        self.assert_fields(result.scenarios[0], question_count=3, scenario_id="growth")
        self.assertEqual([call[2] for call in reader.calls],
                         ["baseline"] * 3 + ["selected"] * 3 + ["cards"] * 3)

    def test_multiple_evaluation_scenarios_have_separate_and_mean_metrics(self):
        class ScenarioReader(RecordingReader):
            def answer(self, scenario, question, stimulus):
                answer = super().answer(scenario, question, stimulus)
                return answer if scenario.id == "plans" else Answer(None, "fixture")

        sources = [self.scenarios["loans"]]
        searched = evaluation.evaluate_generator(sources, self.readers, pool=[])
        artifact = evaluation.freeze_selection(searched, sources, self.readers, True)
        result = evaluation.evaluate_selection(
            artifact, [self.scenarios["plans"], self.scenarios["growth"]], [ScenarioReader()]
        )
        self.assertEqual([item.selected_accuracy for item in result.scenarios], [1.0, 0.0])
        self.assertEqual([item.selected_lift for item in result.scenarios], [1.0, 0.0])
        self.assertEqual(result.aggregate.selected_accuracy, 0.5)
        self.assertEqual(result.aggregate.selected_lift, 0.5)

    def test_invalid_render_mode_rejects_before_work(self):
        reader = RecordingReader()
        with untouched("glassbox.core.render_stimulus"), self.assertRaises(ValueError):
            evaluation.evaluate_selection(self.artifact, [self.scenarios["growth"]], [reader], skip_render="false")
        self.assertEqual(reader.calls, [])


class TestEvaluationCLI(SelectionCase):

    def write_selection(self, name="selection.yaml"):
        path = self.root / name
        evaluation.write_selection(path, self.artifact)
        return path

    def evaluate(self, path, *, scenarios="growth", output=None, overrides=(), expected=0):
        return self.cli([
            "evaluate", "--selection", path, "--scenarios", scenarios,
            "--out", output or self.root / "evaluation", *overrides,
        ], expected)

    def test_optimize_then_evaluate_preserves_frozen_selection(self):
        selected, evaluated = self.root / "selection", self.root / "evaluation"
        self.cli(["optimize", "--readers", "simulated", "--scenarios", "loans,plans",
                  "--skip-render", "--out", selected])
        path = selected / "selection.yaml"
        before = path.read_bytes()
        frozen = evaluation.load_selection(path)
        with untouched("glassbox.evaluation.search_best_interface"):
            console, _ = self.evaluate(path)
        self.assertIn("no search was run", console)
        self.assertEqual(path.read_bytes(), before)
        payload = core.decode_yaml((evaluated / "evaluation.yaml").read_text())
        expected = {
            "selection_sha256": frozen.sha256, "selected_features": sorted(frozen.features),
            "selection_scenario_ids": ["loans", "plans"], "reader_names": list(frozen.reader_names),
            "evaluation_protocol": "frozen_features_disjoint_scenarios.v1",
            "baseline_sampling": "shared_per_scenario", "selection": evaluation.selection_payload(frozen),
        }
        self.assertEqual({key: payload[key] for key in expected}, expected)
        self.assertEqual([item["scenario_id"] for item in payload["scenarios"]], ["growth"])
        self.assertTrue(payload["skip_render"])
        self.assertTrue((evaluated / "evaluation.md").is_file())
        self.assertEqual(list(selected.rglob("*.json")) + list(evaluated.rglob("*.json")), [])

    def test_overlap_rejects_before_building_readers(self):
        path = self.write_selection()
        with untouched("glassbox.core.build_readers"):
            _, error = self.evaluate(path, scenarios="loans", expected=2)
        self.assertIn("overlaps", error)
        self.assertFalse((self.root / "evaluation").exists())

    def test_edited_artifact_rejects_before_building_readers(self):
        path = self.root / "selection.yaml"
        payload = evaluation.selection_payload(self.artifact)
        payload["selection_reward"] = 0.123456
        path.write_text(yaml.safe_dump(payload))
        with untouched("glassbox.core.build_readers"):
            _, error = self.evaluate(path, expected=2)
        self.assertIn("digest", error)

    def test_existing_selection_rejects_before_search(self):
        path = self.write_selection()
        before = path.read_bytes()
        with untouched("glassbox.core.build_readers", "glassbox.evaluation.evaluate_generator"):
            _, error = self.cli(["optimize", "--readers", "simulated", "--skip-render", "--out", self.root], 2)
        self.assertIn("new output directory", error)
        self.assertEqual(path.read_bytes(), before)

    def test_output_alias_cannot_overwrite_selection(self):
        path = self.write_selection("evaluation.yaml")
        before = path.read_bytes()
        with untouched("glassbox.core.build_readers"):
            _, error = self.evaluate(path, output=self.root, expected=2)
        self.assertIn("aliases input", error)
        self.assertEqual(path.read_bytes(), before)

    def test_direct_report_writer_protects_loaded_selection(self):
        path = self.write_selection("evaluation.yaml")
        loaded = evaluation.load_selection(path)
        before = path.read_bytes()
        result = evaluation.evaluate_selection(loaded, [self.scenarios["growth"]], build_readers(["simulated"]))
        with self.assertRaisesRegex(ValueError, "aliases input"):
            evaluation.write_evaluation(self.root, result, loaded)
        self.assertEqual(path.read_bytes(), before)

    def test_report_rejects_mismatched_selection_before_creating_outputs(self):
        result = evaluation.evaluate_selection(self.artifact, [self.scenarios["growth"]], build_readers(["simulated"]))
        output = self.root / "evaluation"
        for fields in (
            {"selection_sha256": "0" * 64}, {"selected_features": frozenset(["polished"])},
            {"selection_scenario_ids": ("growth",)}, {"selection_reward": 0.123456},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(ValueError, "does not match the frozen selection"),
            ):
                evaluation.write_evaluation(output, replace(result, **fields), self.artifact)
            self.assertFalse(output.exists())

    def test_explicit_reader_and_render_overrides_are_recorded(self):
        path = self.write_selection()
        with patch("glassbox.core.render_stimulus", side_effect=lambda view, skip:
                   Stimulus(view, "spec", view.to_text())) as render:
            self.evaluate(path, overrides=["--readers", "simulated:literal", "--render"])
        payload = core.decode_yaml((self.root / "evaluation/evaluation.yaml").read_text())
        self.assertEqual(payload["reader_names"], ["simulated:literal"])
        self.assertFalse(payload["skip_render"])
        self.assertEqual([call.kwargs["skip"] for call in render.call_args_list], [False] * 3)
        self.assertTrue(evaluation.load_selection(path).skip_render)

    def test_evaluation_requires_explicit_scenarios(self):
        with untouched("glassbox.evaluation.load_selection"):
            _, error = self.cli(["evaluate", "--selection", "selection.yaml"], 2)
        self.assertIn("--scenarios", error)


class TestQuestionRepair(unittest.TestCase):
    def test_invalid_render_flag_fails_before_rendering(self):
        scenario = core.load_all_scenarios()["loans"]
        reader = build_reader("simulated:literal")
        for skip in ("false", 0, None):
            with self.subTest(skip_render=skip), untouched("glassbox.core.render_stimulus"):
                self.assertRaisesRegex(ValueError, "skip_render must be a boolean", evaluation.repair,
                                       scenario, reader, scenario.questions[0], skip_render=skip)

    def test_all_nine_questions_and_three_personas_converge_without_source_changes(self):
        count = 0
        for scenario in core.load_all_scenarios().values():
            before = core.scenario_payload(scenario)
            for question in scenario.questions:
                for reader in build_readers(["simulated"]):
                    with self.subTest(
                        scenario=scenario.id, question=question.id, reader=reader.name
                    ):
                        with patch("glassbox.core.render_stimulus", wraps=render_presentation) as render:
                            result = evaluation.repair(scenario, reader, question, skip_render=True)
                        self.assertTrue(result.converged)
                        self.assertIsNotNone(result.turns_to_understanding)
                        self.assertGreater(len(result.turns), 0)
                        self.assertEqual(render.call_count, len(result.turns))
                        self.assertEqual([turn.turn for turn in result.turns], list(range(len(result.turns))))
                        if len(result.turns) > 1:
                            self.assertIn("derived", result.turns[1].features)
                        seen = [call.args[0] for call in render.call_args_list]
                        self.assertTrue(
                            all(p.primary_metric == question.target_field for p in seen)
                        )
                        if question.compute["op"].startswith("count_"):
                            self.assertTrue(all(p.highlight_item is None for p in seen))
                        count += 1
                all_features = frozenset({"detail", "derived", "highlight", "sorted", "polished"})
                view = core.build_presentation(scenario, all_features, target=question)
                self.assertEqual(view.target_question_id, question.id)
                if question.compute["op"] == "rank":
                    self.assertEqual(view.highlight_item, "A")
                    self.assertEqual(view.highlight_label, "rank 2")
                    self.assertNotIn("highlighted as best", view.to_text())
            self.assertEqual(before, core.scenario_payload(scenario))
        self.assertEqual(count, 27)

    def test_percent_target_does_not_inherit_currency_and_plan_argmax_is_correct(self):
        scenarios = core.load_all_scenarios()
        growth = scenarios["growth"]
        question = next(q for q in growth.questions if q.id == "biggest_percent_gain")
        view = core.build_variant(growth, "annotated", target=question)
        self.assertEqual(view.highlight_item, "South")
        self.assertEqual(view.shown["South"]["growth_pct"], 90)
        self.assertEqual(view.unit, "")
        self.assertNotIn("$90k", view.to_text())
        self.assertNotIn("$90k", view.to_html())
        question = next(q for q in scenarios["plans"].questions if q.id == "most_expensive_2yr")
        self.assertEqual(
            core.build_variant(scenarios["plans"], "annotated", target=question).highlight_item, "X"
        )

    def test_nonprimary_raw_headline_needs_explicit_units(self):
        scenario = core.load_all_scenarios()["growth"]
        question = replace(
            scenario.questions[0], compute={"op": "argmax", "field": "end"}, answer="e"
        )
        scenario = replace(scenario, questions=[question])
        view = core.build_variant(scenario, "annotated", target=question)
        self.assertEqual(view.unit, "")
        self.assertNotIn("$345k", view.to_text())
        data = copy.deepcopy(scenario.data)
        data["presentation"]["field_display"] = {
            "end": {"unit": "$k", "value_label": "Ending revenue"}
        }
        view = core.build_variant(replace(scenario, data=data), "annotated", target=question)
        self.assertEqual(view.unit, "$k")
        self.assertIn("Ending revenue = $345k", view.to_text())

    def test_zero_turn_budget_and_stubborn_reader_are_bounded(self):
        scenario = core.load_all_scenarios()["loans"]
        question = scenario.questions[0]

        result = evaluation.repair(scenario, RecordingReader(), question, max_turns=0, skip_render=True)
        self.assertEqual(len(result.turns), 1)
        self.assertFalse(result.converged)
        self.assertIsNone(result.turns_to_understanding)
        result = evaluation.repair(scenario, RecordingReader(), question, max_turns=4, skip_render=True)
        self.assertLessEqual(len(result.turns), 5)
        self.assertFalse(result.converged)
        for bad in (True, -1, 1.5, "2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evaluation.repair(scenario, RecordingReader(), question, max_turns=bad, skip_render=True)

    def test_target_context_survives_feature_additions_and_escaping(self):
        scenario = core.load_all_scenarios()["growth"]
        question = scenario.questions[1]
        base = core.build_variant(scenario, "cards", target=question)
        updated = core.with_features(scenario, base, frozenset({"derived"}))
        self.assertEqual(updated.primary_metric, "growth_pct")
        self.assertEqual(updated.target_question_id, question.id)
        changed = replace(
            updated, title="<script>alert(1)</script>", target_stem="<img src=x onerror=alert(1)>"
        )
        self.assertNotIn("<script>", changed.to_html())
        self.assertNotIn("<img src=x", changed.to_html())
        with self.assertRaises(ValueError):
            core.build_variant(scenario, "cards", target=replace(question, id="missing"))


class ReportCase(WorkflowCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(core.load_all_scenarios().values())
        cls.rows = evaluation.read_all(cls.scenarios, build_readers(["simulated"]), skip_render=True)
        cls.book = evaluation.ScoreBook(cls.rows)
        cls.h1 = evaluation.analyze_h1(cls.book, cls.scenarios, SimulatedJudge())

    def write_run(self, **options):
        return evaluation.write_run(self.root, self.scenarios, self.rows, self.book,
                                self.h1, True, **options)


class TestReport(ReportCase):

    def test_undefined_lift_is_labeled_in_markdown(self):
        scenario = self.h1.scenarios[0]
        row = replace(scenario.rows[0], comp_lift=float("nan"))
        summary = replace(self.h1, scenarios=[replace(scenario, rows=[row])])
        text = evaluation.render_report_md(self.book, summary)
        self.assertIn("undefined", text)
        self.assertNotIn("nan pts", text)

    def test_all_invalid_judgments_stay_missing_through_analysis_and_reports(self):
        pairwise = CountingJudge(PairwiseJudgment("invalid", raw="A or B"))
        pairwise.name = "simulated:invalid"
        pairwise.simulated = pairwise.deterministic = True
        judge = PairwiseRatingJudge(pairwise)
        h1 = evaluation.analyze_h1(self.book, self.scenarios, judge)
        self.assertFalse(h1.divergence_found)

        paths = evaluation.write_run(self.root, self.scenarios, self.rows, self.book, h1, True)
        raw = Path(paths["results"]).read_text(encoding="utf-8")
        markdown = Path(paths["report"]).read_text(encoding="utf-8")

        results = core.decode_yaml(raw)
        self.assertFalse(raw.lstrip().startswith("{"))
        self.assertIsNone(results["h1_summary"]["mean_spearman"])
        self.assertFalse(results["h1_summary"]["divergence_found"])
        for scenario in results["metrics"].values():
            self.assertEqual(scenario["h1"]["reversals"], [])
            for row in scenario["h1"]["rows"]:
                self.assertIsNone(row["pref_score"])
                self.assertIsNone(row["pref_rank"])
                self.assertIsNotNone(row["comp_rank"])

        metadata = results["h1_summary"]["judge"]
        self.assert_fields(metadata, comparison_count=18, invalid_count=18,
                           valid_count=0, tie_count=0, abstained_count=0)
        self.assertEqual(len(metadata["evidence"]), 3)
        for evidence in metadata["evidence"]:
            for score in evidence["variant_scores"]:
                self.assert_fields(score, credit=0, denominator=0, planned_count=4, score=None)
            for comparison in evidence["comparisons"]:
                self.assertEqual(comparison["raw"], "A or B")
                self.assertEqual(comparison["status"], "invalid")
        self.assertNotIn("nan", markdown.lower())
        self.assertIn("judge scores unavailable", markdown)
        self.assertIn("Too few scored interfaces to compare rankings.", markdown)
        self.assertNotIn("**Reversal:**", markdown)


class TestProvenance(ReportCase):

    def analyze(self, rows, judge=None):
        book = evaluation.ScoreBook(rows)
        h1 = evaluation.analyze_h1(book, self.scenarios, judge or SimulatedJudge())
        return book, h1

    def test_reader_simulation_and_judge_determinism_are_independent(self):
        cases = (
            ("real", SimulatedJudge(), False, False, False, "Synthetic judge"),
            ("mixed", SimulatedJudge(), False, True, True, "Mixed readers"),
            ("simulated", PairwiseRatingJudge(CountingJudge()), True, True, False, "judge score"),
            ("unknown", SimulatedJudge(), True, True, False, "Declared deterministic: **no**"),
        )
        for mode, judge, all_simulated, any_simulated, mixed, label in cases:
            rows = []
            for row in self.rows:
                simulated = mode != "real" and (mode != "mixed" or row.reader.endswith("literal"))
                rows.append(replace(row, reader_simulated=simulated,
                                    reader_deterministic=None if mode == "unknown" else simulated))
            with self.subTest(mode=mode):
                book, h1 = self.analyze(rows, judge)
                self.assert_fields(h1, simulated=all_simulated, any_reader_simulated=any_simulated,
                                   mixed_readers=mixed, deterministic=False)
                payload = evaluation.build_results(self.scenarios, rows, book, h1, True)
                self.assertEqual(payload["meta"]["judge"]["simulated"], judge.simulated)
                self.assertEqual(payload["meta"]["judge"]["target"], judge.target)
                self.assertFalse(payload["meta"]["deterministic"])
                markdown = evaluation.render_report_md(book, h1)
                self.assertIn(label, markdown)
                self.assertNotIn("aesthetic preference", markdown)

    def test_task_snapshots_and_stimulus_hashes_are_recorded(self):
        payload = evaluation.build_results(self.scenarios, self.rows, self.book, self.h1, True)
        self.assertTrue({"meta", "metrics", "h1_summary"} <= payload.keys())
        self.assertTrue(payload["h1_summary"]["divergence_found"])
        meta = payload["meta"]
        self.assert_fields(meta, renderer="skip-render", scenario_hash_format="glassbox.scenario.v1", deterministic=True)
        self.assertEqual(meta["scenarios"], {scenario.id: core.scenario_sha256(scenario) for scenario in self.scenarios})
        self.assertEqual(meta["scenario_snapshots"],
                         {scenario.id: core.scenario_payload(scenario) for scenario in self.scenarios})
        self.assertEqual(len(self.rows[0].stimulus_text_sha256), 64)
        self.assert_fields(self.rows[0], stimulus_image_sha256=None)
        markdown = evaluation.render_report_md(self.book, self.h1)
        for label in ("Simulated readers only", "Reversal", "MCQ lift vs synthetic polish",
                      "Synthetic judge"):
            self.assertIn(label, markdown)

    def test_render_metadata_is_snapshotted_for_each_answer(self):
        scenario = self.scenarios[0]
        reader = build_reader("simulated:diligent")
        presentation = core.build_variant(scenario, "table")
        stimulus = Stimulus(
            presentation,
            "image",
            presentation.to_text(),
            "synthetic.png",
            image_sha256="8" * 64,
            render_metadata={"renderer": "synthetic-test", "scale": 2, "window": [1600, 1000]},
        )
        with patch("glassbox.core.render_stimulus", return_value=stimulus):
            rows = evaluation.read_all(
                [scenario], [reader], variants=["table"], skip_render=True
            )
        self.assert_fields(rows[0], reader_provider="simulated", reader_model_family="simulated",
                           stimulus_image_sha256="8" * 64, render_metadata=stimulus.render_metadata)
        rows[0].render_metadata["window"][0] = 1
        self.assertEqual(rows[1].render_metadata["window"], [1600, 1000])
        self.assertEqual(stimulus.render_metadata["window"], [1600, 1000])

    def test_writer_prevalidates_every_output_and_declared_input(self):
        source = self.root / "results.yaml"
        source.write_text("id: source-config\n")
        before = source.read_bytes()
        with self.assertRaises(ValueError):
            self.write_run(protected_paths=[str(source)])
        self.assertEqual(source.read_bytes(), before)
        self.assertFalse((self.root / "report.md").exists())
        source.unlink()
        source = self.root / "source.yaml"
        source.write_text("source")
        (self.root / "report.md").symlink_to(source)
        with self.assertRaises(ValueError):
            self.write_run(protected_paths=[str(source)])
        self.assertFalse((self.root / "results.yaml").exists())
        self.assertEqual(source.read_text(), "source")


if __name__ == "__main__":
    unittest.main()

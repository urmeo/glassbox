"""Strict public configuration and CLI preflight regressions."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from glassbox import anchor, studies, training
from glassbox.cli import build_parser, main
from glassbox.readers import build_reader, expand_reader_specs
from glassbox.readers.anthropic import AnthropicReader
from glassbox.readers.openai_compat import OpenAICompatReader

STUDY = {"id": "test", "description": "test", "readers": ["simulated"]}
ANCHOR = {
    "id": "test",
    "items": [
        {"scenario": "loans", "question": "cheapest_total", "human_accuracy": 0.5}
    ],
}
TRAINING = {
    "id": "test",
    "description": "test",
    "framework": "tinker",
    "base_model": "Qwen/Qwen3-VL-8B-Instruct",
    "generator_family": "qwen",
    "methods": ["sft", "grpo"],
    "reward": "comprehension_lift",
    "reader_pool": ["openrouter:anthropic/claude-sonnet-5"],
}


class TestShapeBoundaries(unittest.TestCase):
    def test_non_object_and_unknown_keys(self):
        for parser, error, valid in (
            (studies.parse_study, studies.StudyError, STUDY),
            (anchor.parse_anchor_set, anchor.AnchorError, ANCHOR),
            (training.parse_training_config, training.TrainingConfigError, TRAINING),
        ):
            for value in (None, [], "x", {1: 2, "x": 3}, dict(valid, unexpected=True)):
                with self.subTest(parser=parser.__name__, value=value):
                    with self.assertRaises(error):
                        parser(value)

    def test_study_typed_nonempty_distinct_fields(self):
        invalid = {
            "readers": (
                [],
                [1],
                [""],
                ["simulated", "simulated:literal"],
                ["unknown:x"],
            ),
            "scenarios": ([], [1], ["loans", "loans"]),
            "interfaces": ([], [1], ["table", "table"], ["raw"]),
            "replicates": (True, 0, 1.5, "2"),
            "skip_render": (0, "false", None),
            "judge": (None, [], "invalid", "pairwise:openrouter:"),
            "description": (None, False),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(studies.StudyError):
                        studies.parse_study(dict(STUDY, **{field: value}))

    def test_direct_study_constructor_has_same_guards(self):
        valid = studies.StudyConfig(**STUDY)
        for fields in (
            {"replicates": True},
            {"skip_render": "false"},
            {"readers": []},
            {"interfaces": []},
            {"scenarios": ["loans", "loans"]},
            {"judge": "unknown"},
        ):
            with self.subTest(fields=fields), self.assertRaises(studies.StudyError):
                replace(valid, **fields)

    def test_nested_anchor_types_and_duplicates(self):
        for item in (
            None,
            [],
            "item",
            dict(ANCHOR["items"][0], extra=1),
            dict(ANCHOR["items"][0], human_accuracy=True),
            dict(ANCHOR["items"][0], human_accuracy=float("nan")),
            dict(ANCHOR["items"][0], human_accuracy=10**1000),
        ):
            with self.subTest(item=item), self.assertRaises(anchor.AnchorError):
                anchor.parse_anchor_set(dict(ANCHOR, items=[item]))
        for values in (
            {"is_fixture": "false"},
            {"note": {}},
            {"source": None},
            {"condition": []},
            {"items": ANCHOR["items"] * 2},
        ):
            with self.subTest(values=values), self.assertRaises(anchor.AnchorError):
                anchor.parse_anchor_set(dict(ANCHOR, **values))

    def test_direct_anchor_item_and_set_guard(self):
        with self.assertRaises(anchor.AnchorError):
            anchor.AnchorItem("loans", "cheapest_total", True)
        with self.assertRaises(anchor.AnchorError):
            anchor.AnchorSet("x", "", "", "raw", [ANCHOR["items"][0]], True)
        item = anchor.AnchorItem("loans", "cheapest_total", 0.5)
        with self.assertRaises(anchor.AnchorError):
            anchor.AnchorSet("x", "", "", "raw", [item, item], True)

    def test_training_structural_boundaries(self):
        for fields in (
            {"status": None},
            {"params": []},
            {"params": {"lr": float("inf")}},
            {"params": {1: 2}},
            {"methods": ["sft", "sft"]},
            {"reader_pool": [False]},
            {"reader_pool": ["simulated", "simulated:literal"]},
        ):
            with (
                self.subTest(fields=fields),
                self.assertRaises(training.TrainingConfigError),
            ):
                training.parse_training_config(dict(TRAINING, **fields))
        valid = training.parse_training_config(TRAINING)
        with self.assertRaises(training.TrainingConfigError):
            replace(valid, params=[])

    def test_cyclic_params_rejected_as_config_error(self):
        params = {}
        params["self"] = params
        with self.assertRaises(training.TrainingConfigError):
            training.parse_training_config(dict(TRAINING, params=params))

    def test_training_underlying_families(self):
        self.assertEqual(
            training.validate_training_config(training.parse_training_config(TRAINING)),
            [],
        )
        for fields in (
            {"reader_pool": ["openrouter:qwen/qwen3-vl-8b"]},
            {"reader_pool": ["anthropic:unrecognized"]},
            {"generator_family": "unrecognized"},
            {"base_model": "unrecognized"},
            {"base_model": "openai:gpt-4o"},
        ):
            with self.subTest(fields=fields):
                self.assertTrue(
                    training.validate_training_config(
                        training.parse_training_config(dict(TRAINING, **fields))
                    )
                )

    def test_reader_pool_validated_before_construction(self):
        for specs in (
            [],
            "simulated",
            [True],
            [""],
            ["simulated", "simulated:literal"],
        ):
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                expand_reader_specs(specs)
        for constructor in (
            lambda: AnthropicReader("a b"),
            lambda: OpenAICompatReader("openai", "a b"),
            lambda: AnthropicReader("claude", True),
            lambda: OpenAICompatReader("openai", "gpt-4o", 0),
            lambda: OpenAICompatReader([], "gpt-4o"),
            lambda: build_reader(None),
        ):
            with self.assertRaises(ValueError):
                constructor()


class TestConfigLoading(unittest.TestCase):
    def test_builtin_names_work_outside_checkout(self):
        self.assertEqual(studies.load_study("offline_demo").id, "offline_demo")
        self.assertEqual(anchor.load_anchor_set("fixture.json").id, "fixture")
        self.assertEqual(
            training.load_training_config("sft_grpo_qwen3vl").generator_family, "qwen"
        )

    def test_explicit_missing_path_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            for loader, error, filename in (
                (studies.load_study, studies.StudyError, "offline_demo.json"),
                (anchor.load_anchor_set, anchor.AnchorError, "fixture.json"),
                (
                    training.load_training_config,
                    training.TrainingConfigError,
                    "sft_grpo_qwen3vl.json",
                ),
            ):
                with self.subTest(loader=loader.__name__), self.assertRaises(error):
                    loader(str(Path(tmp) / filename))

    def test_strict_json_constants_duplicates_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "config.json"
            for raw in (
                "[]",
                '{"id":"a","id":"b"}',
                '{"replicates":NaN}',
                '{"replicates":Infinity}',
            ):
                file.write_text(raw)
                for loader, error in (
                    (studies.load_study, studies.StudyError),
                    (anchor.load_anchor_set, anchor.AnchorError),
                    (training.load_training_config, training.TrainingConfigError),
                ):
                    with (
                        self.subTest(raw=raw, loader=loader.__name__),
                        self.assertRaises(error),
                    ):
                        loader(str(file))

    def test_explicit_file_is_loaded_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "offline_demo.json"
            path.write_text(json.dumps(dict(STUDY, id="external")))
            self.assertEqual(studies.load_study(path).id, "external")


class TestPreflight(unittest.TestCase):
    def test_mutated_study_rejected_before_work(self):
        config = studies.StudyConfig(**STUDY)
        config.readers.append("simulated:literal")
        with (
            patch("glassbox.studies.build_readers") as readers,
            patch("glassbox.scoring.read_all") as read,
        ):
            with self.assertRaises(studies.StudyError):
                studies.run_study(config, "unused")
            readers.assert_not_called()
            read.assert_not_called()

    def test_unknown_scenario_rejected_before_judge_and_readers(self):
        config = studies.StudyConfig(**dict(STUDY, scenarios=["unknown"]))
        with (
            patch("glassbox.studies.build_judge") as judge,
            patch("glassbox.studies.build_readers") as readers,
        ):
            with self.assertRaises(studies.StudyError):
                studies.run_study(config, "unused")
            judge.assert_not_called()
            readers.assert_not_called()

    def test_mutated_anchor_and_invalid_runtime_flags_rejected_before_reading(self):
        config = anchor.parse_anchor_set(ANCHOR)
        readers = [build_reader("simulated:literal")]
        for fields in ({"replicates": True}, {"skip_render": "false"}, {"readers": []}):
            kwargs = dict(readers=readers, skip_render=True)
            kwargs.update(fields)
            with (
                self.subTest(fields=fields),
                patch("glassbox.scoring.read_all") as read,
            ):
                with self.assertRaises(anchor.AnchorError):
                    anchor.run_anchor(config, **kwargs)
                read.assert_not_called()
        config.items.append(config.items[0])
        with (
            patch("glassbox.scoring.read_all") as read,
            self.assertRaises(anchor.AnchorError),
        ):
            anchor.run_anchor(config, readers, skip_render=True)
        read.assert_not_called()

    def test_training_preflight_happens_before_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            bad = copy.deepcopy(TRAINING)
            bad["reader_pool"] = ["openrouter:qwen/qwen3-vl-8b"]
            path.write_text(json.dumps(bad))
            with (
                patch("glassbox.cli.reward_mod.evaluate_generator") as run,
                patch("glassbox.cli.build_readers") as readers,
            ):
                with redirect_stderr(io.StringIO()):
                    status = main(
                        [
                            "optimize",
                            "--training",
                            str(path),
                            "--skip-render",
                            "--out",
                            tmp,
                        ]
                    )
                self.assertEqual(status, 2)
                run.assert_not_called()
                readers.assert_not_called()

    def test_cli_judge_and_selected_variants_are_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "run",
                            "--judge",
                            "pairwise",
                            "--interfaces",
                            "cards,table",
                            "--scenarios",
                            "loans",
                            "--skip-render",
                            "--out",
                            tmp,
                        ]
                    ),
                    0,
                )
            raw = json.loads((Path(tmp) / "results.json").read_text())
            self.assertEqual(raw["meta"]["study"]["judge"], "pairwise")
            self.assertEqual(raw["meta"]["study"]["interfaces"], ["cards", "table"])
            judged = raw["h1_summary"]["judge"]
            self.assertEqual(judged["variants"], ["cards", "table"])
            self.assertEqual(judged["comparison_count"], 1)

    def test_cli_runtime_failures_return_one(self):
        from glassbox.readers._http import ReaderAPIError
        from glassbox.render import RenderError

        for error in (
            ReaderAPIError("fake upstream failure"),
            RenderError("fake render failure"),
            OSError("fake write failure"),
        ):
            output = io.StringIO()
            with (
                self.subTest(error=error),
                patch("glassbox.cli.run_study", side_effect=error),
                redirect_stderr(output),
            ):
                self.assertEqual(main(["run", "--skip-render"]), 1)
                self.assertNotIn("Traceback", output.getvalue())

    def test_cli_status_contract_and_no_tracebacks(self):
        for argv, expected in (
            (["--help"], 0),
            (["run", "--unknown"], 2),
            (["run", "--replicates", "0", "--skip-render"], 2),
            (["run", "--readers", "simulated,", "--skip-render"], 2),
            (["run", "--scenarios", "loans,loans", "--skip-render"], 2),
            (["run", "--judge", "unknown", "--skip-render"], 2),
            (["run", "--interfaces", "", "--skip-render"], 2),
            (["run", "--readers", "anthropic:claude-sonnet-5", "--skip-render"], 1),
        ):
            output = io.StringIO()
            with (
                self.subTest(argv=argv),
                patch.dict("os.environ", {}, clear=True),
                redirect_stdout(output),
                redirect_stderr(output),
            ):
                self.assertEqual(main(argv), expected)
                self.assertNotIn("Traceback", output.getvalue())

    def test_cli_defaults_and_no_abbreviated_flags(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["run"]).out, "outputs")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--rep", "2"]), 2)


class TestOutputPreflight(unittest.TestCase):
    def test_source_directory_rejected_before_reader_build(self):
        import glassbox

        directory = Path(glassbox.__file__).parent / "_data" / "scenarios"
        with (
            patch("glassbox.studies.build_readers") as readers,
            patch("glassbox.scoring.read_all") as read,
        ):
            output = io.StringIO()
            with redirect_stderr(output):
                self.assertEqual(
                    main(["run", "--skip-render", "--out", str(directory)]), 2
                )
            readers.assert_not_called()
            read.assert_not_called()
            self.assertFalse((directory / "results.json").exists())

    def test_external_config_artifact_collisions_fail_before_work(self):
        for command, filename, raw, flag in (
            ("study", "results.json", STUDY, "--config"),
            ("anchor", "anchor.json", ANCHOR, "--set"),
            ("optimize", "optimize.json", TRAINING, "--training"),
        ):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / filename
                config.write_text(json.dumps(raw))
                before = config.read_bytes()
                with (
                    patch("glassbox.scoring.read_all") as read,
                    patch("glassbox.cli.reward_mod.evaluate_generator") as optimize,
                ):
                    args = [command, flag, str(config), "--out", tmp]
                    if command != "study":
                        args.append("--skip-render")
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        self.assertEqual(main(args), 2)
                    read.assert_not_called()
                    optimize.assert_not_called()
                self.assertEqual(config.read_bytes(), before)

    def test_load_then_direct_run_preserves_external_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps(STUDY))
            config = studies.load_study(path)
            with patch("glassbox.studies.build_readers") as readers:
                with self.assertRaises(ValueError):
                    studies.run_study(config, tmp)
                readers.assert_not_called()

    def test_report_symlink_and_hardlink_rejected_before_work(self):
        import os

        for link in ("symbolic", "hard"):
            with self.subTest(link=link), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                source = path / "input.json"
                source.write_text("unchanged")
                out = path / "out"
                out.mkdir()
                target = out / "results.json"
                if link == "symbolic":
                    target.symlink_to(source)
                else:
                    os.link(source, target)
                with (
                    patch("glassbox.studies.build_readers") as readers,
                    patch("glassbox.scoring.read_all") as read,
                ):
                    with redirect_stderr(io.StringIO()):
                        self.assertEqual(
                            main(["run", "--skip-render", "--out", str(out)]), 2
                        )
                    readers.assert_not_called()
                    read.assert_not_called()
                self.assertEqual(source.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()

"""Frozen-selection commands, input protection, and report provenance."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from glassbox import cli, report, reward, schema, selection
from glassbox.readers import build_readers
from glassbox.stimuli import Stimulus


class TestEvaluationCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = schema.load_all_scenarios()
        readers = build_readers(["simulated"])
        sources = [cls.scenarios["loans"], cls.scenarios["plans"]]
        search = reward.evaluate_generator(sources, readers, pool=["detail", "derived"])
        cls.selection = selection.freeze_selection(search, sources, readers, True)

    def run_cli(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = cli.main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_optimize_then_evaluate_preserves_frozen_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected, evaluated = root / "selection", root / "evaluation"
            status, _, error = self.run_cli([
                "optimize", "--readers", "simulated", "--scenarios", "loans,plans",
                "--skip-render", "--out", str(selected),
            ])
            self.assertEqual(status, 0, error)
            artifact_path = selected / "selection.json"
            before = artifact_path.read_bytes()
            frozen = selection.load_selection(artifact_path)
            with patch("glassbox.reward.search_best_interface", side_effect=AssertionError("search reran")):
                status, console, error = self.run_cli([
                    "evaluate", "--selection", str(artifact_path),
                    "--scenarios", "growth", "--out", str(evaluated),
                ])
            self.assertEqual(status, 0, error)
            self.assertIn("no search was run", console)
            self.assertEqual(artifact_path.read_bytes(), before)
            payload = json.loads((evaluated / "evaluation.json").read_text())
            self.assertEqual(payload["selection_sha256"], frozen.sha256)
            self.assertEqual(payload["selected_features"], sorted(frozen.features))
            self.assertEqual(payload["selection_scenario_ids"], ["loans", "plans"])
            self.assertEqual([s["scenario_id"] for s in payload["scenarios"]], ["growth"])
            self.assertEqual(payload["reader_names"], list(frozen.reader_names))
            self.assertTrue(payload["skip_render"])
            self.assertEqual(payload["evaluation_protocol"], "frozen_features_disjoint_scenarios.v1")
            self.assertEqual(payload["baseline_sampling"], "shared_per_scenario")
            self.assertEqual(payload["selection"], selection.selection_payload(frozen))
            self.assertTrue((evaluated / "evaluation.md").is_file())

    def test_overlap_rejects_before_building_readers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "selection.json"
            selection.write_selection(path, self.selection)
            with patch("glassbox.cli.build_readers") as build:
                status, _, error = self.run_cli([
                    "evaluate", "--selection", str(path), "--scenarios", "loans",
                    "--out", str(root / "evaluation"),
                ])
                build.assert_not_called()
            self.assertEqual(status, 2)
            self.assertIn("overlaps", error)
            self.assertFalse((root / "evaluation").exists())

    def test_edited_artifact_rejects_before_building_readers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "selection.json"
            payload = selection.selection_payload(self.selection)
            payload["selection_reward"] = 0.123456
            path.write_text(json.dumps(payload))
            with patch("glassbox.cli.build_readers") as build:
                status, _, error = self.run_cli([
                    "evaluate", "--selection", str(path), "--scenarios", "growth",
                    "--out", str(root / "evaluation"),
                ])
                build.assert_not_called()
            self.assertEqual(status, 2)
            self.assertIn("digest", error)

    def test_existing_selection_rejects_before_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            selection.write_selection(path, self.selection)
            before = path.read_bytes()
            with patch("glassbox.cli.build_readers") as build, patch("glassbox.reward.evaluate_generator") as search:
                status, _, error = self.run_cli([
                    "optimize", "--readers", "simulated", "--skip-render", "--out", temporary,
                ])
                build.assert_not_called()
                search.assert_not_called()
            self.assertEqual(status, 2)
            self.assertIn("new output directory", error)
            self.assertEqual(path.read_bytes(), before)

    def test_output_alias_cannot_overwrite_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.json"
            selection.write_selection(path, self.selection)
            before = path.read_bytes()
            with patch("glassbox.cli.build_readers") as build:
                status, _, error = self.run_cli([
                    "evaluate", "--selection", str(path), "--scenarios", "growth",
                    "--out", temporary,
                ])
                build.assert_not_called()
            self.assertEqual(status, 2)
            self.assertIn("aliases input", error)
            self.assertEqual(path.read_bytes(), before)

    def test_direct_report_writer_protects_loaded_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluation.json"
            selection.write_selection(path, self.selection)
            loaded = selection.load_selection(path)
            before = path.read_bytes()
            result = reward.evaluate_selection(
                loaded, [self.scenarios["growth"]], build_readers(["simulated"])
            )
            with self.assertRaisesRegex(ValueError, "aliases input"):
                report.write_evaluation(temporary, result, loaded)
            self.assertEqual(path.read_bytes(), before)

    def test_report_rejects_mismatched_selection_before_creating_outputs(self):
        result = reward.evaluate_selection(
            self.selection, [self.scenarios["growth"]], build_readers(["simulated"])
        )
        for changed in (
            replace(result, selection_sha256="0" * 64),
            replace(result, selected_features=frozenset(["polished"])),
            replace(result, selection_scenario_ids=("growth",)),
            replace(result, selection_reward=0.123456),
        ):
            with self.subTest(result=changed), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "evaluation"
                with self.assertRaisesRegex(ValueError, "does not match the frozen selection"):
                    report.write_evaluation(output, changed, self.selection)
                self.assertFalse(output.exists())

    def test_explicit_reader_and_render_overrides_are_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "selection.json"
            selection.write_selection(path, self.selection)
            with patch("glassbox.reward.render.render", side_effect=lambda view, skip:
                       Stimulus(view, "spec", view.to_text())) as render:
                status, _, error = self.run_cli([
                    "evaluate", "--selection", str(path), "--scenarios", "growth",
                    "--readers", "simulated:literal", "--render", "--out", str(root / "evaluation"),
                ])
            self.assertEqual(status, 0, error)
            payload = json.loads((root / "evaluation/evaluation.json").read_text())
            self.assertEqual(payload["reader_names"], ["simulated:literal"])
            self.assertFalse(payload["skip_render"])
            self.assertTrue(all(call.kwargs["skip"] is False for call in render.call_args_list))
            self.assertTrue(selection.load_selection(path).skip_render)

    def test_evaluation_requires_explicit_scenarios(self):
        with patch("glassbox.cli.load_selection") as load:
            status, _, error = self.run_cli(["evaluate", "--selection", "selection.json"])
            load.assert_not_called()
        self.assertEqual(status, 2)
        self.assertIn("--scenarios", error)


if __name__ == "__main__":
    unittest.main()

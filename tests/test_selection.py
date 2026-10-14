"""Frozen feature selection, split validation, and held-out evaluation."""

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from glassbox import reward, schema, selection
from glassbox.readers import build_readers
from glassbox.readers.base import Answer, Reader
from glassbox.stimuli import Stimulus


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


class TestSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = schema.load_all_scenarios()
        cls.sources = [cls.scenarios["loans"], cls.scenarios["plans"]]
        cls.readers = build_readers(["simulated"])
        cls.search = reward.evaluate_generator(cls.sources, cls.readers,
                                               pool=["detail", "derived"])
        cls.artifact = selection.freeze_selection(cls.search, cls.sources, cls.readers, True)

    def test_round_trip_preserves_selection_provenance(self):
        payload = selection.selection_payload(self.artifact)
        parsed = selection.parse_selection(payload)
        self.assertEqual(parsed, self.artifact)
        self.assertEqual(parsed.features, self.search.generator_features)
        self.assertEqual(parsed.scenario_ids, ("loans", "plans"))
        self.assertEqual(parsed.reader_names, tuple(reader.name for reader in self.readers))
        self.assertEqual(parsed.selection_reward, self.search.generator_reward)
        self.assertTrue(parsed.skip_render)
        self.assertEqual(len(parsed.sha256), 64)
        self.assertEqual(payload["candidate_count"], 4)
        self.assertEqual(len(payload["reader_prompt_template_sha256"]), 64)
        self.assertEqual(payload["baseline_sampling"], "fresh_per_candidate_and_cards")
        for source, recorded in zip(self.sources, payload["scenarios"]):
            self.assertEqual(recorded["snapshot"], schema.scenario_payload(source))
            self.assertEqual(recorded["sha256"], schema.scenario_sha256(source))

    def test_payload_and_input_mutation_do_not_change_frozen_artifact(self):
        payload = selection.selection_payload(self.artifact)
        parsed = selection.parse_selection(payload)
        payload["features"].clear()
        payload["scenarios"][0]["snapshot"]["data"]["items"][0]["fees"] = 999
        self.assertEqual(parsed.features, self.artifact.features)
        self.assertEqual(parsed.sha256, self.artifact.sha256)
        self.assertEqual(selection.selection_payload(parsed),
                         selection.selection_payload(self.artifact))
        with self.assertRaises(FrozenInstanceError):
            parsed.payload_json = "{}"

    def test_exclusive_writer_and_explicit_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            returned = selection.write_selection(path, self.artifact)
            before = path.read_bytes()
            loaded = selection.load_selection(path)
            self.assertEqual(returned, str(path.resolve()))
            self.assertEqual(loaded.sha256, self.artifact.sha256)
            self.assertEqual(loaded.source_path, str(path.resolve()))
            with self.assertRaises(FileExistsError):
                selection.write_selection(path, self.artifact)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(selection.SelectionError):
                selection.load_selection(Path(temporary) / "missing.json")

    def test_strict_artifact_schema_features_protocol_and_digests(self):
        valid = selection.selection_payload(self.artifact)
        mutations = [
            {"extra": True}, {"features": ["headline"]}, {"features": [True]},
            {"features": ["detail", "detail"]}, {"features": "detail"},
            {"candidate_count": True}, {"candidate_count": 3},
            {"selection_reward": float("nan")}, {"selection_reward": 2},
            {"selection_reward": True}, {"skip_render": "false"},
            {"selection_protocol": "other"}, {"baseline_sampling": "other"},
            {"scenarios": []}, {"readers": []}, {"sha256": "0" * 64},
        ]
        for mutation in mutations:
            raw = dict(valid, **mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                selection.parse_selection(raw)
        for raw in (None, [], "selection", {key: value for key, value in valid.items()
                                         if key != "features"}):
            with self.subTest(raw=type(raw)), self.assertRaises(ValueError):
                selection.parse_selection(raw)

    def test_nested_snapshot_and_reader_edits_reject(self):
        valid = selection.selection_payload(self.artifact)
        for mutate in (
            lambda raw: raw["scenarios"][0].update(id="changed"),
            lambda raw: raw["scenarios"][0].update(sha256="0" * 64),
            lambda raw: raw["scenarios"][0].update(content_sha256="0" * 64),
            lambda raw: raw["scenarios"][0].update(source_sha256="0" * 64),
            lambda raw: raw["scenarios"][0]["snapshot"].update(title="changed"),
            lambda raw: raw["scenarios"][0]["snapshot"]["questions"][0].update(answer="a"),
            lambda raw: raw["readers"][0].update(extra=True),
            lambda raw: raw["readers"][0].update(simulated="false"),
            lambda raw: raw["readers"][0].update(deterministic="true"),
            lambda raw: raw["readers"][1].update(name=raw["readers"][0]["name"]),
        ):
            raw = copy.deepcopy(valid)
            mutate(raw)
            with self.assertRaises(ValueError):
                selection.parse_selection(raw)

    def test_invalid_selection_json_rejects(self):
        for text in ('{"sha256":"a","sha256":"b"}', '{"selection_reward":1e309}', '[]'):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "selection.json"
                path.write_text(text)
                with self.assertRaises(selection.SelectionError):
                    selection.load_selection(path)

    def test_freeze_rejects_mismatched_search_input_counts(self):
        with self.assertRaises(selection.SelectionError):
            selection.freeze_selection(self.search, self.sources[:1], self.readers, True)
        with self.assertRaises(selection.SelectionError):
            selection.freeze_selection(self.search, self.sources, self.readers[:1], True)

    def test_empty_feature_selection_is_valid(self):
        result = reward.evaluate_generator(self.sources, self.readers, pool=[])
        artifact = selection.freeze_selection(result, self.sources, self.readers, True)
        self.assertEqual(artifact.features, frozenset())
        self.assertEqual(selection.selection_payload(artifact)["candidate_count"], 1)

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
            with patch("glassbox.reward.render.render") as render:
                with self.assertRaises(ValueError):
                    reward.evaluate_selection(self.artifact, scenarios, [reader])
                render.assert_not_called()
            self.assertEqual(reader.calls, [])

    def test_artifact_edits_fail_before_rendering_or_answers(self):
        payload = selection.selection_payload(self.artifact)
        payload["features"] = ["polished"]
        artifact = replace(self.artifact, payload_json=json.dumps(payload))
        reader = RecordingReader()
        with patch("glassbox.reward.render.render") as render:
            with self.assertRaisesRegex(ValueError, "digest"):
                reward.evaluate_selection(artifact, [self.scenarios["growth"]], [reader])
            render.assert_not_called()
        self.assertEqual(reader.calls, [])

    def test_numeric_formatting_and_item_order_do_not_create_new_source(self):
        source = self.sources[0]
        snapshot = schema.scenario_payload(source)
        for reorder in (False, True):
            data = copy.deepcopy(source.data)
            data["items"][0]["monthly_payment"] = int(data["items"][0]["monthly_payment"])
            if reorder:
                data["items"].reverse()
            clone = replace(source, id="cosmetic-copy", title="A renamed task", data=data,
                            questions=[replace(q, id=q.id + "_copy", stem=q.stem + " Again.")
                                       for q in source.questions])
            self.assertNotEqual(schema.scenario_sha256(source), schema.scenario_sha256(clone))
            reader = RecordingReader()
            with self.subTest(reorder=reorder), patch("glassbox.reward.render.render") as render:
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    reward.evaluate_selection(self.artifact, [clone], [reader])
                render.assert_not_called()
            self.assertEqual(reader.calls, [])
        self.assertEqual(schema.scenario_payload(source), snapshot)

    def test_all_fixture_selection_leaves_no_fixture_for_evaluation(self):
        scenarios = list(self.scenarios.values())
        result = reward.evaluate_generator(scenarios, self.readers, pool=[])
        artifact = selection.freeze_selection(result, scenarios, self.readers, True)
        for scenario in scenarios:
            with self.assertRaisesRegex(ValueError, "overlaps"):
                selection.validate_evaluation(artifact, [scenario])

    def test_held_out_evaluation_does_not_search_and_shares_one_baseline(self):
        reader = RecordingReader()
        with patch("glassbox.reward.search_best_interface", side_effect=AssertionError("search reran")):
            result = reward.evaluate_selection(self.artifact, [self.scenarios["growth"]], [reader])
        self.assertEqual(result.selected_features, self.artifact.features)
        self.assertEqual(result.selection_sha256, self.artifact.sha256)
        self.assertEqual(result.selection_scenario_ids, ("loans", "plans"))
        self.assertEqual(result.selection_reward, self.search.generator_reward)
        self.assertEqual(result.reader_names, ("recording",))
        self.assertTrue(result.skip_render)
        self.assertTrue(result.all_readers_simulated)
        self.assertEqual(result.baseline_sampling, "shared_per_scenario")
        self.assertEqual(result.aggregate.selected_accuracy, 1.0)
        self.assertEqual(result.aggregate.cards_accuracy, 0.0)
        self.assertEqual(result.aggregate.baseline_accuracy, 0.0)
        self.assertEqual(result.aggregate.selected_lift, 1.0)
        self.assertEqual(result.aggregate.cards_lift, 0.0)
        self.assertEqual(result.scenarios[0].question_count, 3)
        self.assertEqual(result.scenarios[0].scenario_id, "growth")
        self.assertEqual([call[2] for call in reader.calls],
                         ["baseline"] * 3 + ["selected"] * 3 + ["cards"] * 3)

    def test_explicit_evaluation_mode_override_is_recorded(self):
        reader = RecordingReader()
        with patch("glassbox.reward.render.render", side_effect=lambda view, skip:
                   Stimulus(view, "spec", view.to_text())) as render:
            result = reward.evaluate_selection(self.artifact, [self.scenarios["growth"]],
                                               [reader], skip_render=False)
        self.assertFalse(result.skip_render)
        self.assertTrue(self.artifact.skip_render)
        self.assertTrue(all(call.kwargs["skip"] is False for call in render.call_args_list))

    def test_multiple_evaluation_scenarios_have_separate_and_mean_metrics(self):
        class ScenarioReader(RecordingReader):
            def answer(self, scenario, question, stimulus):
                answer = super().answer(scenario, question, stimulus)
                return answer if scenario.id == "plans" else Answer(None, "fixture")

        sources = [self.scenarios["loans"]]
        searched = reward.evaluate_generator(sources, self.readers, pool=[])
        artifact = selection.freeze_selection(searched, sources, self.readers, True)
        result = reward.evaluate_selection(
            artifact, [self.scenarios["plans"], self.scenarios["growth"]], [ScenarioReader()]
        )
        self.assertEqual([item.selected_accuracy for item in result.scenarios], [1.0, 0.0])
        self.assertEqual([item.selected_lift for item in result.scenarios], [1.0, 0.0])
        self.assertEqual(result.aggregate.selected_accuracy, 0.5)
        self.assertEqual(result.aggregate.selected_lift, 0.5)

    def test_invalid_render_mode_rejects_before_work(self):
        reader = RecordingReader()
        with patch("glassbox.reward.render.render") as render:
            with self.assertRaises(ValueError):
                reward.evaluate_selection(self.artifact, [self.scenarios["growth"]],
                                           [reader], skip_render="false")
            render.assert_not_called()
        self.assertEqual(reader.calls, [])


if __name__ == "__main__":
    unittest.main()

"""A stimulus keeps the presentation and metadata captured at construction."""

import copy
import unittest
from dataclasses import asdict, replace

from glassbox import interfaces, schema
from glassbox.readers import build_readers
from glassbox.stimuli import Stimulus


class TestStimulusSnapshot(unittest.TestCase):
    def setUp(self):
        self.scenario = schema.load_all_scenarios()["loans"]
        self.readers = build_readers(["simulated"])

    def _answers(self, stimulus):
        return [
            reader.answer(self.scenario, question, stimulus)
            for reader in self.readers
            for question in self.scenario.questions
        ]

    def test_caller_mutations_preserve_answers_and_recorded_input(self):
        for kind in ("spec", "image"):
            with self.subTest(kind=kind):
                presentation = interfaces.variant(self.scenario, "table")
                # Direct callers may supply a mutable feature set.
                presentation = replace(presentation, features=set(presentation.features))
                text = presentation.to_text()
                image_fields = (
                    {"image_path": "fixture.png", "image_sha256": "a" * 64}
                    if kind == "image"
                    else {}
                )
                stimulus = Stimulus(presentation, kind, text, **image_fields)
                recorded_input = asdict(stimulus.presentation)
                recorded_hash = stimulus.text_sha256
                answers = self._answers(stimulus)

                presentation.shown["A"]["total_cost"] = 1
                presentation.shown["C"] = {"total_cost": 0}
                presentation.order.reverse()
                presentation.features.add("polished")
                presentation.labels["A"] = "Changed offer"
                presentation.field_units["total_cost"] = "EUR"
                presentation.field_labels["total_cost"] = "Changed metric"

                self.assertEqual(self._answers(stimulus), answers)
                self.assertEqual(asdict(stimulus.presentation), recorded_input)
                self.assertEqual(stimulus.presentation.to_text(), text)
                self.assertEqual(stimulus.text, text)
                self.assertEqual(stimulus.text_sha256, recorded_hash)
                self.assertEqual(stimulus.image_sha256, image_fields.get("image_sha256"))

    def test_shared_source_does_not_share_snapshots_or_metadata(self):
        presentation = interfaces.variant(self.scenario, "table")
        metadata = {"window": [820, 620], "capture": {"scale": 2}}
        original_metadata = copy.deepcopy(metadata)
        spec = Stimulus(
            presentation, "spec", presentation.to_text(), render_metadata=metadata
        )
        image = Stimulus(
            presentation,
            "image",
            presentation.to_text(),
            image_path="fixture.png",
            image_sha256="b" * 64,
            render_metadata=metadata,
        )
        image_input = asdict(image.presentation)
        image_answers = self._answers(image)

        metadata["window"][0] = 1
        metadata["capture"]["scale"] = 99
        self.assertEqual(spec.render_metadata, original_metadata)
        self.assertEqual(image.render_metadata, original_metadata)

        spec.presentation.shown["A"]["total_cost"] = 1
        spec.presentation.order.reverse()
        spec.presentation.labels.clear()
        spec.render_metadata["window"][0] = 2
        spec.render_metadata["capture"]["scale"] = 3

        self.assertEqual(asdict(image.presentation), image_input)
        self.assertEqual(self._answers(image), image_answers)
        self.assertEqual(image.render_metadata, original_metadata)
        self.assertEqual(presentation.shown["A"]["total_cost"], 11280)
        self.assertEqual(image.image_path, "fixture.png")
        self.assertEqual(image.image_sha256, "b" * 64)


if __name__ == "__main__":
    unittest.main()

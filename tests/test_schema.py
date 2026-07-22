"""Schema — structural validation, including the guards added at the M1 gate."""

import copy
import unittest

from glassbox import schema
from glassbox.validate import validate_scenario

BASE = {
    "id": "t", "title": "T", "domain": "d", "description": "desc",
    "data": {
        "items": [
            {"id": "A", "label": "Apple", "x": 2, "z": 0},
            {"id": "B", "label": "Banana", "x": 1, "z": 4},
        ],
    },
    "questions": [{
        "id": "q", "stem": "which has the smallest x?",
        "compute": {"op": "argmin", "field": "x"},
        "choices": [{"id": "a", "text": "Apple", "value": "A"},
                    {"id": "b", "text": "Banana", "value": "B"}],
        "answer": "b",
    }],
}


class TestSchema(unittest.TestCase):
    def test_valid_scenario_parses(self):
        s = schema.parse_scenario(copy.deepcopy(BASE))
        self.assertEqual(s.id, "t")
        self.assertEqual(len(s.questions), 1)

    def test_choice_text_value_mismatch_rejected(self):
        raw = copy.deepcopy(BASE)
        # value "A" but text names Banana — a real reader would be marked wrong.
        raw["questions"][0]["choices"][0] = {"id": "a", "text": "Banana", "value": "A"}
        with self.assertRaises(schema.SchemaError):
            schema.parse_scenario(raw)

    def test_answer_not_a_choice_rejected(self):
        raw = copy.deepcopy(BASE)
        raw["questions"][0]["answer"] = "zzz"
        with self.assertRaises(schema.SchemaError):
            schema.parse_scenario(raw)

    def test_bad_presentation_field_rejected(self):
        raw = copy.deepcopy(BASE)
        raw["data"]["presentation"] = {"headline_field": "not_a_field"}
        with self.assertRaises(schema.SchemaError):
            schema.parse_scenario(raw)

    def test_division_by_zero_derived_reported_not_crashed(self):
        raw = copy.deepcopy(BASE)
        raw["data"]["derived"] = {"ratio": "x / z"}  # z=0 on item A
        s = schema.parse_scenario(raw)
        errors = validate_scenario(s)  # must report, not raise
        self.assertTrue(any("ratio" in e for e in errors), errors)


if __name__ == "__main__":
    unittest.main()

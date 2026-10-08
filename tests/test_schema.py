"""Schema: structural validation, including the guards added at the M1 gate."""

import copy
import unittest

from glassbox import schema
from glassbox.validate import validate_scenario

BASE = {
    "id": "t",
    "title": "T",
    "domain": "d",
    "description": "desc",
    "data": {
        "items": [
            {"id": "A", "label": "Apple", "x": 2, "z": 0},
            {"id": "B", "label": "Banana", "x": 1, "z": 4},
        ],
    },
    "questions": [
        {
            "id": "q",
            "stem": "which has the smallest x?",
            "compute": {"op": "argmin", "field": "x"},
            "choices": [
                {"id": "a", "text": "Apple", "value": "A"},
                {"id": "b", "text": "Banana", "value": "B"},
            ],
            "answer": "b",
        }
    ],
}


class TestSchema(unittest.TestCase):
    def test_valid_scenario_parses(self):
        s = schema.parse_scenario(copy.deepcopy(BASE))
        self.assertEqual(s.id, "t")
        self.assertEqual(len(s.questions), 1)

    def test_choice_text_value_mismatch_rejected(self):
        raw = copy.deepcopy(BASE)
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
        raw["data"]["derived"] = {"ratio": "x / z"}
        s = schema.parse_scenario(raw)
        errors = validate_scenario(s)
        self.assertTrue(any("ratio" in e for e in errors), errors)


class TestStrictShapes(unittest.TestCase):
    def _count(self):
        raw = copy.deepcopy(BASE)
        raw["questions"][0] = {
            "id": "count",
            "stem": "How many values exceed one?",
            "compute": {"op": "count_gt", "field": "x", "threshold": 1},
            "choices": [
                {"id": "n0", "text": "0", "value": 0},
                {"id": "n1", "text": "1", "value": 1},
                {"id": "n2", "text": "2", "value": 2},
            ],
            "answer": "n1",
        }
        return raw

    def test_count_labels_match_exact_nonnegative_integer_values(self):
        self.assertEqual(schema.parse_scenario(self._count()).questions[0].answer, "n1")
        for value, text in ((0, "2"), (True, "1"), (0.5, "0.5"), (-1, "-1"), (1, "1 or 2")):
            with self.subTest(value=value, text=text):
                raw = self._count()
                raw["questions"][0]["choices"][0].update(value=value, text=text)
                with self.assertRaises(schema.SchemaError):
                    schema.parse_scenario(raw)

    def test_duplicate_visible_answers_and_values_are_rejected(self):
        for key in ("text", "value"):
            raw = self._count()
            raw["questions"][0]["choices"][1][key] = raw["questions"][0]["choices"][0][key]
            with self.assertRaises(schema.SchemaError):
                schema.parse_scenario(raw)
        raw = copy.deepcopy(BASE)
        raw["questions"][0]["choices"][1] = {"id": "other", "text": "Apple", "value": "A"}
        with self.assertRaises(schema.SchemaError):
            schema.parse_scenario(raw)

    def test_rank_shape_and_thresholds_are_strict(self):
        for k in (True, 0, 1.5, 3, "2"):
            raw = copy.deepcopy(BASE)
            raw["questions"][0]["compute"].update(op="rank", k=k)
            with self.subTest(k=k), self.assertRaises(schema.SchemaError):
                schema.parse_scenario(raw)
        raw = copy.deepcopy(BASE)
        raw["questions"][0]["compute"].update(op="rank", k=1, order="sideways")
        with self.assertRaises(schema.SchemaError):
            schema.parse_scenario(raw)
        for threshold in (True, None, "1", float("nan"), float("inf")):
            raw = self._count()
            raw["questions"][0]["compute"]["threshold"] = threshold
            with self.subTest(threshold=threshold), self.assertRaises(schema.SchemaError):
                schema.parse_scenario(raw)

    def test_malformed_shapes_and_nonfinite_targets_raise_schema_error(self):
        for raw in (None, [], "scenario", True):
            with self.subTest(raw=raw), self.assertRaises(schema.SchemaError):
                schema.parse_scenario(raw)
        for value in (True, "2", float("nan"), float("inf")):
            raw = copy.deepcopy(BASE)
            raw["data"]["items"][0]["x"] = value
            with self.subTest(value=value), self.assertRaises(schema.SchemaError):
                schema.parse_scenario(raw)
        for path, value in (("op", []), ("field", {}), ("answer", [])):
            raw = copy.deepcopy(BASE)
            target = raw["questions"][0]
            (target if path == "answer" else target["compute"])[path] = value
            with self.assertRaises(schema.SchemaError):
                schema.parse_scenario(raw)

    def test_direct_scenario_guards_and_no_parse_aliasing(self):
        from dataclasses import replace

        raw = copy.deepcopy(BASE)
        scenario = schema.parse_scenario(raw)
        raw["data"]["items"][0]["x"] = 99
        self.assertEqual(scenario.items[0]["x"], 2)
        with self.assertRaises(schema.SchemaError):
            schema.validate_structure(replace(scenario, questions=[]))
        self.assertTrue(validate_scenario(replace(scenario, questions=[])))

    def test_choice_count_matches_letter_parser_capacity(self):
        raw = self._count()
        raw["questions"][0]["choices"] = [
            {"id": "n%d" % index, "text": str(index), "value": index} for index in range(27)
        ]
        with self.assertRaisesRegex(schema.SchemaError, "2 to 26"):
            schema.parse_scenario(raw)

    def test_canonical_hash_covers_authored_task_and_display(self):
        from dataclasses import replace

        scenario = schema.parse_scenario(copy.deepcopy(BASE))
        original = schema.scenario_sha256(scenario)
        for key in ("title", "description", "domain", "id"):
            self.assertNotEqual(
                original, schema.scenario_sha256(replace(scenario, **{key: "changed"}))
            )
        q = scenario.questions[0]
        for replacement in (
            replace(q, stem="Changed question"),
            replace(q, answer="a"),
            replace(q, compute={"op": "argmax", "field": "x"}),
            replace(q, choices=[replace(q.choices[0], text="Changed Apple"), q.choices[1]]),
            replace(q, choices=[replace(q.choices[0], id="z"), q.choices[1]]),
            replace(q, choices=[replace(q.choices[0], value="B"), q.choices[1]]),
        ):
            self.assertNotEqual(
                original, schema.scenario_sha256(replace(scenario, questions=[replacement]))
            )
        data = copy.deepcopy(scenario.data)
        data["presentation"] = {"primary_metric": "z", "primary_extreme": "argmax"}
        self.assertNotEqual(original, schema.scenario_sha256(replace(scenario, data=data)))
        snapshot = schema.scenario_payload(scenario)
        snapshot["data"]["items"][0]["x"] = 99
        self.assertEqual(scenario.items[0]["x"], 2)
        count = schema.parse_scenario(self._count())
        question = count.questions[0]
        threshold = replace(question, compute=dict(question.compute, threshold=2))
        self.assertNotEqual(
            schema.scenario_sha256(count),
            schema.scenario_sha256(replace(count, questions=[threshold])),
        )

    def test_key_order_independent_hash_and_default_metric(self):
        from dataclasses import replace

        scenario = schema.parse_scenario(copy.deepcopy(BASE))
        data = copy.deepcopy(scenario.data)
        data["derived"] = {"z_metric": "z * 2", "a_metric": "x * 2"}
        first = replace(scenario, data=data)
        other = copy.deepcopy(data)
        other["derived"] = dict(reversed(list(other["derived"].items())))
        other["items"] = [dict(reversed(list(item.items()))) for item in other["items"]]
        second = replace(scenario, data=dict(reversed(list(other.items()))))
        self.assertEqual(first.presentation.primary_metric, "a_metric")
        self.assertEqual(schema.scenario_sha256(first), schema.scenario_sha256(second))


if __name__ == "__main__":
    unittest.main()

"""Recompute shipped answers and verify protected source bytes."""

import unittest

from glassbox import schema
from glassbox.validate import validate_all, validate_scenario


class TestShippedScenarios(unittest.TestCase):
    def test_all_scenarios_validate(self):
        ok, report = validate_all()
        self.assertTrue(ok, "scenario integrity failed:\n" + "\n".join(report))

    def test_at_least_three_scenarios(self):
        scenarios = schema.load_all_scenarios()
        self.assertGreaterEqual(len(scenarios), 3, "need >= 3 scenarios, found %d" % len(scenarios))

    def test_every_scenario_has_multiple_questions(self):
        for sid, scenario in schema.load_all_scenarios().items():
            self.assertGreaterEqual(len(scenario.questions), 1, "%s has no questions" % sid)
            self.assertEqual(validate_scenario(scenario), [], "%s failed integrity" % sid)

    def test_loans_worked_example(self):
        scenarios = schema.load_all_scenarios()
        self.assertIn("loans", scenarios)
        loans = scenarios["loans"]
        q = next(q for q in loans.questions if q.id == "cheapest_total")
        self.assertEqual(q.correct_choice.value, "B")


class TestPreservedContent(unittest.TestCase):
    def test_scenarios_reader_prompt_and_license_keep_original_bytes(self):
        import hashlib
        from pathlib import Path

        from glassbox.resources import resource_sha256

        expected = {
            "growth": "c29195c09415b4e53e496cdc2d92fc6f91136ad048159eb60753507644d01918",
            "loans": "894c88d07f167306dd7de6aafc79f9caaf69ea12538a3824be1c8b876ea6b745",
            "plans": "3895dd5871b3d6965104c04894ec33a776dad88155f1f1663a88c84916995688",
        }
        for name, digest in expected.items():
            self.assertEqual(resource_sha256("scenarios", name), digest)
        self.assertEqual(
            resource_sha256("prompts", "reader_mcq"),
            "f887de0d6020ef5b2453e8d3dba968e43040589787ca5a10b6b6f805ac27b54a",
        )
        license_path = Path(__file__).resolve().parents[1] / "LICENSE"
        self.assertEqual(
            hashlib.sha256(license_path.read_bytes()).hexdigest(),
            "9257fc0549bacf1cbfb0d15adcdbcf82f6cca00961a95f51972b907410bcd0c1",
        )

    def test_exact_nine_authored_answers_and_source_values(self):
        from glassbox.compute import compute_answer_value

        scenarios = schema.load_all_scenarios()
        expected = {
            ("growth", "biggest_dollar_gain"): "West",
            ("growth", "biggest_percent_gain"): "South",
            ("growth", "count_gain_over_50"): 2,
            ("loans", "cheapest_total"): "B",
            ("loans", "second_cheapest"): "A",
            ("loans", "count_over_11k"): 2,
            ("plans", "cheapest_2yr"): "Y",
            ("plans", "most_expensive_2yr"): "X",
            ("plans", "count_over_1500"): 2,
        }
        observed = {
            (sid, question.id): compute_answer_value(scenario.data, question.compute)
            for sid, scenario in scenarios.items()
            for question in scenario.questions
        }
        self.assertEqual(observed, expected)
        self.assertEqual(sum(len(scenario.questions) for scenario in scenarios.values()), 9)


if __name__ == "__main__":
    unittest.main()

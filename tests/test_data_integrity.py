"""Data integrity — every shipped scenario's answer key recomputes from its source.

This is the test form of ``glassbox validate``: if any authored answer disagrees
with the value derived from the data, the suite fails. It is the guard that keeps
the benchmark's ground truth honest.
"""

import unittest

from glassbox import schema
from glassbox.validate import validate_all, validate_scenario


class TestShippedScenarios(unittest.TestCase):
    def test_all_scenarios_validate(self):
        ok, report = validate_all()
        self.assertTrue(ok, "scenario integrity failed:\n" + "\n".join(report))

    def test_at_least_three_scenarios(self):
        # Success criterion: >= 3 scenarios.
        scenarios = schema.load_all_scenarios()
        self.assertGreaterEqual(len(scenarios), 3, "need >= 3 scenarios, found %d" % len(scenarios))

    def test_every_scenario_has_multiple_questions(self):
        for sid, scenario in schema.load_all_scenarios().items():
            self.assertGreaterEqual(len(scenario.questions), 1, "%s has no questions" % sid)
            self.assertEqual(validate_scenario(scenario), [], "%s failed integrity" % sid)

    def test_loans_worked_example(self):
        # The canonical H1 example: computed cheapest (B) differs from the cheapest
        # by advertised total (A) — the trap the benchmark is built around.
        scenarios = schema.load_all_scenarios()
        self.assertIn("loans", scenarios)
        loans = scenarios["loans"]
        q = next(q for q in loans.questions if q.id == "cheapest_total")
        self.assertEqual(q.correct_choice.value, "B")  # true cheapest by total_cost


if __name__ == "__main__":
    unittest.main()

"""Verify bounded repair views for every authored question."""

import unittest

from glassbox import repair, schema
from glassbox.readers import build_reader, build_readers


class TestRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loans = schema.load_all_scenarios()["loans"]
        cls.q = cls.loans.questions[0]  # cheapest_total

    def test_literal_trips_then_understands(self):
        r = build_reader("simulated:literal")
        res = repair.repair(self.loans, r, self.q, skip_render=True)
        self.assertFalse(res.turns[0].correct, "should start wrong on misleading cards")
        self.assertTrue(res.converged)
        self.assertEqual(res.turns_to_understanding, 1)
        self.assertTrue(res.turns[-1].correct)

    def test_first_repair_shows_the_metric(self):
        r = build_reader("simulated:literal")
        res = repair.repair(self.loans, r, self.q, skip_render=True)
        self.assertIn("derived", res.turns[1].features)  # the computed metric is now shown

    def test_every_simulated_reader_converges(self):
        for r in build_readers(["simulated"]):
            for sid, scenario in schema.load_all_scenarios().items():
                res = repair.repair(scenario, r, scenario.questions[0], skip_render=True)
                self.assertTrue(res.converged, "%s did not converge on %s" % (r.name, sid))
                self.assertIsNotNone(res.turns_to_understanding)

    def test_turn_indices_are_sequential(self):
        r = build_reader("simulated:diligent")
        res = repair.repair(self.loans, r, self.q, skip_render=True)
        self.assertEqual([t.turn for t in res.turns], list(range(len(res.turns))))


class TestQuestionRepair(unittest.TestCase):
    def test_all_nine_questions_and_three_personas_converge_without_source_changes(self):
        from unittest.mock import patch

        from glassbox import interfaces

        count = 0
        for scenario in schema.load_all_scenarios().values():
            before = schema.scenario_payload(scenario)
            for question in scenario.questions:
                for reader in build_readers(["simulated"]):
                    with self.subTest(
                        scenario=scenario.id, question=question.id, reader=reader.name
                    ):
                        seen = []
                        from glassbox.render import render as original_render

                        def record(presentation, **kwargs):
                            seen.append(presentation)
                            return original_render(presentation, **kwargs)

                        with patch("glassbox.repair.render.render", side_effect=record):
                            result = repair.repair(scenario, reader, question, skip_render=True)
                        self.assertTrue(result.converged)
                        self.assertTrue(
                            all(p.primary_metric == question.target_field for p in seen)
                        )
                        if question.compute["op"].startswith("count_"):
                            self.assertTrue(all(p.highlight_item is None for p in seen))
                        count += 1
                all_features = frozenset({"detail", "derived", "highlight", "sorted", "polished"})
                view = interfaces.build(scenario, all_features, target=question)
                self.assertEqual(view.target_question_id, question.id)
                if question.compute["op"] == "rank":
                    self.assertEqual(view.highlight_item, "A")
                    self.assertEqual(view.highlight_label, "rank 2")
                    self.assertNotIn("highlighted as best", view.to_text())
            self.assertEqual(before, schema.scenario_payload(scenario))
        self.assertEqual(count, 27)

    def test_percent_target_does_not_inherit_currency_and_plan_argmax_is_correct(self):
        from glassbox import interfaces

        scenarios = schema.load_all_scenarios()
        growth = scenarios["growth"]
        question = next(q for q in growth.questions if q.id == "biggest_percent_gain")
        view = interfaces.variant(growth, "annotated", target=question)
        self.assertEqual(view.highlight_item, "South")
        self.assertEqual(view.shown["South"]["growth_pct"], 90)
        self.assertEqual(view.unit, "")
        self.assertNotIn("$90k", view.to_text())
        self.assertNotIn("$90k", view.to_html())
        question = next(q for q in scenarios["plans"].questions if q.id == "most_expensive_2yr")
        self.assertEqual(
            interfaces.variant(scenarios["plans"], "annotated", target=question).highlight_item, "X"
        )

    def test_nonprimary_raw_headline_needs_explicit_units(self):
        import copy
        from dataclasses import replace

        from glassbox import interfaces

        scenario = schema.load_all_scenarios()["growth"]
        question = replace(
            scenario.questions[0], compute={"op": "argmax", "field": "end"}, answer="e"
        )
        scenario = replace(scenario, questions=[question])
        view = interfaces.variant(scenario, "annotated", target=question)
        self.assertEqual(view.unit, "")
        self.assertNotIn("$345k", view.to_text())
        data = copy.deepcopy(scenario.data)
        data["presentation"]["field_display"] = {
            "end": {"unit": "$k", "value_label": "Ending revenue"}
        }
        view = interfaces.variant(replace(scenario, data=data), "annotated", target=question)
        self.assertEqual(view.unit, "$k")
        self.assertIn("Ending revenue = $345k", view.to_text())

    def test_zero_turn_budget_and_stubborn_reader_are_bounded(self):
        from glassbox.readers.base import Answer, Reader

        scenario = schema.load_all_scenarios()["loans"]
        question = scenario.questions[0]

        class Stubborn(Reader):
            name = "stubborn"

            def answer(self, *args):
                return Answer(None)

        result = repair.repair(scenario, Stubborn(), question, max_turns=0, skip_render=True)
        self.assertEqual(len(result.turns), 1)
        self.assertFalse(result.converged)
        self.assertIsNone(result.turns_to_understanding)
        result = repair.repair(scenario, Stubborn(), question, max_turns=4, skip_render=True)
        self.assertLessEqual(len(result.turns), 5)
        self.assertFalse(result.converged)
        for bad in (True, -1, 1.5, "2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                repair.repair(scenario, Stubborn(), question, max_turns=bad, skip_render=True)

    def test_target_context_survives_feature_additions_and_escaping(self):
        from dataclasses import replace

        from glassbox import interfaces

        scenario = schema.load_all_scenarios()["growth"]
        question = scenario.questions[1]
        base = interfaces.variant(scenario, "cards", target=question)
        updated = interfaces.with_features(scenario, base, frozenset({"derived"}))
        self.assertEqual(updated.primary_metric, "growth_pct")
        self.assertEqual(updated.target_question_id, question.id)
        changed = replace(
            updated, title="<script>alert(1)</script>", target_stem="<img src=x onerror=alert(1)>"
        )
        self.assertNotIn("<script>", changed.to_html())
        self.assertNotIn("<img src=x", changed.to_html())
        with self.assertRaises(ValueError):
            interfaces.variant(scenario, "cards", target=replace(question, id="missing"))


if __name__ == "__main__":
    unittest.main()

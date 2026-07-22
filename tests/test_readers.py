"""Simulated readers — deterministic behavior and the H1 fixture properties."""

import unittest

from glassbox import interfaces, render, schema
from glassbox.readers import build_reader, build_readers, expand_reader_specs


def _score(reader, scenario, variant):
    p = interfaces.variant(scenario, variant)
    st = render.render(p, skip=True)
    return sum(1 for q in scenario.questions
               if reader.answer(scenario, q, st).choice_id == q.answer)


class TestReaderRegistry(unittest.TestCase):
    def test_expand_simulated(self):
        specs = expand_reader_specs(["simulated"])
        self.assertEqual(specs, ["simulated:literal", "simulated:diligent", "simulated:careless"])

    def test_bare_simulated_rejected(self):
        with self.assertRaises(ValueError):
            build_reader("simulated")

    def test_unknown_family_rejected(self):
        with self.assertRaises(ValueError):
            build_reader("martian:x")

    def test_at_least_three_readers(self):
        # Success criterion: >= 3 readers usable offline.
        self.assertGreaterEqual(len(build_readers(["simulated"])), 3)


class TestDeterminism(unittest.TestCase):
    def test_same_answer_every_time(self):
        s = schema.load_all_scenarios()["loans"]
        r = build_reader("simulated:careless")
        p = interfaces.variant(s, "cards")
        st = render.render(p, skip=True)
        q = s.questions[0]
        answers = {r.answer(s, q, st).choice_id for _ in range(5)}
        self.assertEqual(len(answers), 1)


class TestCarelessSlip(unittest.TestCase):
    """The careless persona's defining behavior: slip on a close call, rescued by a highlight."""

    def setUp(self):
        self.loans = schema.load_all_scenarios()["loans"]
        self.reader = build_reader("simulated:careless")
        self.q = next(q for q in self.loans.questions if q.id == "cheapest_total")

    def _answer(self, variant):
        st = render.render(interfaces.variant(self.loans, variant), skip=True)
        return self.reader.answer(self.loans, self.q, st).choice_id

    def test_slips_to_runner_up_on_close_call_without_highlight(self):
        # loans cheapest: true B vs A within 5% and the table shows no highlight.
        self.assertEqual(self._answer("table"), "a")   # slip to the runner-up (wrong)

    def test_highlight_of_true_extreme_rescues_the_slip(self):
        self.assertEqual(self._answer("annotated"), "b")  # trusts the highlight (correct)


class TestH1Fixture(unittest.TestCase):
    """The polished cards must transfer *less* understanding than plain data."""

    def setUp(self):
        self.scenarios = schema.load_all_scenarios()
        self.literal = build_reader("simulated:literal")
        self.diligent = build_reader("simulated:diligent")

    def test_literal_fails_on_polished_cards_everywhere(self):
        for sid, s in self.scenarios.items():
            self.assertEqual(_score(self.literal, s, "cards"), 0,
                             "%s: literal reader should score 0 on misleading cards" % sid)

    def test_diligent_computes_from_plain_data(self):
        # With every raw field visible, a diligent reader answers perfectly.
        for sid, s in self.scenarios.items():
            self.assertEqual(_score(self.diligent, s, "baseline"), len(s.questions),
                             "%s: diligent should ace the plain-text baseline" % sid)

    def test_cards_hurt_a_capable_reader(self):
        # The heart of H1: a capable reader does WORSE on cards than on plain text.
        for sid, s in self.scenarios.items():
            cards = _score(self.diligent, s, "cards")
            baseline = _score(self.diligent, s, "baseline")
            self.assertLess(cards, baseline,
                            "%s: diligent should score lower on cards than baseline" % sid)

    def test_honest_table_helps_literal_reader(self):
        # Showing the computed metric lets even a literal reader answer its question.
        for sid, s in self.scenarios.items():
            self.assertGreater(_score(self.literal, s, "table"),
                               _score(self.literal, s, "baseline"),
                               "%s: table should lift the literal reader over baseline" % sid)


if __name__ == "__main__":
    unittest.main()

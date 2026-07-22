"""H4 repair loop — the reader trips, the interface regenerates, understanding lands."""

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
                self.assertTrue(res.converged,
                                "%s did not converge on %s" % (r.name, sid))
                self.assertIsNotNone(res.turns_to_understanding)

    def test_turn_indices_are_sequential(self):
        r = build_reader("simulated:diligent")
        res = repair.repair(self.loans, r, self.q, skip_render=True)
        self.assertEqual([t.turn for t in res.turns], list(range(len(res.turns))))


if __name__ == "__main__":
    unittest.main()

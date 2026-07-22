"""Scoring metrics — accuracy, ceiling, baseline, lift — over the real pipeline."""

import unittest

from glassbox import schema, scoring
from glassbox.readers import build_readers


class TestScoreBook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(schema.load_all_scenarios().values())
        cls.readers = build_readers(["simulated"])
        cls.book = scoring.ScoreBook(
            scoring.read_all(cls.scenarios, cls.readers, skip_render=True))

    def test_accuracy_bounds(self):
        for sid in self.book.scenarios:
            for v in self.book.variants:
                for r in self.book.readers:
                    a = self.book.accuracy(r, sid, v)
                    self.assertGreaterEqual(a, 0.0)
                    self.assertLessEqual(a, 1.0)

    def test_lift_is_accuracy_minus_baseline(self):
        r, sid, v = self.book.readers[0], "loans", "table"
        expected = self.book.accuracy(r, sid, v) - self.book.baseline(r, sid)
        self.assertAlmostEqual(self.book.lift(r, sid, v), expected)

    def test_cards_have_negative_mean_lift(self):
        # The polished cards mislead: negative comprehension lift in every scenario.
        for sid in self.book.scenarios:
            self.assertLess(self.book.mean_lift(sid, "cards"), 0.0,
                            "%s: cards should have negative mean lift" % sid)

    def test_table_beats_cards_on_comprehension(self):
        for sid in self.book.scenarios:
            self.assertGreater(self.book.mean_lift(sid, "table"),
                               self.book.mean_lift(sid, "cards"))

    def test_all_simulated_flag(self):
        self.assertTrue(self.book.all_simulated())
        self.assertFalse(self.book.any_real())

    def test_dimensions_meet_success_criteria(self):
        # >= 3 scenarios x >= 3 interface variants x >= 3 readers.
        self.assertGreaterEqual(len(self.book.scenarios), 3)
        self.assertGreaterEqual(len(self.book.interface_variants()), 3)
        self.assertGreaterEqual(len(self.book.readers), 3)


if __name__ == "__main__":
    unittest.main()

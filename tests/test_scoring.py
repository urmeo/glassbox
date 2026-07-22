"""Scoring metrics — accuracy, ceiling, baseline, lift — over the real pipeline."""

import unittest

from glassbox import schema, scoring
from glassbox.readers import build_reader, build_readers
from glassbox.readers.base import Answer, Reader


class _FlakyReader(Reader):
    """A synthetic reader that flips answers across calls — stands in for a
    stochastic real reader so the replicate machinery can be tested offline."""
    simulated = True
    family = "synthetic"
    name = "synthetic:flaky"

    def __init__(self):
        self._n = 0

    def answer(self, scenario, question, stimulus):
        self._n += 1
        choice = question.choices[self._n % len(question.choices)]
        return Answer(choice_id=choice.id, method="synthetic")


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


class TestReplicates(unittest.TestCase):
    def setUp(self):
        self.loans = schema.load_all_scenarios()["loans"]

    def test_deterministic_reader_has_zero_spread(self):
        book = scoring.ScoreBook(scoring.read_all(
            [self.loans], [build_reader("simulated:literal")],
            variants=["table"], skip_render=True, replicates=3))
        self.assertEqual(book.replicate_count(), 3)
        self.assertEqual(book.accuracy_std("simulated:literal", "loans", "table"), 0.0)
        self.assertEqual(book.answer_stability("simulated:literal", "loans", "table"), 1.0)

    def test_stochastic_reader_shows_spread(self):
        book = scoring.ScoreBook(scoring.read_all(
            [self.loans], [_FlakyReader()], variants=["table"],
            skip_render=True, replicates=4))
        self.assertEqual(book.replicate_count(), 4)
        # A reader that flips answers is not perfectly stable.
        self.assertLess(book.answer_stability("synthetic:flaky", "loans", "table"), 1.0)

    def test_family_grouping(self):
        book = scoring.ScoreBook(scoring.read_all(
            [self.loans], build_readers(["simulated"]), variants=["table"], skip_render=True))
        self.assertEqual(book.families(), ["simulated"])
        self.assertEqual(len(book.readers_in_family("simulated")), 3)


if __name__ == "__main__":
    unittest.main()

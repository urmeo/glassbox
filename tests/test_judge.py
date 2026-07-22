"""Preference judges — the pairwise upgrade (simulated + real adapter contract)."""

import os
import unittest

from glassbox import analysis, interfaces, schema, scoring
from glassbox.judge import (ApiPairwiseJudge, PairwiseJudge, PairwiseRatingJudge,
                            SimulatedPairwiseJudge, build_judge, parse_ab)
from glassbox.readers import build_readers
from glassbox.readers._http import MissingKeyError


class TestPairwise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loans = schema.load_all_scenarios()["loans"]

    def test_parse_ab(self):
        self.assertEqual(parse_ab("A"), "A")
        self.assertEqual(parse_ab("I prefer B."), "B")
        self.assertIsNone(parse_ab("neither, honestly"))

    def test_simulated_polished_wins(self):
        j = SimulatedPairwiseJudge()
        cards = interfaces.variant(self.loans, "cards")
        table = interfaces.variant(self.loans, "table")
        self.assertEqual(j.compare(self.loans, cards, table), "A")   # cards more polished
        self.assertEqual(j.compare(self.loans, table, cards), "B")   # order-independent

    def test_winrate_ranking(self):
        j = PairwiseRatingJudge(SimulatedPairwiseJudge())
        wr = {v: j.preference(self.loans, interfaces.variant(self.loans, v))
              for v in ("cards", "table", "annotated")}
        self.assertEqual(wr["annotated"], 1.0)   # beats both
        self.assertEqual(wr["table"], 0.0)        # beats none
        self.assertAlmostEqual(wr["cards"], 0.5)  # beats table only

    def test_build_judge_pairwise(self):
        self.assertIsInstance(build_judge("pairwise"), PairwiseRatingJudge)

    def test_tie_splits_the_point_no_positional_bias(self):
        class _TieJudge(PairwiseJudge):
            simulated = True
            def compare(self, scenario, a, b):
                return "tie"
        j = PairwiseRatingJudge(_TieJudge())
        wr = {v: j.preference(self.loans, interfaces.variant(self.loans, v))
              for v in ("cards", "table", "annotated")}
        for v, rate in wr.items():
            self.assertAlmostEqual(rate, 0.5, msg=v)  # all ties -> no variant favored

    def test_pairwise_preserves_divergence(self):
        scenarios = list(schema.load_all_scenarios().values())
        book = scoring.ScoreBook(scoring.read_all(
            scenarios, build_readers(["simulated"]), skip_render=True))
        rep = analysis.analyze_h1(book, scenarios, build_judge("pairwise"))
        self.assertTrue(rep.divergence_found)


class TestApiPairwiseJudge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loans = schema.load_all_scenarios()["loans"]

    def test_payload_is_text_only(self):
        j = ApiPairwiseJudge("anthropic:claude-sonnet-5")
        a = interfaces.variant(self.loans, "cards")
        b = interfaces.variant(self.loans, "table")
        payload = j.build_payload(a, b)
        self.assertEqual(payload["model"], "claude-sonnet-5")
        text = payload["messages"][0]["content"][0]["text"]
        self.assertIn("Interface A", text)
        self.assertIn("Interface B", text)

    def test_missing_key_raises(self):
        j = ApiPairwiseJudge("anthropic:claude-sonnet-5")
        a = interfaces.variant(self.loans, "cards")
        b = interfaces.variant(self.loans, "table")
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(MissingKeyError):
                j.compare(self.loans, a, b)
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

    def test_build_judge_real_pairwise(self):
        j = build_judge("pairwise:openrouter:qwen/qwen3-vl-8b-instruct")
        self.assertIsInstance(j, PairwiseRatingJudge)
        self.assertFalse(j.simulated)


if __name__ == "__main__":
    unittest.main()

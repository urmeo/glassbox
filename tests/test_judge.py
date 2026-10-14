"""Polish and text pairwise judge contracts."""

import hashlib
import math
import os
import unittest
from unittest.mock import patch

from glassbox import analysis, interfaces, schema, scoring
from glassbox.judge import (
    ApiPairwiseJudge,
    PairwiseJudge,
    PairwiseJudgment,
    PairwiseRatingJudge,
    SimulatedPairwiseJudge,
    build_judge,
    parse_ab,
)
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
        self.assertEqual(
            j.compare(self.loans, cards, table), "A"
        )
        self.assertEqual(j.compare(self.loans, table, cards), "B")

    def test_winrate_ranking(self):
        j = PairwiseRatingJudge(SimulatedPairwiseJudge())
        wr = {
            v: j.preference(self.loans, interfaces.variant(self.loans, v))
            for v in ("cards", "table", "annotated")
        }
        self.assertEqual(wr["annotated"], 1.0)
        self.assertEqual(wr["table"], 0.0)
        self.assertAlmostEqual(wr["cards"], 0.5)

    def test_build_judge_pairwise(self):
        self.assertIsInstance(build_judge("pairwise"), PairwiseRatingJudge)

    def test_tie_splits_the_point_no_positional_bias(self):
        class _TieJudge(PairwiseJudge):
            simulated = True

            def compare(self, scenario, a, b):
                return "tie"

        j = PairwiseRatingJudge(_TieJudge())
        wr = {
            v: j.preference(self.loans, interfaces.variant(self.loans, v))
            for v in ("cards", "table", "annotated")
        }
        for v, rate in wr.items():
            self.assertAlmostEqual(rate, 0.5, msg=v)
        self.assertEqual(j.metadata().valid_count, 6)
        self.assertEqual(j.metadata().tie_count, 6)

    def test_always_first_choice_does_not_favor_a_variant(self):
        class AlwaysFirst(PairwiseJudge):
            def compare(self, scenario, a, b):
                return "A"

        for variants in (("cards", "table"), ("table", "cards")):
            judge = PairwiseRatingJudge(AlwaysFirst(), variants)
            for variant in variants:
                self.assertEqual(
                    judge.preference(
                        self.loans, interfaces.variant(self.loans, variant)
                    ),
                    0.5,
                )
            metadata = judge.metadata()
            self.assertEqual(metadata.comparison_count, 2)
            self.assertEqual(metadata.protocol, "counterbalanced_pairs_winrate.v2")
            evidence = metadata.evidence[0]
            self.assertEqual(
                [(item.variant_a, item.variant_b) for item in evidence.comparisons],
                [variants, tuple(reversed(variants))],
            )
            self.assertTrue(all(
                item.denominator == 2 for item in evidence.variant_scores
            ))

    def test_mixed_outcomes_record_credit_and_coverage_without_cache_duplicates(self):
        class MixedJudge(PairwiseJudge):
            def __init__(self):
                self.replies = iter((
                    "A",
                    PairwiseJudgment("invalid", "A or B"),
                    PairwiseJudgment("abstained", "ABSTAIN"),
                    "tie",
                    "B",
                    PairwiseJudgment("invalid", "unclear"),
                ))
                self.calls = 0

            def compare(self, scenario, a, b):
                self.calls += 1
                return next(self.replies)

        pairwise = MixedJudge()
        judge = PairwiseRatingJudge(pairwise)
        expected = {"cards": 0.75, "table": 0.0, "annotated": 0.75}
        for _ in range(2):
            for variant, score in expected.items():
                self.assertEqual(
                    judge.preference(
                        self.loans, interfaces.variant(self.loans, variant)
                    ),
                    score,
                )
        metadata = judge.metadata()
        self.assertEqual(pairwise.calls, 6)
        self.assertEqual(metadata.comparison_count, 6)
        self.assertEqual(
            (metadata.valid_count, metadata.invalid_count,
             metadata.abstained_count, metadata.tie_count),
            (3, 2, 1, 1),
        )
        evidence = metadata.evidence[0]
        self.assertEqual(
            {item.variant: item.credit for item in evidence.variant_scores},
            {"cards": 1.5, "table": 0.0, "annotated": 1.5},
        )
        self.assertTrue(all(item.denominator == 2 for item in evidence.variant_scores))
        self.assertTrue(all(
            item.planned_count == 4 for item in evidence.variant_scores
        ))
        self.assertEqual(evidence.comparisons[1].raw, "A or B")
        self.assertEqual(evidence.comparisons[2].status, "abstained")

    def test_pairwise_preserves_divergence(self):
        scenarios = list(schema.load_all_scenarios().values())
        book = scoring.ScoreBook(
            scoring.read_all(scenarios, build_readers(["simulated"]), skip_render=True)
        )
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

    def test_api_reply_status_and_actual_sent_prompt_hash(self):
        judge = ApiPairwiseJudge("anthropic:test-model")
        a = interfaces.variant(self.loans, "cards")
        b = interfaces.variant(self.loans, "table")
        payload = {
            "messages": [{"content": [{"type": "text", "text": "Actual sent prompt"}]}]
        }
        for raw, outcome in (
            ("B", "B"),
            ("A or B", "invalid"),
            ("tie", "invalid"),
            ("I cannot decide", "invalid"),
            ("", "invalid"),
            ("ABSTAIN", "abstained"),
            ("abstained", "abstained"),
        ):
            with (
                self.subTest(raw=raw),
                patch.object(judge, "build_payload", return_value=payload) as build,
                patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
                patch(
                    "glassbox.readers._http.post_json",
                    return_value={"content": [{"type": "text", "text": raw}]},
                ) as send,
            ):
                result = judge.compare(self.loans, a, b)
            self.assertEqual(result.outcome, outcome)
            self.assertEqual(result.raw, raw)
            self.assertEqual(
                result.prompt_sha256,
                hashlib.sha256(b"Actual sent prompt").hexdigest(),
            )
            self.assertIs(send.call_args.args[2], payload)
            build.assert_called_once()

    def test_all_invalid_api_replies_have_no_credit_or_score(self):
        judge = PairwiseRatingJudge(
            ApiPairwiseJudge("anthropic:test-model"), ["cards", "table"]
        )
        replies = [
            {"content": [{"type": "text", "text": text}]}
            for text in ("A or B", "unparseable")
        ]
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake-key"}),
            patch("glassbox.readers._http.post_json", side_effect=replies) as send,
        ):
            for variant in ("cards", "table", "cards"):
                self.assertTrue(math.isnan(judge.preference(
                    self.loans, interfaces.variant(self.loans, variant)
                )))
        self.assertEqual(send.call_count, 2)
        metadata = judge.metadata()
        self.assertEqual(metadata.invalid_count, 2)
        self.assertEqual(metadata.valid_count, 0)
        self.assertEqual(metadata.tie_count, 0)
        evidence = metadata.evidence[0]
        for item in evidence.variant_scores:
            self.assertEqual(
                (item.credit, item.denominator, item.planned_count), (0, 0, 2)
            )
            self.assertTrue(math.isnan(item.score))
        self.assertEqual(
            [item.raw for item in evidence.comparisons], ["A or B", "unparseable"]
        )


if __name__ == "__main__":
    unittest.main()

"""Explicit answer grammar and provider-independent family identities."""

import unittest

from glassbox.answer_parsing import parse_choice_letter
from glassbox.families import canonical_family, model_family, resolve_family
from glassbox.judge import parse_ab
from glassbox.readers import build_reader
from glassbox.readers.base import Reader, parse_choice
from glassbox.schema import load_all_scenarios


class TestChoiceGrammar(unittest.TestCase):
    def test_declared_choices_and_final_answer_precedence(self):
        cases = {
            "B": "B",
            "b) Offer B": "B",
            "The answer is C.": "C",
            "Answer: A": "A",
            "A clear answer is hard. My choice is B.": "B",
            "I prefer B.": "B",
            "Comparing Offer C to B, I would pick B.": "B",
            "Between A and C, C is cheapest, so C.": "C",
            "Choice: 'B'.": "B",
            "Final answer: B. Earlier choice: A.": "B",
            "Select C.": "C",
            "b)OfferB": "B",
            "Final answer: B\nExplanation: fees are lower.": "B",
            "Answer: B because fees are lower.": "B",
            "Answer: B (fees are lower).": "B",
        }
        for reply, expected in cases.items():
            with self.subTest(reply=reply):
                self.assertEqual(parse_choice_letter(reply, ("A", "B", "C")), expected)

    def test_incidental_ambiguous_or_disallowed_letters_are_unparseable(self):
        replies = [
            "A chart is easier to read.",
            "A or B",
            "The chart shows A and B.",
            "The answer is A. My choice is B.",
            "Final answer: A. Final answer: B.",
            "Choice: B. Final answer: A or B",
            "I choose D.",
            "The answer is a complicated chart.",
            "I cannot choose B.",
            "Some people choose A.",
            "A comparison",
            "A. Earlier answer. My choice is B.",
            "AB",
            "",
            None,
            "unparseable",
        ]
        for reply in replies:
            with self.subTest(reply=reply):
                self.assertIsNone(parse_choice_letter(reply, ("A", "B", "C")))

    def test_same_grammar_for_pairwise_and_mcq(self):
        question = load_all_scenarios()["loans"].questions[0]
        for reply in ("A chart is hard to read. My choice is B.", "I prefer B."):
            self.assertEqual(parse_ab(reply), "B")
            self.assertEqual(parse_choice(reply, question), "b")
        for reply in ("A chart is hard to read.", "A or B", "I choose D."):
            self.assertIsNone(parse_ab(reply))
            self.assertIsNone(parse_choice(reply, question))
        self.assertEqual(parse_choice("Offer A", question), "a")
        self.assertIsNone(parse_choice("Offer A and Offer B", question))

    def test_allowed_letters_are_validated(self):
        for allowed in ([], "AB", ["a"], ["AA"], [True], ["A", "A"]):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                parse_choice_letter("A", allowed)


class TestFamilies(unittest.TestCase):
    def test_provider_does_not_define_model_family(self):
        direct = resolve_family("anthropic:claude-sonnet-5")
        routed = resolve_family("openrouter:anthropic/claude-sonnet-5")
        self.assertEqual(
            (direct.provider, routed.provider), ("anthropic", "openrouter")
        )
        self.assertEqual(direct.family, routed.family)
        self.assertEqual(direct.family, "anthropic")
        self.assertTrue(direct.resolved and routed.resolved)
        self.assertEqual(routed.basis, "model-name")
        self.assertEqual(model_family("openrouter:openai/gpt-4o"), "openai")

    def test_all_recognized_families_and_aliases(self):
        for spec, family in {
            "Qwen/Qwen3-VL-8B-Instruct": "qwen",
            "openai:o3": "openai",
            "openrouter:meta-llama/llama-3.3": "llama",
            "openrouter:google/gemini-2.5": "gemini",
            "openrouter:mistralai/mixtral-8x7b": "mistral",
            "openrouter:deepseek/deepseek-r1": "deepseek",
        }.items():
            with self.subTest(spec=spec):
                self.assertEqual(model_family(spec), family)
        self.assertEqual(canonical_family("Claude"), "anthropic")
        self.assertEqual(canonical_family("GPT"), "openai")

    def test_unknown_identity_is_not_a_family(self):
        for spec in (
            None,
            "",
            "anthropic:unrecognized",
            "openai:private-model",
            "openrouter:alibaba/vl-max",
            "openrouter:vendor/qwenterprise",
            "openrouter:vendor/llamazing",
        ):
            with self.subTest(spec=spec):
                identity = resolve_family(spec)
                self.assertEqual(identity.family, "unknown")
                self.assertFalse(identity.resolved)
                self.assertEqual(identity.basis, "unknown")
        self.assertEqual(canonical_family("private"), "unknown")

    def test_reader_metadata_determinism_and_legacy_provider(self):
        routed = build_reader("openrouter:anthropic/claude-sonnet-5")
        self.assertEqual(routed.family, "openrouter")
        self.assertEqual(routed.provider, "openrouter")
        self.assertEqual(routed.model_family, "anthropic")
        self.assertTrue(routed.family_resolved)
        self.assertFalse(routed.deterministic)
        simulated = build_reader("simulated:literal")
        self.assertTrue(simulated.simulated and simulated.deterministic)
        self.assertEqual(simulated.model_family, "simulated")
        self.assertIsNone(Reader.deterministic)


if __name__ == "__main__":
    unittest.main()

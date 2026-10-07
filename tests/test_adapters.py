"""Provider payloads, prompt hashes and replies, checked without network calls."""

import base64
import hashlib
import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from glassbox import interfaces, schema
from glassbox.readers import build_reader
from glassbox.readers._http import MissingKeyError
from glassbox.readers.anthropic import AnthropicReader, extract_text as anthropic_text
from glassbox.readers.openai_compat import (
    OpenAICompatReader,
    extract_text as openai_text,
)
from glassbox.readers.base import build_mcq_prompt, parse_choice
from glassbox.stimuli import Stimulus

PNG_BYTES = b"\x89PNG\r\n\x1a\nFAKEPNGDATA"


def _image_stimulus(scenario, variant="cards"):
    p = interfaces.variant(scenario, variant)
    fh = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fh.write(PNG_BYTES)
    fh.close()
    return Stimulus(presentation=p, kind="image", text=p.to_text(), image_path=fh.name)


def _spec_stimulus(scenario, variant="baseline"):
    p = interfaces.variant(scenario, variant)
    return Stimulus(presentation=p, kind="spec", text=p.to_text())


class AdapterTestBase(unittest.TestCase):
    def setUp(self):
        self.loans = schema.load_all_scenarios()["loans"]
        self.q = self.loans.questions[0]  # cheapest_total, choices a/b/c


class TestAnthropicPayload(AdapterTestBase):
    def test_raw_base64_image_first(self):
        stim = _image_stimulus(self.loans)
        self.addCleanup(os.unlink, stim.image_path)
        payload = AnthropicReader("claude-sonnet-5").build_payload(self.q, stim)
        self.assertEqual(payload["model"], "claude-sonnet-5")
        self.assertEqual(payload["max_tokens"], 512)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image")  # image before text
        self.assertEqual(content[0]["source"]["type"], "base64")
        self.assertEqual(content[0]["source"]["media_type"], "image/png")
        # raw base64, no data: prefix
        self.assertEqual(
            content[0]["source"]["data"], base64.b64encode(PNG_BYTES).decode()
        )
        self.assertNotIn("data:", content[0]["source"]["data"])
        self.assertEqual(content[1]["type"], "text")
        self.assertIn(self.q.stem, content[1]["text"])

    def test_extract_text(self):
        self.assertEqual(
            anthropic_text({"content": [{"type": "text", "text": "B"}]}), "B"
        )
        self.assertEqual(anthropic_text({"content": []}), "")


class TestOpenAICompatPayload(AdapterTestBase):
    def test_data_uri_image(self):
        stim = _image_stimulus(self.loans)
        self.addCleanup(os.unlink, stim.image_path)
        reader = OpenAICompatReader("openrouter", "qwen/qwen3-vl-8b-instruct")
        payload = reader.build_payload(self.q, stim)
        self.assertEqual(payload["model"], "qwen/qwen3-vl-8b-instruct")
        parts = payload["messages"][0]["content"]
        img = next(p for p in parts if p["type"] == "image_url")
        self.assertTrue(img["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertIn(base64.b64encode(PNG_BYTES).decode(), img["image_url"]["url"])

    def test_openrouter_headers_have_title_no_referer_by_default(self):
        reader = OpenAICompatReader("openrouter", "qwen/qwen3-vl-8b-instruct")
        headers = reader._headers("sk-test")
        self.assertEqual(headers["authorization"], "Bearer sk-test")
        self.assertEqual(headers["X-Title"], "Glass Box")
        self.assertNotIn("HTTP-Referer", headers)  # only when GLASSBOX_APP_URL is set

    def test_extract_text(self):
        self.assertEqual(openai_text({"choices": [{"message": {"content": "A"}}]}), "A")
        self.assertEqual(openai_text({"choices": []}), "")


class TestChoiceParsing(AdapterTestBase):
    def test_letters_and_text(self):
        cases = {
            "B": "b",
            "b) Offer B": "b",
            "The answer is C.": "c",
            "Answer: A": "a",
            "Offer A": "a",  # text match
        }
        for reply, expected in cases.items():
            self.assertEqual(parse_choice(reply, self.q), expected, "reply=%r" % reply)

    def test_unparseable_is_none(self):
        self.assertIsNone(parse_choice("", self.q))
        self.assertIsNone(parse_choice("purple monkey dishwasher", self.q))

    def test_verbose_reply_uses_the_concluding_letter(self):
        # A reasoning reply concludes with its pick: "…Offer C vs B, I'd pick B" is B,
        # not the first-mentioned C (which would silently mis-score a correct reader).
        self.assertEqual(
            parse_choice("Comparing Offer C to B, I would pick B.", self.q), "b"
        )
        self.assertEqual(
            parse_choice("Between A and C, C is cheapest, so C.", self.q), "c"
        )

    def test_skip_render_prompt_inlines_interface_text(self):
        stim = _spec_stimulus(self.loans)
        prompt = build_mcq_prompt(self.q, stim, image=False)
        self.assertIn("Offer A", prompt)  # the interface text is present
        self.assertIn(self.q.stem, prompt)


class TestKeysAndRegistry(AdapterTestBase):
    def test_missing_key_raises_clearly(self):
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(MissingKeyError):
                AnthropicReader("claude-sonnet-5").answer(
                    self.loans, self.q, _spec_stimulus(self.loans)
                )
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

    def test_registry_selects_adapter(self):
        a = build_reader("anthropic:claude-opus-4-8")
        self.assertIsInstance(a, AnthropicReader)
        self.assertEqual(a.name, "anthropic:claude-opus-4-8")
        r = build_reader("openrouter:qwen/qwen3-vl-8b-instruct")
        self.assertIsInstance(r, OpenAICompatReader)
        self.assertEqual(r.family, "openrouter")


class TestSentPromptHash(AdapterTestBase):
    def test_actual_payload_text_is_hashed_once(self):
        stimulus = _spec_stimulus(self.loans)
        adapters = [
            (
                AnthropicReader("claude-sonnet-5"),
                "anthropic",
                "ANTHROPIC_API_KEY",
                {"content": [{"type": "text", "text": "B"}]},
            ),
            (
                OpenAICompatReader("openrouter", "anthropic/claude-sonnet-5"),
                "openai_compat",
                "OPENROUTER_API_KEY",
                {"choices": [{"message": {"content": "B"}}]},
            ),
        ]
        for reader, module, key_name, response in adapters:
            with self.subTest(module=module):
                payload = {
                    "messages": [
                        {"content": [{"type": "text", "text": "Exact sent text"}]}
                    ]
                }
                with (
                    patch.object(
                        reader, "build_payload", return_value=payload
                    ) as build,
                    patch(
                        "glassbox.readers." + module + ".post_json",
                        return_value=response,
                    ) as send,
                    patch.dict(os.environ, {key_name: "fake-key"}),
                ):
                    answer = reader.answer(self.loans, self.q, stimulus)
                build.assert_called_once()
                self.assertIs(send.call_args.args[2], payload)
                self.assertEqual(
                    answer.prompt_sha256, hashlib.sha256(b"Exact sent text").hexdigest()
                )
                self.assertEqual(answer.raw, "B")
                self.assertEqual(answer.choice_id, "b")

    def test_direct_unsupported_mcq_is_rejected_before_request(self):
        stimulus = _spec_stimulus(self.loans)
        for reader, module in (
            (AnthropicReader("claude-sonnet-5"), "anthropic"),
            (OpenAICompatReader("openai", "gpt-4o"), "openai_compat"),
        ):
            for choices in ([self.q.choices[0]], self.q.choices * 9):
                question = replace(self.q, choices=choices)
                with (
                    self.subTest(module=module, count=len(choices)),
                    patch("glassbox.readers." + module + ".post_json") as send,
                ):
                    with self.assertRaises(ValueError):
                        reader.answer(self.loans, question, stimulus)
                    send.assert_not_called()


if __name__ == "__main__":
    unittest.main()

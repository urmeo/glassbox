"""Real-reader adapters — request shape and reply parsing, verified without network.

Live calls are deferred to M2 (no keys in this environment). These tests pin the two
wire contracts (D-0006): Anthropic raw base64 vs OpenAI-compatible data-URI, reply
extraction, choice parsing, key handling, and registry selection.
"""

import base64
import os
import tempfile
import unittest

from glassbox import interfaces, schema
from glassbox.readers import build_reader
from glassbox.readers._http import MissingKeyError
from glassbox.readers.anthropic import AnthropicReader, extract_text as anthropic_text
from glassbox.readers.openai_compat import OpenAICompatReader, extract_text as openai_text
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
        payload = AnthropicReader("claude-sonnet-5").build_payload(self.q, stim)
        self.assertEqual(payload["model"], "claude-sonnet-5")
        self.assertEqual(payload["max_tokens"], 512)
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image")           # image before text
        self.assertEqual(content[0]["source"]["type"], "base64")
        self.assertEqual(content[0]["source"]["media_type"], "image/png")
        # raw base64, no data: prefix
        self.assertEqual(content[0]["source"]["data"], base64.b64encode(PNG_BYTES).decode())
        self.assertNotIn("data:", content[0]["source"]["data"])
        self.assertEqual(content[1]["type"], "text")
        self.assertIn(self.q.stem, content[1]["text"])

    def test_extract_text(self):
        self.assertEqual(anthropic_text({"content": [{"type": "text", "text": "B"}]}), "B")
        self.assertEqual(anthropic_text({"content": []}), "")


class TestOpenAICompatPayload(AdapterTestBase):
    def test_data_uri_image(self):
        stim = _image_stimulus(self.loans)
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
            "Offer A": "a",           # text match
        }
        for reply, expected in cases.items():
            self.assertEqual(parse_choice(reply, self.q), expected, "reply=%r" % reply)

    def test_unparseable_is_none(self):
        self.assertIsNone(parse_choice("", self.q))
        self.assertIsNone(parse_choice("purple monkey dishwasher", self.q))

    def test_skip_render_prompt_inlines_interface_text(self):
        stim = _spec_stimulus(self.loans)
        prompt = build_mcq_prompt(self.q, stim, image=False)
        self.assertIn("Offer A", prompt)   # the interface text is present
        self.assertIn(self.q.stem, prompt)


class TestKeysAndRegistry(AdapterTestBase):
    def test_missing_key_raises_clearly(self):
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(MissingKeyError):
                AnthropicReader("claude-sonnet-5").answer(self.loans, self.q, _spec_stimulus(self.loans))
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


if __name__ == "__main__":
    unittest.main()

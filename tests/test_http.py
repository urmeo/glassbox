"""HTTP validation and fake response tests; no network requests."""

import http.client
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from glassbox.readers._http import ReaderAPIError, post_json
from glassbox.readers.anthropic import extract_text as anthropic_text
from glassbox.readers.openai_compat import extract_text as openai_text

URL = "https://example.invalid/api"


class TruncatedResponse(io.BytesIO):
    def read(self, *args):
        raise http.client.IncompleteRead(b"partial response", 100)


class TestHTTP(unittest.TestCase):
    def test_valid_wire_payload_and_object_response(self):
        with patch(
            "urllib.request.urlopen", return_value=io.BytesIO(b'{"ok":true}')
        ) as send:
            self.assertEqual(
                post_json(URL, {"authorization": "Bearer key"}, {"prompt": "B"}),
                {"ok": True},
            )
            request = send.call_args.args[0]
            self.assertEqual(json.loads(request.data), {"prompt": "B"})
            self.assertEqual(request.get_header("Content-type"), "application/json")
            self.assertEqual(send.call_args.kwargs["timeout"], 90)

    def test_explicit_content_type_is_preserved_without_mutating_headers(self):
        headers = {"Content-Type": "application/custom+json"}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"{}")) as send:
            post_json(URL, headers, {})
        request = send.call_args.args[0]
        self.assertEqual(request.get_header("Content-type"), "application/custom+json")
        self.assertEqual(headers, {"Content-Type": "application/custom+json"})

    def test_nonfinite_and_wrong_json_response_rejected(self):
        for response in (
            b"[]",
            b"null",
            b'{"x":NaN}',
            b'{"x":1e999}',
            b'{"x":1,"x":2}',
            b"not json",
            b"\xff",
        ):
            with (
                self.subTest(response=response),
                patch(
                    "urllib.request.urlopen", return_value=io.BytesIO(response)
                ) as send,
            ):
                with self.assertRaises(ReaderAPIError):
                    post_json(URL, {}, {})
                self.assertEqual(send.call_count, 1)

    def test_response_nesting_error_is_wrapped_without_retrying(self):
        response = b'{"nested":' * 2000 + b"null" + b"}" * 2000
        with patch("urllib.request.urlopen", return_value=io.BytesIO(response)) as send:
            with self.assertRaisesRegex(ReaderAPIError, "invalid JSON response"):
                post_json(URL, {}, {})
        send.assert_called_once()

    def test_retryable_failure_can_recover(self):
        for failure in (
            urllib.error.URLError("temporary connection failure"),
            urllib.error.HTTPError(URL, 429, "busy", {}, io.BytesIO(b"retry later")),
        ):
            with self.subTest(failure=type(failure).__name__), patch(
                "urllib.request.urlopen",
                side_effect=[failure, io.BytesIO(b'{"ok":true}')],
            ) as send:
                self.assertEqual(post_json(URL, {}, {}, retries=1), {"ok": True})
                self.assertEqual(send.call_count, 2)

    def test_http_protocol_errors_can_recover(self):
        for failure in (
            TruncatedResponse(),
            http.client.BadStatusLine("invalid status line"),
        ):
            with self.subTest(failure=type(failure).__name__), patch(
                "urllib.request.urlopen",
                side_effect=[failure, io.BytesIO(b'{"ok":true}')],
            ) as send:
                self.assertEqual(post_json(URL, {}, {}, retries=1), {"ok": True})
                self.assertEqual(send.call_count, 2)
            if isinstance(failure, TruncatedResponse):
                self.assertTrue(failure.closed)

    def test_truncated_responses_obey_retry_limit(self):
        responses = [TruncatedResponse(), TruncatedResponse()]
        with patch("urllib.request.urlopen", side_effect=responses) as send:
            with self.assertRaisesRegex(ReaderAPIError, "request to .* failed"):
                post_json(URL, {}, {}, retries=1)
            self.assertEqual(send.call_count, 2)
        self.assertTrue(all(response.closed for response in responses))

    def test_unreadable_http_error_body_preserves_status_and_retry_rules(self):
        for code, attempts in ((400, 1), (503, 2)):
            bodies = [TruncatedResponse() for _ in range(attempts)]
            failures = [
                urllib.error.HTTPError(URL, code, "failed", {}, body)
                for body in bodies
            ]
            with self.subTest(code=code), patch(
                "urllib.request.urlopen", side_effect=failures
            ) as send:
                with self.assertRaises(ReaderAPIError) as context:
                    post_json(URL, {"authorization": "Bearer fake-key"}, {}, retries=1)
                self.assertIn("HTTP %d" % code, str(context.exception))
                self.assertIn("body could not be read", str(context.exception))
                self.assertNotIn("fake-key", str(context.exception))
                self.assertEqual(send.call_count, attempts)
            self.assertTrue(all(body.closed for body in bodies))

    def test_invalid_request_numbers_fail_before_http(self):
        for kwargs in (
            {"timeout": True},
            {"timeout": 0},
            {"timeout": float("inf")},
            {"timeout": 10**1000},
            {"retries": True},
            {"retries": -1},
            {"retries": 1.5},
        ):
            with self.subTest(kwargs=kwargs), patch("urllib.request.urlopen") as send:
                with self.assertRaises(ValueError):
                    post_json(URL, {}, {}, **kwargs)
                send.assert_not_called()
        with patch("urllib.request.urlopen") as send:
            with self.assertRaises(ValueError):
                post_json(URL, {}, {"value": float("nan")})
            send.assert_not_called()

    def test_retry_limit_and_secret_error_body(self):
        failures = [
            urllib.error.HTTPError(URL, 503, "busy", {}, io.BytesIO(b"token-123 busy")),
            urllib.error.HTTPError(URL, 503, "busy", {}, io.BytesIO(b"token-123 busy")),
        ]
        with patch("urllib.request.urlopen", side_effect=failures) as send:
            with self.assertRaises(ReaderAPIError) as context:
                post_json(URL, {"authorization": "Bearer token-123"}, {}, retries=1)
            self.assertEqual(send.call_count, 2)
            self.assertNotIn("token-123", str(context.exception))
            self.assertIn("[redacted]", str(context.exception))

    def test_nonretryable_error_has_one_attempt(self):
        failure = urllib.error.HTTPError(URL, 400, "bad", {}, io.BytesIO(b"bad input"))
        with patch("urllib.request.urlopen", side_effect=failure) as send:
            with self.assertRaises(ReaderAPIError):
                post_json(URL, {}, {}, retries=2)
            self.assertEqual(send.call_count, 1)

    def test_malformed_provider_text_is_unparseable(self):
        for response in (
            None,
            [],
            {"content": "B"},
            {"content": [{"type": "text", "text": []}]},
        ):
            self.assertEqual(anthropic_text(response), "")
        for response in (
            None,
            [],
            {"choices": None},
            {"choices": [{"message": {"content": []}}]},
        ):
            self.assertEqual(openai_text(response), "")


if __name__ == "__main__":
    unittest.main()

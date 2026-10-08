"""Truncated errors must not expose a prefix of an echoed credential."""

import io
import unittest
import urllib.error
from unittest.mock import patch

from glassbox.readers._http import ReaderAPIError, post_json


class TestHTTPRedaction(unittest.TestCase):
    def test_secret_crossing_error_body_limit_is_redacted_before_truncation(self):
        token = "sample-private-token-123"
        url = "https://example.invalid/api"
        for headers in (
            {"authorization": "Bearer " + token},
            {"X-API-Key": token},
        ):
            with self.subTest(headers=headers):
                body = io.BytesIO(("x" * 490 + token + " trailing text").encode())
                failure = urllib.error.HTTPError(url, 400, "bad", {}, body)
                with patch("urllib.request.urlopen", side_effect=failure) as send:
                    with self.assertRaises(ReaderAPIError) as context:
                        post_json(url, headers, {}, retries=2)
                message = str(context.exception)
                self.assertNotIn("sample-pri", message)
                self.assertIn("[redacted]", message)
                self.assertEqual(send.call_count, 1)
                self.assertTrue(body.closed)
                self.assertLessEqual(len(message.split(": ", 1)[1]), 500)

"""Retry timing tests use mocked clocks and responses; no network or real waits."""

import io
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime, formatdate
from unittest.mock import call, patch

from glassbox.readers._http import ReaderAPIError, post_json

URL = "https://example.invalid/api"


class _Clock:
    epoch = 1_700_000_000.0

    def __init__(self):
        self.elapsed = 0.0

    def monotonic(self):
        return self.elapsed

    def time(self):
        return self.epoch + self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds


def _http_failure(retry_after=None, *, body=b"retry later"):
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return urllib.error.HTTPError(URL, 503, "busy", headers, io.BytesIO(body))


class TestHTTPTiming(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.timeouts = []
        for name in ("monotonic", "time", "sleep"):
            patcher = patch(
                "glassbox.readers._http.time." + name,
                side_effect=getattr(self.clock, name),
            )
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def _responses(self, *attempts):
        attempts = iter(attempts)

        def send(request, timeout):
            self.timeouts.append(timeout)
            duration, response = next(attempts)
            self.clock.elapsed += duration
            if isinstance(response, Exception):
                raise response
            return response

        return send

    def test_exponential_backoff_is_capped_and_retries_are_limited(self):
        attempts = [
            (0, urllib.error.URLError("temporary failure")) for _ in range(7)
        ] + [(0, io.BytesIO(b"{}"))]
        with patch("urllib.request.urlopen", side_effect=self._responses(*attempts)) as send:
            self.assertEqual(post_json(URL, {}, {}, timeout=200, retries=7), {})
        self.assertEqual(self.sleep.call_args_list, [
            call(1), call(2), call(4), call(8), call(16), call(30), call(30),
        ])
        self.assertEqual(self.timeouts, [200, 199, 197, 193, 185, 169, 139, 109])
        self.assertEqual(send.call_count, 8)

    def test_delta_seconds_header_is_case_insensitive(self):
        failure = urllib.error.HTTPError(
            URL, 429, "busy", {"rEtRy-AfTeR": "7"}, io.BytesIO(b"retry later")
        )
        with patch("urllib.request.urlopen", side_effect=self._responses(
            (0, failure), (0, io.BytesIO(b"{}")),
        )):
            self.assertEqual(post_json(URL, {}, {}, retries=1), {})
        self.sleep.assert_called_once_with(7)
        self.assertEqual(self.timeouts, [90, 83])
        self.assertTrue(failure.closed)

    def test_repeated_429_with_one_second_cooldown_cannot_retry_immediately(self):
        failures = [
            urllib.error.HTTPError(
                URL, 429, "busy", {"Retry-After": "1"}, io.BytesIO(b"retry later")
            )
            for _ in range(3)
        ]
        with patch("urllib.request.urlopen", side_effect=self._responses(
            *((0, failure) for failure in failures)
        )) as send:
            with self.assertRaisesRegex(ReaderAPIError, "HTTP 429"):
                post_json(URL, {}, {}, retries=2)
        self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])
        self.assertEqual(self.timeouts, [90, 89, 87])
        self.assertEqual(send.call_count, 3)
        self.assertTrue(all(failure.closed for failure in failures))

    def test_http_dates_use_wall_time_and_their_timezone(self):
        retry_at = self.clock.epoch + 12
        headers = (
            formatdate(retry_at, usegmt=True),
            format_datetime(datetime.fromtimestamp(
                retry_at, timezone(timedelta(hours=2)),
            )),
            datetime.fromtimestamp(retry_at, timezone.utc).strftime(
                "%a, %d %b %Y %H:%M:%S"
            ),
        )
        for header in headers:
            with self.subTest(header=header):
                self.clock.elapsed = 0
                self.sleep.reset_mock()
                self.timeouts.clear()
                with patch("urllib.request.urlopen", side_effect=self._responses(
                    (3, _http_failure(header)), (0, io.BytesIO(b"{}")),
                )):
                    self.assertEqual(post_json(URL, {}, {}, retries=1), {})
                self.sleep.assert_called_once_with(9)
                self.assertEqual(self.timeouts, [90, 78])

    def test_missing_invalid_negative_and_past_headers_use_backoff(self):
        for header in (
            None, "", "invalid", "-2", "NaN", "inf", "-inf", "1e999", "1.5", "+4",
            formatdate(self.clock.epoch - 100, usegmt=True),
        ):
            with self.subTest(header=header):
                self.clock.elapsed = 0
                self.sleep.reset_mock()
                with patch("urllib.request.urlopen", side_effect=self._responses(
                    (0, _http_failure(header)), (0, io.BytesIO(b"{}")),
                )):
                    self.assertEqual(post_json(URL, {}, {}, retries=1), {})
                self.sleep.assert_called_once_with(1)

    def test_long_provider_cooldown_stops_instead_of_retrying_early(self):
        for header in (
            "60", "1000000", "9" * 5000,
            formatdate(self.clock.epoch + 86400, usegmt=True),
        ):
            with self.subTest(header=header):
                self.sleep.reset_mock()
                failure = _http_failure(header, body=b"secret-token busy")
                with patch("urllib.request.urlopen", side_effect=self._responses(
                    (0, failure),
                )) as send:
                    with self.assertRaises(ReaderAPIError) as context:
                        post_json(
                            URL, {"x-api-key": "secret-token"}, {}, retries=1,
                        )
                self.assertIn("HTTP 503", str(context.exception))
                self.assertIn("[redacted]", str(context.exception))
                self.assertNotIn("secret-token", str(context.exception))
                self.sleep.assert_not_called()
                send.assert_called_once()
                self.assertTrue(failure.closed)

    def test_provider_cooldown_at_local_limit_is_allowed(self):
        with patch("urllib.request.urlopen", side_effect=self._responses(
            (0, _http_failure("30")), (0, io.BytesIO(b"{}")),
        )):
            self.assertEqual(post_json(URL, {}, {}, retries=1), {})
        self.sleep.assert_called_once_with(30)
        self.assertEqual(self.timeouts, [90, 60])

    def test_retry_after_does_not_shorten_exponential_backoff(self):
        with patch("urllib.request.urlopen", side_effect=self._responses(
            (0, urllib.error.URLError("temporary failure")),
            (0, _http_failure("0")),
            (0, io.BytesIO(b"{}")),
        )):
            self.assertEqual(post_json(URL, {}, {}, retries=2), {})
        self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])

    def test_attempt_time_and_wait_both_reduce_the_next_timeout(self):
        with patch("urllib.request.urlopen", side_effect=self._responses(
            (4, urllib.error.URLError("temporary failure")),
            (5, io.BytesIO(b"{}")),
        )):
            self.assertEqual(post_json(URL, {}, {}, timeout=20, retries=1), {})
        self.assertEqual(self.timeouts, [20, 15])
        self.assertEqual(self.clock.elapsed, 10)

    def test_wait_that_uses_remaining_budget_stops_with_redacted_http_error(self):
        for header in ("8", "20"):
            with self.subTest(header=header):
                self.clock.elapsed = 0
                self.sleep.reset_mock()
                failure = _http_failure(header, body=b"secret-token busy")
                with patch("urllib.request.urlopen", side_effect=self._responses(
                    (2, failure),
                )) as send:
                    with self.assertRaises(ReaderAPIError) as context:
                        post_json(
                            URL, {"Authorization": "Bearer secret-token"}, {},
                            timeout=10, retries=2,
                        )
                self.assertIn("HTTP 503", str(context.exception))
                self.assertIn("[redacted]", str(context.exception))
                self.assertNotIn("secret-token", str(context.exception))
                self.sleep.assert_not_called()
                send.assert_called_once()
                self.assertTrue(failure.closed)

    def test_attempt_that_exhausts_budget_is_not_retried(self):
        with patch("urllib.request.urlopen", side_effect=self._responses(
            (10, urllib.error.URLError("temporary failure")),
        )) as send:
            with self.assertRaisesRegex(ReaderAPIError, "temporary failure"):
                post_json(URL, {}, {}, timeout=10, retries=2)
        self.sleep.assert_not_called()
        send.assert_called_once()

    def test_oversleep_does_not_start_an_attempt_after_the_deadline(self):
        def oversleep(seconds):
            self.clock.elapsed += seconds + 10

        self.sleep.side_effect = oversleep
        failure = _http_failure()
        with patch("urllib.request.urlopen", side_effect=self._responses(
            (0, failure),
        )) as send:
            with self.assertRaisesRegex(ReaderAPIError, "HTTP 503"):
                post_json(URL, {}, {}, timeout=10, retries=2)
        self.sleep.assert_called_once_with(1)
        send.assert_called_once()
        self.assertTrue(failure.closed)

    def test_late_success_is_rejected_and_response_is_closed(self):
        response = io.BytesIO(b"{}")
        with patch("urllib.request.urlopen", side_effect=self._responses(
            (11, response),
        )) as send:
            with self.assertRaisesRegex(ReaderAPIError, "exceeded timeout"):
                post_json(URL, {}, {}, timeout=10)
        self.sleep.assert_not_called()
        send.assert_called_once()
        self.assertTrue(response.closed)

    def test_retry_limit_stops_before_waiting(self):
        failure = _http_failure("20")
        with patch("urllib.request.urlopen", side_effect=self._responses(
            (0, failure),
        )) as send:
            with self.assertRaisesRegex(ReaderAPIError, "HTTP 503"):
                post_json(URL, {}, {}, retries=0)
        self.sleep.assert_not_called()
        send.assert_called_once()


if __name__ == "__main__":
    unittest.main()

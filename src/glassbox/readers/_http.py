"""Stdlib HTTP and image encoding; API keys are read only at call time."""

from __future__ import annotations

import base64
import http.client
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional

from .._config import strict_json

_RETRYABLE = {429, 500, 502, 503, 529}
_MAX_RETRY_DELAY = 30.0


class MissingKeyError(RuntimeError):
    """A real reader was invoked without its API key in the environment."""


class ReaderAPIError(RuntimeError):
    """A real reader's API call failed."""


def require_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise MissingKeyError(
            f"{env_name} is not set. Export it before running this API reader."
        )
    return key


def encode_png_base64(path: str) -> str:
    """Base64-encode a PNG file (raw, no data: prefix)."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _retry_delay(retry_after: Optional[str], backoff: float) -> float:
    if not isinstance(retry_after, str) or not retry_after.strip():
        return backoff
    retry_after = retry_after.strip()
    if retry_after.isascii() and retry_after.isdecimal():
        # Huge valid integers become infinity and exceed the local wait limit.
        return max(backoff, float(retry_after))
    try:
        retry_at = parsedate_to_datetime(retry_after)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = retry_at.timestamp() - time.time()
    except (TypeError, ValueError, OverflowError, OSError):
        return backoff
    if not math.isfinite(delay) or delay < 0:
        return backoff
    return max(backoff, delay)


def post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float = 90,
    retries: int = 2,
) -> Dict[str, Any]:
    """POST JSON with a shared timeout budget for attempts and retry waits."""
    try:
        finite_timeout = math.isfinite(timeout)
    except (TypeError, OverflowError):
        finite_timeout = False
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not finite_timeout
        or timeout <= 0
    ):
        raise ValueError("timeout must be a finite positive number")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    try:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("payload must contain finite JSON values") from None
    request_headers = {name.lower(): value for name, value in headers.items()}
    request_headers.setdefault("content-type", "application/json")

    deadline = time.monotonic() + timeout
    retries_left = retries
    backoff = 1.0
    failure = "request to %s exceeded timeout" % url
    while True:
        request = urllib.request.Request(
            url, data=data, headers=request_headers, method="POST"
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ReaderAPIError(failure) from None
        retry_after = None
        try:
            with urllib.request.urlopen(request, timeout=remaining) as response:
                try:
                    result = strict_json(response.read().decode("utf-8"))
                except (UnicodeError, ValueError, RecursionError) as exc:
                    raise ReaderAPIError(
                        "invalid JSON response from %s: %s" % (url, exc)
                    ) from None
                if not isinstance(result, dict):
                    raise ReaderAPIError("response from %s must be a JSON object" % url)
                if time.monotonic() >= deadline:
                    raise ReaderAPIError("request to %s exceeded timeout" % url)
                return result
        except urllib.error.HTTPError as exc:
            retry_after = next(
                (value for name, value in (exc.headers or {}).items()
                 if name.lower() == "retry-after"),
                None,
            )
            try:
                body = exc.read().decode("utf-8", "replace")
            except (http.client.HTTPException, OSError):
                body = "error response body could not be read"
            finally:
                exc.close()
            for name, value in headers.items():
                if name.lower() in {"authorization", "x-api-key"}:
                    secret = value.removeprefix("Bearer ")
                    if secret:
                        body = body.replace(secret, "[redacted]")
            failure = "HTTP %s from %s: %s" % (exc.code, url, body[:500])
            if exc.code not in _RETRYABLE:
                raise ReaderAPIError(failure) from None
        except (http.client.HTTPException, urllib.error.URLError, OSError) as exc:
            failure = "request to %s failed: %s" % (url, exc)
        if retries_left == 0:
            raise ReaderAPIError(failure) from None
        delay = _retry_delay(retry_after, backoff)
        if delay > _MAX_RETRY_DELAY or delay >= deadline - time.monotonic():
            raise ReaderAPIError(failure) from None
        time.sleep(delay)
        retries_left -= 1
        backoff = min(backoff * 2, _MAX_RETRY_DELAY)

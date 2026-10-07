"""Stdlib HTTP and image encoding; API keys are read only at call time."""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Dict

from .._config import strict_json

_RETRYABLE = {429, 500, 502, 503, 529}


class MissingKeyError(RuntimeError):
    """A real reader was invoked without its API key in the environment."""


class ReaderAPIError(RuntimeError):
    """A real reader's API call failed."""


def require_key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise MissingKeyError(
            "%s is not set. Real readers read their key from the environment at call "
            "time (nothing is stored); export it to run this reader." % env_name
        )
    return key


def encode_png_base64(path: str) -> str:
    """Base64-encode a PNG file (raw, no data: prefix)."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int = 90,
    retries: int = 2,
) -> Dict[str, Any]:
    """POST ``payload`` as JSON and return the parsed response, with bounded retries."""
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
    merged = dict(headers)
    merged.setdefault("content-type", "application/json")

    last: Exception = ReaderAPIError("no attempt made")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=merged, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                try:
                    result = strict_json(resp.read().decode("utf-8"))
                except (UnicodeError, ValueError) as exc:
                    raise ReaderAPIError(
                        "invalid JSON response from %s: %s" % (url, exc)
                    ) from None
                if not isinstance(result, dict):
                    raise ReaderAPIError("response from %s must be a JSON object" % url)
                return result
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", "replace")[:500]
            finally:
                exc.close()
            for name, value in headers.items():
                if name.lower() in {"authorization", "x-api-key"}:
                    secret = value.removeprefix("Bearer ")
                    if secret:
                        body = body.replace(secret, "[redacted]")
            last = ReaderAPIError("HTTP %s from %s: %s" % (exc.code, url, body))
            if exc.code in _RETRYABLE and attempt < retries:
                continue
            raise last
        except (urllib.error.URLError, OSError) as exc:
            last = ReaderAPIError("request to %s failed: %s" % (url, exc))
            if attempt < retries:
                continue
            raise last
    raise last

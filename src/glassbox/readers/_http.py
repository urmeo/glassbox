"""Shared plumbing for the real-API readers — stdlib HTTP + image encoding.

Keys are read from the environment at call time and never persisted (a project
requirement). A missing key raises a clear error rather than failing silently.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

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


def post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any],
              timeout: int = 90, retries: int = 2) -> Dict[str, Any]:
    """POST ``payload`` as JSON and return the parsed response, with bounded retries."""
    data = json.dumps(payload).encode("utf-8")
    merged = dict(headers)
    merged.setdefault("content-type", "application/json")

    last: Exception = ReaderAPIError("no attempt made")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers=merged, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
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

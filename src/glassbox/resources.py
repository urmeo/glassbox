"""Read packaged content through individual Python 3.9 resource files."""

from __future__ import annotations

import hashlib
import json
import math
from importlib import resources
from typing import Any, Tuple

_KINDS = ("scenarios", "prompts", "studies", "anchors", "training")


class ResourceError(ValueError):
    """A packaged resource is missing or malformed."""


def _directory(kind: str):
    if not isinstance(kind, str) or kind not in _KINDS:
        raise ResourceError("unknown resource kind %r" % (kind,))
    return resources.files("glassbox").joinpath("_data").joinpath(kind)


def _filename(kind: str, name: str) -> str:
    _directory(kind)
    if (
        not isinstance(name, str)
        or not name.strip()
        or name in (".", "..")
        or any(char in name for char in ("/", "\\", "\0", ":"))
    ):
        raise ResourceError("resource name must be a basename")
    suffix = ".txt" if kind == "prompts" else ".json"
    return name if name.endswith(suffix) else name + suffix


def resource_names(kind: str) -> Tuple[str, ...]:
    suffix = ".txt" if kind == "prompts" else ".json"
    try:
        return tuple(
            sorted(
                file.name
                for file in _directory(kind).iterdir()
                if file.is_file() and file.name.endswith(suffix)
            )
        )
    except (OSError, FileNotFoundError) as exc:
        raise ResourceError("missing packaged resource directory %r" % kind) from exc


def resource_bytes(kind: str, name: str) -> bytes:
    filename = _filename(kind, name)
    try:
        return _directory(kind).joinpath(filename).read_bytes()
    except OSError as exc:
        raise ResourceError("missing packaged resource %s/%s" % (kind, filename)) from exc


def resource_text(kind: str, name: str) -> str:
    try:
        return resource_bytes(kind, name).decode("utf-8")
    except UnicodeError as exc:
        raise ResourceError("resource %s/%s is not UTF-8 text" % (kind, name)) from exc


def decode_json(text: str, label: str = "JSON") -> Any:
    """Reject duplicate keys and nonstandard numeric JSON constants."""

    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ResourceError("%s: duplicate JSON key %r" % (label, key))
            value[key] = item
        return value

    def constant(value):
        raise ResourceError("%s: nonstandard JSON constant %s" % (label, value))

    def number(value, converter):
        try:
            parsed = converter(value)
            if not math.isfinite(parsed):
                raise ValueError("nonfinite number")
            return parsed
        except (ValueError, OverflowError) as exc:
            raise ResourceError("%s: numeric JSON values must be finite" % label) from exc

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=constant,
            parse_float=lambda value: number(value, float),
            parse_int=lambda value: number(value, int),
        )
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise ResourceError("%s: invalid JSON: %s" % (label, exc)) from exc


def resource_json(kind: str, name: str) -> Any:
    return decode_json(resource_text(kind, name), "%s/%s" % (kind, name))


def resource_sha256(kind: str, name: str) -> str:
    return hashlib.sha256(resource_bytes(kind, name)).hexdigest()

"""Pure configuration guards and packaged or explicit-file JSON loading."""

from __future__ import annotations

import json
import math
from pathlib import Path


def text(value, name, error=ValueError, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise error("%s must be a %sstring" % (name, "" if empty else "nonempty "))
    return value.strip() if not empty else value


def strings(values, name, error=ValueError, optional=False):
    if values is None and optional:
        return None
    if not isinstance(values, (list, tuple)) or not values:
        raise error("%s must be a nonempty list" % name)
    result = [text(value, name + " item", error) for value in values]
    if len(set(result)) != len(result):
        raise error("%s contains duplicates" % name)
    return result


def positive_integer(value, name, error=ValueError):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise error("%s must be an integer >= 1" % name)
    return value


def boolean(value, name, error=ValueError):
    if not isinstance(value, bool):
        raise error("%s must be boolean" % name)
    return value


def finite_json(value, error=ValueError, depth=0):
    if depth > 100:
        raise error("configuration JSON nesting exceeds 100 levels")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: finite_json(item, error, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item, error, depth + 1) for item in value]
    raise error("configuration must contain finite JSON values")


def object_fields(raw, fields, error):
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise error("configuration must be an object with string keys")
    unknown = set(raw) - set(fields)
    if unknown:
        raise error("unknown configuration fields: %s" % sorted(unknown))


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field %r" % key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON value %s" % value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def load_config(kind, source, error):
    """Resolve a bundled basename or an explicit path without checkout fallback."""
    from .resources import resource_names, resource_text

    try:
        path_argument = isinstance(source, Path)
        source = str(source) if path_argument else text(source, "config source", error)
        path = Path(source)
        explicit = (
            path_argument or path.is_absolute() or "/" in source or "\\" in source
        )
        name = source if source.endswith(".json") else source + ".json"
        if not explicit and name in resource_names(kind):
            raw = resource_text(kind, source)
        else:
            raw = path.read_text(encoding="utf-8")
        return strict_json(raw)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise error("%s: %s" % (source, exc)) from None


def config_input_path(kind, source):
    """Return the external config path, or None for a bundled name."""
    from .resources import resource_names

    path_argument = isinstance(source, Path)
    value = str(source) if path_argument else text(source, "config source")
    path = Path(value)
    explicit = path_argument or path.is_absolute() or "/" in value or "\\" in value
    name = value if value.endswith(".json") else value + ".json"
    if not explicit and name in resource_names(kind):
        return None
    try:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("invalid config path: %s" % exc) from None

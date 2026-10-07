"""Preflight output roots and complete artifact sets before execution."""

from __future__ import annotations

import os
from pathlib import Path


def _within(path: Path, parent: Path) -> bool:
    child = tuple(part.casefold() for part in path.parts)
    ancestor = tuple(part.casefold() for part in parent.parts)
    return child[: len(ancestor)] == ancestor


def validate_output_dir(out_dir, *, protected_paths=()) -> Path:
    """Protect package/source trees; return the resolved writable namespace."""
    try:
        if not isinstance(out_dir, (str, Path)) or not str(out_dir).strip():
            raise ValueError("output directory must be a nonempty path")
        original = Path(out_dir)
        if original.is_symlink():
            raise ValueError("output directory must not be a symlink")
        output = original.resolve()
        if output.exists() and not output.is_dir():
            raise ValueError("output directory must be a directory")
        package = Path(__file__).resolve().parent
        roots = [package]
        repository = package.parent.parent
        if (repository / "pyproject.toml").is_file():
            roots.extend(
                repository / name
                for name in ("src", "data", "docs", "tests", "scripts", ".github")
            )
        roots.extend(
            Path(path).resolve() for path in protected_paths if Path(path).is_dir()
        )
        if any(_within(output, root) or _within(root, output) for root in roots):
            raise ValueError("output directory intersects package or input sources")
        return output
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("cannot resolve output directory: %s" % error) from error


def validate_output_files(out_dir, names, *, protected_paths=()) -> tuple:
    """Reject planned file links and aliases; ordinary output replacement is allowed."""
    try:
        inputs = tuple(Path(path) for path in protected_paths)
        directory = validate_output_dir(out_dir, protected_paths=inputs)
        if isinstance(names, (str, bytes)):
            raise ValueError("output names must be a sequence of basenames")
        names = tuple(names)
        if not names or any(
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or any(c in name for c in ("/", "\\", "\0"))
            for name in names
        ):
            raise ValueError("output names must be nonempty basenames")
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("output filenames alias each other")
        destinations = tuple(directory / name for name in names)
        resolved_inputs = tuple(path.resolve() for path in inputs)
        for destination in destinations:
            if destination.is_symlink():
                raise ValueError(
                    "output artifact must not be a symlink: %s" % destination
                )
            if destination.exists() and (
                not destination.is_file() or destination.stat().st_nlink > 1
            ):
                raise ValueError("output artifact must be a regular unlinked file")
            for source, resolved in zip(inputs, resolved_inputs):
                alias = str(destination).casefold() == str(resolved).casefold()
                if not alias and destination.exists() and source.exists():
                    alias = os.path.samefile(destination, source)
                if alias:
                    raise ValueError("output artifact aliases input: %s" % source)
        return destinations
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("cannot resolve output artifacts: %s" % error) from error

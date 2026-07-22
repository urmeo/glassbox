"""Rendering — turn an interface into a stimulus, via headless Chrome or a spec.

The offline path (``skip=True`` or no Chrome) returns a ``spec`` stimulus and needs
no browser, so ``scripts/verify.sh`` is fully reproducible with no dependencies. When
Chrome is present, ``render`` screenshots the interface's self-contained HTML to a PNG.

Determinism (a named project risk): a fixed device-scale and window size, a system
font stack, self-contained CSS with no external resources, and the renderer version
recorded per run keep screenshots stable across machines.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional

from .interfaces import Presentation
from .stimuli import Stimulus

DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
_WINDOW = "820,1240"
_SCALE = "2"


class RenderError(RuntimeError):
    """Chrome was asked to render and failed."""


def chrome_path() -> str:
    return os.environ.get("GLASSBOX_CHROME", DEFAULT_CHROME)


def chrome_available() -> bool:
    path = chrome_path()
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def renderer_version(skip: bool = False) -> str:
    """A short string identifying the renderer, for run provenance."""
    if skip or not chrome_available():
        return "skip-render"
    try:
        out = subprocess.run([chrome_path(), "--version"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or "chrome (version unknown)"
    except (OSError, subprocess.SubprocessError):
        return "chrome (version unavailable)"


def _screenshot(html: str, png_path: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as fh:
        html_path = fh.name
        fh.write(html)
    try:
        cmd = [
            chrome_path(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--no-first-run", "--no-default-browser-check",
            "--force-device-scale-factor=" + _SCALE, "--window-size=" + _WINDOW,
            "--default-background-color=FFFFFFFF",
            "--screenshot=" + png_path, "file://" + html_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if not os.path.isfile(png_path) or os.path.getsize(png_path) == 0:
            raise RenderError("chrome produced no screenshot: %s" % (result.stderr.strip()[:400]))
    finally:
        try:
            os.unlink(html_path)
        except OSError:
            pass


def render(presentation: Presentation, out_dir: Optional[str] = None,
           skip: bool = False) -> Stimulus:
    """Render ``presentation`` to a :class:`Stimulus`.

    ``skip`` (or the absence of Chrome) yields a ``spec`` stimulus; otherwise a PNG is
    written under ``out_dir`` (or a temp dir) and an ``image`` stimulus is returned.
    """
    text = presentation.to_text()
    if skip or not chrome_available():
        return Stimulus(presentation=presentation, kind="spec", text=text)

    directory = out_dir or tempfile.mkdtemp(prefix="glassbox_render_")
    os.makedirs(directory, exist_ok=True)
    stem = "%s__%s" % (presentation.scenario_id, presentation.variant.replace("+", "_"))
    png_path = os.path.join(directory, stem + ".png")
    _screenshot(presentation.to_html(), png_path)
    return Stimulus(presentation=presentation, kind="image", text=text, image_path=png_path)

"""Stage an isolated Chrome screenshot and replace outputs only after validation."""

from __future__ import annotations

import hashlib
import os
import re
import signal
import struct
import subprocess
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Optional

from .interfaces import Presentation
from .output_paths import validate_output_files
from .stimuli import Stimulus

DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
_WINDOW = "820,620"
_SCALE = "2"
_RENDER_TIMEOUT = 15.0
_EXIT_GRACE = 1.0
_STOP_TIMEOUT = 2.0
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class RenderError(RuntimeError):
    """A requested screenshot failed or produced invalid image data."""


def chrome_path() -> str:
    return os.environ.get("GLASSBOX_CHROME", DEFAULT_CHROME)


def chrome_available() -> bool:
    path = chrome_path()
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def renderer_version(skip: bool = False) -> str:
    if skip or not chrome_available():
        return "skip-render"
    try:
        result = subprocess.run(
            [chrome_path(), "--version"], capture_output=True, text=True, timeout=20
        )
        return (
            result.stdout.strip()
            if result.returncode == 0 and result.stdout.strip()
            else "chrome (version unknown)"
        )
    except (OSError, subprocess.SubprocessError):
        return "chrome (version unavailable)"


def _validate_png(data: bytes) -> Dict[str, int]:
    """Check complete noninterlaced PNG chunks, CRCs and bounded decoded rows."""
    if not data.startswith(_PNG_SIGNATURE) or len(data) > 64 * 1024 * 1024:
        raise RenderError("chrome output is not a bounded PNG")
    offset, header, image_data, ended, saw_data = 8, None, bytearray(), False, False
    palette, data_ended = None, False
    while offset < len(data):
        if offset + 12 > len(data):
            raise RenderError("PNG chunk is truncated")
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        if not re.fullmatch(b"[A-Za-z]{4}", kind):
            raise RenderError("PNG chunk type is invalid")
        end = offset + size + 12
        if end > len(data):
            raise RenderError("PNG chunk is truncated")
        content = data[offset + 8 : end - 4]
        crc = struct.unpack(">I", data[end - 4 : end])[0]
        if (zlib.crc32(kind + content) & 0xFFFFFFFF) != crc:
            raise RenderError("PNG chunk CRC is invalid")
        if header is None and kind != b"IHDR":
            raise RenderError("PNG must start with IHDR")
        if kind == b"IHDR":
            if header is not None or size != 13:
                raise RenderError("PNG IHDR is invalid")
            header = struct.unpack(">IIBBBBB", content)
        elif kind == b"PLTE":
            if saw_data or palette is not None or size == 0 or size > 768 or size % 3:
                raise RenderError("PNG palette is invalid")
            palette = size // 3
        elif kind == b"IDAT":
            if data_ended:
                raise RenderError("PNG image chunks must be consecutive")
            saw_data = True
            image_data.extend(content)
        elif kind == b"IEND":
            if size != 0 or end != len(data):
                raise RenderError("PNG IEND is invalid")
            ended = True
            break
        elif not kind[0] & 32:
            raise RenderError("PNG has an unknown critical chunk")
        if saw_data and kind != b"IDAT":
            data_ended = True
        offset = end
    if header is None or not saw_data or not ended:
        raise RenderError("PNG lacks required image chunks")
    width, height, depth, color, compression, filtering, interlace = header
    depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
    if (
        not width
        or not height
        or width * height > 32_000_000
        or color not in depths
        or depth not in depths[color]
        or (compression, filtering, interlace) != (0, 0, 0)
    ):
        raise RenderError("PNG IHDR dimensions/encoding are invalid")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color]
    if color == 3 and (palette is None or palette > 2**depth):
        raise RenderError("indexed PNG requires a valid palette")
    row_size = (width * channels * depth + 7) // 8 + 1
    expected = row_size * height
    if expected > 128 * 1024 * 1024:
        raise RenderError("PNG decoded data exceeds the screenshot limit")
    try:
        decoder = zlib.decompressobj()
        rows = decoder.decompress(bytes(image_data), expected + 1)
        if (
            len(rows) != expected
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
            or any(rows[index] > 4 for index in range(0, len(rows), row_size))
        ):
            raise RenderError("PNG image rows are invalid")
    except zlib.error as exc:
        raise RenderError("PNG image compression is invalid") from exc
    return {"width": width, "height": height}


def _cleanup_chrome(process: subprocess.Popen) -> None:
    """Stop only the browser and process group created for this capture."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_STOP_TIMEOUT)
    if os.name == "posix":
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break


def _capture(command, staged: Path, log_path: Path) -> Dict[str, Any]:
    """Confirm capture before accepting natural or controlled zero exit."""
    process = None
    started, confirmed_at, confirmed_data = time.monotonic(), None, None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            while True:
                status = process.poll()
                now = time.monotonic()
                messages = log_path.read_bytes()
                if status is not None and status != 0:
                    raise RenderError(
                        "chrome exited %s: %s"
                        % (status, messages.decode("utf-8", errors="replace")[-400:])
                    )
                data, dimensions = None, None
                if staged.is_file():
                    data = staged.read_bytes()
                    try:
                        dimensions = _validate_png(data)
                    except RenderError:
                        if status is not None:
                            raise
                marker = (
                    ("%d bytes written to file %s" % (len(data), staged)).encode("utf-8")
                    if dimensions is not None
                    else None
                )
                confirmed = marker is not None and marker in messages.splitlines()
                if status == 0:
                    if dimensions is None:
                        raise RenderError("chrome exited without a complete screenshot")
                    return {
                        **dimensions,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "completion": "natural-exit",
                    }
                if now - started >= _RENDER_TIMEOUT:
                    raise RenderError(
                        "chrome did not finish capture within %g seconds" % _RENDER_TIMEOUT
                    )
                if confirmed:
                    if confirmed_at is None:
                        confirmed_at, confirmed_data = now, data
                    if data != confirmed_data:
                        raise RenderError("chrome screenshot changed after capture confirmation")
                    if now - confirmed_at >= _EXIT_GRACE:
                        status = process.poll()
                        if status is not None:
                            continue
                        process.terminate()
                        status = process.wait(timeout=_STOP_TIMEOUT)
                        if status != 0:
                            raise RenderError("owned chrome shutdown exited %s" % status)
                        if staged.read_bytes() != confirmed_data:
                            raise RenderError("chrome screenshot changed during shutdown")
                        return {
                            **dimensions,
                            "sha256": hashlib.sha256(data).hexdigest(),
                            "completion": "validated-capture-controlled-shutdown",
                        }
                time.sleep(0.05)
    finally:
        if process is not None:
            _cleanup_chrome(process)


def _screenshot(html: str, png_path: str) -> Dict[str, Any]:
    destination = Path(png_path)
    html_path, staged = None, None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", dir=destination.parent, delete=False, encoding="utf-8"
        ) as stream:
            html_path = Path(stream.name)
            stream.write(html)
        descriptor, temporary = tempfile.mkstemp(suffix=".png", dir=destination.parent)
        os.close(descriptor)
        staged = Path(temporary)
        staged.unlink()
        with tempfile.TemporaryDirectory(
            prefix="glassbox_chrome_", dir=destination.parent
        ) as profile:
            command = [
                chrome_path(),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--disable-extensions",
                "--password-store=basic",
                "--use-mock-keychain",
                "--user-data-dir=" + profile,
                "--force-device-scale-factor=" + _SCALE,
                "--window-size=" + _WINDOW,
                "--default-background-color=FFFFFFFF",
                "--screenshot=" + str(staged),
                html_path.resolve().as_uri(),
            ]
            image = _capture(command, staged, Path(profile) / "capture.log")
        data = staged.read_bytes()
        dimensions = _validate_png(data)
        if hashlib.sha256(data).hexdigest() != image["sha256"]:
            raise RenderError("chrome screenshot changed before replacement")
        os.replace(str(staged), str(destination))
        return {**dimensions, "sha256": image["sha256"], "completion": image["completion"]}
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderError("chrome render failed: %s" % exc) from exc
    finally:
        for path in (html_path, staged):
            if path is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


def _component(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", value).strip("_") or "view"
    if safe != value or value != value.casefold() or len(safe) > 80:
        safe = safe[:80] + "_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return safe


def render(
    presentation: Presentation, out_dir: Optional[str] = None, skip: bool = False
) -> Stimulus:
    text = presentation.to_text()
    if skip or not chrome_available():
        return Stimulus(
            presentation, "spec", text, render_metadata={"mode": "spec", "name": "skip-render"}
        )
    directory = (
        Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="glassbox_render_"))
    )
    try:
        stem = _component(presentation.scenario_id) + "__" + _component(presentation.variant)
        if presentation.target_question_id is not None:
            stem += "__" + _component(presentation.target_question_id)
        destination = validate_output_files(directory, [stem + ".png"])[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = _screenshot(presentation.to_html(), str(destination))
    except OSError as exc:
        raise RenderError("cannot create screenshot output: %s" % exc) from exc
    metadata = {
        "mode": "image",
        "name": "chrome",
        "version": renderer_version(),
        "window": [int(value) for value in _WINDOW.split(",")],
        "scale": int(_SCALE),
        "completion": image["completion"],
        "color_scheme": "system",
        "font_stack": "system",
        "png_dimensions": [image["width"], image["height"]],
    }
    return Stimulus(
        presentation,
        "image",
        text,
        image_path=str(destination),
        image_sha256=image["sha256"],
        render_metadata=metadata,
    )

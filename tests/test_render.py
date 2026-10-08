"""Rendering: skip-render degradation and clean RenderError on Chrome failure."""

import os
import tempfile
import unittest

from glassbox import interfaces, render, schema


class TestRender(unittest.TestCase):
    def setUp(self):
        self.loans = schema.load_all_scenarios()["loans"]

    def test_skip_render_needs_no_chrome(self):
        st = render.render(interfaces.variant(self.loans, "table"), skip=True)
        self.assertEqual(st.kind, "spec")
        self.assertIsNone(st.image_path)

    def test_screenshot_wraps_exec_failure_as_render_error(self):
        p = interfaces.variant(self.loans, "table")
        saved = os.environ.get("GLASSBOX_CHROME")
        os.environ["GLASSBOX_CHROME"] = "/nonexistent/chrome-binary-xyz"
        try:
            with self.assertRaises(render.RenderError):
                render._screenshot(p.to_html(), os.path.join(tempfile.mkdtemp(), "x.png"))
        finally:
            if saved is None:
                os.environ.pop("GLASSBOX_CHROME", None)
            else:
                os.environ["GLASSBOX_CHROME"] = saved


def _png(width=1, height=1):
    import struct
    import zlib

    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    rows = b"".join(b"\0" + b"\x00\x00\x00\xff" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class TestAtomicRender(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch

        self.processes = []
        group = patch("glassbox.render.os.killpg", create=True)
        self.kill_group = group.start()
        self.addCleanup(group.stop)

    def test_protected_output_roots_and_linked_pngs_fail_before_chrome(self):
        from pathlib import Path
        from unittest.mock import patch

        presentation = interfaces.variant(schema.load_all_scenarios()["growth"], "annotated")
        with (
            patch("glassbox.render.chrome_available", return_value=True),
            patch("glassbox.render.subprocess.Popen") as chrome,
        ):
            with self.assertRaisesRegex(ValueError, "sources"):
                render.render(presentation, out_dir=str(Path(render.__file__).parent))
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "input.txt"
                source.write_bytes(b"protected source bytes")
                destination = Path(directory) / "growth__annotated.png"
                destination.symlink_to(source)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    render.render(presentation, out_dir=directory)
                destination.unlink()
                os.link(source, destination)
                with self.assertRaisesRegex(ValueError, "unlinked"):
                    render.render(presentation, out_dir=directory)
                self.assertEqual(source.read_bytes(), b"protected source bytes")
            chrome.assert_not_called()

    def _run(
        self,
        payload,
        code=0,
        exception=None,
        confirmation="correct",
        shutdown_code=0,
        wait_timeout=False,
        poll_states=(),
        changed_on_stop=None,
    ):
        import subprocess
        from pathlib import Path
        from urllib.parse import unquote, urlparse

        observed = []
        states = iter(poll_states)

        class Process:
            pid = 999999

            def __init__(self, image):
                self.status = code
                self.image = image
                self.terminations = self.kills = 0

            def poll(self):
                self.status = next(states, self.status)
                return self.status

            def wait(self, timeout):
                if wait_timeout and self.status is None:
                    raise subprocess.TimeoutExpired("owned chrome", timeout)
                return self.status

            def terminate(self):
                self.terminations += 1
                self.status = None if wait_timeout else shutdown_code
                if changed_on_stop is not None:
                    self.image.write_bytes(changed_on_stop)

            def kill(self):
                self.kills += 1
                self.status = -9

        def fake(command, **kwargs):
            image = Path(
                next(arg.split("=", 1)[1] for arg in command if arg.startswith("--screenshot="))
            )
            profile = Path(
                next(arg.split("=", 1)[1] for arg in command if arg.startswith("--user-data-dir="))
            )
            html_file = Path(unquote(urlparse(command[-1]).path))
            self.assertTrue(profile.is_dir())
            self.assertTrue(html_file.is_file())
            self.assertFalse(image.exists())
            self.assertEqual(kwargs["start_new_session"], os.name == "posix")
            self.assertIn("--use-mock-keychain", command)
            self.assertIn("--password-store=basic", command)
            observed.extend([image, profile, html_file])
            if payload is not None:
                image.write_bytes(payload)
                if confirmation is not None:
                    size = len(payload) if confirmation != "wrong-size" else len(payload) + 1
                    path = image if confirmation != "wrong-path" else image.with_name("other.png")
                    kwargs["stdout"].write(
                        ("%d bytes written to file %s\n" % (size, path)).encode()
                    )
                    kwargs["stdout"].flush()
            if exception is not None:
                raise exception
            process = Process(image)
            self.processes.append(process)
            return process

        return fake, observed

    def test_stale_png_nonzero_exit_missing_and_invalid_outputs_preserve_old(self):
        from pathlib import Path
        from unittest.mock import patch

        for payload, code in (
            (_png(), 1),
            (None, 0),
            (b"not PNG", 0),
            (_png()[:-1] + b"x", 0),
            (_png(0, 1), 0),
        ):
            with (
                self.subTest(code=code, bytes=payload is not None),
                tempfile.TemporaryDirectory() as directory,
            ):
                destination = Path(directory) / "old.png"
                destination.write_bytes(b"previous image")
                fake, observed = self._run(payload, code)
                with patch("glassbox.render.subprocess.Popen", side_effect=fake):
                    with self.assertRaises(render.RenderError):
                        render._screenshot("<h1>view</h1>", str(destination))
                self.assertEqual(destination.read_bytes(), b"previous image")
                self.assertTrue(all(not path.exists() for path in observed))
                self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_timeout_exec_failure_and_replace_failure_clean_staging(self):
        import subprocess
        from pathlib import Path
        from unittest.mock import patch

        for exception in (OSError("fake exec failure"), subprocess.TimeoutExpired("chrome", 60)):
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "old.png"
                destination.write_bytes(b"previous")
                fake, observed = self._run(None, exception=exception)
                with patch("glassbox.render.subprocess.Popen", side_effect=fake):
                    with self.assertRaises(render.RenderError):
                        render._screenshot("view", str(destination))
                self.assertEqual(destination.read_bytes(), b"previous")
                self.assertTrue(all(not path.exists() for path in observed))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "old.png"
            destination.write_bytes(b"previous")
            fake, observed = self._run(_png())
            with (
                patch("glassbox.render.subprocess.Popen", side_effect=fake),
                patch("glassbox.render.os.replace", side_effect=OSError("fake replace")),
            ):
                with self.assertRaises(render.RenderError):
                    render._screenshot("view", str(destination))
            self.assertEqual(destination.read_bytes(), b"previous")
            self.assertTrue(all(not path.exists() for path in observed))

    def test_success_hash_metadata_and_filename_confinement(self):
        import hashlib
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch

        presentation = interfaces.variant(schema.load_all_scenarios()["growth"], "annotated")
        presentation = replace(
            presentation,
            scenario_id="../../outside",
            variant="../custom",
            target_question_id="../target",
        )
        with tempfile.TemporaryDirectory(prefix="glassbox test ") as directory:
            fake, observed = self._run(_png(2, 1))
            with (
                patch("glassbox.render.chrome_available", return_value=True),
                patch("glassbox.render.renderer_version", return_value="fake chrome"),
                patch("glassbox.render.subprocess.Popen", side_effect=fake),
            ):
                stimulus = render.render(presentation, out_dir=directory)
            image = Path(stimulus.image_path)
            self.assertEqual(image.parent, Path(directory).resolve())
            self.assertEqual(image.read_bytes(), _png(2, 1))
            self.assertEqual(stimulus.image_sha256, hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertEqual(stimulus.render_metadata["png_dimensions"], [2, 1])
            self.assertEqual(stimulus.render_metadata["window"], [820, 620])
            self.assertEqual(stimulus.render_metadata["scale"], 2)
            self.assertEqual(stimulus.render_metadata["completion"], "natural-exit")
            self.assertTrue(all(not path.exists() for path in observed))
        self.assertNotEqual(render._component("cards+derived"), render._component("cards_derived"))
        self.assertNotEqual(render._component("Growth"), render._component("growth"))

    def test_complete_capture_requires_exact_browser_confirmation(self):
        from pathlib import Path
        from unittest.mock import patch

        for confirmation in (None, "wrong-path", "wrong-size"):
            with (
                self.subTest(confirmation=confirmation),
                tempfile.TemporaryDirectory() as directory,
            ):
                destination = Path(directory) / "old.png"
                destination.write_bytes(b"previous image")
                fake, observed = self._run(_png(), code=None, confirmation=confirmation)
                with (
                    patch("glassbox.render.subprocess.Popen", side_effect=fake),
                    patch("glassbox.render._RENDER_TIMEOUT", 0),
                ):
                    with self.assertRaises(render.RenderError):
                        render._screenshot("view", str(destination))
                self.assertEqual(destination.read_bytes(), b"previous image")
                self.assertTrue(all(not path.exists() for path in observed))

    def test_natural_zero_exit_valid_fresh_png_needs_no_log_wording(self):
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "view.png"
            fake, observed = self._run(_png(), confirmation=None)
            with patch("glassbox.render.subprocess.Popen", side_effect=fake):
                metadata = render._screenshot("view", str(destination))
            self.assertEqual(destination.read_bytes(), _png())
            self.assertEqual(metadata["completion"], "natural-exit")
            self.assertEqual(self.processes[-1].terminations, 0)
            self.assertTrue(all(not path.exists() for path in observed))

    def test_controlled_capture_requires_graceful_zero_exit_and_records_method(self):
        from unittest.mock import patch

        presentation = interfaces.variant(schema.load_all_scenarios()["loans"], "cards")
        with tempfile.TemporaryDirectory() as directory:
            fake, observed = self._run(_png(), code=None)
            with (
                patch("glassbox.render.subprocess.Popen", side_effect=fake),
                patch("glassbox.render._EXIT_GRACE", 0),
                patch("glassbox.render.chrome_available", return_value=True),
                patch("glassbox.render.renderer_version", return_value="fake chrome"),
            ):
                stimulus = render.render(presentation, out_dir=directory)
            self.assertEqual(stimulus.kind, "image")
            self.assertEqual(
                stimulus.render_metadata["completion"], "validated-capture-controlled-shutdown"
            )
            self.assertEqual(self.processes[-1].terminations, 1)
            self.assertEqual(self.processes[-1].kills, 0)
            self.assertEqual(self.processes[-1].status, 0)
            self.assertTrue(all(not path.exists() for path in observed))
            if os.name == "posix":
                self.assertTrue(
                    all(
                        call.args[0] == self.processes[-1].pid
                        for call in self.kill_group.call_args_list
                    )
                )

    def test_natural_nonzero_or_failed_controlled_shutdown_never_replace_image(self):
        from pathlib import Path
        from unittest.mock import patch

        cases = (
            {"poll_states": (None, 1)},
            {"shutdown_code": -15},
            {"shutdown_code": -9},
            {"shutdown_code": 1},
            {"wait_timeout": True},
            {"changed_on_stop": _png(2, 1)},
        )
        for options in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "old.png"
                destination.write_bytes(b"previous image")
                fake, observed = self._run(_png(), code=None, **options)
                with (
                    patch("glassbox.render.subprocess.Popen", side_effect=fake),
                    patch("glassbox.render._EXIT_GRACE", 0),
                ):
                    with self.assertRaises(render.RenderError):
                        render._screenshot("view", str(destination))
                self.assertEqual(destination.read_bytes(), b"previous image")
                self.assertTrue(all(not path.exists() for path in observed))

    def test_skip_provenance_and_metadata_are_detached(self):
        import hashlib

        from glassbox.stimuli import Stimulus

        presentation = interfaces.variant(schema.load_all_scenarios()["loans"], "table")
        stimulus = render.render(presentation, skip=True)
        self.assertEqual(
            stimulus.text_sha256, hashlib.sha256(stimulus.text.encode("utf-8")).hexdigest()
        )
        self.assertIsNone(stimulus.image_sha256)
        metadata = {"window": [1, 2]}
        copied = Stimulus(presentation, "spec", "text", render_metadata=metadata)
        metadata["window"][0] = 999
        self.assertEqual(copied.render_metadata["window"], [1, 2])

    def test_hashseed_does_not_change_actual_shown_text_or_html(self):
        import json
        import subprocess
        import sys
        from pathlib import Path

        code = """import json
from glassbox import interfaces, schema
rows=[]
for scenario in schema.load_all_scenarios().values():
    for name in interfaces.ALL_VARIANTS:
        p=interfaces.variant(scenario,name)
        rows.append([p.shown,p.to_text(),p.to_html()])
    for question in scenario.questions:
        p=interfaces.build(scenario,frozenset({'detail','derived','headline','cards_context','highlight','sorted'}),target=question)
        rows.append([p.shown,p.to_text(),p.to_html()])
print(json.dumps(rows))"""
        outputs = []
        for seed in ("1", "3"):
            env = dict(
                os.environ,
                PYTHONHASHSEED=seed,
                PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
            )
            result = subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
            )
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(json.loads(outputs[0])), 24)


if __name__ == "__main__":
    unittest.main()

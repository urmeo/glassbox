import copy
import hashlib
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.parse import unquote, urlparse

from glassbox import core


class TestReaders(unittest.TestCase):
    def test_simulated_reader_identities(self):
        readers = core.build_readers(["simulated"])
        self.assertEqual([reader.name for reader in readers],
                         ["simulated:literal", "simulated:diligent", "simulated:careless"])
        for reader in readers:
            with self.subTest(reader=reader.name):
                identity = core.resolve_family(reader.name)
                self.assertEqual(identity.family, "simulated")
                self.assertTrue(identity.resolved)
                self.assertTrue(reader.simulated)
                self.assertTrue(reader.deterministic)
        for spec in (None, "", "unknown", "simulated:unknown"):
            with self.subTest(spec=spec):
                self.assertEqual(core.resolve_family(spec).family, "unknown")

    def test_live_reader_specs_are_rejected(self):
        for spec in (
            "hosted:model", "proxy:third-party-model",
            "local:model", "remote:model", "proxy:model", "simulated:unknown",
        ):
            with self.subTest(spec=spec):
                self.assertRaises(ValueError, core.build_reader, spec)
                self.assertRaises(ValueError, core.build_readers, [spec])


class TestRender(unittest.TestCase):
    def setUp(self):
        self.loans = core.load_all_scenarios()["loans"]

    def test_explicit_field_display_applies_to_html_and_text(self):
        data = copy.deepcopy(self.loans.data)
        data["presentation"]["field_display"] = {
            "fees": {"unit": "$", "value_label": "Setup fee"}
        }
        scenario = replace(self.loans, data=data)
        for name in ("table", "annotated"):
            with self.subTest(variant=name):
                presentation = core.build_variant(scenario, name)
                for view in (presentation.to_html(), presentation.to_text()):
                    self.assertIn("Setup fee", view)
                    self.assertIn("$300", view)


def _png(width=1, height=1):
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
        self.processes = []
        group = patch("glassbox.core.os.killpg", create=True)
        self.kill_group = group.start()
        self.addCleanup(group.stop)

    def test_protected_output_roots_and_linked_pngs_fail_before_chrome(self):
        presentation = core.build_variant(core.load_all_scenarios()["growth"], "annotated")
        with (
            patch("glassbox.core.chrome_available", return_value=True),
            patch("glassbox.core.subprocess.Popen") as chrome,
        ):
            with self.assertRaisesRegex(ValueError, "sources"):
                core.render_stimulus(presentation, out_dir=str(Path(core.__file__).parent))
            with tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "input.txt"
                source.write_bytes(b"protected source bytes")
                destination = Path(directory) / "growth__annotated.png"
                destination.symlink_to(source)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    core.render_stimulus(presentation, out_dir=directory)
                destination.unlink()
                os.link(source, destination)
                with self.assertRaisesRegex(ValueError, "unlinked"):
                    core.render_stimulus(presentation, out_dir=directory)
                self.assertEqual(source.read_bytes(), b"protected source bytes")
            chrome.assert_not_called()

    @contextmanager
    def capture(
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

        with (
            patch("glassbox.core.chrome_available", return_value=True),
            patch("glassbox.core.renderer_version", return_value="fake chrome"),
            patch("glassbox.core.subprocess.Popen", side_effect=fake),
        ):
            yield
        self.assertTrue(all(not path.exists() for path in observed))

    def assert_capture_fails(self, payload, **options):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "old.png"
            destination.write_bytes(b"previous image")
            with self.capture(payload, **options), self.assertRaises(core.RenderError):
                core._screenshot("view", str(destination))
            self.assertEqual(destination.read_bytes(), b"previous image")
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_invalid_capture_preserves_previous_image(self):
        for payload, options in (
            (_png(), {"code": 1}), (None, {}), (b"not PNG", {}),
            (_png()[:-1] + b"x", {}), (_png(0, 1), {}),
            (None, {"exception": OSError("exec failure")}),
            (None, {"exception": subprocess.TimeoutExpired("chrome", 60)}),
        ):
            with self.subTest(options=options, payload=payload):
                self.assert_capture_fails(payload, **options)
        with patch("glassbox.core.os.replace", side_effect=OSError("replace failure")):
            self.assert_capture_fails(_png())

    def test_cleanup_failure_preserves_previous_image(self):
        temporary_directory = tempfile.TemporaryDirectory

        class CleanupFailure:
            def __init__(self, *args, **kwargs):
                self.directory = temporary_directory(*args, **kwargs)

            def __enter__(self):
                return self.directory.__enter__()

            def __exit__(self, *args):
                self.directory.__exit__(*args)
                raise OSError("cleanup failed")

        with temporary_directory() as directory:
            destination = Path(directory) / "old.png"
            destination.write_bytes(b"previous image")
            with self.capture(_png()), patch("glassbox.core.tempfile.TemporaryDirectory", CleanupFailure):
                with self.assertRaisesRegex(core.RenderError, "cleanup failed"):
                    core._screenshot("view", str(destination))
            self.assertEqual(destination.read_bytes(), b"previous image")
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_success_hash_metadata_and_filename_confinement(self):
        presentation = core.build_variant(core.load_all_scenarios()["growth"], "annotated")
        presentation = replace(presentation, scenario_id="../../outside", variant="../custom",
                               target_question_id="../target")
        with tempfile.TemporaryDirectory(prefix="glassbox test ") as directory:
            with self.capture(_png(2, 1), confirmation=None):
                stimulus = core.render_stimulus(presentation, out_dir=directory)
            image = Path(stimulus.image_path)
            self.assertEqual(image.parent, Path(directory).resolve())
            self.assertEqual(image.read_bytes(), _png(2, 1))
            self.assertEqual(stimulus.image_sha256, hashlib.sha256(image.read_bytes()).hexdigest())
            self.assertEqual(self.processes[-1].terminations, 0)
            for field, expected in {
                "png_dimensions": [2, 1], "window": [820, 620],
                "scale": 2, "completion": "natural-exit",
            }.items():
                with self.subTest(field=field):
                    self.assertEqual(stimulus.render_metadata[field], expected)
        self.assertNotEqual(core._component("cards+derived"), core._component("cards_derived"))
        self.assertNotEqual(core._component("Growth"), core._component("growth"))

    def test_long_identifiers_produce_distinct_bounded_filenames(self):
        presentation = core.build_variant(core.load_all_scenarios()["growth"], "table")
        presentation = replace(presentation, scenario_id="s" * 120, variant="v" * 120)
        with tempfile.TemporaryDirectory() as directory:
            images = []
            for suffix in ("first", "second"):
                with self.capture(_png()):
                    stimulus = core.render_stimulus(
                        replace(presentation, target_question_id="q" * 120 + suffix),
                        out_dir=directory,
                    )
                image = Path(stimulus.image_path)
                self.assertLessEqual(len(image.name.encode("utf-8")), 255)
                self.assertEqual(image.read_bytes(), _png())
                images.append(image)
            self.assertNotEqual(*images)
            self.assertEqual(set(Path(directory).resolve().iterdir()), set(images))

    def test_complete_capture_requires_exact_browser_confirmation(self):
        for confirmation in (None, "wrong-path", "wrong-size"):
            with self.subTest(confirmation=confirmation), patch("glassbox.core._RENDER_TIMEOUT", 0):
                self.assert_capture_fails(_png(), code=None, confirmation=confirmation)

    def test_controlled_capture_requires_graceful_zero_exit_and_records_method(self):
        presentation = core.build_variant(core.load_all_scenarios()["loans"], "cards")
        with tempfile.TemporaryDirectory() as directory:
            with self.capture(_png(), code=None), patch("glassbox.core._EXIT_GRACE", 0):
                stimulus = core.render_stimulus(presentation, out_dir=directory)
            self.assertEqual(stimulus.kind, "image")
            self.assertEqual(stimulus.render_metadata["completion"], "validated-capture-controlled-shutdown")
            process = self.processes[-1]
            self.assertEqual((process.terminations, process.kills, process.status), (1, 0, 0))
            if os.name == "posix":
                self.kill_group.assert_called()
                self.assertTrue(all(call.args[0] == process.pid for call in self.kill_group.call_args_list))

    def test_failed_controlled_shutdown_never_replaces_image(self):
        for options in (
            {"poll_states": (None, 1)}, {"shutdown_code": -15},
            {"shutdown_code": -9}, {"shutdown_code": 1},
            {"wait_timeout": True}, {"changed_on_stop": _png(2, 1)},
        ):
            with self.subTest(options=options), patch("glassbox.core._EXIT_GRACE", 0):
                self.assert_capture_fails(_png(), code=None, **options)

    def test_skip_render_provenance(self):
        presentation = core.build_variant(core.load_all_scenarios()["loans"], "table")
        stimulus = core.render_stimulus(presentation, skip=True)
        self.assertEqual(
            stimulus.text_sha256, hashlib.sha256(stimulus.text.encode("utf-8")).hexdigest()
        )
        self.assertEqual((stimulus.kind, stimulus.image_path, stimulus.image_sha256), ("spec", None, None))

    def test_separate_processes_render_identical_text_and_html(self):
        code = """from glassbox import core
rows = []
features = frozenset({'detail', 'derived', 'headline', 'cards_context', 'highlight', 'sorted'})
for scenario in core.load_all_scenarios().values():
    for name in core.ALL_VARIANTS:
        presentation = core.build_variant(scenario, name)
        rows.append([presentation.shown, presentation.to_text(), presentation.to_html()])
    for question in scenario.questions:
        presentation = core.build_presentation(scenario, features, target=question)
        rows.append([presentation.shown, presentation.to_text(), presentation.to_html()])
print(core.encode_yaml(rows))"""
        outputs = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-B", "-c", code],
                cwd=Path(__file__).resolve().parents[1] / "src",
                capture_output=True, text=True, check=True,
            )
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(core.decode_yaml(outputs[0])), 24)


class TestRenderSnapshot(unittest.TestCase):
    def test_lossy_numeric_display_is_rejected_before_capture(self):
        presentation = core.build_variant(core.load_all_scenarios()["loans"], "table")
        for skip in (True, False):
            with (
                self.subTest(skip=skip),
                patch("glassbox.core._num", return_value="0"),
                patch("glassbox.core.chrome_available", return_value=True),
                patch("glassbox.core._screenshot") as capture,
            ):
                with self.assertRaisesRegex(ValueError, "numeric display"):
                    core.render_stimulus(presentation, skip=skip)
                capture.assert_not_called()

    def test_caller_mutation_during_capture_does_not_change_recorded_input(self):
        scenario = core.load_all_scenarios()["loans"]
        presentation = core.build_variant(scenario, "table")
        expected = copy.deepcopy(presentation)

        def screenshot(document, destination):
            self.assertEqual(document, expected.to_html())
            presentation.shown["A"]["total_cost"] = -1
            presentation.order.reverse()
            presentation.labels["A"] = "Changed after capture started"
            return {"width": 820, "height": 620, "sha256": "a" * 64,
                    "completion": "natural-exit"}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("glassbox.core.chrome_available", return_value=True),
            patch("glassbox.core.renderer_version", return_value="test-browser"),
            patch("glassbox.core._screenshot", side_effect=screenshot) as capture,
        ):
            stimulus = core.render_stimulus(presentation, out_dir=directory)
        capture.assert_called_once()
        self.assertEqual(stimulus.presentation, expected)
        self.assertEqual(stimulus.text, expected.to_text())
        self.assertEqual(stimulus.text, stimulus.presentation.to_text())
        self.assertNotEqual(stimulus.presentation.shown, presentation.shown)


class TestStimulusBoundaries(unittest.TestCase):
    def test_image_and_spec_fields_agree_with_the_declared_kind(self):
        presentation = core.build_variant(core.load_all_scenarios()["loans"], "table")
        for fields in (
            {"kind": "image"},
            {"kind": "image", "image_path": ""},
            {"kind": "image", "image_path": True},
            {"kind": "spec", "image_path": "image.png"},
            {"kind": "spec", "image_sha256": "0" * 64},
            {"kind": "spec", "presentation": None},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                arguments = {"presentation": presentation, "text": "text"}
                arguments.update(fields)
                core.Stimulus(**arguments)


class TestStimulusSnapshot(unittest.TestCase):
    def setUp(self):
        self.scenario = core.load_all_scenarios()["loans"]
        self.readers = core.build_readers(["simulated"])

    def answers(self, stimulus):
        return [reader.answer(self.scenario, question, stimulus)
                for reader in self.readers for question in self.scenario.questions]

    def test_caller_mutation_preserves_recorded_inputs_and_answers(self):
        for kind in ("spec", "image"):
            with self.subTest(kind=kind):
                presentation = core.build_variant(self.scenario, "table")
                presentation = replace(presentation, features=set(presentation.features))
                image_fields = {"image_path": "fixture.png", "image_sha256": "a" * 64} if kind == "image" else {}
                stimulus = core.Stimulus(presentation, kind, presentation.to_text(), **image_fields)
                recorded, answers = copy.deepcopy(stimulus), self.answers(stimulus)
                presentation.shown["A"]["total_cost"] = 1
                presentation.shown["C"] = {"total_cost": 0}
                presentation.order.reverse()
                presentation.features.add("polished")
                presentation.labels["A"] = "Changed offer"
                presentation.field_units["total_cost"] = "EUR"
                presentation.field_labels["total_cost"] = "Changed metric"
                self.assertEqual(stimulus, recorded)
                self.assertEqual(self.answers(stimulus), answers)
                self.assertEqual(stimulus.presentation.to_text(), stimulus.text)
                self.assertEqual(stimulus.text_sha256, recorded.text_sha256)

    def test_shared_source_does_not_share_snapshots_or_metadata(self):
        presentation = core.build_variant(self.scenario, "table")
        metadata = {"window": [820, 620], "capture": {"scale": 2}}
        spec = core.Stimulus(presentation, "spec", presentation.to_text(), render_metadata=metadata)
        image = replace(spec, kind="image", image_path="fixture.png", image_sha256="b" * 64)
        recorded, answers = copy.deepcopy(image), self.answers(image)
        metadata["window"][0] = 1
        metadata["capture"]["scale"] = 99
        self.assertEqual(spec.render_metadata, recorded.render_metadata)
        self.assertEqual(image.render_metadata, recorded.render_metadata)
        spec.presentation.shown["A"]["total_cost"] = 1
        spec.presentation.order.reverse()
        spec.presentation.labels.clear()
        spec.render_metadata["window"][0] = 2
        spec.render_metadata["capture"]["scale"] = 3
        self.assertEqual(image, recorded)
        self.assertEqual(self.answers(image), answers)
        self.assertEqual(presentation.shown["A"]["total_cost"], 11280)


if __name__ == "__main__":
    unittest.main()

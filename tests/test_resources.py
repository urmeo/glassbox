"""Packaged resources retain content and load independently of checkout paths."""

import os
import tempfile
import unittest
from unittest.mock import patch

from glassbox import schema
from glassbox.prompts import load_prompt
from glassbox.resources import (
    ResourceError,
    decode_json,
    resource_bytes,
    resource_json,
    resource_names,
    resource_sha256,
    resource_text,
)


class TestResources(unittest.TestCase):
    def test_all_eight_packaged_resources_and_builtin_config_names(self):
        counts = {
            kind: len(resource_names(kind))
            for kind in ("scenarios", "prompts", "studies", "anchors", "training")
        }
        self.assertEqual(
            counts, {"scenarios": 3, "prompts": 2, "studies": 1, "anchors": 1, "training": 1}
        )
        self.assertEqual(resource_json("studies", "offline_demo")["id"], "offline_demo")
        self.assertTrue(resource_json("anchors", "fixture")["is_fixture"])
        self.assertEqual(resource_json("training", "sft_grpo_qwen3vl")["id"], "sft_grpo_qwen3vl_8b")
        self.assertEqual(
            resource_sha256("prompts", "reader_mcq"),
            "f887de0d6020ef5b2453e8d3dba968e43040589787ca5a10b6b6f805ac27b54a",
        )
        self.assertIn("\n", load_prompt("reader_mcq"))
        self.assertEqual(
            resource_bytes("prompts", "reader_mcq").decode("utf-8"),
            resource_text("prompts", "reader_mcq.txt"),
        )

    def test_unrelated_working_directory_needs_no_checkout_data(self):
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                self.assertEqual(len(schema.load_all_scenarios()), 3)
                self.assertIn("{stem}", load_prompt("reader_mcq"))
        finally:
            os.chdir(previous)

    def test_resource_names_reject_traversal_and_json_is_strict(self):
        for name in ("../reader_mcq", "/tmp/file", "..\\file", "", "C:escape"):
            with self.subTest(name=name), self.assertRaises(ResourceError):
                resource_text("prompts", name)
        for kind in ("other", None, []):
            with self.subTest(kind=kind), self.assertRaises(ResourceError):
                resource_names(kind)
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'):
            with self.subTest(text=text), self.assertRaises(ResourceError):
                decode_json(text)
        with self.assertRaises(ResourceError):
            resource_text("prompts", "missing")

    def test_uses_single_child_traversable_access_without_as_file(self):
        class File:
            def read_bytes(self):
                return b"resource bytes"

        class Directory:
            def __init__(self, path=()):
                self.path = path

            def joinpath(self, child):
                if len(self.path) == 2:
                    self_test.assertEqual(self.path, ("_data", "prompts"))
                    self_test.assertEqual(child, "reader_mcq.txt")
                    return File()
                return Directory(self.path + (child,))

        self_test = self
        with patch("glassbox.resources.resources.files", return_value=Directory()):
            self.assertEqual(resource_text("prompts", "reader_mcq"), "resource bytes")

    def test_explicit_missing_external_directory_does_not_fall_back(self):
        with self.assertRaises(schema.SchemaError):
            schema.load_all_scenarios("")
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(schema.load_all_scenarios(directory), {})

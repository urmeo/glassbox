"""Study configs; load, validate, and run a reproducible H1 study offline."""

import json
import os
import tempfile
import unittest

from glassbox import studies
from glassbox.cli import main

SHIPPED_STUDY = "offline_demo"


class TestStudyConfig(unittest.TestCase):
    def test_load_shipped_study(self):
        config = studies.load_study(SHIPPED_STUDY)
        self.assertEqual(config.id, "offline_demo")
        self.assertEqual(config.readers, ["simulated"])
        self.assertTrue(config.skip_render)

    def test_missing_readers_rejected(self):
        with self.assertRaises(studies.StudyError):
            studies.parse_study({"id": "x", "description": "d"})

    def test_bad_interface_rejected(self):
        with self.assertRaises(studies.StudyError):
            studies.parse_study(
                {
                    "id": "x",
                    "description": "d",
                    "readers": ["simulated"],
                    "interfaces": ["not_a_variant"],
                }
            )

    def test_bad_replicates_rejected(self):
        with self.assertRaises(studies.StudyError):
            studies.parse_study(
                {
                    "id": "x",
                    "description": "d",
                    "readers": ["simulated"],
                    "replicates": 0,
                }
            )


class TestRunStudy(unittest.TestCase):
    def test_run_shipped_study_offline(self):
        config = studies.load_study(SHIPPED_STUDY)
        with tempfile.TemporaryDirectory() as tmp:
            result = studies.run_study(config, tmp)
            self.assertTrue(result.h1.divergence_found)
            self.assertTrue(os.path.isfile(result.paths["report"]))
            with open(result.paths["results"]) as fh:
                data = json.load(fh)
            self.assertEqual(data["meta"]["study"]["id"], "offline_demo")
            self.assertIn("cross_family", data)

    def test_unknown_scenario_rejected(self):
        config = studies.StudyConfig(
            id="x", description="d", readers=["simulated"], scenarios=["nope"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(studies.StudyError):
                studies.run_study(config, tmp)

    def test_cli_study_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                main(["study", "--config", SHIPPED_STUDY, "--out", tmp]), 0
            )
            self.assertTrue(os.path.isfile(os.path.join(tmp, "report.md")))


if __name__ == "__main__":
    unittest.main()

"""CLI — the offline end-to-end path that scripts/verify.sh exercises."""

import json
import os
import tempfile
import unittest

from glassbox.cli import main


class TestCLI(unittest.TestCase):
    def test_validate_ok(self):
        self.assertEqual(main(["validate"]), 0)

    def test_run_offline_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            self.assertEqual(main(["run", "--readers", "simulated", "--skip-render", "--out", out]), 0)
            self.assertTrue(os.path.isfile(os.path.join(out, "report.md")))
            with open(os.path.join(out, "results.json")) as fh:
                data = json.load(fh)
            self.assertTrue(data["h1_summary"]["divergence_found"])
            self.assertTrue(data["h1_summary"]["simulated"])

    def test_run_single_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "run")
            self.assertEqual(
                main(["run", "--readers", "simulated", "--scenarios", "loans",
                      "--skip-render", "--out", out]), 0)
            with open(os.path.join(out, "results.json")) as fh:
                data = json.load(fh)
            self.assertEqual(list(data["metrics"].keys()), ["loans"])

    def test_repair_offline_writes_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "repair")
            self.assertEqual(
                main(["repair", "--scenario", "loans", "--reader", "simulated:literal",
                      "--skip-render", "--out", out]), 0)
            self.assertTrue(os.path.isfile(os.path.join(out, "transcript.md")))

    def test_unknown_scenario_errors(self):
        with self.assertRaises(SystemExit):
            main(["run", "--scenarios", "nonexistent", "--skip-render",
                  "--out", tempfile.mkdtemp()])


if __name__ == "__main__":
    unittest.main()

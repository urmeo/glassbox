"""Check wheel/source resources and offline commands outside the checkout."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]

SMOKE = """
import importlib, json, pathlib, pkgutil, subprocess, sys
import glassbox
from glassbox.resources import resource_names, resource_bytes
from glassbox.schema import load_all_scenarios
root = pathlib.Path(sys.argv[1])
assert glassbox.__version__ == "0.1.1"
assert not pathlib.Path(glassbox.__file__).resolve().is_relative_to(root)
for module in pkgutil.walk_packages(glassbox.__path__, glassbox.__name__ + "."):
    importlib.import_module(module.name)
count = 0
for kind in ("scenarios", "prompts", "studies", "anchors", "training"):
    for name in resource_names(kind):
        assert resource_bytes(kind, name) == (root / "src/glassbox/_data" / kind / name).read_bytes()
        count += 1
assert count == 8
scenarios = load_all_scenarios()
assert len(scenarios) == 3 and sum(len(s.questions) for s in scenarios.values()) == 9
def run(args, status=0, console=False):
    command = [str(pathlib.Path(sys.executable).with_name("glassbox"))] if console else [sys.executable, "-m", "glassbox"]
    result = subprocess.run(command + args, capture_output=True, text=True)
    assert result.returncode == status, (args, result.stdout, result.stderr)
    assert "Traceback" not in result.stderr
run(["validate"], console=True)
run(["validate"])
run(["--help"])
run(["run", "--readers", "simulated", "--skip-render", "--out", "demo"])
run(["run", "--readers", "simulated", "--skip-render", "--judge", "pairwise", "--out", "pairwise"])
run(["study", "--config", "offline_demo", "--out", "study"])
run(["repair", "--scenario", "growth", "--question", "biggest_percent_gain", "--reader", "simulated:literal", "--skip-render", "--out", "repair"])
run(["anchor", "--set", "fixture", "--skip-render", "--out", "anchor"])
run(["optimize", "--readers", "simulated", "--skip-render", "--out", "search"])
run(["run", "--replicates", "0", "--skip-render"], 2)
def invalid_constant(value):
    raise ValueError("nonfinite JSON: " + value)
for path in pathlib.Path(".").rglob("*.json"):
    json.loads(path.read_text(), parse_constant=invalid_constant)
assert pathlib.Path("demo/results.json").is_file()
assert pathlib.Path("repair/transcript.md").is_file()
print("Installed checks: 8 resources; 3 scenarios; 9 answers; 10 CLI cases; strict JSON.")
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "python", type=Path, help="Python from the clean installed environment"
    )
    args = parser.parse_args()
    wheels = list((ROOT / "dist").glob("*.whl"))
    sdists = list((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        parser.error("dist requires exactly one wheel and one source distribution")
    files = [
        p
        for p in (ROOT / "src/glassbox").rglob("*")
        if p.is_file() and (p.suffix == ".py" or "_data" in p.parts)
    ]
    files = [p for p in files if "__pycache__" not in p.parts]
    with zipfile.ZipFile(wheels[0]) as wheel, tarfile.open(sdists[0]) as sdist:
        members = sdist.getnames()
        wheel_licenses = [
            name
            for name in wheel.namelist()
            if name.endswith(".dist-info/licenses/LICENSE")
        ]
        source_licenses = [name for name in members if name.endswith("/LICENSE")]
        if len(wheel_licenses) != 1 or len(source_licenses) != 1:
            raise ValueError("distributions require one MIT LICENSE")
        with sdist.extractfile(source_licenses[0]) as handle:
            if handle.read() != (ROOT / "LICENSE").read_bytes():
                raise ValueError("source distribution license differs")
        if wheel.read(wheel_licenses[0]) != (ROOT / "LICENSE").read_bytes():
            raise ValueError("wheel license differs")
        for path in files:
            name = path.relative_to(ROOT / "src").as_posix()
            if wheel.read(name) != path.read_bytes():
                raise ValueError("wheel bytes differ: " + name)
            matches = [m for m in members if m.endswith("/src/" + name)]
            if len(matches) != 1:
                raise ValueError("source distribution missing/duplicating: " + name)
            with sdist.extractfile(matches[0]) as handle:
                if handle.read() != path.read_bytes():
                    raise ValueError("source distribution bytes differ: " + name)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="glassbox-installed-") as directory:
        subprocess.run(
            [os.path.abspath(args.python), "-W", "error", "-", str(ROOT)],
            input=SMOKE,
            text=True,
            cwd=directory,
            env=environment,
            check=True,
        )
    print("Wheel/source bytes matched: %d files." % len(files))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print("error: %s" % error, file=sys.stderr)
        raise SystemExit(1) from None

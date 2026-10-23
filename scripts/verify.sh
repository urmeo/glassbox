#!/usr/bin/env bash
set -euo pipefail

require_outputs() {
    local directory="$1"
    shift
    local filename
    for filename in "$@"; do
        if [[ ! -f "$directory/$filename" || ! -s "$directory/$filename" ]]; then
            printf 'Missing or empty output: %s\n' "$directory/$filename" >&2
            return 1
        fi
    done
}

glassbox_python="${1:-python3}"
if [[ "$glassbox_python" == */* && "$glassbox_python" != /* ]]; then
    glassbox_python="$PWD/$glassbox_python"
fi
cd "$(dirname "$0")/../src"

printf '1/3 unit tests\n'
"$glassbox_python" -B -W error - <<'PYTHON'
import importlib.util
import sys
import unittest
from pathlib import Path

spec = importlib.util.find_spec("glassbox")
locations = set() if spec is None else {
    Path(path).resolve() for path in spec.submodule_search_locations or ()
}
if locations != {(Path.cwd() / "glassbox").resolve()}:
    sys.exit("glassbox resolves outside this checkout; use a clean Python environment.")
suite = unittest.defaultTestLoader.discover("../tests", pattern="test_*.py")
sys.exit(not unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful())
PYTHON

printf '2/3 answer validation\n'
"$glassbox_python" -B -m glassbox validate

glassbox_check_dir="$(mktemp -d)"
trap 'rm -rf "$glassbox_check_dir"' EXIT

printf '3/3 offline commands\n'
"$glassbox_python" -B -m glassbox run \
    --readers simulated --skip-render --out "$glassbox_check_dir/demo"
require_outputs "$glassbox_check_dir/demo" report.md results.yaml

"$glassbox_python" -B -m glassbox repair \
    --scenario growth --question biggest_percent_gain --reader simulated:literal \
    --skip-render --out "$glassbox_check_dir/repair"
require_outputs "$glassbox_check_dir/repair" transcript.md repair.yaml

"$glassbox_python" -B -m glassbox optimize \
    --readers simulated --scenarios loans,plans --skip-render \
    --out "$glassbox_check_dir/selection"
require_outputs "$glassbox_check_dir/selection" optimize.md optimize.yaml selection.yaml

"$glassbox_python" -B -m glassbox evaluate \
    --selection "$glassbox_check_dir/selection/selection.yaml" \
    --scenarios growth --out "$glassbox_check_dir/evaluation"
require_outputs "$glassbox_check_dir/evaluation" evaluation.md evaluation.yaml

printf 'Verified tests, source answers and offline commands.\n'

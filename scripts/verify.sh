#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
glassbox_python="${GLASSBOX_PYTHON:-python3}"
glassbox_check_dir="$(mktemp -d)"
trap 'rm -rf "$glassbox_check_dir"' EXIT
printf '1/3 unit tests\n'
"$glassbox_python" -W error -m unittest discover -s tests -p 'test_*.py' -q
printf '2/3 answer validation\n'
"$glassbox_python" -m glassbox validate
printf '3/3 offline commands\n'
"$glassbox_python" -m glassbox run --readers simulated --skip-render --out "$glassbox_check_dir/demo"
"$glassbox_python" -m glassbox repair --scenario growth --question biggest_percent_gain --reader simulated:literal --skip-render --out "$glassbox_check_dir/repair"
test -s "$glassbox_check_dir/demo/results.json"
test -s "$glassbox_check_dir/repair/transcript.md"
printf 'Verified tests, source answers and offline commands.\n'

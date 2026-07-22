#!/usr/bin/env bash
# verify.sh — the one green light for Glass Box.
# Runs, in order: the unit-test suite, data-integrity validation (every answer key
# recomputed from source), and an offline end-to-end demo (render→read→score→judge→
# H1 analysis, plus the H4 repair loop). Fully offline: no network, no API keys.
# Exit 0 with "VERIFY GREEN" only if all stages pass.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

bold() { printf '\n\033[1m══ %s ══\033[0m\n' "$1"; }
fail() { printf '\033[31m✗ %s\033[0m\n' "$1" >&2; exit 1; }

bold "unit tests"
python3 -m unittest discover -s tests -p "test_*.py" -v

bold "data integrity — recompute every answer key from source"
python3 -m glassbox validate

bold "offline end-to-end demo — render(skip) → read(simulated) → score → judge → H1 analysis"
demo="runs/_verify_demo"
rm -rf "$demo"
python3 -m glassbox run --readers simulated --skip-render --out "$demo"
[ -f "$demo/report.md" ]    || fail "missing $demo/report.md"
[ -f "$demo/results.json" ] || fail "missing $demo/results.json"

bold "offline repair loop — H4 turns to understanding"
repair="runs/_verify_repair"
rm -rf "$repair"
python3 -m glassbox repair --scenario loans --reader simulated:literal --skip-render --out "$repair"
[ -f "$repair/transcript.md" ] || fail "missing $repair/transcript.md"

bold "VERIFY GREEN"
echo "✓ tests + validate + offline e2e demo (H1 report + repair transcript) all passed"

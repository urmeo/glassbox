"""Data integrity — recompute every answer key from source and confirm it matches.

This is the guarantee behind "no unverifiable answer keys" (a project success
criterion). ``glassbox validate`` runs :func:`validate_all`; so does the test suite
and ``scripts/verify.sh``. A scenario whose authored ``answer`` disagrees with the
value recomputed from its data is a hard failure, not a warning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import compute
from .schema import Scenario, load_all_scenarios


def validate_scenario(scenario: Scenario) -> List[str]:
    """Return a list of integrity errors for one scenario (empty means clean)."""
    errors: List[str] = []

    # Every derived expression must evaluate for every item.
    for name, expr in scenario.derived.items():
        for item in scenario.items:
            try:
                compute.evaluate_expr(expr, item)
            except compute.ExpressionError as exc:
                errors.append(
                    "%s: derived %r fails on item %r: %s"
                    % (scenario.id, name, item["id"], exc)
                )

    # Every question's authored answer must equal the recomputed answer.
    for q in scenario.questions:
        try:
            recomputed = compute.recompute(scenario.data, {
                "id": q.id,
                "compute": q.compute,
                "choices": [{"id": c.id, "value": c.value} for c in q.choices],
            })
        except (compute.ExpressionError, ValueError, KeyError) as exc:
            errors.append("%s/%s: cannot recompute answer: %s" % (scenario.id, q.id, exc))
            continue
        if recomputed != q.answer:
            errors.append(
                "%s/%s: authored answer %r but source data recomputes to %r"
                % (scenario.id, q.id, q.answer, recomputed)
            )
    return errors


def validate_all(directory: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Validate every scenario. Returns (ok, human-readable report lines)."""
    scenarios: Dict[str, Scenario] = load_all_scenarios(directory)
    if not scenarios:
        return False, ["no scenarios found in %s" % (directory or "data/scenarios")]

    report: List[str] = []
    all_errors: List[str] = []
    for sid, scenario in scenarios.items():
        errors = validate_scenario(scenario)
        all_errors.extend(errors)
        status = "ok" if not errors else "FAIL"
        report.append("%-16s %s  (%d question%s)"
                      % (sid, status, len(scenario.questions),
                         "" if len(scenario.questions) == 1 else "s"))
        report.extend("    - " + e for e in errors)
    return (not all_errors), report

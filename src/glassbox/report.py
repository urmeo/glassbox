"""Report — write a run's results to ``results.json`` + a human ``report.md``, and a
repair loop to ``transcript.md``.

Run provenance (models, renderer, versions, a content hash per scenario) is recorded so
a result is always traceable to the exact data and setup that produced it. When every
reader is simulated, the report says so at the top and frames the finding as a pipeline
demonstration, never evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from . import __version__, interfaces
from .analysis import CrossFamilyAnalysis, H1Analysis
from .render import renderer_version
from .repair import RepairResult
from .schema import Question, Scenario
from .scoring import ScoreBook, QuestionResult

SIMULATED_CAVEAT = (
    "> **Simulated readers only.** These readers are deterministic fixtures whose "
    "behavior is designed, not observed. This report shows that the harness detects a "
    "divergence when one is present by construction; it is **not** evidence about real "
    "readers or real interfaces."
)


def _scenario_hash(scenario: Scenario) -> str:
    blob = json.dumps(scenario.data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def run_metadata(scenarios: Sequence[Scenario], reader_names: Sequence[str],
                 skip_render: bool) -> Dict[str, Any]:
    return {
        "glassbox_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "renderer": renderer_version(skip=skip_render),
        "readers": list(reader_names),
        "scenarios": {s.id: _scenario_hash(s) for s in scenarios},
        "deterministic": all(r.startswith("simulated:") for r in reader_names),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "argv": " ".join(sys.argv),
    }


def _pct(x: float) -> str:
    return "%+.0f pts" % (x * 100)


# --- results.json -----------------------------------------------------------

def build_results(scenarios: Sequence[Scenario], results: Sequence[QuestionResult],
                  book: ScoreBook, h1: H1Analysis, cross: CrossFamilyAnalysis,
                  skip_render: bool) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    scen_h1 = {s.scenario_id: s for s in h1.scenarios}
    for sid in book.scenarios:
        per_reader = {}
        for r in book.readers:
            per_reader[r] = {
                "ceiling": book.ceiling(r, sid),
                "baseline": book.baseline(r, sid),
                "variants": {v: {"accuracy": book.accuracy(r, sid, v),
                                 "lift": book.lift(r, sid, v),
                                 "ceiling_gap": book.ceiling_gap(r, sid, v)}
                             for v in book.variants},
            }
        aggregate = {v: {"mean_lift": book.mean_lift(sid, v),
                         "mean_accuracy": book.mean_accuracy(sid, v)}
                     for v in book.variants}
        sh = scen_h1[sid]
        metrics[sid] = {
            "readers": per_reader,
            "aggregate": aggregate,
            "h1": {
                "spearman": sh.spearman,
                "reader_agreement": sh.reader_agreement,
                "rows": [row.__dict__ for row in sh.rows],
                "reversals": [rv.__dict__ for rv in sh.reversals],
            },
        }
    return {
        "meta": run_metadata(scenarios, book.readers, skip_render),
        "per_question": [r.__dict__ for r in results],
        "metrics": metrics,
        "h1_summary": {
            "mean_spearman": h1.mean_spearman,
            "divergence_found": h1.divergence_found,
            "simulated": h1.simulated,
        },
        "cross_family": {
            "n_families": cross.n_families,
            "families": [{"family": f.family, "readers": f.readers,
                          "comp_lift": f.comp_lift, "reversal_holds": f.reversal_holds}
                         for f in cross.families],
            "preference": cross.preference,
            "cross_family_agreement": cross.cross_family_agreement,
            "survives_across_families": cross.survives_across_families,
        },
    }


# --- report.md --------------------------------------------------------------

def render_cross_family_md(cross: CrossFamilyAnalysis) -> str:
    lines: List[str] = ["## Cross-family agreement", ""]
    if cross.n_families < 2:
        fam = cross.families[0].family if cross.families else "none"
        lines.append("Only **1** reader family (`%s`). Cross-family agreement needs ≥2 "
                     "independent families — add real readers (M2, needs API keys). The "
                     "reversal within this family: **%s**."
                     % (fam, "holds" if cross.families and cross.families[0].reversal_holds
                        else "not present"))
        lines.append("")
        return "\n".join(lines)

    verdict = ("survives across families ✓" if cross.survives_across_families
               else "does NOT survive across all families")
    lines.append("Divergence **%s** · cross-family agreement (Spearman of comprehension "
                 "rankings) = **%.2f** across %d families."
                 % (verdict, cross.cross_family_agreement, cross.n_families))
    lines.append("")
    header = "| interface | preference | " + " | ".join(f.family for f in cross.families) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * cross.n_families)
    for v in cross.interface_variants:
        row = "| %s | %.2f | " % (v, cross.preference[v])
        row += " | ".join(_pct(f.comp_lift[v]) for f in cross.families) + " |"
        lines.append(row)
    lines.append("")
    for f in cross.families:
        lines.append("- family `%s` (%d readers): reversal %s"
                     % (f.family, len(f.readers),
                        "holds ✓" if f.reversal_holds else "absent ✗"))
    lines.append("")
    return "\n".join(lines)


def render_report_md(book: ScoreBook, h1: H1Analysis,
                     cross: Optional[CrossFamilyAnalysis] = None) -> str:
    lines: List[str] = ["# Glass Box — comprehension vs preference (H1)", ""]
    lines.append("Readers: " + ", ".join("`%s`" % r for r in book.readers))
    lines.append("")
    if h1.simulated:
        lines.append(SIMULATED_CAVEAT)
        lines.append("")

    verdict = ("preference and comprehension diverge"
               if h1.divergence_found else "no divergence detected")
    lines.append("**Headline:** %s — mean Spearman(preference, comprehension) = %.2f "
                 "across %d scenario(s)." % (verdict, h1.mean_spearman, len(h1.scenarios)))
    lines.append("")

    label = {sh.scenario_id: sh for sh in h1.scenarios}
    for sid in book.scenarios:
        sh = label[sid]
        lines.append("## %s" % sid)
        lines.append("")
        lines.append("Spearman(pref, comp) = **%.2f** · cross-reader agreement = %.2f"
                     % (sh.spearman, sh.reader_agreement))
        lines.append("")
        lines.append("| interface | comprehension lift | mean accuracy | preference | understood rank | preferred rank |")
        lines.append("|---|---:|---:|---:|:---:|:---:|")
        for row in sorted(sh.rows, key=lambda r: r.comp_rank):
            lines.append("| %s | %s | %.0f%% | %.2f | %d | %d |"
                         % (row.variant, _pct(row.comp_lift), row.mean_accuracy * 100,
                            row.pref_score, row.comp_rank, row.pref_rank))
        lines.append("")
        if sh.reversals:
            for rv in sh.reversals:
                lines.append("- **Reversal:** `%s` is preferred over `%s` (preference "
                             "+%.2f) yet transfers **less** understanding "
                             "(comprehension %s)." % (rv.preferred, rv.understood,
                                                      rv.pref_gap, _pct(-rv.comp_gap)))
        else:
            lines.append("- No reversal: preference and comprehension agree here.")
        lines.append("")
    if cross is not None:
        lines.append(render_cross_family_md(cross))
    return "\n".join(lines)


def write_run(out_dir: str, scenarios: Sequence[Scenario],
              results: Sequence[QuestionResult], book: ScoreBook, h1: H1Analysis,
              cross: CrossFamilyAnalysis, skip_render: bool) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.json")
    report_path = os.path.join(out_dir, "report.md")
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(build_results(scenarios, results, book, h1, cross, skip_render), fh, indent=2)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_report_md(book, h1, cross))
    return {"results": results_path, "report": report_path}


# --- transcript.md (repair) -------------------------------------------------

def render_transcript_md(result: RepairResult, scenario: Scenario, question: Question) -> str:
    lines: List[str] = ["# Repair transcript (H4) — turns to understanding", ""]
    lines.append("Scenario `%s` · reader `%s` · question `%s`"
                 % (result.scenario_id, result.reader, result.question_id))
    lines.append("")
    lines.append("> %s" % question.stem)
    correct_text = next(c.text for c in question.choices if c.id == question.answer)
    lines.append("")
    lines.append("Correct answer: **%s**" % correct_text)
    lines.append("")
    if result.reader.startswith("simulated:"):
        lines.append(SIMULATED_CAVEAT)
        lines.append("")
    for t in result.turns:
        chosen_text = next((c.text for c in question.choices if c.id == t.chosen), "—")
        mark = "✓ understood" if t.correct else "✗ wrong"
        header = "**Turn %d** — interface `%s` (%s)" % (t.turn, t.variant, ", ".join(t.features))
        if t.transform:
            header += " — repair: *%s*" % t.transform
        lines.append(header)
        lines.append("")
        lines.append("- reader chose: %s → %s" % (chosen_text, mark))
        lines.append("")
    if result.converged:
        lines.append("**Result:** understood after **%d** repair turn(s)."
                     % result.turns_to_understanding)
    else:
        lines.append("**Result:** did not converge within the turn budget.")
    return "\n".join(lines)


def write_repair(out_dir: str, result: RepairResult, scenario: Scenario,
                 question: Question) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    transcript_path = os.path.join(out_dir, "transcript.md")
    json_path = os.path.join(out_dir, "repair.json")
    with open(transcript_path, "w", encoding="utf-8") as fh:
        fh.write(render_transcript_md(result, scenario, question))
    with open(json_path, "w", encoding="utf-8") as fh:
        payload = {"scenario": result.scenario_id, "reader": result.reader,
                   "question": result.question_id, "converged": result.converged,
                   "turns_to_understanding": result.turns_to_understanding,
                   "turns": [t.__dict__ for t in result.turns]}
        json.dump(payload, fh, indent=2)
    return {"transcript": transcript_path, "repair": json_path}

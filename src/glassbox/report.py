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
import math
import os
import platform
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from . import __version__, interfaces
from .analysis import CrossFamilyAnalysis, H1Analysis
from .anchor import AnchorResult
from .reward import EvalResult
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


def _json_safe(obj: Any) -> Any:
    """Recursively replace NaN/Infinity floats with None — bare NaN is invalid JSON
    (RFC 8259) and a strict parser rejects it. Undefined correlations become null."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _dump(payload: Any, fh) -> None:
    json.dump(_json_safe(payload), fh, indent=2)


def _scenario_hash(scenario: Scenario) -> str:
    blob = json.dumps(scenario.data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def run_metadata(scenarios: Sequence[Scenario], reader_names: Sequence[str],
                 skip_render: bool, study: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = {
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
    if study is not None:
        meta["study"] = study
    return meta


def _pct(x: float) -> str:
    return "%+.0f pts" % (x * 100)


# --- results.json -----------------------------------------------------------

def build_results(scenarios: Sequence[Scenario], results: Sequence[QuestionResult],
                  book: ScoreBook, h1: H1Analysis, cross: CrossFamilyAnalysis,
                  skip_render: bool, study: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        "meta": run_metadata(scenarios, book.readers, skip_render, study),
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
              cross: CrossFamilyAnalysis, skip_render: bool,
              study: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.json")
    report_path = os.path.join(out_dir, "report.md")
    with open(results_path, "w", encoding="utf-8") as fh:
        _dump(build_results(scenarios, results, book, h1, cross, skip_render, study), fh)
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
        _dump(payload, fh)
    return {"transcript": transcript_path, "repair": json_path}


# --- anchor report (H3) -----------------------------------------------------

ANCHOR_FIXTURE_CAVEAT = (
    "> **Fixture anchor set.** The human numbers here are synthetic, invented only to "
    "verify the harness. This is **not** an H3 result — replace with published CALVI / "
    "Cleveland & McGill per-item human accuracy for a real anchor run."
)


def _corr(x: float) -> str:
    return "—" if x != x else "%.2f" % x  # nan -> em dash


def render_anchor_md(result: AnchorResult) -> str:
    a = result.anchor
    lines: List[str] = ["# Glass Box — H3 anchoring (model reader vs human)", ""]
    lines.append("Anchor set: `%s` — %s" % (a.id, a.source))
    lines.append("Readers: " + ", ".join("`%s`" % r for r in result.readers))
    lines.append("Condition: readers see the `%s` presentation." % a.condition)
    lines.append("")
    if a.is_fixture:
        lines.append(ANCHOR_FIXTURE_CAVEAT)
        lines.append("")
    if result.readers and all(r.startswith("simulated:") for r in result.readers):
        lines.append(SIMULATED_CAVEAT)  # model side is simulated, too
        lines.append("")
    lines.append("**Correlation (model vs human accuracy, %d items):** Spearman %s · "
                 "Pearson %s · Kendall %s." % (result.n_items, _corr(result.spearman),
                                               _corr(result.pearson), _corr(result.kendall)))
    lines.append("")
    lines.append("| item | human | model | gap |")
    lines.append("|---|---:|---:|---:|")
    for p in result.points:
        lines.append("| %s/%s | %.0f%% | %.0f%% | %.0f%% |"
                     % (p.scenario, p.question, p.human_accuracy * 100,
                        p.model_accuracy * 100, p.abs_gap * 100))
    lines.append("")
    lines.append("**Where the proxy breaks (largest gaps):**")
    for p in result.breaks():
        lines.append("- `%s/%s` — human %.0f%% vs model %.0f%% (gap %.0f pts)"
                     % (p.scenario, p.question, p.human_accuracy * 100,
                        p.model_accuracy * 100, p.abs_gap * 100))
    lines.append("")
    return "\n".join(lines)


def write_anchor(out_dir: str, result: AnchorResult) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "anchor.md")
    json_path = os.path.join(out_dir, "anchor.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_anchor_md(result))
    with open(json_path, "w", encoding="utf-8") as fh:
        _dump({
            "anchor_id": result.anchor.id, "source": result.anchor.source,
            "is_fixture": result.anchor.is_fixture, "condition": result.anchor.condition,
            "readers": result.readers,
            "correlations": {"spearman": result.spearman, "pearson": result.pearson,
                             "kendall": result.kendall},
            "points": [p.__dict__ for p in result.points],
        }, fh)
    return {"report": report_path, "anchor": json_path}


# --- optimize report (M4, offline generator search) -------------------------

OPTIMIZE_CAVEAT = (
    "> **Offline search generator.** The \"generator\" here is an exhaustive search over "
    "the interface feature-lattice, not a trained model — it proves the comprehension "
    "reward is optimizable. The real generator (SFT → GRPO on a VLM) is gated on paid "
    "training (Tinker credits + GPU)."
)


def render_optimize_md(ev: EvalResult, readers: Sequence[str],
                       training: Optional[Dict[str, Any]] = None) -> str:
    lines: List[str] = ["# Glass Box — comprehension-optimized interface (M4)", ""]
    lines.append("Readers scoring the reward: " + ", ".join("`%s`" % r for r in readers))
    lines.append("")
    lines.append(OPTIMIZE_CAVEAT)
    lines.append("")
    if readers and all(r.startswith("simulated:") for r in readers):
        lines.append(SIMULATED_CAVEAT)  # the reward numbers come from designed fixtures
        lines.append("")
    lines.append("**Generator selected:** `%s`" % (", ".join(sorted(ev.generator_features)) or "(none)"))
    lines.append("")
    lines.append("| interface | comprehension lift |")
    lines.append("|---|---:|")
    lines.append("| generator (reward-optimized) | %s |" % _pct(ev.generator_reward))
    lines.append("| polished cards (preference-tuned analog) | %s |" % _pct(ev.cards_reward))
    lines.append("| plain-text baseline | +0 pts |")
    lines.append("")
    lines.append("**Verdict:** beats the preference-tuned baseline on comprehension: **%s** · "
                 "beats plain text: **%s**."
                 % ("yes" if ev.beats_preference_tuned else "no",
                    "yes" if ev.beats_plaintext else "no"))
    lines.append("")
    if training is not None:
        state = "ready to run" if not training["errors"] else "BLOCKED"
        lines.append("**Training plan** `%s` — %s. Pre-run validation: %s."
                     % (training["id"], training["status"], state))
        for e in training["errors"]:
            lines.append("- ✗ %s" % e)
        lines.append("")
    return "\n".join(lines)


def write_optimize(out_dir: str, ev: EvalResult, readers: Sequence[str],
                   training: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "optimize.md")
    json_path = os.path.join(out_dir, "optimize.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_optimize_md(ev, readers, training))
    with open(json_path, "w", encoding="utf-8") as fh:
        _dump({
            "generator_features": sorted(ev.generator_features),
            "generator_reward": ev.generator_reward, "cards_reward": ev.cards_reward,
            "beats_preference_tuned": ev.beats_preference_tuned,
            "beats_plaintext": ev.beats_plaintext, "readers": list(readers),
            "training": training,
        }, fh)
    return {"report": report_path, "optimize": json_path}

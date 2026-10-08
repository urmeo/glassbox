"""Strict JSON and concise reports with separate reader and judge provenance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .analysis import CrossFamilyAnalysis, H1Analysis
from .anchor import AnchorResult
from .reward import EvalResult
from .render import renderer_version
from .repair import RepairResult
from .schema import (
    Question,
    Scenario,
    scenario_payload,
    scenario_sha256,
    SCENARIO_HASH_FORMAT,
)
from .scoring import ScoreBook, QuestionResult

SIMULATED_CAVEAT = (
    "> **Simulated readers only.** These readers are deterministic fixtures whose "
    "behavior is designed, not observed. This report shows that the harness detects a "
    "divergence when one is present by construction; it is **not** evidence about real "
    "readers or real interfaces."
)


def _json_safe(obj: Any) -> Any:
    """Undefined floating measures become JSON null."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _dump(payload: Any, fh) -> None:
    json.dump(_json_safe(payload), fh, indent=2, ensure_ascii=False, allow_nan=False)


def _scenario_hash(scenario: Scenario) -> str:
    return scenario_sha256(scenario)


def run_metadata(
    scenarios: Sequence[Scenario],
    reader_names: Sequence[str],
    skip_render: bool,
    study: Optional[Dict[str, Any]] = None,
    *,
    book: Optional[ScoreBook] = None,
    h1: Optional[H1Analysis] = None,
) -> Dict[str, Any]:
    from .prompts import load_prompt
    from .resources import resource_sha256
    from .answer_parsing import PARSER_VERSION

    meta = {
        "glassbox_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "renderer": renderer_version(skip=skip_render),
        "readers": list(reader_names),
        "scenarios": {s.id: _scenario_hash(s) for s in scenarios},
        "scenario_hash_format": SCENARIO_HASH_FORMAT,
        "scenario_snapshots": {s.id: scenario_payload(s) for s in scenarios},
        "prompt_resource_sha256": {
            name: resource_sha256("prompts", name)
            for name in ("reader_mcq", "judge_pairwise")
        },
        "prompt_template_sha256": {
            name: hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()
            for name in ("reader_mcq", "judge_pairwise")
        },
        "answer_parser_version": PARSER_VERSION,
        "deterministic": h1.deterministic if h1 is not None else None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
    }
    if book is not None:
        actual_variants = list(dict.fromkeys(row.variant for row in book.results))
        meta["resolved_selection"] = {
            "scenarios": [s.id for s in scenarios],
            "interfaces": [v for v in actual_variants if v in book.interface_variants()],
            "variants": actual_variants,
            "readers": list(reader_names),
        }
        meta["reader_identity"] = {
            name: {
                "provider": row.reader_provider,
                "model": row.reader_model,
                "model_family": row.reader_model_family,
                "family_resolved": row.reader_family_resolved,
                "family_basis": row.reader_family_basis,
                "simulated": row.reader_simulated,
                "deterministic": row.reader_deterministic,
            }
            for name in book.readers
            for row in [next(r for r in book.results if r.reader == name)]
        }
        meta["all_readers_simulated"] = book.all_simulated()
        meta["any_readers_simulated"] = book.any_simulated()
        meta["mixed_readers"] = book.any_simulated() and book.any_real()
    if h1 is not None:
        meta["judge"] = asdict(h1.judge)
    if study is not None:
        meta["study"] = study
    return meta


def _pct(x: float) -> str:
    return "%+.0f pts" % (x * 100)




def build_results(
    scenarios: Sequence[Scenario],
    results: Sequence[QuestionResult],
    book: ScoreBook,
    h1: H1Analysis,
    cross: CrossFamilyAnalysis,
    skip_render: bool,
    study: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    scen_h1 = {s.scenario_id: s for s in h1.scenarios}
    for sid in book.scenarios:
        per_reader = {}
        for r in book.readers:
            per_reader[r] = {
                "ceiling": book.ceiling(r, sid),
                "baseline": book.baseline(r, sid),
                "variants": {
                    v: {
                        "accuracy": book.accuracy(r, sid, v),
                        "lift": book.lift(r, sid, v),
                        "ceiling_gap": book.ceiling_gap(r, sid, v),
                    }
                    for v in book.variants
                },
            }
        aggregate = {
            v: {
                "mean_lift": book.mean_lift(sid, v),
                "mean_accuracy": book.mean_accuracy(sid, v),
            }
            for v in book.variants
        }
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
        "meta": run_metadata(
            scenarios, book.readers, skip_render, study, book=book, h1=h1
        ),
        "per_question": [asdict(r) for r in results],
        "metrics": metrics,
        "h1_summary": {
            "mean_spearman": h1.mean_spearman,
            "divergence_found": h1.divergence_found,
            "simulated": h1.simulated,
            "any_reader_simulated": h1.any_reader_simulated,
            "mixed_readers": h1.mixed_readers,
            "judge": asdict(h1.judge),
            "deterministic": h1.deterministic,
        },
        "cross_family": {
            "n_families": cross.n_families,
            "families": [
                {
                    "family": f.family,
                    "readers": f.readers,
                    "comp_lift": f.comp_lift,
                    "reversal_holds": f.reversal_holds,
                    "reversals": f.reversals,
                }
                for f in cross.families
            ],
            "preference": cross.preference,
            "cross_family_agreement": cross.cross_family_agreement,
            "survives_across_families": cross.survives_across_families,
            "survival_definition": "any_reversal_in_every_resolved_family",
            "common_reversals": cross.common_reversals,
            "unresolved_readers": cross.unresolved_readers,
            "judge_target": cross.judge_target,
        },
    }




def render_cross_family_md(cross: CrossFamilyAnalysis) -> str:
    lines = [
        "## Cross-family agreement",
        "",
        "%d resolved model families; unresolved readers excluded: %s."
        % (cross.n_families, ", ".join(cross.unresolved_readers) or "none"),
        "",
    ]
    lines.append(
        "Any reversal in every resolved family: **%s** (requires at least 2)."
        % ("yes" if cross.survives_across_families else "no")
    )
    lines.append(
        "Common reversed pairs: %s."
        % (
            ", ".join("`%s` over `%s`" % pair for pair in cross.common_reversals)
            or "none"
        )
    )
    lines.append(
        "Spearman agreement of family lift vectors: **%s**."
        % _corr(cross.cross_family_agreement)
    )
    lines.append(
        "Resolved labels describe model names; synthetic fixtures provide no independent empirical evidence."
    )
    lines.append("")
    if cross.families:
        lines.extend(
            [
                "| interface | judge score | "
                + " | ".join(f.family for f in cross.families)
                + " |",
                "|---|---:|" + "---:|" * cross.n_families,
            ]
        )
        for v in cross.interface_variants:
            lines.append(
                "| %s | %.2f | %s |"
                % (
                    v,
                    cross.preference[v],
                    " | ".join(_pct(f.comp_lift[v]) for f in cross.families),
                )
            )
    return "\n".join(lines) + "\n"


def render_report_md(
    book: ScoreBook, h1: H1Analysis, cross: Optional[CrossFamilyAnalysis] = None
) -> str:
    target = h1.judge.score_label
    lines = [
        "# Glass Box: MCQ lift vs %s (H1)" % target,
        "",
        "Readers: " + ", ".join("`%s`" % r for r in book.readers),
        "",
    ]
    if h1.simulated:
        lines.extend([SIMULATED_CAVEAT, ""])
    elif h1.any_reader_simulated:
        lines.extend(
            [
                "> **Mixed readers.** Some scores come from designed synthetic fixtures.",
                "",
            ]
        )
    lines.append(
        "Judge: `%s`; target **%s**; modality `%s`; protocol `%s`; comparisons %d."
        % (
            h1.judge.name,
            target,
            h1.judge.modality,
            h1.judge.protocol,
            h1.judge.comparison_count,
        )
    )
    if h1.judge.simulated:
        lines.append(
            "> **Synthetic judge.** These scores use fixed fixture rules, including when readers are real."
        )
    lines.append(
        "All components declared deterministic: **%s**."
        % ("yes" if h1.deterministic else "no")
    )
    lines.extend(
        [
            "",
            "Judge score and MCQ lift: **%s**; mean Spearman %s across %d scenarios."
            % (
                "a reversal is present"
                if h1.divergence_found
                else "no reversal detected",
                _corr(h1.mean_spearman),
                len(h1.scenarios),
            ),
            "",
        ]
    )
    for sh in h1.scenarios:
        lines.extend(
            [
                "## %s" % sh.scenario_id,
                "",
                "Spearman(judge, lift) = **%s**; reader agreement = **%s**."
                % (_corr(sh.spearman), _corr(sh.reader_agreement)),
                "",
                "| interface | MCQ lift | accuracy | %s | lift rank | judge rank |"
                % target,
                "|---|---:|---:|---:|:---:|:---:|",
            ]
        )
        for row in sorted(sh.rows, key=lambda r: r.comp_rank):
            lines.append(
                "| %s | %s | %.0f%% | %.2f | %d | %d |"
                % (
                    row.variant,
                    _pct(row.comp_lift),
                    row.mean_accuracy * 100,
                    row.pref_score,
                    row.comp_rank,
                    row.pref_rank,
                )
            )
        lines.append("")
        for rv in sh.reversals:
            lines.append(
                "- **Reversal:** `%s` scores above `%s` on %s (+%.2f), with lower MCQ lift (%s)."
                % (rv.preferred, rv.understood, target, rv.pref_gap, _pct(-rv.comp_gap))
            )
        if not sh.reversals:
            lines.append("No reversed pair.")
        lines.append("")
    if cross is not None:
        lines.append(render_cross_family_md(cross))
    return "\n".join(lines)


def write_run(
    out_dir: str,
    scenarios: Sequence[Scenario],
    results: Sequence[QuestionResult],
    book: ScoreBook,
    h1: H1Analysis,
    cross: CrossFamilyAnalysis,
    skip_render: bool,
    study: Optional[Dict[str, Any]] = None,
    *,
    protected_paths: Sequence[str] = (),
) -> Dict[str, str]:
    results_path, report_path = _destinations(
        out_dir, ("results.json", "report.md"), protected_paths
    )
    with open(results_path, "w", encoding="utf-8") as fh:
        _dump(
            build_results(scenarios, results, book, h1, cross, skip_render, study), fh
        )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_report_md(book, h1, cross))
    return {"results": results_path, "report": report_path}




def render_transcript_md(
    result: RepairResult, scenario: Scenario, question: Question
) -> str:
    lines: List[str] = ["# Repair transcript (H4): answer accuracy", ""]
    lines.append(
        "Scenario `%s` · reader `%s` · question `%s`"
        % (result.scenario_id, result.reader, result.question_id)
    )
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
        chosen_text = next(
            (c.text for c in question.choices if c.id == t.chosen), "unknown"
        )
        mark = "✓ understood" if t.correct else "✗ wrong"
        header = "**Turn %d**: interface `%s` (%s)" % (
            t.turn,
            t.variant,
            ", ".join(t.features),
        )
        if t.transform:
            header += "; repair: *%s*" % t.transform
        lines.append(header)
        lines.append("")
        lines.append("- reader chose: %s → %s" % (chosen_text, mark))
        lines.append("")
    if result.converged:
        lines.append(
            "**Result:** understood after **%d** repair turn(s)."
            % result.turns_to_understanding
        )
    else:
        lines.append("**Result:** did not converge within the turn budget.")
    return "\n".join(lines)


def write_repair(
    out_dir: str,
    result: RepairResult,
    scenario: Scenario,
    question: Question,
    *,
    protected_paths: Sequence[str] = (),
) -> Dict[str, str]:
    transcript_path, json_path = _destinations(
        out_dir, ("transcript.md", "repair.json"), protected_paths
    )
    with open(transcript_path, "w", encoding="utf-8") as fh:
        fh.write(render_transcript_md(result, scenario, question))
    with open(json_path, "w", encoding="utf-8") as fh:
        payload = {
            "scenario": result.scenario_id,
            "reader": result.reader,
            "question": result.question_id,
            "converged": result.converged,
            "turns_to_understanding": result.turns_to_understanding,
            "turns": [t.__dict__ for t in result.turns],
        }
        _dump(payload, fh)
    return {"transcript": transcript_path, "repair": json_path}



ANCHOR_FIXTURE_CAVEAT = (
    "> **Fixture anchor set.** The human numbers here are synthetic, invented only to "
    "verify the harness. This is a fixture check. Use verified per-item human accuracy for a human comparison: "
    "declare its source and conditions."
)


def _corr(x: float) -> str:
    return "undefined" if not math.isfinite(x) else "%.2f" % x


def render_anchor_md(result: AnchorResult) -> str:
    a = result.anchor
    lines: List[str] = ["# Glass Box: model and human accuracy (H3)", ""]
    lines.append("Anchor set: `%s`; %s" % (a.id, a.source))
    lines.append("Readers: " + ", ".join("`%s`" % r for r in result.readers))
    lines.append("Condition: readers see the `%s` presentation." % a.condition)
    lines.append("")
    if a.is_fixture:
        lines.append(ANCHOR_FIXTURE_CAVEAT)
        lines.append("")
    if result.readers and all(r.startswith("simulated:") for r in result.readers):
        lines.append(SIMULATED_CAVEAT)
        lines.append("")
    lines.append(
        "**Correlation (model vs human accuracy, %d items):** Spearman %s · "
        "Pearson %s · Kendall %s."
        % (
            result.n_items,
            _corr(result.spearman),
            _corr(result.pearson),
            _corr(result.kendall),
        )
    )
    lines.append("")
    lines.append("| item | human | model | gap |")
    lines.append("|---|---:|---:|---:|")
    for p in result.points:
        lines.append(
            "| %s/%s | %.0f%% | %.0f%% | %.0f%% |"
            % (
                p.scenario,
                p.question,
                p.human_accuracy * 100,
                p.model_accuracy * 100,
                p.abs_gap * 100,
            )
        )
    lines.append("")
    lines.append("**Where the proxy breaks (largest gaps):**")
    for p in result.breaks():
        lines.append(
            "- `%s/%s`: human %.0f%% vs model %.0f%% (gap %.0f pts)"
            % (
                p.scenario,
                p.question,
                p.human_accuracy * 100,
                p.model_accuracy * 100,
                p.abs_gap * 100,
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_anchor(
    out_dir: str, result: AnchorResult, *, protected_paths: Sequence[str] = ()
) -> Dict[str, str]:
    report_path, json_path = _destinations(
        out_dir, ("anchor.md", "anchor.json"), protected_paths
    )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_anchor_md(result))
    with open(json_path, "w", encoding="utf-8") as fh:
        _dump(
            {
                "anchor_id": result.anchor.id,
                "source": result.anchor.source,
                "is_fixture": result.anchor.is_fixture,
                "condition": result.anchor.condition,
                "readers": result.readers,
                "correlations": {
                    "spearman": result.spearman,
                    "pearson": result.pearson,
                    "kendall": result.kendall,
                },
                "points": [p.__dict__ for p in result.points],
            },
            fh,
        )
    return {"report": report_path, "anchor": json_path}



OPTIMIZE_CAVEAT = (
    "> **In-sample feature search.** Candidate selection and reported scores use the "
    "same scenarios, questions and readers. A fresh plain-text baseline is sampled "
    "for every candidate and the cards comparison. There is no held-out evaluation "
    "or trained generator result. Model-family exclusion and restricted features "
    "do not prove resistance to answer leakage or reward gaming."
)


def render_optimize_md(
    ev: EvalResult, readers: Sequence[str], training: Optional[Dict[str, Any]] = None
) -> str:
    lines: List[str] = ["# Glass Box: in-sample feature search (M4)", ""]
    lines.append(
        "Readers scoring the reward: " + ", ".join("`%s`" % r for r in readers)
    )
    lines.append("")
    lines.append(OPTIMIZE_CAVEAT)
    lines.append("")
    if ev.all_readers_simulated:
        lines.append(SIMULATED_CAVEAT)
        lines.append("")
    elif ev.any_reader_simulated:
        lines.extend(
            ["> **Mixed readers.** Some rewards come from synthetic fixtures.", ""]
        )
    lines.append(
        "Protocol `%s`; %d candidates; %d scenarios; %d readers; baseline `%s`."
        % (
            ev.selection_protocol,
            ev.candidate_count,
            ev.scenario_count,
            ev.reader_count,
            ev.baseline_sampling,
        )
    )
    lines.append("")
    lines.append(
        "**Generator selected:** `%s`"
        % (", ".join(sorted(ev.generator_features)) or "(none)")
    )
    lines.append("")
    lines.append("| interface | comprehension lift |")
    lines.append("|---|---:|")
    lines.append("| generator (reward-optimized) | %s |" % _pct(ev.generator_reward))
    lines.append("| fixed polished cards | %s |" % _pct(ev.cards_reward))
    lines.append("| plain-text baseline | +0 pts |")
    lines.append("")
    lines.append(
        "Higher in-sample lift than fixed cards: **%s**; "
        "positive measured plain-text lift: **%s**."
        % (
            "yes" if ev.beats_preference_tuned else "no",
            "yes" if ev.beats_plaintext else "no",
        )
    )
    lines.append("")
    if training is not None:
        state = "ready to run" if not training["errors"] else "BLOCKED"
        lines.append(
            "**Training plan** `%s`: %s. Pre-run validation: %s."
            % (training["id"], training["status"], state)
        )
        for e in training["errors"]:
            lines.append("- ✗ %s" % e)
        lines.append("")
    return "\n".join(lines)


def write_optimize(
    out_dir: str,
    ev: EvalResult,
    readers: Sequence[str],
    training: Optional[Dict[str, Any]] = None,
    *,
    protected_paths: Sequence[str] = (),
) -> Dict[str, str]:
    report_path, json_path = _destinations(
        out_dir, ("optimize.md", "optimize.json"), protected_paths
    )
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_optimize_md(ev, readers, training))
    with open(json_path, "w", encoding="utf-8") as fh:
        _dump(
            {
                "generator_features": sorted(ev.generator_features),
                "generator_reward": ev.generator_reward,
                "cards_reward": ev.cards_reward,
                "beats_preference_tuned": ev.beats_preference_tuned,
                "beats_plaintext": ev.beats_plaintext,
                "readers": list(readers),
                "selection_protocol": ev.selection_protocol,
                "baseline_sampling": ev.baseline_sampling,
                "candidate_count": ev.candidate_count,
                "scenario_count": ev.scenario_count,
                "reader_count": ev.reader_count,
                "all_readers_simulated": ev.all_readers_simulated,
                "any_reader_simulated": ev.any_reader_simulated,
                "training": training,
            },
            fh,
        )
    return {"report": report_path, "optimize": json_path}


def _destinations(
    out_dir: str, names: Sequence[str], protected_paths: Sequence[str]
) -> List[str]:
    from .output_paths import validate_output_files

    paths = validate_output_files(out_dir, names, protected_paths=protected_paths)
    os.makedirs(paths[0].parent, exist_ok=True)
    return [str(path) for path in paths]

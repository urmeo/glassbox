"""Strict JSON and Markdown reports with separate reader and judge provenance."""

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
from .reward import EvalResult, HeldOutResult
from .selection import (
    SelectionArtifact, selection_payload, validate_selection, write_selection,
)
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
    "> **Simulated readers only.** Scores follow fixture rules and do not measure "
    "model or human comprehension."
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


def _pct(value: float) -> str:
    return "%+.0f pts" % (value * 100) if math.isfinite(value) else "undefined"


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
            "reversal_comparison_available": cross.reversal_comparison_available,
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
    if not cross.reversal_comparison_available:
        reversal_status = "unavailable"
    else:
        reversal_status = "yes" if cross.survives_across_families else "no"
    lines.append(
        "Any reversal in every resolved family: **%s** (requires at least 2)."
        % reversal_status
    )
    lines.append(
        "Common reversed pairs: %s."
        % (
            (", ".join("`%s` over `%s`" % pair for pair in cross.common_reversals) or "none")
            if cross.reversal_comparison_available else "unavailable"
        )
    )
    lines.append(
        "Spearman agreement of family lift vectors: **%s**."
        % _corr(cross.cross_family_agreement)
    )
    lines.append(
        "Families are inferred from model names; synthetic fixtures are not independent evidence."
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
                "| %s | %s | %s |"
                % (
                    v,
                    _corr(cross.preference[v]),
                    " | ".join(_pct(f.comp_lift[v]) for f in cross.families),
                )
            )
    return "\n".join(lines) + "\n"


def render_report_md(
    book: ScoreBook, h1: H1Analysis, cross: Optional[CrossFamilyAnalysis] = None
) -> str:
    target = h1.judge.score_label
    lines = [
        "# GlassBox: MCQ lift vs %s" % target,
        "",
        "Readers: " + ", ".join("`%s`" % r for r in book.readers),
        "",
    ]
    if h1.simulated:
        lines.extend([SIMULATED_CAVEAT, ""])
    elif h1.any_reader_simulated:
        lines.extend(
            [
                "> **Mixed readers.** Some scores come from synthetic fixtures.",
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
    if h1.judge.evidence:
        lines.append(
            "Judgments: %d valid (including %d ties), %d invalid, %d abstained."
            % (h1.judge.valid_count, h1.judge.tie_count,
               h1.judge.invalid_count, h1.judge.abstained_count)
        )
        lines.append(
            "Invalid and abstained replies receive no credit and are excluded from rate denominators."
        )
    if h1.judge.simulated:
        lines.append(
            "> **Synthetic judge.** These scores use fixed fixture rules, including when readers are real."
        )
    lines.append(
        "Declared deterministic: **%s**."
        % ("yes" if h1.deterministic else "no")
    )
    if h1.divergence_found:
        ranking_status = "a reversal is present"
    elif any(math.isfinite(row.pref_score) for s in h1.scenarios for row in s.rows):
        ranking_status = "no reversal among scored interfaces"
    else:
        ranking_status = "judge scores unavailable"
    lines.extend([
        "",
        "Judge score and MCQ lift: **%s**; mean Spearman %s across %d scenarios."
        % (ranking_status, _corr(h1.mean_spearman), len(h1.scenarios)),
        "",
    ])
    if any(not math.isfinite(row.pref_score) for s in h1.scenarios for row in s.rows):
        lines.extend([
            "Judge scores with no valid comparisons are undefined. Ranks and correlations use finite scores only.",
            "",
        ])
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
        for row in sorted(sh.rows, key=lambda r: (r.comp_rank is None, r.comp_rank or 0)):
            lines.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (
                    row.variant,
                    _pct(row.comp_lift),
                    _percent(row.mean_accuracy),
                    _corr(row.pref_score),
                    row.comp_rank if row.comp_rank is not None else "undefined",
                    row.pref_rank if row.pref_rank is not None else "undefined",
                )
            )
        lines.append("")
        for rv in sh.reversals:
            lines.append(
                "- **Reversal:** `%s` scores above `%s` on %s (+%.2f), with lower MCQ lift (%s)."
                % (rv.preferred, rv.understood, target, rv.pref_gap, _pct(-rv.comp_gap))
            )
        if not sh.reversals:
            scored = sum(
                math.isfinite(row.pref_score) and math.isfinite(row.comp_lift)
                for row in sh.rows
            )
            lines.append(
                "No reversed pair among scored interfaces." if scored >= 2
                else "Too few scored interfaces to compare rankings."
            )
        lines.append("")
    if h1.judge.evidence:
        lines.extend([
            "## Judge coverage", "",
            "Rates use valid comparisons only; coverage can differ after invalid or abstained replies.", "",
            "| scenario | interface | credit | valid | planned |",
            "|---|---|---:|---:|---:|",
        ])
        for evidence in h1.judge.evidence:
            for score in evidence.variant_scores:
                lines.append("| %s | %s | %.2f | %d | %d |" % (
                    evidence.scenario_id, score.variant, score.credit,
                    score.denominator, score.planned_count,
                ))
        lines.extend([
            "", "A/B order, outcomes, raw replies and prompt hashes are in `results.json` under `meta.judge.evidence`.", "",
            "Cross-family judge means require a finite score in every selected scenario.", "",
        ])
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
    lines: List[str] = ["# Repair transcript: answer accuracy", ""]
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
        mark = "correct" if t.correct else "incorrect"
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
            "**Result:** correct answer after **%d** repair turn(s)."
            % result.turns_to_understanding
        )
    else:
        lines.append("**Result:** no correct answer within the turn budget.")
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
    "> **Fixture anchor set.** Human accuracy values are synthetic. Measured "
    "comparisons need matching questions and presentation conditions, with a "
    "documented source."
)


def _percent(value: float) -> str:
    return "%.0f%%" % (value * 100) if math.isfinite(value) else "undefined"


def _corr(x: float) -> str:
    return "undefined" if not math.isfinite(x) else "%.2f" % x


def render_anchor_md(result: AnchorResult) -> str:
    a = result.anchor
    lines: List[str] = ["# GlassBox: model and human accuracy", ""]
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
    lines.append("| item | human | model | gap (percentage points) |")
    lines.append("|---|---:|---:|---:|")
    for p in result.points:
        lines.append(
            "| %s/%s | %.0f%% | %.0f%% | %.0f |"
            % (
                p.scenario,
                p.question,
                p.human_accuracy * 100,
                p.model_accuracy * 100,
                p.abs_gap * 100,
            )
        )
    lines.append("")
    lines.append("**Largest accuracy gaps:**")
    lines.append("")
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
    lines: List[str] = ["# GlassBox: in-sample feature search", ""]
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
        "**Selected features:** `%s`"
        % (", ".join(sorted(ev.generator_features)) or "(none)")
    )
    lines.append("")
    lines.append("| interface | comprehension lift |")
    lines.append("|---|---:|")
    lines.append("| selected interface | %s |" % _pct(ev.generator_reward))
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
        state = "config valid" if not training["errors"] else "config invalid"
        lines.append(
            "**Training plan** `%s`: %s. Configuration check: %s. No trainer is included."
            % (training["id"], training["status"], state)
        )
        if training["errors"]:
            lines.append("")
        for error in training["errors"]:
            lines.append("- %s" % error)
        lines.append("")
    return "\n".join(lines)


def write_optimize(
    out_dir: str,
    ev: EvalResult,
    readers: Sequence[str],
    training: Optional[Dict[str, Any]] = None,
    *,
    protected_paths: Sequence[str] = (),
    selection: Optional[SelectionArtifact] = None,
) -> Dict[str, str]:
    names = ("optimize.md", "optimize.json")
    if selection is not None:
        names += ("selection.json",)
    destinations = _destinations(out_dir, names, protected_paths)
    report_path, json_path = destinations[:2]
    if selection is not None:
        write_selection(destinations[2], selection)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render_optimize_md(ev, readers, training))
        if selection is not None:
            fh.write("\nFrozen selection: [selection.json](selection.json). Evaluate it on separate scenarios.\n")
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
                "selection_sha256": selection.sha256 if selection is not None else None,
                "selection_file": "selection.json" if selection is not None else None,
            },
            fh,
        )
    paths = {"report": report_path, "optimize": json_path}
    if selection is not None:
        paths["selection"] = destinations[2]
    return paths


def render_evaluation_md(result: HeldOutResult) -> str:
    lines = [
        "# GlassBox: frozen feature evaluation", "",
        "Selection scenarios: " + ", ".join(result.selection_scenario_ids),
        "Evaluation scenarios: " + ", ".join(s.scenario_id for s in result.scenarios),
        "Readers: " + ", ".join(result.reader_names),
        "Selection SHA-256: `%s`" % result.selection_sha256,
        "Protocol: `%s`; baseline: `%s`; skip render: `%s`." % (
            result.evaluation_protocol, result.baseline_sampling, result.skip_render,
        ), "",
        "Features were frozen before evaluation. Scenario IDs and task content are disjoint from the recorded selection set.",
        "This checks recorded selection history; it cannot verify access outside this workflow.", "",
    ]
    if result.all_readers_simulated:
        lines.extend([SIMULATED_CAVEAT, ""])
    elif result.any_reader_simulated:
        lines.extend(["> **Mixed readers.** Some scores come from synthetic fixtures.", ""])
    lines.extend([
        "Selected features: " + (", ".join(sorted(result.selected_features)) or "(none)"),
        "Selection-set lift: %s (used for selection, not held-out)." % _pct(result.selection_reward), "",
        "| evaluation scenario | questions | selected accuracy | cards accuracy | baseline accuracy | selected lift | cards lift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for scenario in result.scenarios:
        lines.append("| %s | %d | %s | %s | %s | %s | %s |" % (
            scenario.scenario_id, scenario.question_count,
            _percent(scenario.selected_accuracy), _percent(scenario.cards_accuracy),
            _percent(scenario.baseline_accuracy), _pct(scenario.selected_lift),
            _pct(scenario.cards_lift),
        ))
    aggregate = result.aggregate
    lines.extend([
        "", "Mean held-out lift: selected %s; cards %s." % (
            _pct(aggregate.selected_lift), _pct(aggregate.cards_lift),
        ),
        "Scenarios receive equal weight. The baseline is shared by the fixed selected and cards views within each evaluation scenario.",
        "Scores describe these scenarios and readers; they do not establish population generalization.",
    ])
    return "\n".join(lines) + "\n"


def write_evaluation(
    out_dir: str,
    result: HeldOutResult,
    selection: SelectionArtifact,
    *,
    protected_paths: Sequence[str] = (),
) -> Dict[str, str]:
    selection = validate_selection(selection)
    if (
        result.selection_sha256 != selection.sha256
        or result.selected_features != selection.features
        or result.selection_scenario_ids != selection.scenario_ids
        or result.selection_reward != selection.selection_reward
    ):
        raise ValueError("evaluation result does not match the frozen selection")
    if selection.source_path is not None:
        protected_paths = tuple(protected_paths) + (selection.source_path,)
    report_path, json_path = _destinations(
        out_dir, ("evaluation.md", "evaluation.json"), protected_paths
    )
    payload = asdict(result)
    payload["selected_features"] = sorted(result.selected_features)
    payload["selection"] = selection_payload(selection)
    with open(json_path, "w", encoding="utf-8") as stream:
        _dump(payload, stream)
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write(render_evaluation_md(result))
    return {"report": report_path, "evaluation": json_path}


def _destinations(
    out_dir: str, names: Sequence[str], protected_paths: Sequence[str]
) -> List[str]:
    from .output_paths import validate_output_files

    paths = validate_output_files(out_dir, names, protected_paths=protected_paths)
    os.makedirs(paths[0].parent, exist_ok=True)
    return [str(path) for path in paths]

"""Offline and explicitly selected reader commands."""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Sequence

from . import report, reward as reward_mod, validate as validate_mod
from . import repair as repair_mod
from ._config import strings
from .anchor import load_anchor_set, run_anchor
from .output_paths import validate_output_files
from .readers import build_reader, build_readers
from .schema import Scenario, load_all_scenarios
from .selection import freeze_selection, load_selection, validate_evaluation
from .studies import StudyConfig, StudyError, StudyResult, load_study, run_study
from .training import (
    TrainingConfigError,
    load_training_config,
    validate_training_config,
)


def _csv(spec: str, name: str) -> List[str]:
    if not isinstance(spec, str):
        raise ValueError("%s must be comma-separated strings" % name)
    return strings(spec.split(","), name)


def _select_scenarios(spec: Optional[str]) -> List[Scenario]:
    all_scenarios = load_all_scenarios()
    if spec is None:
        return list(all_scenarios.values())
    chosen = []
    for sid in _csv(spec, "scenarios"):
        if sid not in all_scenarios:
            raise StudyError(
                "unknown scenario %r (have: %s)" % (sid, ", ".join(all_scenarios))
            )
        chosen.append(all_scenarios[sid])
    return chosen


def cmd_validate(args: argparse.Namespace) -> int:
    ok, lines = validate_mod.validate_all()
    print("\n".join(lines))
    print("VALIDATE OK" if ok else "VALIDATE FAILED")
    return 0 if ok else 1


def _print_summary(result: StudyResult) -> None:
    book, h1, cross, paths = result.book, result.h1, result.cross, result.paths
    print("readers: %s" % ", ".join(book.readers))
    if h1.any_reader_simulated:
        print("simulated reader answers are fixture outputs")
    print("judge: %s (%s)" % (h1.judge.name, h1.judge.score_label))
    if h1.judge.evidence:
        print(
            "judgments: %d valid (%d ties), %d invalid, %d abstained"
            % (h1.judge.valid_count, h1.judge.tie_count,
               h1.judge.invalid_count, h1.judge.abstained_count)
        )
    if h1.judge.simulated:
        print("judge scores are synthetic")
    if h1.divergence_found:
        ranking_status = "divergence found"
    elif any(math.isfinite(row.pref_score) for s in h1.scenarios for row in s.rows):
        ranking_status = "no reversal among scored interfaces"
    else:
        ranking_status = "judge scores unavailable"
    print("Rankings: %s (mean Spearman = %s)" % (
        ranking_status, _corr(h1.mean_spearman),
    ))
    for scenario in h1.scenarios:
        for reversal in scenario.reversals:
            print(
                "  %s: %s scores above %s on the judge and below it on question accuracy"
                % (scenario.scenario_id, reversal.preferred, reversal.understood)
            )
    print("resolved model families: %d" % cross.n_families)
    if cross.n_families >= 2:
        print(
            "a reversal in every resolved family: %s (agreement = %s)"
            % (
                cross.survives_across_families
                if cross.reversal_comparison_available else "unavailable",
                _corr(cross.cross_family_agreement),
            )
        )
    print("wrote %s and %s" % (paths["report"], paths["results"]))


def cmd_run(args: argparse.Namespace) -> int:
    config = StudyConfig(
        id="adhoc",
        description="CLI study",
        readers=_csv(args.readers, "readers"),
        scenarios=_csv(args.scenarios, "scenarios")
        if args.scenarios is not None
        else None,
        interfaces=_csv(args.interfaces, "interfaces")
        if args.interfaces is not None
        else None,
        replicates=args.replicates,
        judge=args.judge,
        skip_render=args.skip_render,
    )
    _print_summary(run_study(config, args.out))
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    config = load_study(args.config)
    print("study: %s" % config.id)
    _print_summary(run_study(config, args.out))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    training = None
    if args.training is not None:
        config = load_training_config(args.training)
        errors = validate_training_config(config)
        if errors:
            raise TrainingConfigError("; ".join(errors))
        training = {"id": config.id, "status": config.status, "errors": errors}
    protected = (
        (config._source_path,)
        if training is not None and config._source_path is not None
        else ()
    )
    destinations = validate_output_files(
        args.out, ("optimize.md", "optimize.json", "selection.json"),
        protected_paths=protected,
    )
    if destinations[2].exists():
        raise ValueError("selection.json already exists; use a new output directory")
    scenarios = _select_scenarios(args.scenarios)
    readers = build_readers(_csv(args.readers, "readers"))
    evaluation = reward_mod.evaluate_generator(
        scenarios, readers, skip_render=args.skip_render
    )
    selection = freeze_selection(evaluation, scenarios, readers, args.skip_render)
    paths = report.write_optimize(
        args.out,
        evaluation,
        [r.name for r in readers],
        training,
        protected_paths=protected,
        selection=selection,
    )
    print("Search and evaluation use the same questions; no model is trained.")
    if any(reader.simulated for reader in readers):
        print("simulated reader answers are fixture outputs")
    print(
        "selected features: %s"
        % (", ".join(sorted(evaluation.generator_features)) or "(none)")
    )
    print(
        "comprehension lift: selected %+.0f pts; cards %+.0f pts"
        % (evaluation.generator_reward * 100, evaluation.cards_reward * 100)
    )
    if training is not None:
        print("training configuration: %s (%s)" % (training["id"], training["status"]))
    print("wrote %s and %s" % (paths["report"], paths["selection"]))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    selection = load_selection(args.selection)
    protected = (selection.source_path,)
    validate_output_files(
        args.out, ("evaluation.md", "evaluation.json"), protected_paths=protected
    )
    scenarios = _select_scenarios(args.scenarios)
    validate_evaluation(selection, scenarios)
    specs = (
        _csv(args.readers, "readers")
        if args.readers is not None else selection.reader_names
    )
    readers = build_readers(specs)
    result = reward_mod.evaluate_selection(
        selection, scenarios, readers, skip_render=args.skip_render
    )
    paths = report.write_evaluation(
        args.out, result, selection, protected_paths=protected
    )
    print("frozen features evaluated on disjoint scenarios; no search was run")
    if result.any_reader_simulated:
        print("simulated reader answers are fixture outputs")
    print(
        "held-out lift: selected %+.0f pts; cards %+.0f pts"
        % (result.aggregate.selected_lift * 100, result.aggregate.cards_lift * 100)
    )
    print("wrote %s and %s" % (paths["report"], paths["evaluation"]))
    return 0


def cmd_anchor(args: argparse.Namespace) -> int:
    anchor = load_anchor_set(args.set)
    protected = (anchor._source_path,) if anchor._source_path is not None else ()
    validate_output_files(
        args.out, ("anchor.md", "anchor.json"), protected_paths=protected
    )
    readers = build_readers(_csv(args.readers, "readers"))
    result = run_anchor(
        anchor,
        readers,
        skip_render=args.skip_render,
        replicates=args.replicates,
        out_dir=args.out,
    )
    paths = report.write_anchor(args.out, result, protected_paths=protected)
    if anchor.is_fixture:
        print("synthetic human-accuracy fixture; no human study")
    print(
        "anchor %s: Spearman=%s Pearson=%s Kendall=%s (%d items)"
        % (
            anchor.id,
            _corr(result.spearman),
            _corr(result.pearson),
            _corr(result.kendall),
            result.n_items,
        )
    )
    print("wrote %s" % paths["report"])
    return 0


def _corr(value: float) -> str:
    return "unknown" if not math.isfinite(value) else "%.2f" % value


def cmd_repair(args: argparse.Namespace) -> int:
    scenarios = _select_scenarios(args.scenario)
    if len(scenarios) != 1:
        raise ValueError("repair requires one scenario")
    scenario = scenarios[0]
    if args.question is not None:
        question = next((q for q in scenario.questions if q.id == args.question), None)
        if question is None:
            raise ValueError(
                "unknown question %r in scenario %r" % (args.question, scenario.id)
            )
    else:
        question = scenario.questions[0]
    validate_output_files(args.out, ("transcript.md", "repair.json"))
    reader = build_reader(args.reader)
    result = repair_mod.repair(
        scenario, reader, question, skip_render=args.skip_render, out_dir=args.out
    )
    paths = report.write_repair(args.out, result, scenario, question)
    print(
        "correct after %d repair turn(s)" % result.turns_to_understanding
        if result.converged
        else "no correct answer within the turn budget"
    )
    print("wrote %s" % paths["transcript"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glassbox",
        allow_abbrev=False,
        description="Score questions about data interfaces.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="run a selected study", allow_abbrev=False)
    p_run.add_argument(
        "--readers", default="simulated", help="comma-separated reader specs"
    )
    p_run.add_argument("--scenarios", default=None, help="comma-separated scenario ids")
    p_run.add_argument(
        "--interfaces", default=None, help="comma-separated testable interfaces"
    )
    p_run.add_argument(
        "--judge", default="polish", help="polish, pairwise, or pairwise:<reader>"
    )
    p_run.add_argument("--replicates", type=int, default=1)
    p_run.add_argument("--skip-render", action="store_true")
    p_run.add_argument("--out", default="outputs", help="output directory")
    p_run.set_defaults(func=cmd_run)
    p_study = sub.add_parser("study", help="run a JSON study", allow_abbrev=False)
    p_study.add_argument(
        "--config", required=True, help="bundled name or explicit JSON path"
    )
    p_study.add_argument("--out", default="outputs")
    p_study.set_defaults(func=cmd_study)
    p_opt = sub.add_parser(
        "optimize", help="select features on scored questions", allow_abbrev=False
    )
    p_opt.add_argument("--readers", default="simulated")
    p_opt.add_argument("--scenarios", default=None)
    p_opt.add_argument(
        "--training", default=None, help="training configuration to validate"
    )
    p_opt.add_argument("--skip-render", action="store_true")
    p_opt.add_argument("--out", default="outputs")
    p_opt.set_defaults(func=cmd_optimize)
    p_eval = sub.add_parser(
        "evaluate", help="score frozen features on separate scenarios", allow_abbrev=False
    )
    p_eval.add_argument("--selection", required=True, help="frozen selection JSON path")
    p_eval.add_argument(
        "--scenarios", required=True, help="comma-separated evaluation scenario ids"
    )
    p_eval.add_argument(
        "--readers", default=None, help="defaults to the selection reader pool"
    )
    mode = p_eval.add_mutually_exclusive_group()
    mode.add_argument("--skip-render", dest="skip_render", action="store_true")
    mode.add_argument("--render", dest="skip_render", action="store_false")
    p_eval.set_defaults(func=cmd_evaluate, skip_render=None)
    p_eval.add_argument("--out", default="outputs/evaluation")
    p_anchor = sub.add_parser(
        "anchor", help="compare per-item accuracies", allow_abbrev=False
    )
    p_anchor.add_argument(
        "--set", required=True, help="bundled name or explicit JSON path"
    )
    p_anchor.add_argument("--readers", default="simulated")
    p_anchor.add_argument("--replicates", type=int, default=1)
    p_anchor.add_argument("--skip-render", action="store_true")
    p_anchor.add_argument("--out", default="outputs")
    p_anchor.set_defaults(func=cmd_anchor)
    p_val = sub.add_parser(
        "validate", help="recompute source answer keys", allow_abbrev=False
    )
    p_val.set_defaults(func=cmd_validate)
    p_rep = sub.add_parser(
        "repair", help="run one question's repair loop", allow_abbrev=False
    )
    p_rep.add_argument("--scenario", default="loans")
    p_rep.add_argument("--reader", default="simulated:literal")
    p_rep.add_argument("--question", default=None)
    p_rep.add_argument("--skip-render", action="store_true")
    p_rep.add_argument("--out", default="outputs")
    p_rep.set_defaults(func=cmd_repair)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print("configuration error: %s" % exc, file=sys.stderr)
        return 2
    except (RuntimeError, OSError) as exc:
        print("runtime error: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

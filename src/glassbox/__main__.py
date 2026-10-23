from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import core, evaluation


def _csv(spec: str, name: str) -> List[str]:
    if not isinstance(spec, str):
        raise ValueError("%s must be comma-separated strings" % name)
    return core.strings(spec.split(","), name)


def _select_scenarios(spec: Optional[str]) -> List[core.Scenario]:
    return evaluation.select_scenarios(None if spec is None else _csv(spec, "scenarios"))


def cmd_validate(args: argparse.Namespace) -> int:
    ok, lines = core.validate_all()
    print("\n".join(lines))
    print("VALIDATE OK" if ok else "VALIDATE FAILED")
    return 0 if ok else 1


def _print_summary(result: evaluation.StudyResult) -> None:
    book, h1, paths = result.book, result.h1, result.paths
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
    print("wrote %s and %s" % (paths["report"], paths["results"]))


def cmd_run(args: argparse.Namespace) -> int:
    config = evaluation.StudyConfig(
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
    _print_summary(evaluation.run_study(config, args.out))
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    config = evaluation.load_study(args.config)
    print("study: %s" % config.id)
    _print_summary(evaluation.run_study(config, args.out))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    destinations = core.validate_output_files(
        args.out, ("optimize.md", "optimize.yaml", "selection.yaml"),
    )
    if destinations[2].exists():
        raise ValueError("selection.yaml already exists; use a new output directory")
    scenarios = _select_scenarios(args.scenarios)
    readers = core.build_readers(_csv(args.readers, "readers"))
    search_result = evaluation.evaluate_generator(
        scenarios, readers, skip_render=args.skip_render
    )
    selection = evaluation.freeze_selection(search_result, scenarios, readers, args.skip_render)
    paths = evaluation.write_optimize(
        args.out,
        search_result,
        [r.name for r in readers],
        selection=selection,
    )
    print("Search and evaluation use the same questions; no model is trained.")
    if any(reader.simulated for reader in readers):
        print("simulated reader answers are fixture outputs")
    print(
        "selected features: %s"
        % (", ".join(sorted(search_result.generator_features)) or "(none)")
    )
    print(
        "comprehension lift: selected %+.0f pts; cards %+.0f pts"
        % (search_result.generator_reward * 100, search_result.cards_reward * 100)
    )
    print("wrote %s and %s" % (paths["report"], paths["selection"]))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    selection = evaluation.load_selection(args.selection)
    protected = (selection.source_path,)
    core.validate_output_files(
        args.out, ("evaluation.md", "evaluation.yaml"), protected_paths=protected
    )
    scenarios = _select_scenarios(args.scenarios)
    evaluation.validate_evaluation(selection, scenarios)
    specs = (
        _csv(args.readers, "readers")
        if args.readers is not None else selection.reader_names
    )
    readers = core.build_readers(specs)
    result = evaluation.evaluate_selection(
        selection, scenarios, readers, skip_render=args.skip_render
    )
    paths = evaluation.write_evaluation(
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
    anchor = evaluation.load_anchor_set(args.set)
    protected = (anchor._source_path,) if anchor._source_path is not None else ()
    core.validate_output_files(
        args.out, ("anchor.md", "anchor.yaml"), protected_paths=protected
    )
    readers = core.build_readers(_csv(args.readers, "readers"))
    result = evaluation.run_anchor(
        anchor,
        readers,
        skip_render=args.skip_render,
        replicates=args.replicates,
        out_dir=args.out,
    )
    paths = evaluation.write_anchor(args.out, result, protected_paths=protected)
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
    core.validate_output_files(args.out, ("transcript.md", "repair.yaml"))
    reader = core.build_reader(args.reader)
    result = evaluation.repair(
        scenario, reader, question, skip_render=args.skip_render, out_dir=args.out
    )
    paths = evaluation.write_repair(args.out, result, scenario, question)
    print(
        "correct after %d repair turn(s)" % result.turns_to_understanding
        if result.converged
        else "no correct answer within the turn budget"
    )
    print("wrote %s" % paths["transcript"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    parser = argparse.ArgumentParser(
        prog="glassbox",
        allow_abbrev=False,
        description="Score questions about data interfaces.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, description, handler):
        command_parser = sub.add_parser(name, help=description, allow_abbrev=False)
        command_parser.set_defaults(func=handler)
        return command_parser

    p_run = command("run", "run a selected study", cmd_run)
    p_run.add_argument(
        "--readers", default="simulated", help="simulated or comma-separated simulated personas"
    )
    p_run.add_argument("--scenarios", default=None, help="comma-separated scenario ids")
    p_run.add_argument(
        "--interfaces", default=None, help="comma-separated testable interfaces"
    )
    p_run.add_argument(
        "--judge", default="polish", choices=("polish", "pairwise"), help="fixed polish score or pairwise comparison"
    )
    p_run.add_argument("--replicates", type=int, default=1)
    p_run.add_argument("--skip-render", action="store_true")
    p_run.add_argument("--out", default=str(output_dir), help="output directory")
    p_study = command("study", "run a YAML study", cmd_study)
    p_study.add_argument(
        "--config", required=True, help="bundled name or explicit YAML path"
    )
    p_study.add_argument("--out", default=str(output_dir))
    p_opt = command("optimize", "select features on scored questions", cmd_optimize)
    p_opt.add_argument("--readers", default="simulated")
    p_opt.add_argument("--scenarios", default=None)
    p_opt.add_argument("--skip-render", action="store_true")
    p_opt.add_argument("--out", default=str(output_dir))
    p_eval = command("evaluate", "score frozen features on separate scenarios", cmd_evaluate)
    p_eval.add_argument("--selection", required=True, help="frozen selection YAML path")
    p_eval.add_argument(
        "--scenarios", required=True, help="comma-separated evaluation scenario ids"
    )
    p_eval.add_argument(
        "--readers", default=None, help="defaults to the selection reader pool"
    )
    mode = p_eval.add_mutually_exclusive_group()
    mode.add_argument("--skip-render", dest="skip_render", action="store_true")
    mode.add_argument("--render", dest="skip_render", action="store_false")
    p_eval.set_defaults(skip_render=None)
    p_eval.add_argument("--out", default=str(output_dir / "evaluation"))
    p_anchor = command("anchor", "compare per-item accuracies", cmd_anchor)
    p_anchor.add_argument(
        "--set", required=True, help="bundled name or explicit YAML path"
    )
    p_anchor.add_argument("--readers", default="simulated")
    p_anchor.add_argument("--replicates", type=int, default=1)
    p_anchor.add_argument("--skip-render", action="store_true")
    p_anchor.add_argument("--out", default=str(output_dir))
    command("validate", "recompute source answer keys", cmd_validate)
    p_rep = command("repair", "run one question's repair loop", cmd_repair)
    p_rep.add_argument("--scenario", default="loans")
    p_rep.add_argument("--reader", default="simulated:literal")
    p_rep.add_argument("--question", default=None)
    p_rep.add_argument("--skip-render", action="store_true")
    p_rep.add_argument("--out", default=str(output_dir))
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

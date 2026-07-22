"""Command-line interface: ``glassbox run | study | validate | repair``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from . import report, reward as reward_mod, validate as validate_mod
from . import repair as repair_mod
from .anchor import AnchorError, load_anchor_set, run_anchor
from .readers import build_reader, build_readers
from .readers._http import MissingKeyError
from .render import RenderError
from .schema import Scenario, load_all_scenarios
from .studies import StudyConfig, StudyError, StudyResult, load_study, run_study
from .training import TrainingConfigError, load_training_config, validate_training_config


def _select_scenarios(spec: Optional[str]) -> List[Scenario]:
    all_scenarios = load_all_scenarios()
    if not spec:
        return list(all_scenarios.values())
    chosen = []
    for sid in spec.split(","):
        sid = sid.strip()
        if sid not in all_scenarios:
            raise SystemExit("unknown scenario %r (have: %s)"
                             % (sid, ", ".join(all_scenarios)))
        chosen.append(all_scenarios[sid])
    return chosen


def cmd_validate(args: argparse.Namespace) -> int:
    ok, lines = validate_mod.validate_all()
    print("\n".join(lines))
    print("VALIDATE OK" if ok else "VALIDATE FAILED")
    return 0 if ok else 1


def _run_config(config: StudyConfig, out_dir: str) -> StudyResult:
    try:
        return run_study(config, out_dir)
    except MissingKeyError as exc:
        raise SystemExit("cannot run a real reader: %s" % exc)
    except RenderError as exc:
        raise SystemExit("rendering failed: %s\n(try --skip-render to run without a browser)" % exc)
    except StudyError as exc:
        raise SystemExit("study error: %s" % exc)


def _print_summary(result: StudyResult) -> None:
    book, h1, cross, paths = result.book, result.h1, result.cross, result.paths
    print("readers: %s" % ", ".join(book.readers))
    if h1.simulated:
        print("(simulated readers — a demonstration, not evidence about real readers)")
    print("H1: %s (mean Spearman = %.2f)"
          % ("divergence found" if h1.divergence_found else "no divergence",
             h1.mean_spearman))
    for sh in h1.scenarios:
        for rv in sh.reversals:
            print("  %s: '%s' preferred over '%s' but understood less"
                  % (sh.scenario_id, rv.preferred, rv.understood))
    if cross.n_families >= 2:
        print("cross-family: divergence %s (agreement = %.2f across %d families)"
              % ("survives" if cross.survives_across_families else "does not survive",
                 cross.cross_family_agreement, cross.n_families))
    else:
        print("cross-family: 1 family only — needs >=2 for a cross-family result")
    print("wrote %s and %s" % (paths["report"], paths["results"]))


def cmd_run(args: argparse.Namespace) -> int:
    config = StudyConfig(
        id="adhoc", description="ad-hoc CLI run",
        readers=args.readers.split(","),
        scenarios=args.scenarios.split(",") if args.scenarios else None,
        replicates=args.replicates, skip_render=args.skip_render)
    _print_summary(_run_config(config, args.out))
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    try:
        config = load_study(args.config)
    except (StudyError, OSError) as exc:
        raise SystemExit("cannot load study %r: %s" % (args.config, exc))
    print("study: %s — %s" % (config.id, config.description))
    _print_summary(_run_config(config, args.out))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    scenarios = _select_scenarios(args.scenarios)
    readers = build_readers(args.readers.split(","))
    try:
        ev = reward_mod.evaluate_generator(scenarios, readers, skip_render=args.skip_render)
    except MissingKeyError as exc:
        raise SystemExit("cannot run a real reader: %s" % exc)
    except RenderError as exc:
        raise SystemExit("rendering failed: %s\n(try --skip-render to run without a browser)" % exc)

    training = None
    if args.training:
        try:
            cfg = load_training_config(args.training)
        except (TrainingConfigError, OSError) as exc:
            raise SystemExit("cannot load training config %r: %s" % (args.training, exc))
        training = {"id": cfg.id, "status": cfg.status,
                    "errors": validate_training_config(cfg)}

    paths = report.write_optimize(args.out, ev, [r.name for r in readers], training)
    print("(offline search generator — proves the reward is optimizable, not a trained model)")
    if all(r.name.startswith("simulated:") for r in readers):
        print("(simulated readers — the comprehension numbers come from designed fixtures, not evidence)")
    print("generator selected: %s" % (", ".join(sorted(ev.generator_features)) or "(none)"))
    print("comprehension lift: generator %+.0f pts vs polished cards %+.0f pts"
          % (ev.generator_reward * 100, ev.cards_reward * 100))
    print("beats preference-tuned: %s · beats plain-text: %s"
          % (ev.beats_preference_tuned, ev.beats_plaintext))
    if training is not None:
        print("training plan %s: %s"
              % (training["id"], "ready" if not training["errors"] else "BLOCKED"))
    print("wrote %s" % paths["report"])
    return 0


def cmd_anchor(args: argparse.Namespace) -> int:
    try:
        anchor = load_anchor_set(args.set)
    except (AnchorError, OSError) as exc:
        raise SystemExit("cannot load anchor set %r: %s" % (args.set, exc))
    readers = build_readers(args.readers.split(","))
    try:
        result = run_anchor(anchor, readers, skip_render=args.skip_render,
                            replicates=args.replicates, out_dir=args.out)
    except MissingKeyError as exc:
        raise SystemExit("cannot run a real reader: %s" % exc)
    except RenderError as exc:
        raise SystemExit("rendering failed: %s\n(try --skip-render to run without a browser)" % exc)

    paths = report.write_anchor(args.out, result)
    if anchor.is_fixture:
        print("(fixture anchor set — synthetic human numbers, not an H3 result)")
    print("anchor %s: Spearman=%s Pearson=%s Kendall=%s (%d items)"
          % (anchor.id, _corr(result.spearman), _corr(result.pearson),
             _corr(result.kendall), result.n_items))
    print("wrote %s" % paths["report"])
    return 0


def _corr(x: float) -> str:
    return "nan" if x != x else "%.2f" % x


def cmd_repair(args: argparse.Namespace) -> int:
    scenarios = _select_scenarios(args.scenario)
    scenario = scenarios[0]
    reader = build_reader(args.reader)
    if args.question:
        question = next((q for q in scenario.questions if q.id == args.question), None)
        if question is None:
            raise SystemExit("unknown question %r in scenario %r"
                             % (args.question, scenario.id))
    else:
        question = scenario.questions[0]

    try:
        result = repair_mod.repair(scenario, reader, question,
                                   skip_render=args.skip_render, out_dir=args.out)
    except MissingKeyError as exc:
        raise SystemExit("cannot run a real reader: %s" % exc)
    except RenderError as exc:
        raise SystemExit("rendering failed: %s\n(try --skip-render to run without a browser)" % exc)

    paths = report.write_repair(args.out, result, scenario, question)
    if result.converged:
        print("understood after %d repair turn(s)" % result.turns_to_understanding)
    else:
        print("did not converge within the turn budget")
    print("wrote %s" % paths["transcript"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glassbox",
        description="Measure whether a generated interface conveys understanding.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="render, read, score, and report H1 divergence")
    p_run.add_argument("--readers", default="simulated",
                       help="comma-separated reader specs (default: simulated)")
    p_run.add_argument("--scenarios", default=None,
                       help="comma-separated scenario ids (default: all)")
    p_run.add_argument("--replicates", type=int, default=1,
                       help="answers per question per reader (default: 1)")
    p_run.add_argument("--skip-render", action="store_true",
                       help="skip PNG rendering; readers use the spec/text view")
    p_run.add_argument("--out", default="runs/run", help="output directory")
    p_run.set_defaults(func=cmd_run)

    p_study = sub.add_parser("study", help="run a study defined by a JSON config file")
    p_study.add_argument("--config", required=True, help="path to a study config JSON")
    p_study.add_argument("--out", default="runs/study", help="output directory")
    p_study.set_defaults(func=cmd_study)

    p_opt = sub.add_parser("optimize", help="M4: search the interface lattice for the highest comprehension reward")
    p_opt.add_argument("--readers", default="simulated", help="comma-separated reader specs")
    p_opt.add_argument("--scenarios", default=None, help="comma-separated scenario ids (default: all)")
    p_opt.add_argument("--training", default=None, help="optional training config JSON to validate")
    p_opt.add_argument("--skip-render", action="store_true", help="skip PNG rendering")
    p_opt.add_argument("--out", default="runs/optimize", help="output directory")
    p_opt.set_defaults(func=cmd_optimize)

    p_anchor = sub.add_parser("anchor", help="H3: correlate model vs human accuracy on an anchor set")
    p_anchor.add_argument("--set", required=True, help="path to an anchor set JSON")
    p_anchor.add_argument("--readers", default="simulated", help="comma-separated reader specs")
    p_anchor.add_argument("--replicates", type=int, default=1, help="answers per question per reader")
    p_anchor.add_argument("--skip-render", action="store_true", help="skip PNG rendering")
    p_anchor.add_argument("--out", default="runs/anchor", help="output directory")
    p_anchor.set_defaults(func=cmd_anchor)

    p_val = sub.add_parser("validate", help="recompute every answer key from source")
    p_val.set_defaults(func=cmd_validate)

    p_rep = sub.add_parser("repair", help="run the H4 repair loop on one question")
    p_rep.add_argument("--scenario", default="loans", help="scenario id (default: loans)")
    p_rep.add_argument("--reader", default="simulated:literal",
                       help="a single reader spec (default: simulated:literal)")
    p_rep.add_argument("--question", default=None,
                       help="question id (default: the scenario's first question)")
    p_rep.add_argument("--skip-render", action="store_true", help="skip PNG rendering")
    p_rep.add_argument("--out", default="runs/repair", help="output directory")
    p_rep.set_defaults(func=cmd_repair)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Command-line interface: ``glassbox run | validate | repair``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence

from . import analysis, report, scoring, validate as validate_mod
from . import repair as repair_mod
from .judge import SimulatedJudge
from .readers import build_reader, build_readers
from .readers._http import MissingKeyError
from .render import RenderError
from .schema import Scenario, load_all_scenarios


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


def cmd_run(args: argparse.Namespace) -> int:
    scenarios = _select_scenarios(args.scenarios)
    readers = build_readers(args.readers.split(","))
    try:
        results = scoring.read_all(scenarios, readers, skip_render=args.skip_render,
                                   out_dir=args.out)
    except MissingKeyError as exc:
        raise SystemExit("cannot run a real reader: %s" % exc)
    except RenderError as exc:
        raise SystemExit("rendering failed: %s\n(try --skip-render to run without a browser)" % exc)

    book = scoring.ScoreBook(results)
    h1 = analysis.analyze_h1(book, scenarios, SimulatedJudge())
    paths = report.write_run(args.out, scenarios, results, book, h1, args.skip_render)

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
    print("wrote %s and %s" % (paths["report"], paths["results"]))
    return 0


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
    p_run.add_argument("--skip-render", action="store_true",
                       help="skip PNG rendering; readers use the spec/text view")
    p_run.add_argument("--out", default="runs/run", help="output directory")
    p_run.set_defaults(func=cmd_run)

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

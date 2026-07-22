# Glass Box

**A benchmark that measures whether a generated interface conveys *understanding* — not whether it is merely preferred — and an open harness to run it.**

When an AI explains something today — a lab result, a set of loan offers, its own reasoning — it increasingly draws you an interface instead of writing a paragraph. The field judges those interfaces by asking which one people *prefer*, and people pick the polished one most of the time even when the content is identical. But liking an explanation and understanding it are not the same thing.

Glass Box scores an interface by the **comprehension it transfers**: a vision-language model reads *only the rendered interface* and answers objective, single-answer questions about the underlying data. Each interface is scored against that reader's own **ceiling** (how well it answers the same questions from the raw data) and reported as **comprehension lift** over a plain-text baseline. A preference judge runs alongside — never as the score — so the harness can show where *preferred* and *understood* come apart.

This repository is the measurement harness (Phase 1). Later phases train a generator on the comprehension signal and add a live repair loop.

## The pipeline

```
  data/scenarios/*.json          interfaces.py            render.py           readers/            scoring.py
 ┌─────────────────────┐      ┌──────────────────┐    ┌────────────┐    ┌───────────────┐    ┌──────────────┐
 │  source data +      │      │  interface       │    │  stimulus  │    │  reader        │    │ comprehension│
 │  MCQ questions +    │─────▶│  cards · table · │───▶│  PNG  or   │───▶│  VLM  or       │──▶ │ lift vs      │
 │  recomputable answers│     │  annotated       │    │  text spec │    │  simulated     │    │ baseline,    │
 │                     │      │  (+ raw/baseline │    │            │    │  → MCQ answer  │    │ ceiling-     │
 └─────────────────────┘      │   references)    │    └────────────┘    └───────────────┘    │ controlled   │
                              └──────────────────┘                                            └──────┬───────┘
                                       │                                                             │
                                 judge.py (preference, alongside — never the score) ─────────────────┤
                                                                                                     ▼
                                                                    analysis.py: H1 divergence report
                                                                    repair.py:   H4 turns to understanding
```

**Render → read → score.** An interface is rendered to what a reader actually sees; the reader answers objective questions; the score is how much the interface helped, over a plain-text baseline and against the reader's own ceiling. The preference judge runs in parallel so the harness can report where *preferred* and *understood* disagree (H1), and the repair loop regenerates an interface until a tripped reader understands (H4).

## Why it's built this way

- **Objective questions, one correct answer.** Every question is multiple-choice with exactly one right choice, and that choice is **recomputed from the source data** by a validator — there are no unverifiable answer keys.
- **Aggregate & relational, not lookups.** Questions ask "which option costs least over its term," not "what is in cell B2" — a generator can't win by printing the answer.
- **Ceiling-controlled.** The score reflects what the *interface* added, not how good the reader is at arithmetic.
- **Cross-family readers.** Real adapters exist for independent API families (Anthropic; OpenAI-compatible, including OpenRouter → Qwen3-VL), selectable per run. No keys are stored.
- **Verifiable offline.** Deterministic *simulated* readers exercise the whole pipeline with no network and no keys, so `verify.sh` is fully reproducible. Simulated readers are pipeline fixtures — every report they appear in is labeled `simulated` and they never stand in for a scientific result.

## Requirements

- **Python 3.9+** — the core uses only the standard library. No `pip install` step.
- **(Optional) Google Chrome** — for rendering interfaces to PNG. Without it, the harness runs in `--skip-render` mode and readers consume a text/structured spec of each interface. Override the binary with `GLASSBOX_CHROME`.

## Quickstart

```sh
# Verify everything: unit tests + data-integrity validation + an offline end-to-end demo
bash scripts/verify.sh

# Score interfaces end to end with the deterministic offline readers (writes an H1 divergence report)
PYTHONPATH=src python3 -m glassbox run --readers simulated --out runs/demo

# Recompute every answer key from source data (fails loudly on any mismatch)
PYTHONPATH=src python3 -m glassbox validate

# Run the H4 repair loop on one scenario (writes a repair transcript)
PYTHONPATH=src python3 -m glassbox repair --scenario loans --reader simulated:literal --out runs/repair
```

To use real readers, export the relevant key and name the reader on the command line — for example `--readers anthropic:claude-sonnet-5,openrouter:qwen/qwen3-vl-8b-instruct`. Keys are read from the environment at call time and never written to disk.

## Layout

```
data/scenarios/*.json   source data + questions + recomputable answers
data/prompts/*.txt       the reader's MCQ prompt template
src/glassbox/            the package: schema · stimuli · interfaces · render · readers · judge · scoring · analysis · repair · report · cli
tests/                   stdlib unittest suite, incl. answer-key recomputation
scripts/verify.sh        the one verify command
runs/                    run outputs (gitignored)
```

## Status

Phase 1 (measurement). See `CHANGELOG.md` for what has landed.

## License

MIT — see [LICENSE](LICENSE).

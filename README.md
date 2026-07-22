# Glass Box

**Measure whether a generated interface conveys understanding — not just whether it's liked.**

A reader (a vision-language model) sees only a rendered interface and answers objective questions about the data behind it. The interface's score is **comprehension lift**: how much better the reader does with it than with a plain-text baseline of the same data, measured against the reader's own ceiling on the raw numbers. A preference judge runs alongside — never as the score — so the harness can show where *preferred* and *understood* pull apart.

This is Phase 1, the measurement harness. Later phases train a generator on the signal and add a live repair loop.

## Pipeline

```
scenario ──▶ interface ──▶ stimulus ──▶ reader ──▶ answer ──▶ comprehension lift
 data      cards/table/    PNG or text   VLM or     MCQ        vs baseline,
           annotated       (skip-render) simulated  choice     ceiling-controlled
                                                                      │
        preference judge (alongside, never the score) ───────────────┤
                                                                      ▼
                                            H1 divergence report · H4 repair loop
```

**Worked example.** Three loan offers. The polished card advertises the lowest headline total but hides the term and fees; the plain table shows the real cost. A reader gets it right from the table and wrong from the card — the card is *preferred* yet transfers *less* understanding. That reversal, with no recruiting, is H1.

## Why it holds up

- **Recomputable answers.** Every answer is re-derived from source by `glassbox validate` — no unverifiable keys.
- **Aggregate, not lookups.** "Which costs least over its term," not "what's in cell B2" — a generator can't win by printing the answer.
- **Ceiling-controlled.** The score is what the interface *added*, not the reader's arithmetic.
- **Cross-family, key-safe.** Adapters for Anthropic and OpenAI-compatible APIs (incl. OpenRouter → Qwen3-VL); keys are read from the environment, never stored.
- **Offline-verifiable.** Deterministic simulated readers run the whole pipeline with no network. They are fixtures — always labeled `simulated`, never evidence.

## Use

Python 3.9+, standard library only (no install step). Chrome is optional for PNG rendering; without it, `--skip-render` feeds readers a text view of each interface.

```sh
bash scripts/verify.sh                                   # tests + validation + offline demo
PYTHONPATH=src python3 -m glassbox validate              # recompute every answer key
PYTHONPATH=src python3 -m glassbox run --readers simulated --out runs/demo          # H1 report
PYTHONPATH=src python3 -m glassbox repair --scenario loans --reader simulated:literal --out runs/repair   # H4 transcript
```

Real readers, per run: `--readers anthropic:claude-sonnet-5,openrouter:qwen/qwen3-vl-8b-instruct` (needs the matching key in your environment).

More commands: `glassbox study --config data/studies/offline_demo.json` (a reproducible, pre-registerable run across reader families) · `glassbox anchor --set data/anchors/fixture.json` (H3: correlate model vs human accuracy) · `glassbox optimize` (M4: search the interface space for the highest comprehension reward).

## Layout

```
data/scenarios/*.json   source data + questions + recomputable answers
data/prompts/*.txt      the reader's MCQ prompt template
src/glassbox/           schema · compute · interfaces · render · readers · judge · scoring · analysis · repair · report · cli
tests/                  stdlib unittest suite (incl. answer-key recomputation)
scripts/verify.sh       the one verify command
```

## License

MIT — see [LICENSE](LICENSE).

# Changelog

All notable changes to Glass Box are documented here. This project follows
[Semantic Versioning](https://semver.org).

## [0.1.0] — 2026-07-23

First release: the measurement harness (render → read → score).

### Added
- End-to-end pipeline: render an interface → a reader answers multiple-choice
  questions from it alone → score comprehension lift over a plain-text baseline,
  under ceiling control.
- Deterministic simulated readers (`literal`, `diligent`, `careless`) that make the
  whole pipeline verifiable offline with no API keys.
- Real reader adapters for two independent API families — Anthropic Messages and an
  OpenAI-compatible path (OpenAI + OpenRouter → Qwen3-VL) — selectable per run, no
  keys stored.
- A preference judge that runs alongside the readers and never enters the score.
- H1 analysis: rank interfaces by comprehension and by preference, report the
  Spearman correlation and every reversal (preferred more, understood less).
- H4 repair loop: regenerate an interface to fix a reader's specific mistake and
  count turns to understanding.
- Three scenarios (loan offers, phone plans, regional revenue growth), each with a
  headline that can mislead; every answer key is recomputed from source by
  `glassbox validate`.
- Headless-Chrome rendering (HTML → PNG) with graceful `--skip-render` degradation.
- `scripts/verify.sh`: unit tests + data-integrity validation + an offline
  end-to-end demo. Stdlib-only core, Python 3.9+.

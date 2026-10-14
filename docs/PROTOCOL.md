# Measurement protocol

## Reference tasks

| Source | Questions |
| :--- | :--- |
| [Loans](../src/glassbox/_data/scenarios/loans.json) | Total cost · threshold count · rank |
| [Plans](../src/glassbox/_data/scenarios/plans.json) | 2-year cost · threshold count · maximum |
| [Growth](../src/glassbox/_data/scenarios/growth.json) | Dollar gain · percentage gain · threshold count |

`glassbox validate` recomputes all 9 source answers. Packaged resources include questions,
choices, and prompt templates; external scenarios use the same schema. The loan fixture
contrasts advertised and computed totals; scoring uses computed cost.

## Measures

| Quantity | Computation |
| :--- | :--- |
| Accuracy | Correct answers / attempts, with replicate spread |
| Comprehension lift | Interface accuracy minus plain-text baseline accuracy |
| Ceiling gap | Interface accuracy minus raw-data accuracy |
| Reversal | Higher judge score with lower answer accuracy |

Raw-data accuracy is a separate diagnostic, not a normalization or control for lift.
Missing cells and undefined correlations remain undefined.

Judge metadata is separate from reader metadata. The default uses fixed polish weights;
the [pairwise prompt](../src/glassbox/_data/prompts/judge_pairwise.txt) asks for perceived
comprehension from text, not screenshots or aesthetic preference. Tournaments compare
selected interfaces in both orders: six judgments for three views. Full task content,
selected variants, and protocol identify the cache. Invalid replies and abstentions get
no credit and are excluded from denominators. Genuine custom-judge ties receive half
credit. Rates are conditional on accepted judgments; coverage can differ by view.

`valid_count` includes ties; `tie_count` is a subset. Invalid and abstained counts are
separate. Pairwise scores and ranks are undefined without accepted judgments. Raw replies,
A/B order, outcomes, and prompt hashes are retained in `results.json` at
`meta.judge.evidence`.

Cross-family reports group known model families across providers using name heuristics,
not availability or training-data checks. Unresolved identities do not establish independent
families. “Any reversal per family” may involve different pairs; common pairs are reported.
Ranks omit nonfinite scores; correlations use finite paired observations. Cross-family preference means
require finite judge scores for every selected scenario. Cross-family reversal status needs at
least two resolved families and two finite judge/lift pairs in each family; otherwise it is unavailable.

## Commands

```sh
glassbox study --config offline_demo --out outputs/study
glassbox run --readers simulated --skip-render --judge pairwise --out outputs/pairwise
glassbox repair --scenario growth --question biggest_percent_gain --reader simulated:literal --skip-render --out outputs/repair
glassbox anchor --set fixture --skip-render --out outputs/anchor
glassbox optimize --readers simulated --scenarios loans,plans --skip-render --out outputs/selection
glassbox evaluate --selection outputs/selection/selection.json --scenarios growth --out outputs/evaluation
```

`optimize` writes in-sample `optimize.md`/`optimize.json` plus a frozen `selection.json`.
That file is created exclusively; repeat searches need a new output directory. `evaluate`
requires explicit `--scenarios`, uses frozen features without search, and inherits reader
and rendering settings unless overridden. Overlap checks reject shared scenario IDs,
task fingerprints, or source-data fingerprints. `evaluation.md` and `evaluation.json`
report a shared baseline for selected/cards views within each scenario; scenarios receive
equal weight.

“Disjoint” refers to the recorded selection set. Exact fingerprints cannot certify no
prior access elsewhere or detect every semantic copy. The shipped synthetic evaluation
is a pilot, not proof of generalization.

Installed wheels support built-in names and explicit external paths; missing paths fail.
Live readers need a provider/model specification and environment credential. Verification
sends no provider requests.

`--skip-render` gives simulated readers structured values and live readers text. PNGs use
the [Chrome screenshot command](https://developer.chrome.com/docs/automation-and-testing/headless-cli)
via `GLASSBOX_CHROME`, with an isolated profile. Failed or invalid captures preserve existing
files. Completion records natural exit or confirmed capture with graceful shutdown.
Fixed dimensions do not imply identical pixels across platforms.

## Interpretation

- Human anchor values are synthetic; correlations establish no human study result.
- Optimization searches 32 feature subsets on the reported questions, with a fresh
  plain-text baseline per candidate. Selection scores remain in sample; use the frozen
  artifact for disjoint evaluation. Training declarations are checked; no trainer is included.
- Repair follows the selected question's field and operation. Derived values and highlights
  can expose answers; feature restrictions and family exclusions do not prove resistance
  to answer leakage.

Metadata records task, prompt, and stimulus hashes, reader/provider identity, judge criterion,
and simulation status. Source and generated-image hashes describe different artifacts.

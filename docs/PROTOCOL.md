# Measurement protocol

## Reference tasks

| Source | Questions |
| :--- | :--- |
| [Loans](../src/glassbox/_data/scenarios/loans.json) | Total cost · threshold count · rank |
| [Plans](../src/glassbox/_data/scenarios/plans.json) | 2-year cost · threshold count · maximum |
| [Growth](../src/glassbox/_data/scenarios/growth.json) | Dollar gain · percentage gain · threshold count |

`glassbox validate` recomputes all 9 answers from source values. Packaged resources include authored questions, choices and prompt templates. External scenarios use the same schema checks.

The loan fixture intentionally contrasts advertised and computed totals. Scoring uses computed cost.

## Measures

| Quantity | Computation |
| :--- | :--- |
| Accuracy | Correct answers / attempts, with replicate spread |
| Comprehension lift | Interface accuracy minus plain-text baseline accuracy |
| Ceiling gap | Interface accuracy minus raw-data accuracy |
| Reversal | Higher judge score with lower answer accuracy |

The raw-data ceiling is a separate diagnostic. It does not normalize or control lift. Missing cells and undefined correlations remain undefined in reports.

Judge metadata is independent of reader metadata. The default judge uses designed polish weights. The [pairwise prompt](../src/glassbox/_data/prompts/judge_pairwise.txt) asks for perceived comprehension from text; it does not judge screenshots or aesthetic preference. A tournament compares only the selected interfaces. Full task content, selected variants and protocol identify its cache.

Cross-family reporting groups known underlying model families across providers. Identity resolution is a heuristic, not an availability or training-data check. Unresolved identities do not establish independent families. “Any reversal per family” may involve different interface pairs; common pairs are reported separately.

## Commands

```sh
glassbox study --config offline_demo --out outputs/study
glassbox run --readers simulated --skip-render --judge pairwise --out outputs/pairwise
glassbox repair --scenario growth --question biggest_percent_gain --reader simulated:literal --skip-render --out outputs/repair
glassbox anchor --set fixture --skip-render --out outputs/anchor
glassbox optimize --readers simulated --skip-render --out outputs/search
```

Built-in names work from an installed wheel. Explicit external paths remain supported; missing paths fail. Live readers require an explicit provider/model specification and its environment credential. Verification sends no provider requests.

`--skip-render` uses structured values for simulated readers and text for live readers. PNG rendering uses the [Chrome screenshot command](https://developer.chrome.com/docs/automation-and-testing/headless-cli), selected by `GLASSBOX_CHROME`. Its profile is isolated; failed or invalid screenshots cannot replace existing files. Completion records natural exit or confirmed capture followed by graceful browser shutdown. Fixed dimensions do not imply identical pixels across platforms.

## Interpretation

- The human-accuracy anchor is synthetic. Its correlations establish no human study result.
- Optimization searches 32 feature subsets on the reported questions, with a new plain-text baseline per candidate. This is in-sample selection; training is declared and gated, not executed.
- Repair adds features for the selected question's field and operation. Derived values and highlights can expose answers; feature restrictions and family exclusions do not prove resistance to answer leakage.

Result metadata records task, prompt and stimulus hashes, reader/provider identity, judge criterion and simulation status. Source hashes and generated-image hashes describe different artifacts.

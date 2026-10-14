# Changes

## Unreleased

- Preserve numeric display precision and use field names when display labels are omitted.
- Copy presentations before rendering and when constructing stimuli.
- Respect provider cooldowns and add backoff within the HTTP retry budget.
- Compare pairwise views in both orders and retain invalid, abstained, and genuine tie outcomes.
- Report accepted-judgment coverage; leave missing judge scores and ranks undefined.
- Freeze selected features and evaluate them on scenarios disjoint by ID, task content, and source data.
- Reject null primary metrics and retry truncated responses or malformed HTTP status lines.
- Simplify reader selection, choice parsing, arithmetic, ranking, and scoring loops.
- Fix iterator input handling, numeric overflow, HTTP response errors, and anchor subsets.
- Validate stimulus modes and direct-call inputs before evaluation.
- Apply display labels and units consistently in HTML and text.
- Organize documentation, charts, and screenshots under `docs/`.

## 0.1.1

- Package source fixtures; validate installed commands.
- Separate judge and reader metadata; record complete task hashes.
- Correct selected-question repair, answer parsing and output handling.

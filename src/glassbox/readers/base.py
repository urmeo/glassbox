"""Reader base types and the shared logic for turning values into a choice."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from ..prompts import load_prompt
from ..schema import Question, Scenario
from ..stimuli import Stimulus


@dataclass(frozen=True)
class Answer:
    """A reader's response to one question."""
    choice_id: Optional[str]   # the chosen choice id, or None if unparseable
    method: str = ""           # how it was derived ("read"/"computed"/"headline"/"guess"/"api")
    raw: str = ""              # raw model output, for real readers


class Reader:
    """Interface every reader implements."""

    name: str = "reader"
    family: str = ""
    simulated: bool = False

    def answer(self, scenario: Scenario, question: Question, stimulus: Stimulus) -> Answer:
        raise NotImplementedError


# --- shared: apply a question's op to a set of per-item values ---------------

def apply_op(compute_spec: Mapping[str, Any], values: Mapping[str, Any],
             order: List[str]) -> Any:
    """Apply argmin/argmax/rank/count_* to ``values`` (item id -> value)."""
    op = compute_spec["op"]
    if op in ("argmin", "argmax"):
        chooser = min if op == "argmin" else max
        return chooser(order, key=lambda i: values[i])
    if op == "rank":
        k = int(compute_spec["k"])
        reverse = compute_spec.get("order", "asc") == "desc"
        ordered = sorted(order, key=lambda i: values[i], reverse=reverse)
        k = max(1, min(k, len(ordered)))
        return ordered[k - 1]
    if op.startswith("count_"):
        threshold = compute_spec["threshold"]
        cmp = {
            "count_ge": lambda v: v >= threshold,
            "count_le": lambda v: v <= threshold,
            "count_gt": lambda v: v > threshold,
            "count_lt": lambda v: v < threshold,
        }[op]
        return sum(1 for i in order if cmp(values[i]))
    raise ValueError("unknown op %r" % op)


def value_to_choice(question: Question, value: Any) -> Optional[str]:
    """Map a computed answer value to the choice id whose ``value`` equals it."""
    for c in question.choices:
        if c.value == value:
            return c.id
    return None


def guess_choice(question: Question) -> str:
    """The deterministic fallback when a reader cannot answer: the first choice."""
    return question.choices[0].id


# --- shared: MCQ prompt building and answer parsing (real readers) -----------

def _letter(index: int) -> str:
    return chr(ord("A") + index)


def build_mcq_prompt(question: Question, stimulus: Stimulus, image: bool) -> str:
    """Render the MCQ prompt. Choices are lettered A, B, C … for a terse reply.

    In image mode the interface is the picture; otherwise its text is inlined so a
    real reader can still answer with ``--skip-render`` and no browser.
    """
    template = load_prompt("reader_mcq")
    options = "\n".join("%s. %s" % (_letter(i), c.text) for i, c in enumerate(question.choices))
    context_block = "" if image else ("The interface is shown below as text:\n\n"
                                       + stimulus.text + "\n\n")
    # Single pass so a literal "{options}" in a stem or interface text is not re-substituted.
    fields = {"context_block": context_block, "stem": question.stem, "options": options}
    return re.sub(r"\{(context_block|stem|options)\}", lambda m: fields[m.group(1)], template)


def parse_choice(reply: str, question: Question) -> Optional[str]:
    """Map a model's free-text reply to a choice id, or None if unparseable."""
    text = (reply or "").strip()
    if not text:
        return None
    letters = [_letter(i) for i in range(len(question.choices))]

    # 1. A leading option letter ("A", "A.", "A) Offer A", "a").
    m = re.match(r"\s*([A-Za-z])\b", text)
    if m and m.group(1).upper() in letters:
        return question.choices[letters.index(m.group(1).upper())].id

    # 2. "answer is X" / "option X" / "choice: X".
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*([A-Za-z])\b", text, re.I)
    if m and m.group(1).upper() in letters:
        return question.choices[letters.index(m.group(1).upper())].id

    # 3. Any standalone uppercase option letter elsewhere.
    for mm in re.finditer(r"\b([A-Z])\b", text):
        if mm.group(1) in letters:
            return question.choices[letters.index(mm.group(1))].id

    # 4. Fall back to matching the choice's own text (longest match wins).
    low = text.lower()
    hits = [c for c in question.choices if c.text.lower() in low]
    if hits:
        return max(hits, key=lambda c: len(c.text)).id

    return None

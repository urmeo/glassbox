"""Reader base types and the shared logic for turning values into a choice."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from ..answer_parsing import parse_choice_letter
from ..prompts import load_prompt
from ..schema import Question, Scenario
from ..stimuli import Stimulus


@dataclass(frozen=True)
class Answer:
    """A reader's response to one question."""

    choice_id: Optional[str]
    method: str = ""
    raw: str = ""
    prompt_sha256: Optional[str] = None


class Reader:
    """Interface every reader implements."""

    name: str = "reader"
    family: str = ""
    simulated: bool = False
    deterministic: Optional[bool] = None
    provider: str = "unknown"
    model_family: str = "unknown"
    family_resolved: bool = False
    family_basis: str = "unknown"

    def answer(
        self, scenario: Scenario, question: Question, stimulus: Stimulus
    ) -> Answer:
        raise NotImplementedError




def apply_op(
    compute_spec: Mapping[str, Any], values: Mapping[str, Any], order: List[str]
) -> Any:
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




def _letter(index: int) -> str:
    return chr(ord("A") + index)


def build_mcq_prompt(question: Question, stimulus: Stimulus, image: bool) -> str:
    """Format lettered choices with the image or inline interface text."""
    if not 2 <= len(question.choices) <= 26:
        raise ValueError("MCQ prompts require between 2 and 26 choices")
    template = load_prompt("reader_mcq")
    options = "\n".join(
        "%s. %s" % (_letter(i), c.text) for i, c in enumerate(question.choices)
    )
    context_block = (
        ""
        if image
        else ("The interface is shown below as text:\n\n" + stimulus.text + "\n\n")
    )
    fields = {"context_block": context_block, "stem": question.stem, "options": options}
    return re.sub(
        r"\{(context_block|stem|options)\}", lambda m: fields[m.group(1)], template
    )


def parse_choice(reply: str, question: Question) -> Optional[str]:
    """Map a declared letter or exact authored choice text to its ID."""
    letters = [_letter(index) for index in range(len(question.choices))]
    chosen = parse_choice_letter(reply, letters)
    if chosen is not None:
        return question.choices[letters.index(chosen)].id
    if not isinstance(reply, str):
        return None
    hits = [
        choice.id
        for choice in question.choices
        if choice.text.casefold() == reply.strip().casefold()
    ]
    return hits[0] if len(hits) == 1 else None

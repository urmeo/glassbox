"""Simulated readers — deterministic fixtures that make the pipeline verifiable offline.

These are **not** stand-ins for real readers: every report they appear in is labeled
``simulated`` and they never support a scientific claim. Their only job is to
prove that, given a known interface, the render->read->score->analyze pipeline computes
the right numbers. Each persona models a distinct, general failure mode:

* ``literal``  — reads only the values the interface shows; never computes. If the
  answer field is not on screen, it falls back to the interface's *emphasized* value
  (which may mislead), else guesses.
* ``diligent`` — computes the honest metric whenever every input it needs is visible;
  otherwise falls back like ``literal``.
* ``careless`` — like ``diligent``, but on a close argmin/argmax it slips to the
  runner-up unless the interface highlights the true winner.

The logic is scenario-agnostic: it reads only what a presentation exposes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .. import compute
from ..interfaces import Presentation
from ..schema import Question, Scenario
from ..stimuli import Stimulus
from .base import Answer, Reader, apply_op, guess_choice, value_to_choice

SIMULATED_PERSONAS = ("literal", "diligent", "careless")
CLOSE_MARGIN = 0.05  # relative gap under which a "careless" reader may slip


class SimulatedReader(Reader):
    simulated = True

    def __init__(self, persona: str):
        if persona not in SIMULATED_PERSONAS:
            raise ValueError("unknown simulated persona %r (have %s)"
                             % (persona, ", ".join(SIMULATED_PERSONAS)))
        self.persona = persona
        self.family = "simulated"
        self.name = "simulated:" + persona

    def answer(self, scenario: Scenario, question: Question, stimulus: Stimulus) -> Answer:
        p = stimulus.presentation
        values, method = self._value_source(scenario, question, p)
        if values is None:
            return Answer(choice_id=guess_choice(question), method="guess")

        answer_value = apply_op(question.compute, values, p.order)
        if self.persona == "careless" and question.compute["op"] in ("argmin", "argmax"):
            answer_value = self._careless_adjust(question, values, p, answer_value)

        choice = value_to_choice(question, answer_value)
        if choice is None:
            return Answer(choice_id=guess_choice(question), method="guess")
        return Answer(choice_id=choice, method=method)

    def _value_source(self, scenario: Scenario, question: Question,
                      p: Presentation) -> Tuple[Optional[Dict[str, Any]], str]:
        """Decide which per-item values this persona reasons over, and how it got them."""
        target = question.compute.get("field")
        order = p.order

        # 1. Read the answer field directly if the interface shows it.
        if target and all(p.shows_field(i, target) for i in order):
            return {i: p.shown[i][target] for i in order}, "read"

        # 2. Compute a derived answer field, if the reader computes and every input is shown.
        if self.persona in ("diligent", "careless") and target in scenario.derived:
            inputs = compute.expr_fields(scenario.derived[target])
            if all(all(f in p.shown[i] for f in inputs) for i in order):
                return ({i: compute.evaluate_expr(scenario.derived[target], p.shown[i])
                         for i in order}, "computed")

        # 3. Fall back to the interface's emphasized value (the headline — may mislead).
        emph = p.emphasized_field
        if emph and all(p.shows_field(i, emph) for i in order):
            return {i: p.shown[i][emph] for i in order}, "headline"

        # 4. Nothing usable on screen.
        return None, "guess"

    def _careless_adjust(self, question: Question, values: Dict[str, Any],
                         p: Presentation, answer_value: Any) -> Any:
        op = question.compute["op"]
        order = p.order
        if len(order) < 2:
            return answer_value
        ranked = sorted(order, key=lambda i: values[i], reverse=(op == "argmax"))
        top, runner = ranked[0], ranked[1]
        v1, v2 = values[top], values[runner]
        gap = abs(v1 - v2) / max(abs(v1), abs(v2), 1)
        if gap >= CLOSE_MARGIN:
            return answer_value  # a comfortable margin — even a careless reader is fine
        # A close call: trust a highlight only when it marks this question's true extreme.
        if p.highlight_item is not None and p.highlight_item == top:
            return top
        return runner

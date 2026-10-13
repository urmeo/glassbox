"""Three deterministic reader fixtures with explicit computational rules."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .. import compute
from ..interfaces import Presentation
from ..schema import Question, Scenario
from ..stimuli import Stimulus
from .base import Answer, Reader, apply_op, guess_choice, value_to_choice

SIMULATED_PERSONAS = ("literal", "diligent", "careless")
CLOSE_MARGIN = 0.05


class SimulatedReader(Reader):
    simulated = True
    deterministic = True
    provider = "simulated"
    model_family = "simulated"
    family_resolved = True
    family_basis = "simulated-fixture"

    def __init__(self, persona: str):
        if persona not in SIMULATED_PERSONAS:
            raise ValueError(
                "unknown simulated persona %r (have %s)"
                % (persona, ", ".join(SIMULATED_PERSONAS))
            )
        self.persona = persona
        self.family = "simulated"
        self.name = "simulated:" + persona

    def answer(
        self, scenario: Scenario, question: Question, stimulus: Stimulus
    ) -> Answer:
        presentation = stimulus.presentation
        values, method = self._value_source(scenario, question, presentation)
        if values is None:
            return Answer(choice_id=guess_choice(question), method="guess")

        answer_value = apply_op(question.compute, values, presentation.order)
        if self.persona == "careless" and question.compute["op"] in (
            "argmin",
            "argmax",
        ):
            answer_value = self._careless_adjust(
                question, values, presentation, answer_value
            )

        choice = value_to_choice(question, answer_value)
        if choice is None:
            return Answer(choice_id=guess_choice(question), method="guess")
        return Answer(choice_id=choice, method=method)

    def _value_source(
        self, scenario: Scenario, question: Question, presentation: Presentation
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Choose visible values or compute a target from visible inputs."""
        target = question.compute.get("field")
        order = presentation.order

        if target and all(
            presentation.shows_field(item_id, target) for item_id in order
        ):
            return {
                item_id: presentation.shown[item_id][target] for item_id in order
            }, "read"

        if self.persona in ("diligent", "careless") and target in scenario.derived:
            expression = scenario.derived[target]
            inputs = compute.expr_fields(expression)
            if all(
                field in presentation.shown[item_id]
                for item_id in order
                for field in inputs
            ):
                return (
                    {
                        item_id: compute.evaluate_expr(
                            expression, presentation.shown[item_id]
                        )
                        for item_id in order
                    },
                    "computed",
                )

        emphasized = presentation.emphasized_field
        if emphasized and all(
            presentation.shows_field(item_id, emphasized) for item_id in order
        ):
            return {
                item_id: presentation.shown[item_id][emphasized] for item_id in order
            }, "headline"

        return None, "guess"

    def _careless_adjust(
        self,
        question: Question,
        values: Dict[str, Any],
        presentation: Presentation,
        answer_value: Any,
    ) -> Any:
        op = question.compute["op"]
        order = presentation.order
        if len(order) < 2:
            return answer_value
        ranked = sorted(
            order, key=lambda item_id: values[item_id], reverse=(op == "argmax")
        )
        winner, runner_up = ranked[:2]
        first_value, second_value = values[winner], values[runner_up]
        gap = abs(first_value - second_value) / max(
            abs(first_value), abs(second_value), 1
        )
        if gap >= CLOSE_MARGIN:
            return answer_value
        if presentation.highlight_item == winner:
            return winner
        return runner_up

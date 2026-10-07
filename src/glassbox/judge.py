"""Synthetic polish ratings and text perceived-comprehension comparisons."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import interfaces
from .answer_parsing import parse_choice_letter
from .interfaces import INTERFACE_VARIANTS, Presentation
from .schema import Scenario, scenario_sha256


@dataclass(frozen=True)
class Preference:
    scenario_id: str
    variant: str
    score: float
    judge: str
    simulated: bool


@dataclass(frozen=True)
class JudgeMetadata:
    name: str
    spec: str
    target: str
    modality: str
    protocol: str
    simulated: bool
    deterministic: bool
    variants: Tuple[str, ...] = ()
    comparison_count: int = 0
    prompt_sha256: Optional[str] = None

    @property
    def score_label(self) -> str:
        return {
            "synthetic_polish": "synthetic polish",
            "perceived_comprehension": "perceived comprehension",
        }.get(self.target, "judge score")


class Judge:
    name = "judge"
    spec = "custom"
    simulated = False
    deterministic = False
    target = "unknown"
    modality = "unknown"
    protocol = "custom"

    def preference(self, scenario: Scenario, presentation: Presentation) -> float:
        raise NotImplementedError

    def metadata(self) -> JudgeMetadata:
        return JudgeMetadata(
            self.name,
            self.spec,
            self.target,
            self.modality,
            self.protocol,
            self.simulated,
            self.deterministic,
        )


class SimulatedJudge(Judge):
    """Fixed polish weights for the offline fixture."""

    name = "simulated:polish"
    spec = "polish"
    simulated = True
    deterministic = True
    target = "synthetic_polish"
    modality = "presentation_features"
    protocol = "fixed_polish_weights.v1"

    def preference(self, scenario: Scenario, presentation: Presentation) -> float:
        f = presentation.features
        score = 0.30
        for feature, weight in (
            ("polished", 0.35),
            ("highlight", 0.08),
            ("derived", 0.05),
            ("sorted", 0.04),
            ("as_text", -0.15),
        ):
            if feature in f:
                score += weight
        if presentation.order:
            avg_fields = statistics.fmean(
                len(presentation.shown[i]) for i in presentation.order
            )
            score -= 0.015 * avg_fields
        return max(0.0, min(1.0, score))


def validate_judge_spec(spec: str) -> str:
    """Validate a selector without loading keys or making requests."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("judge must be a nonempty string")
    spec = spec.strip()
    if spec in ("polish", "simulated", "simulated:polish"):
        return "polish"
    if spec in ("pairwise", "pairwise:simulated", "pairwise:polish"):
        return "pairwise"
    if spec.startswith("pairwise:"):
        reader_spec = spec[len("pairwise:") :]
        provider, _, model = reader_spec.partition(":")
        if provider not in ("anthropic", "openai", "openrouter"):
            raise ValueError("pairwise provider must be anthropic/openai/openrouter")
        if not model or model != model.strip() or any(c.isspace() for c in model):
            raise ValueError("pairwise judge needs a model id without whitespace")
        return spec
    raise ValueError(
        "unknown judge %r (have: polish, pairwise, pairwise:<reader>)" % spec
    )


def _variants(variants: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if variants is None:
        return tuple(INTERFACE_VARIANTS)
    if isinstance(variants, (str, bytes)):
        raise ValueError("judge variants must be a nonempty sequence")
    result = tuple(variants)
    if (
        not result
        or any(not isinstance(v, str) or v not in INTERFACE_VARIANTS for v in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError("judge variants must be nonempty, known and distinct")
    return result


def build_judge(spec: str, variants: Optional[Sequence[str]] = None) -> Judge:
    spec = validate_judge_spec(spec)
    selected = _variants(variants)
    if spec == "polish":
        return SimulatedJudge()
    pairwise = (
        SimulatedPairwiseJudge()
        if spec == "pairwise"
        else ApiPairwiseJudge(spec[len("pairwise:") :])
    )
    return PairwiseRatingJudge(pairwise, selected, spec=spec)


def judge_all(
    judge: Judge, scenario: Scenario, presentations: Sequence[Presentation]
) -> List[Preference]:
    return [
        Preference(
            scenario.id,
            p.variant,
            judge.preference(scenario, p),
            judge.name,
            judge.simulated,
        )
        for p in presentations
    ]


def parse_ab(reply: str) -> Optional[str]:
    return parse_choice_letter(reply, ("A", "B"))


class PairwiseJudge:
    name = "pairwise"
    simulated = False
    deterministic = False
    target = "unknown"
    modality = "unknown"
    protocol = "ordered_pairs_winrate.v1"

    @property
    def prompt_sha256(self) -> Optional[str]:
        return None

    def compare(self, scenario: Scenario, a: Presentation, b: Presentation) -> str:
        raise NotImplementedError


class SimulatedPairwiseJudge(PairwiseJudge):
    name = "simulated:pairwise-polish"
    simulated = True
    deterministic = True
    target = "synthetic_polish"
    modality = "presentation_features"

    def __init__(self):
        self._rating = SimulatedJudge()

    def compare(self, scenario: Scenario, a: Presentation, b: Presentation) -> str:
        sa, sb = (
            self._rating.preference(scenario, a),
            self._rating.preference(scenario, b),
        )
        if sa != sb:
            return "A" if sa > sb else "B"
        return "A" if a.variant <= b.variant else "B"


class ApiPairwiseJudge(PairwiseJudge):
    """Compare text presentations using the shipped comprehension prompt."""

    target = "perceived_comprehension"
    modality = "text"

    def __init__(self, reader_spec: str, max_tokens: int = 8):
        if not isinstance(reader_spec, str):
            raise ValueError("pairwise reader spec must be a string")
        spec = validate_judge_spec("pairwise:" + reader_spec)
        self.family, _, self.model = spec[len("pairwise:") :].partition(":")
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        self.name, self.max_tokens = spec, max_tokens

    @property
    def prompt_sha256(self) -> str:
        from .prompts import load_prompt

        return hashlib.sha256(load_prompt("judge_pairwise").encode("utf-8")).hexdigest()

    def _prompt(self, a: Presentation, b: Presentation) -> str:
        from .prompts import load_prompt

        fields = {"a": a.to_text(), "b": b.to_text()}
        return re.sub(
            r"\{(a|b)\}", lambda m: fields[m.group(1)], load_prompt("judge_pairwise")
        )

    def build_payload(self, a: Presentation, b: Presentation) -> Dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self._prompt(a, b)}],
                }
            ],
        }

    def compare(self, scenario: Scenario, a: Presentation, b: Presentation) -> str:
        from .readers._http import post_json, require_key

        if self.family == "anthropic":
            from .readers.anthropic import ENDPOINT, API_VERSION, extract_text

            key = require_key("ANTHROPIC_API_KEY")
            resp = post_json(
                ENDPOINT,
                {"x-api-key": key, "anthropic-version": API_VERSION},
                self.build_payload(a, b),
            )
        else:
            from .readers.openai_compat import _ENDPOINTS, _KEY_ENV, extract_text

            key = require_key(_KEY_ENV[self.family])
            resp = post_json(
                _ENDPOINTS[self.family],
                {"authorization": "Bearer " + key},
                self.build_payload(a, b),
            )
        return parse_ab(extract_text(resp)) or "tie"


class PairwiseRatingJudge(Judge):
    """Aggregate one ordered comparison per selected pair; ties split the point."""

    def __init__(
        self,
        pairwise: PairwiseJudge,
        variants: Optional[Sequence[str]] = None,
        *,
        spec: str = "custom",
    ):
        self._pw, self._variants = pairwise, _variants(variants)
        self.name, self.spec = "winrate(%s)" % pairwise.name, spec
        self.simulated, self.deterministic = pairwise.simulated, pairwise.deterministic
        self.target, self.modality, self.protocol = (
            pairwise.target,
            pairwise.modality,
            pairwise.protocol,
        )
        self.comparison_count = 0
        self._cache: Dict[str, Dict[str, float]] = {}

    def metadata(self) -> JudgeMetadata:
        return JudgeMetadata(
            self.name,
            self.spec,
            self.target,
            self.modality,
            self.protocol,
            self.simulated,
            self.deterministic,
            self._variants,
            self.comparison_count,
            self._pw.prompt_sha256,
        )

    def _winrates(self, scenario: Scenario) -> Dict[str, float]:
        identity = (
            scenario_sha256(scenario),
            self._variants,
            self.name,
            self.protocol,
            self.target,
            self.modality,
            self._pw.prompt_sha256,
            getattr(self._pw, "max_tokens", None),
        )
        key = hashlib.sha256(
            json.dumps(identity, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]
        pres = {v: interfaces.variant(scenario, v) for v in self._variants}
        wins = {v: 0.0 for v in self._variants}
        for i, va in enumerate(self._variants):
            for vb in self._variants[i + 1 :]:
                winner = self._pw.compare(scenario, pres[va], pres[vb])
                if winner not in ("A", "B", "tie"):
                    raise ValueError("pairwise result must be A, B or tie")
                self.comparison_count += 1
                if winner == "A":
                    wins[va] += 1.0
                elif winner == "B":
                    wins[vb] += 1.0
                else:
                    wins[va] += 0.5
                    wins[vb] += 0.5
        k = len(self._variants)
        rates = {v: (wins[v] / (k - 1) if k > 1 else 0.0) for v in self._variants}
        self._cache[key] = rates
        return rates

    def preference(self, scenario: Scenario, presentation: Presentation) -> float:
        if presentation.variant not in self._variants:
            raise ValueError("presentation variant was not selected for the judge")
        return self._winrates(scenario)[presentation.variant]

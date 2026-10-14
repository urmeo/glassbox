"""Synthetic polish ratings and text perceived-comprehension comparisons."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import interfaces
from ._config import positive_integer
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
class PairwiseJudgment:
    """One parsed choice, genuine tie, invalid reply, or explicit abstention."""

    outcome: str
    raw: str = ""
    prompt_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if self.outcome not in ("A", "B", "tie", "invalid", "abstained"):
            raise ValueError("pairwise outcome must be A, B, tie, invalid or abstained")
        if not isinstance(self.raw, str):
            raise ValueError("pairwise raw reply must be text")

    @property
    def status(self) -> str:
        return "valid" if self.outcome in ("A", "B") else self.outcome


@dataclass(frozen=True)
class PairwiseComparison:
    scenario_id: str
    variant_a: str
    variant_b: str
    outcome: str
    status: str
    raw: str
    prompt_sha256: Optional[str] = None


@dataclass(frozen=True)
class VariantJudgeScore:
    variant: str
    credit: float
    denominator: int
    planned_count: int
    score: float


@dataclass(frozen=True)
class ScenarioJudgeEvidence:
    scenario_id: str
    scenario_sha256: str
    cache_key: str
    protocol: str
    comparisons: Tuple[PairwiseComparison, ...]
    variant_scores: Tuple[VariantJudgeScore, ...]
    valid_count: int
    invalid_count: int
    abstained_count: int
    tie_count: int


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
    valid_count: int = 0
    invalid_count: int = 0
    abstained_count: int = 0
    tie_count: int = 0
    evidence: Tuple[ScenarioJudgeEvidence, ...] = ()

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
        features = presentation.features
        score = 0.30
        for feature, weight in (
            ("polished", 0.35),
            ("highlight", 0.08),
            ("derived", 0.05),
            ("sorted", 0.04),
            ("as_text", -0.15),
        ):
            if feature in features:
                score += weight
        if presentation.order:
            avg_fields = statistics.fmean(
                len(presentation.shown[item_id]) for item_id in presentation.order
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

    def compare(
        self, scenario: Scenario, a: Presentation, b: Presentation
    ) -> Union[PairwiseJudgment, str]:
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
        first_score = self._rating.preference(scenario, a)
        second_score = self._rating.preference(scenario, b)
        if first_score != second_score:
            return "A" if first_score > second_score else "B"
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
        positive_integer(max_tokens, "max_tokens")
        self.name = spec
        self.max_tokens = max_tokens

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

    def build_payload(self, a: Presentation, b: Presentation) -> Dict[str, Any]:
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

    def compare(
        self, scenario: Scenario, a: Presentation, b: Presentation
    ) -> PairwiseJudgment:
        from .readers._http import post_json, require_key

        payload = self.build_payload(a, b)
        prompt = next(
            block["text"]
            for block in payload["messages"][0]["content"]
            if block["type"] == "text"
        )
        if self.family == "anthropic":
            from .readers.anthropic import ENDPOINT, API_VERSION, extract_text

            key = require_key("ANTHROPIC_API_KEY")
            response = post_json(
                ENDPOINT,
                {"x-api-key": key, "anthropic-version": API_VERSION},
                payload,
            )
        else:
            from .readers.openai_compat import _ENDPOINTS, _KEY_ENV, extract_text

            key = require_key(_KEY_ENV[self.family])
            response = post_json(
                _ENDPOINTS[self.family],
                {"authorization": "Bearer " + key},
                payload,
            )
        raw = extract_text(response)
        outcome = parse_ab(raw)
        if outcome is None:
            outcome = (
                "abstained" if raw.strip().casefold() in ("abstain", "abstained")
                else "invalid"
            )
        return PairwiseJudgment(
            outcome, raw, hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        )


class PairwiseRatingJudge(Judge):
    """Judge both pair orders; exclude invalid replies and abstentions from rates."""

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
        self.target, self.modality = (
            pairwise.target,
            pairwise.modality,
        )
        self.protocol = "counterbalanced_pairs_winrate.v2"
        self.comparison_count = 0
        self._cache: Dict[str, ScenarioJudgeEvidence] = {}

    def metadata(self) -> JudgeMetadata:
        evidence = tuple(self._cache.values())
        return JudgeMetadata(
            name=self.name,
            spec=self.spec,
            target=self.target,
            modality=self.modality,
            protocol=self.protocol,
            simulated=self.simulated,
            deterministic=self.deterministic,
            variants=self._variants,
            comparison_count=self.comparison_count,
            prompt_sha256=self._pw.prompt_sha256,
            valid_count=sum(item.valid_count for item in evidence),
            invalid_count=sum(item.invalid_count for item in evidence),
            abstained_count=sum(item.abstained_count for item in evidence),
            tie_count=sum(item.tie_count for item in evidence),
            evidence=evidence,
        )

    def _evidence(self, scenario: Scenario) -> ScenarioJudgeEvidence:
        scenario_hash = scenario_sha256(scenario)
        presentations = {
            variant: interfaces.variant(scenario, variant) for variant in self._variants
        }
        identity = (
            scenario_hash,
            self._variants,
            tuple(
                (
                    variant,
                    {**asdict(presentation), "features": sorted(presentation.features)},
                    presentation.to_text(),
                )
                for variant, presentation in presentations.items()
            ),
            self.name,
            self.protocol,
            self._pw.name,
            self._pw.protocol,
            self.target,
            self.modality,
            self._pw.prompt_sha256,
            getattr(self._pw, "max_tokens", None),
            getattr(self._pw, "family", None),
            getattr(self._pw, "model", None),
        )
        key = hashlib.sha256(
            json.dumps(
                identity, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]
        credits = {variant: 0.0 for variant in self._variants}
        denominators = {variant: 0 for variant in self._variants}
        comparisons = []
        for first, second in combinations(self._variants, 2):
            for variant_a, variant_b in ((first, second), (second, first)):
                result = self._pw.compare(
                    scenario, presentations[variant_a], presentations[variant_b]
                )
                if isinstance(result, str):
                    result = PairwiseJudgment(result, raw=result)
                if not isinstance(result, PairwiseJudgment):
                    raise ValueError(
                        "pairwise judge must return a judgment or outcome string"
                    )
                comparisons.append(PairwiseComparison(
                    scenario_id=scenario.id,
                    variant_a=variant_a,
                    variant_b=variant_b,
                    outcome=result.outcome,
                    status=result.status,
                    raw=result.raw,
                    prompt_sha256=result.prompt_sha256,
                ))
                if result.status in ("invalid", "abstained"):
                    continue
                denominators[variant_a] += 1
                denominators[variant_b] += 1
                if result.outcome == "A":
                    credits[variant_a] += 1.0
                elif result.outcome == "B":
                    credits[variant_b] += 1.0
                else:
                    credits[variant_a] += 0.5
                    credits[variant_b] += 0.5
        evidence = ScenarioJudgeEvidence(
            scenario_id=scenario.id,
            scenario_sha256=scenario_hash,
            cache_key=key,
            protocol=self.protocol,
            comparisons=tuple(comparisons),
            variant_scores=tuple(
                VariantJudgeScore(
                    variant=variant,
                    credit=credits[variant],
                    denominator=denominators[variant],
                    planned_count=2 * (len(self._variants) - 1),
                    score=(credits[variant] / denominators[variant]
                           if denominators[variant] else float("nan")),
                )
                for variant in self._variants
            ),
            valid_count=sum(item.status in ("valid", "tie") for item in comparisons),
            invalid_count=sum(item.status == "invalid" for item in comparisons),
            abstained_count=sum(item.status == "abstained" for item in comparisons),
            tie_count=sum(item.status == "tie" for item in comparisons),
        )
        self._cache[key] = evidence
        self.comparison_count += len(comparisons)
        return evidence

    def preference(self, scenario: Scenario, presentation: Presentation) -> float:
        if presentation.variant not in self._variants:
            raise ValueError("presentation variant was not selected for the judge")
        return next(
            item.score for item in self._evidence(scenario).variant_scores
            if item.variant == presentation.variant
        )

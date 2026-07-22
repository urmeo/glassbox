"""The preference judge — measures what is *liked*, never what is *understood*.

It runs alongside the readers and is deliberately kept out of the score: comprehension
is measured only by readers answering questions. Keeping preference and comprehension
separate is what lets the harness show them diverge.

The v1 judge is a deterministic **rating** judge over aesthetic features — a stand-in
that reproduces the field's "people prefer the polished one" finding offline. It is a
placeholder that a later version upgrades to real pairwise API judges. It looks only at
how an interface *presents*, never at whether its content is correct.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from . import interfaces
from .interfaces import INTERFACE_VARIANTS, Presentation
from .schema import Scenario


@dataclass(frozen=True)
class Preference:
    scenario_id: str
    variant: str
    score: float
    judge: str
    simulated: bool


class Judge:
    name = "judge"
    simulated = False

    def preference(self, scenario: Scenario, presentation: Presentation) -> float:
        raise NotImplementedError


class SimulatedJudge(Judge):
    """Rates polish from presentation features — the more styled and less busy, the higher."""

    name = "simulated:polish"
    simulated = True

    # Aesthetic weights: polish dominates (the field's 70-83% preference for polished UIs),
    # small bonuses for helpful-looking touches, a penalty for plain text and for clutter.
    def preference(self, scenario: Scenario, presentation: Presentation) -> float:
        f = presentation.features
        score = 0.30
        if "polished" in f:
            score += 0.35
        if "highlight" in f:
            score += 0.08
        if "derived" in f:
            score += 0.05
        if "sorted" in f:
            score += 0.04
        if "as_text" in f:
            score -= 0.15
        if presentation.order:  # busier interfaces read as less clean
            avg_fields = statistics.fmean(len(presentation.shown[i]) for i in presentation.order)
            score -= 0.015 * avg_fields
        return max(0.0, min(1.0, score))


def build_judge(spec: str) -> Judge:
    """Construct a judge from a spec string.

    ``polish`` — deterministic rating judge (v1). ``pairwise`` — deterministic pairwise
    judge as win-rate. ``pairwise:<reader-spec>`` — a real pairwise API judge, e.g.
    ``pairwise:anthropic:claude-sonnet-5`` (needs the matching key)."""
    if spec in ("polish", "simulated", "simulated:polish"):
        return SimulatedJudge()
    if spec in ("pairwise", "pairwise:simulated", "pairwise:polish"):
        return PairwiseRatingJudge(SimulatedPairwiseJudge())
    if spec.startswith("pairwise:"):
        return PairwiseRatingJudge(ApiPairwiseJudge(spec[len("pairwise:"):]))
    raise ValueError("unknown judge %r (have: polish, pairwise, pairwise:<reader>)" % spec)


def judge_all(judge: Judge, scenario: Scenario,
              presentations: Sequence[Presentation]) -> List[Preference]:
    """Score a set of presentations for one scenario."""
    return [Preference(scenario_id=scenario.id, variant=p.variant,
                       score=judge.preference(scenario, p), judge=judge.name,
                       simulated=judge.simulated)
            for p in presentations]


# --- pairwise judging (M2 upgrade: compare two interfaces, aggregate to a ranking) ---

def parse_ab(reply: str) -> Optional[str]:
    """Extract 'A' or 'B' from a pairwise reply, or None."""
    m = re.search(r"\b([AB])\b", (reply or "").upper())
    return m.group(1) if m else None


class PairwiseJudge:
    """Compares two interfaces and picks the one that conveys more understanding."""

    name = "pairwise"
    simulated = False

    def compare(self, scenario: Scenario, a: Presentation, b: Presentation) -> str:
        """Return 'A' if the first interface wins, 'B' if the second, 'tie' if neither."""
        raise NotImplementedError


class SimulatedPairwiseJudge(PairwiseJudge):
    """Deterministic pairwise judge — the higher polish score wins (ties by variant name)."""

    name = "simulated:pairwise-polish"
    simulated = True

    def __init__(self):
        self._rating = SimulatedJudge()

    def compare(self, scenario: Scenario, a: Presentation, b: Presentation) -> str:
        sa = self._rating.preference(scenario, a)
        sb = self._rating.preference(scenario, b)
        if sa != sb:
            return "A" if sa > sb else "B"
        return "A" if a.variant <= b.variant else "B"


class ApiPairwiseJudge(PairwiseJudge):
    """Real pairwise judge over a text description of each interface. Needs a key."""

    def __init__(self, reader_spec: str, max_tokens: int = 8):
        family, _, model = reader_spec.partition(":")
        if family not in ("anthropic", "openai", "openrouter"):
            raise ValueError("pairwise judge family must be anthropic/openai/openrouter")
        if not model:
            raise ValueError("pairwise judge needs a model id, e.g. pairwise:anthropic:claude-sonnet-5")
        self.family = family
        self.model = model
        self.name = "pairwise:" + reader_spec
        self.simulated = False
        self.max_tokens = max_tokens

    def _prompt(self, a: Presentation, b: Presentation) -> str:
        from .prompts import load_prompt
        # Single pass so a literal "{b}" inside interface A's text is not re-substituted.
        fields = {"a": a.to_text(), "b": b.to_text()}
        return re.sub(r"\{(a|b)\}", lambda m: fields[m.group(1)], load_prompt("judge_pairwise"))

    def build_payload(self, a: Presentation, b: Presentation) -> Dict:
        text = self._prompt(a, b)
        return {"model": self.model, "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}

    def compare(self, scenario: Scenario, a: Presentation, b: Presentation) -> str:
        from .readers._http import post_json, require_key
        if self.family == "anthropic":
            from .readers.anthropic import ENDPOINT, API_VERSION, extract_text
            key = require_key("ANTHROPIC_API_KEY")
            resp = post_json(ENDPOINT, {"x-api-key": key, "anthropic-version": API_VERSION},
                             self.build_payload(a, b))
            text = extract_text(resp)
        else:
            from .readers.openai_compat import _ENDPOINTS, _KEY_ENV, extract_text
            key = require_key(_KEY_ENV[self.family])
            resp = post_json(_ENDPOINTS[self.family], {"authorization": "Bearer " + key},
                             self.build_payload(a, b))
            text = extract_text(resp)
        # An unparseable reply is a tie, never a default win for A — otherwise a parse
        # failure would systematically credit the positionally-earlier interface.
        return parse_ab(text) or "tie"


class PairwiseRatingJudge(Judge):
    """Turns a pairwise judge into a per-interface win-rate, usable wherever a rating
    judge is — so the H1 and cross-family analyses need no change to use pairwise."""

    def __init__(self, pairwise: PairwiseJudge, variants: Optional[Sequence[str]] = None):
        self._pw = pairwise
        self._variants = list(variants) if variants else list(INTERFACE_VARIANTS)
        self.name = "winrate(%s)" % pairwise.name
        self.simulated = pairwise.simulated
        self._cache: Dict[str, Dict[str, float]] = {}

    def _winrates(self, scenario: Scenario) -> Dict[str, float]:
        if scenario.id in self._cache:
            return self._cache[scenario.id]
        pres = {v: interfaces.variant(scenario, v) for v in self._variants}
        wins = {v: 0.0 for v in self._variants}
        for i in range(len(self._variants)):
            for j in range(i + 1, len(self._variants)):
                va, vb = self._variants[i], self._variants[j]
                winner = self._pw.compare(scenario, pres[va], pres[vb])
                if winner == "A":
                    wins[va] += 1.0
                elif winner == "B":
                    wins[vb] += 1.0
                else:  # a tie splits the point — no positional bias
                    wins[va] += 0.5
                    wins[vb] += 0.5
        k = len(self._variants)
        rates = {v: (wins[v] / (k - 1) if k > 1 else 0.0) for v in self._variants}
        self._cache[scenario.id] = rates
        return rates

    def preference(self, scenario: Scenario, presentation: Presentation) -> float:
        return self._winrates(scenario).get(presentation.variant, 0.0)

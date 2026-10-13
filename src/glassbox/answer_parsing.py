"""Conservative option-letter parsing for MCQ readers and pairwise judges."""

from __future__ import annotations

import re
from typing import Optional, Sequence

PARSER_VERSION = "1.0.0"


def parse_choice_letter(reply: str, allowed: Sequence[str]) -> Optional[str]:
    """Return one declared option; incidental or conflicting mentions are unparseable."""
    if (
        not isinstance(allowed, (list, tuple))
        or not allowed
        or any(
            not isinstance(letter, str) or not re.fullmatch(r"[A-Z]", letter)
            for letter in allowed
        )
        or len(set(allowed)) != len(allowed)
    ):
        raise ValueError("allowed letters must be unique uppercase A-Z options")
    if not isinstance(reply, str) or not reply.strip():
        return None
    text = reply.strip()
    option_pattern = (
        r"[\s\"'([{]*([A-Za-z])\b[\"')\]}]*"
        r"(?=$|[ \t]*[.,;!?)(]|[ \t]*\n|[ \t]+(?:because|since|due\s+to|as\b|over\b))"
    )
    final_answers = re.findall(
        r"\bfinal\s+(?:answer|choice|selection)\b\s*(?:is|:|=)?" + option_pattern,
        text,
        re.I,
    )
    if re.search(r"\bfinal\s+(?:answer|choice|selection)\b", text, re.I):
        return _unique(final_answers, allowed) if final_answers else None
    declaration_patterns = [
        r"\b(?:answer|choice|selection)\b\s*(?:is|:|=)?" + option_pattern,
        r"(?:\bI\s+(?:would\s+)?|\bI'd\s+|(?:^|[.!?\n])\s*)"
        r"(?:choose|pick|select|prefer)\s+(?:option\s+|offer\s+)?" + option_pattern,
        r"\bso\s+(?:option\s+)?" + option_pattern + r"[\s.!)]*$",
    ]
    declared = [
        item
        for pattern in declaration_patterns
        for item in re.findall(pattern, text, re.I)
    ]
    leading = re.match(r"^([A-Za-z])[.)]\s*", text)
    if leading:
        declared.append(leading.group(1))
    if declared:
        return _unique(declared, allowed)
    exact = re.fullmatch(r"(?:option\s+)?([A-Za-z])[\s.!)]*", text, re.I)
    return _unique([exact.group(1)], allowed) if exact else None


def _unique(candidates: Sequence[str], allowed: Sequence[str]) -> Optional[str]:
    choices = {item.upper() for item in candidates}
    return (
        next(iter(choices)) if len(choices) == 1 and choices <= set(allowed) else None
    )

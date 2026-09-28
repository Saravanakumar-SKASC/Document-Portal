"""
Regex-based PII redaction.

A deliberately conservative baseline: it masks identifiers that are unambiguous
(emails, international phone numbers, Luhn-valid payment cards, Hong Kong ID numbers,
passport numbers that are labelled as such). It does NOT try to catch every name or
address - production systems should add an NER-based tool such as Microsoft Presidio.

Aviation note: part numbers (e.g. 2315-0045-001), ATA chapters (32-41-00) and flight
numbers (CX123) look like IDs, so patterns are chosen to leave them untouched.
"""

import re
from dataclasses import dataclass, field

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# International format only (leading +), or a number explicitly labelled as a phone
PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s-]?\d){7,12}\b")
PHONE_LABELLED = re.compile(r"(?i)\b(phone|tel|mobile|mob|contact no)(\.?\s*[:#]?\s*)(\(?\d[\d\s()-]{6,16}\d)")
CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
HKID = re.compile(r"\b[A-Z]{1,2}\d{6}\s?\(?[0-9A]\)?(?![\w-])")
PASSPORT_LABELLED = re.compile(r"(?i)\b(passport(?:\s*(?:no|number|#))?\.?\s*[:#]?\s*)([A-Z0-9]{6,9})\b")


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


@dataclass
class RedactionResult:
    text: str
    counts: dict = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def redact(text: str) -> RedactionResult:
    counts: dict[str, int] = {}

    def bump(kind: str):
        counts[kind] = counts.get(kind, 0) + 1

    def sub(pattern, kind, replacement):
        nonlocal text

        def _repl(m):
            bump(kind)
            return replacement(m) if callable(replacement) else replacement

        text = pattern.sub(_repl, text)

    sub(EMAIL, "email", "[EMAIL]")

    def _card(m):
        if _luhn_valid(m.group(0)):
            bump("card")
            return "[CARD]"
        return m.group(0)

    text = CARD_CANDIDATE.sub(_card, text)
    sub(PHONE_INTL, "phone", "[PHONE]")
    sub(PHONE_LABELLED, "phone", lambda m: f"{m.group(1)}{m.group(2)}[PHONE]")
    sub(HKID, "hkid", "[HKID]")
    sub(PASSPORT_LABELLED, "passport", lambda m: f"{m.group(1)}[PASSPORT]")
    return RedactionResult(text=text, counts=counts)


def redact_text(text: str) -> str:
    return redact(text).text

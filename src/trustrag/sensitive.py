"""Sensitive-data and instruction detection for ingested text and generated answers.

Deterministic, dependency-free and conservative. Secrets are always redacted before text is indexed,
so they can never be retrieved, embedded or quoted. Personal data is redacted unless a deployment
decides otherwise. Instruction-like text is not removed (it may be legitimate content) but the chunk
is flagged, and the answer step presents it to a model as quoted data.
"""

from __future__ import annotations

import re
import unicodedata

SECRETS: dict[str, re.Pattern[str]] = {
    "aws-access-key-id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b"),
    "private-key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?"
        r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    "password-assignment": re.compile(r"(?i)\b(?:password|passwd|pwd|secret)\s*[:=]\s*\S{4,}"),
}

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")

INSTRUCTIONS: dict[str, re.Pattern[str]] = {
    "override": re.compile(
        r"(?i)\b(?:ignore|disregard|forget)\b.{0,30}\b(?:previous|prior|above|all|earlier)\b.{0,20}"
        r"\b(?:instructions?|rules?|prompts?)\b"
    ),
    "role-hijack": re.compile(r"(?i)\byou are now\b|\bnew system prompt\b|\bsystem override\b"),
    "access-claim": re.compile(
        r"(?i)\b(?:grant|give)\b.{0,30}\b(?:access|permission|clearance)\b|\b(?:i am|user is)\b.{0,20}\badmin\b"
    ),
    "exfiltration": re.compile(r"(?i)\b(?:send|post|upload|email)\b.{0,40}\b(?:to|via)\b.{0,15}(?:https?://|webhook)"),
}


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def redact(text: str, *, pii: bool = True) -> tuple[str, list[str]]:
    """Return (clean_text, labels_redacted). Secrets always; e-mail, phone and card numbers if pii=True."""
    found: list[str] = []
    for label, rx in SECRETS.items():
        text, n = rx.subn(f"[REDACTED:{label}]", text)
        found += [label] * n
    if pii:
        text, n = EMAIL.subn("[REDACTED:email]", text)
        found += ["email"] * n

        def card(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and _luhn(digits):
                found.append("payment-card")
                return "[REDACTED:payment-card]"
            return m.group()

        text = CARD.sub(card, text)

        def phone(m: re.Match[str]) -> str:
            if len(re.sub(r"\D", "", m.group())) >= 10:
                found.append("phone")
                return "[REDACTED:phone]"
            return m.group()

        text = PHONE.sub(phone, text)
    return text, found


def instruction_flags(text: str) -> list[str]:
    normalised = unicodedata.normalize("NFKC", re.sub(r"[​-‏⁠-⁤﻿]", "", text))
    return sorted(name for name, rx in INSTRUCTIONS.items() if rx.search(normalised))

"""Privacy redaction and secret-request detection."""
import re

_PATTERNS = [
    (re.compile(r"(?<!\d)(?:\+?234|0)[789][01]\d{8}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{10,11}(?!\d)"), "[ID_NUMBER]"),          # NIN / BVN / NUBAN
    (re.compile(r"(?<!\d)\d{13,19}(?!\d)"), "[CARD_NUMBER]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
]

SECRET_WORDS = re.compile(r"\b(pin|otp|one[- ]time|password|cvv|token|passcode)\b", re.I)


def redact(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def mentions_secret(text: str) -> bool:
    return bool(SECRET_WORDS.search(text or ""))


import re
from typing import List


def normalize_whitespace(text: str) -> str:
    """Replace repeated whitespace with a single space."""
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str) -> List[str]:
    """
    Split text into words, punctuation, and whitespace tokens.
    Preserve every character for exact reconstruction.
    """
    return re.findall(r"\s+|\w+|[^\w\s]", text, flags=re.UNICODE)


def detokenize_words(tokens: List[str]) -> str:
    """Reconstruct text without changing whitespace or punctuation."""
    return "".join(tokens)


def clean_review(text: str) -> str:
    """Basic cleaning without removing meaningful sentiment information."""
    return normalize_whitespace(str(text))

from __future__ import annotations

import re
from functools import cache
from typing import Literal

import fasttext
from resiliparse.extract.html2text import extract_plain_text

from cs336_data.common import get_shared_assets_path

EMAIL_MASK = "|||EMAIL_ADDRESS|||"
PHONE_MASK = "|||PHONE_NUMBER|||"
IP_MASK = "|||IP_ADDRESS|||"

ClassifierLabel = Literal["cc", "wiki", "non-nsfw", "nsfw", "non-toxic", "toxic"]

_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.+-])")
_PHONE_RE = re.compile(r"(?<!\d)(?:\(\d{3}\)|\d{3})[-\s]?\d{3}[-\s]?\d{4}(?!\d)")
_IP_RE = re.compile(
    r"(?<![\d.])"
    r"(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?!\d)"
)
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_ALPHA_RE = re.compile(r"[A-Za-z]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    return extract_plain_text(html_bytes.decode("utf-8", errors="replace"))


def identify_language(text: str) -> tuple[str, float]:
    model = _load_fasttext_model("classifiers/lid.176.bin")
    if model is not None:
        labels, scores = model.predict(_single_line(text), k=1)
        return _strip_fasttext_label(labels[0]), float(scores[0])

    if _CJK_RE.search(text):
        return "zh", 1.0
    return "en", _english_likelihood(text)


def is_english(text: str, *, threshold: float = 0.7) -> bool:
    language, score = identify_language(text)
    return language == "en" and score >= threshold


def mask_emails(text: str) -> tuple[str, int]:
    return _sub_with_count(_EMAIL_RE, EMAIL_MASK, text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return _sub_with_count(_PHONE_RE, PHONE_MASK, text)


def mask_ips(text: str) -> tuple[str, int]:
    return _sub_with_count(_IP_RE, IP_MASK, text)


def classify_nsfw(text: str) -> tuple[ClassifierLabel, float]:
    model = _load_fasttext_model("classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
    if model is not None:
        label, score = _predict_binary_fasttext(model, text)
        return ("nsfw" if label in {"nsfw", "obscene", "1"} else "non-nsfw", score)

    score = _keyword_score(
        text,
        [
            r"\bf[\W_]*u?[\W_]*c[\W_]*k",
            r"\bc[\W_]*o?[\W_]*c[\W_]*k",
            r"\bc[\W_]*u?[\W_]*n[\W_]*t",
            r"\bass[\W_]*hole",
        ],
    )
    return ("nsfw" if score >= 2.0 else "non-nsfw", max(score, 0.1))


def classify_toxic_speech(text: str) -> tuple[ClassifierLabel, float]:
    model = _load_fasttext_model("classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
    if model is not None:
        label, score = _predict_binary_fasttext(model, text)
        return ("toxic" if label in {"toxic", "hateful", "hate", "1"} else "non-toxic", score)

    score = _keyword_score(
        text,
        [
            r"\bidiot\b",
            r"\bmoron\b",
            r"\btwat\b",
            r"\bf[\W_]*u?[\W_]*c[\W_]*k(?:er|ers|ing)?\b",
        ],
    )
    return ("toxic" if score >= 2.0 else "non-toxic", max(score, 0.1))


def classify_quality(text: str) -> tuple[ClassifierLabel, float]:
    lower_text = text.lower()
    wiki_markers = ("first published", "substantive revision", "bibliography", "references")
    cc_markers = ("forum index", "registerregister", "memberlistmemberlist", "log in")

    wiki_score = sum(marker in lower_text for marker in wiki_markers) + min(len(text) / 20_000, 1.0)
    cc_score = sum(marker in lower_text for marker in cc_markers)

    if wiki_score >= cc_score:
        return "wiki", float(max(wiki_score, 0.1))
    return "cc", float(max(cc_score, 0.1))


def gopher_quality_filter(text: str) -> bool:
    words = _WORD_RE.findall(text)
    if not 50 <= len(words) <= 100_000:
        return False

    average_word_length = sum(len(word) for word in words) / len(words)
    if not 3 <= average_word_length <= 10:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_fraction = sum(line.endswith("...") for line in lines) / len(lines)
        if ellipsis_fraction > 0.3:
            return False

    alphabetic_fraction = sum(bool(_ALPHA_RE.search(word)) for word in words) / len(words)
    return alphabetic_fraction >= 0.8


def _sub_with_count(pattern: re.Pattern[str], replacement: str, text: str) -> tuple[str, int]:
    masked_text, count = pattern.subn(replacement, text)
    return masked_text, count


@cache
def _load_fasttext_model(relative_path: str):
    model_path = get_shared_assets_path() / relative_path
    if not model_path.exists():
        return None
    return fasttext.load_model(str(model_path))


def _strip_fasttext_label(label: str) -> str:
    return label.removeprefix("__label__")


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _predict_binary_fasttext(model, text: str) -> tuple[str, float]:
    labels, scores = model.predict(_single_line(text), k=1)
    return _strip_fasttext_label(labels[0]), float(scores[0])


def _keyword_score(text: str, patterns: list[str]) -> float:
    lower_text = text.lower()
    return float(sum(len(re.findall(pattern, lower_text)) for pattern in patterns))


def _english_likelihood(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    ascii_letters = sum(sum(ch.isascii() and ch.isalpha() for ch in word) for word in words)
    all_letters = sum(sum(ch.isalpha() for ch in word) for word in words)
    if all_letters == 0:
        return 0.0
    return ascii_letters / all_letters

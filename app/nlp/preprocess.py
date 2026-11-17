"""Text preprocessing: cleaning, Hinglish normalization, spaCy lemmatization."""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import spacy
from spacy.language import Language

NEGATIONS = {"not", "no", "never", "n't", "nor", "neither", "none", "nobody", "nothing", "nowhere"}

# Common Hinglish function words: used for language detection and dropped like stopwords.
# Words that are also common English words (main, to, na, me) are left out on purpose.
HINGLISH_MARKERS = {
    "hai", "hain", "tha", "thi", "ka", "ki", "ke", "ko", "se", "kya", "kyu", "kyun",
    "mera", "meri", "mere", "tera", "teri", "aur", "bhi", "yaar", "ekdum", "bilkul",
    "wala", "wali", "raha", "rahi", "gaya", "gayi", "kar", "karo", "kiya", "diya", "hua", "hui",
    "kuch", "sab", "bhai", "liye", "lekin", "abhi", "kaafi", "kafi",
}

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML_RE = re.compile(r"<[^>]+>")
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W\d_]+")


@dataclass
class Preprocessed:
    cleaned_text: str
    tokens: list[str]
    language: str


@lru_cache
def get_nlp() -> Language:
    # Parser and NER are not needed for lemmatization; excluding them saves memory.
    return spacy.load("en_core_web_sm", exclude=["parser", "ner"])


@lru_cache
def _hinglish_map() -> tuple[dict[str, str], re.Pattern]:
    path = Path(__file__).with_name("hinglish_map.json")
    mapping = {k.lower(): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    # Longest keys first so multi-word phrases ("paisa vasool") win over single words.
    keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    return mapping, pattern


def basic_clean(text: str) -> str:
    """Lowercase, strip HTML/URLs/emails, collapse repeated chars and whitespace."""
    text = (text or "").lower()
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    return _SPACE_RE.sub(" ", text).strip()


def detect_language(cleaned: str) -> str:
    """Heuristic: 'hinglish' if enough Hinglish words, 'en' if mostly Latin words, else 'unknown'."""
    words = _WORD_RE.findall(cleaned)
    if not words:
        return "unknown"
    mapping, _ = _hinglish_map()
    hits = sum(1 for w in words if w in HINGLISH_MARKERS or w in mapping)
    phrase_hits = sum(1 for k in mapping if " " in k and k in cleaned)
    hits += phrase_hits
    if hits >= 2 or (hits >= 1 and hits / len(words) >= 0.2):
        return "hinglish"
    latin = sum(1 for w in words if w.isascii())
    if latin / len(words) >= 0.6:
        return "en"
    return "unknown"


def apply_hinglish_map(cleaned: str) -> str:
    mapping, pattern = _hinglish_map()
    return pattern.sub(lambda m: mapping[m.group(1)], cleaned)


def _tokens_from_doc(doc) -> list[str]:
    tokens: list[str] = []
    for tok in doc:
        lemma = tok.lemma_.lower()
        if tok.lower_ in NEGATIONS or lemma in NEGATIONS:
            tokens.append("not" if lemma == "n't" else lemma)
            continue
        if not tok.is_alpha or tok.is_stop or tok.is_punct or tok.lower_ in HINGLISH_MARKERS:
            continue
        tokens.append(lemma)
    return tokens


def preprocess_many(texts: Iterable[str]) -> list[Preprocessed]:
    texts = list(texts)
    cleaned = [basic_clean(t) for t in texts]
    langs = [detect_language(c) for c in cleaned]
    mapped = [apply_hinglish_map(c) for c in cleaned]
    out: list[Preprocessed] = []
    for doc, lang in zip(get_nlp().pipe(mapped, batch_size=64), langs):
        tokens = _tokens_from_doc(doc)
        out.append(Preprocessed(cleaned_text=" ".join(tokens), tokens=tokens, language=lang))
    return out


def preprocess(text: str) -> Preprocessed:
    return preprocess_many([text])[0]

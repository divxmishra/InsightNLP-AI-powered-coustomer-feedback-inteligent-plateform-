"""Lightweight extractive summarization: keyword-frequency + position scoring, no LLM."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.nlp.preprocess import preprocess_many

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
FIRST_SENTENCE_BONUS = 0.25
MIN_TOKENS = 2


@dataclass
class Summary:
    summary: str
    sentences: list[str]
    num_texts: int
    num_sentences: int


@dataclass
class _Candidate:
    text_idx: int
    sent_idx: int
    sentence: str
    tokens: list[str]
    score: float = 0.0


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s and s.strip()]


def summarize(texts: list[str], max_sentences: int = 3) -> Summary:
    """Pick the top sentences by (normalized keyword frequency) x (position bonus).

    Keyword frequency is computed over all lemmatized tokens of all texts, so sentences
    about the most-discussed subjects win. The first sentence of each text gets a bonus.
    Output keeps the original order of the chosen sentences.
    """
    sentences: list[tuple[int, int, str]] = []
    for ti, text in enumerate(texts):
        for si, sent in enumerate(split_sentences(text)):
            sentences.append((ti, si, sent))

    pre = preprocess_many(s for _, _, s in sentences)
    candidates = [
        _Candidate(ti, si, sent, p.tokens)
        for (ti, si, sent), p in zip(sentences, pre)
        if len(p.tokens) >= MIN_TOKENS
    ]
    if not candidates:  # very short input: fall back to every non-empty sentence
        candidates = [_Candidate(ti, si, sent, p.tokens) for (ti, si, sent), p in zip(sentences, pre)]

    freq = Counter(tok for c in candidates for tok in c.tokens)
    top = max(freq.values(), default=1)
    for c in candidates:
        weight = sum(freq[t] / top for t in c.tokens) / math.sqrt(max(len(c.tokens), 1))
        c.score = weight * (1 + FIRST_SENTENCE_BONUS if c.sent_idx == 0 else 1)

    chosen: list[_Candidate] = []
    seen: set[str] = set()
    for c in sorted(candidates, key=lambda c: (-c.score, c.text_idx, c.sent_idx)):
        key = " ".join(c.tokens) or c.sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(c)
        if len(chosen) >= max_sentences:
            break

    chosen.sort(key=lambda c: (c.text_idx, c.sent_idx))
    picked = [c.sentence for c in chosen]
    return Summary(
        summary=" ".join(picked),
        sentences=picked,
        num_texts=len(texts),
        num_sentences=len(sentences),
    )

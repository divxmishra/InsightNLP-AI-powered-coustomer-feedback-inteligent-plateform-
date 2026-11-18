"""Rule-based topic extraction using keyword lists from topics.json."""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.nlp.preprocess import Preprocessed, apply_hinglish_map, basic_clean, preprocess_many


@dataclass
class TopicMatch:
    topic: str
    count: int
    keywords: list[str] = field(default_factory=list)


@lru_cache
def _topic_rules() -> tuple[dict[str, set[str]], dict[str, list[tuple[str, re.Pattern]]]]:
    """Split keywords into single lemmas (matched on tokens) and phrases (matched on text)."""
    raw = json.loads(Path(__file__).with_name("topics.json").read_text(encoding="utf-8"))
    words: dict[str, set[str]] = {}
    phrases: dict[str, list[tuple[str, re.Pattern]]] = {}
    for topic, keywords in raw.items():
        words[topic] = {k.lower() for k in keywords if " " not in k}
        phrases[topic] = [
            (k.lower(), re.compile(r"\b" + re.escape(k.lower()) + r"\b")) for k in keywords if " " in k
        ]
    return words, phrases


def topic_names() -> list[str]:
    return list(_topic_rules()[0])


def _match_one(raw_text: str, pre: Preprocessed) -> dict[str, Counter]:
    words, phrases = _topic_rules()
    text = apply_hinglish_map(basic_clean(raw_text))
    found: dict[str, Counter] = {}
    for topic in words:
        hits: Counter = Counter()
        for tok in pre.tokens:
            if tok in words[topic]:
                hits[tok] += 1
        for phrase, pattern in phrases[topic]:
            n = len(pattern.findall(text))
            if n:
                hits[phrase] += n
        if hits:
            found[topic] = hits
    return found


def _to_matches(found: dict[str, Counter]) -> list[TopicMatch]:
    matches = [TopicMatch(t, sum(c.values()), sorted(c)) for t, c in found.items()]
    return sorted(matches, key=lambda m: (-m.count, m.topic))


def _found_each(texts: list[str], pre: list[Preprocessed] | None = None) -> list[dict[str, Counter]]:
    pre = pre if pre is not None else preprocess_many(texts)
    return [_match_one(raw, p) for raw, p in zip(texts, pre)]


def extract_topics_each(
    texts: list[str], pre: list[Preprocessed] | None = None
) -> list[list[TopicMatch]]:
    """Topics per text, sorted by count (desc). Pass `pre` to reuse preprocessing."""
    return [_to_matches(f) for f in _found_each(texts, pre)]


def extract_topics(texts: list[str], pre: list[Preprocessed] | None = None) -> list[TopicMatch]:
    """Topics aggregated over all texts, sorted by count (desc)."""
    totals: dict[str, Counter] = {}
    for found in _found_each(texts, pre):
        for topic, hits in found.items():
            totals.setdefault(topic, Counter()).update(hits)
    return _to_matches(totals)

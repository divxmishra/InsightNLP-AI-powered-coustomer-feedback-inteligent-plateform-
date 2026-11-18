"""Plain (non-graph) analysis pipeline: preprocess -> sentiment/emotion -> topics -> summary -> save."""

from dataclasses import asdict

from sqlalchemy.orm import Session

from app.db import AnalysisResult, Feedback
from app.nlp.classifiers import analyze_sentiment_emotion
from app.nlp.preprocess import preprocess_many
from app.nlp.summarize import summarize
from app.nlp.topics import extract_topics_each

ITEM_SUMMARY_SENTENCES = 2


def analyze_texts(texts: list[str]) -> list[dict]:
    """Run P2-P5 on each text. Returns one result dict per text (not saved)."""
    pre = preprocess_many(texts)
    preds = analyze_sentiment_emotion(texts)
    topics = extract_topics_each(texts, pre)
    items = []
    for text, p, (sent, emo), tps in zip(texts, pre, preds, topics):
        items.append(
            {
                "text": text,
                "cleaned_text": p.cleaned_text,
                "language": p.language,
                "sentiment": {"label": sent.label, "score": sent.score},
                "emotion": {"label": emo.label, "score": emo.score},
                "topics": [asdict(t) for t in tps],
                "summary": summarize([text], ITEM_SUMMARY_SENTENCES).summary,
            }
        )
    return items


def save_item(db: Session, item: dict) -> int:
    """Persist one analyzed item; returns the new feedback id. Caller commits."""
    fb = Feedback(text=item["text"], cleaned_text=item["cleaned_text"], language=item["language"])
    db.add(fb)
    db.flush()
    db.add(
        AnalysisResult(
            feedback_id=fb.id,
            sentiment=item["sentiment"]["label"],
            sentiment_score=item["sentiment"]["score"],
            emotion=item["emotion"]["label"],
            emotion_score=item["emotion"]["score"],
            topics=item["topics"],
            summary=item["summary"],
        )
    )
    return fb.id


def run_pipeline(texts: list[str], db: Session) -> list[dict]:
    """Analyze and save every text; adds `feedback_id` to each item."""
    items = analyze_texts(texts)
    for item in items:
        item["feedback_id"] = save_item(db, item)
    db.commit()
    return items

from collections import Counter
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.db import AnalysisResult, get_db
from app.nlp.summarize import summarize
from app.routers.pipeline import result_to_out
from app.schemas import StatsOut, TopicCount

router = APIRouter(tags=["dashboard"])

INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"
TOPIC_WINDOW = 1000  # topics are aggregated over the most recent N results
SUMMARY_WINDOW = 50  # summary covers the most recent N reviews
RECENT = 10


@router.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@router.get("/results/stats", response_model=StatsOut)
def stats(db: Session = Depends(get_db)) -> StatsOut:
    total = db.scalar(select(func.count(AnalysisResult.id))) or 0
    sentiment_counts = dict(
        db.execute(select(AnalysisResult.sentiment, func.count()).group_by(AnalysisResult.sentiment)).all()
    )
    emotion_counts = dict(
        db.execute(select(AnalysisResult.emotion, func.count()).group_by(AnalysisResult.emotion)).all()
    )

    latest = db.scalars(
        select(AnalysisResult)
        .options(joinedload(AnalysisResult.feedback))
        .order_by(AnalysisResult.id.desc())
        .limit(TOPIC_WINDOW)
    ).all()
    topic_counts: Counter = Counter()
    for r in latest:
        for t in r.topics or []:
            topic_counts[t["topic"]] += 1  # number of reviews mentioning the topic
    texts = [r.feedback.text for r in latest[:SUMMARY_WINDOW]]

    return StatsOut(
        total=total,
        sentiment_counts=sentiment_counts,
        emotion_counts=emotion_counts,
        top_topics=[TopicCount(topic=t, count=c) for t, c in topic_counts.most_common(10)],
        recent=[result_to_out(r) for r in latest[:RECENT]],
        summary=summarize(list(reversed(texts))).summary if texts else "",
    )

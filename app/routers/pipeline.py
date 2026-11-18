from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.db import AnalysisResult, get_db
from app.nlp.summarize import summarize
from app.pipeline import run_pipeline
from app.schemas import PipelineOut, ResultOut, ResultsOut, TextsIn

router = APIRouter(tags=["pipeline"])


@router.post("/pipeline/run", response_model=PipelineOut)
def pipeline_run(payload: TextsIn, db: Session = Depends(get_db)) -> PipelineOut:
    texts = payload.all_texts()
    items = run_pipeline(texts, db)
    return PipelineOut(items=items, batch_summary=summarize(texts).summary)


def result_to_out(r: AnalysisResult) -> ResultOut:
    return ResultOut(
        id=r.id,
        feedback_id=r.feedback_id,
        text=r.feedback.text,
        cleaned_text=r.feedback.cleaned_text,
        language=r.feedback.language,
        sentiment={"label": r.sentiment, "score": r.sentiment_score},
        emotion={"label": r.emotion, "score": r.emotion_score},
        topics=r.topics or [],
        summary=r.summary,
        created_at=r.created_at,
    )


@router.get("/results", response_model=ResultsOut)
def list_results(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> ResultsOut:
    total = db.scalar(select(func.count(AnalysisResult.id))) or 0
    rows = db.scalars(
        select(AnalysisResult)
        .options(joinedload(AnalysisResult.feedback))
        .order_by(AnalysisResult.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return ResultsOut(total=total, results=[result_to_out(r) for r in rows])

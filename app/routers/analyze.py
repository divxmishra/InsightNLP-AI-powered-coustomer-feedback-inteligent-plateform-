from fastapi import APIRouter

from app.nlp.classifiers import analyze_sentiment_emotion
from app.schemas import FeedbackIn, LabelScore, SentimentEmotionOut

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/sentiment-emotion", response_model=SentimentEmotionOut)
def sentiment_emotion(payload: FeedbackIn) -> SentimentEmotionOut:
    sentiment, emotion = analyze_sentiment_emotion([payload.text])[0]
    return SentimentEmotionOut(
        sentiment=LabelScore(label=sentiment.label, score=sentiment.score),
        emotion=LabelScore(label=emotion.label, score=emotion.score),
    )

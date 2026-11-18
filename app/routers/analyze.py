from dataclasses import asdict

from fastapi import APIRouter

from app.nlp.classifiers import analyze_sentiment_emotion
from app.nlp.topics import extract_topics
from app.schemas import FeedbackIn, LabelScore, SentimentEmotionOut, TextsIn, TopicOut, TopicsOut

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/sentiment-emotion", response_model=SentimentEmotionOut)
def sentiment_emotion(payload: FeedbackIn) -> SentimentEmotionOut:
    sentiment, emotion = analyze_sentiment_emotion([payload.text])[0]
    return SentimentEmotionOut(
        sentiment=LabelScore(label=sentiment.label, score=sentiment.score),
        emotion=LabelScore(label=emotion.label, score=emotion.score),
    )


@router.post("/topics", response_model=TopicsOut)
def topics(payload: TextsIn) -> TopicsOut:
    texts = payload.all_texts()
    return TopicsOut(num_texts=len(texts), topics=[TopicOut(**asdict(m)) for m in extract_topics(texts)])

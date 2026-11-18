from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.db import init_db
from app.nlp.classifiers import get_emotion_model, get_sentiment_model
from app.nlp.preprocess import get_nlp
from app.routers import analyze, feedback, pipeline


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load models once at startup so the first request is not slow.
    get_nlp()
    get_sentiment_model()
    get_emotion_model()
    init_db()  # create tables if missing (no Alembic; see README)
    yield


app = FastAPI(title="InsightNLP", version=__version__, lifespan=lifespan)
app.include_router(feedback.router)
app.include_router(analyze.router)
app.include_router(pipeline.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}

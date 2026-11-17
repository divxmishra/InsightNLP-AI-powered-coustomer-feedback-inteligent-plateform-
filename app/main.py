from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.nlp.preprocess import get_nlp
from app.routers import feedback


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_nlp()  # load spaCy once at startup so the first request is not slow
    yield


app = FastAPI(title="InsightNLP", version=__version__, lifespan=lifespan)
app.include_router(feedback.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}

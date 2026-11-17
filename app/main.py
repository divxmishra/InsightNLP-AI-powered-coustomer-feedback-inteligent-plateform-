from fastapi import FastAPI

from app import __version__
from app.routers import feedback

app = FastAPI(title="InsightNLP", version=__version__)
app.include_router(feedback.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}

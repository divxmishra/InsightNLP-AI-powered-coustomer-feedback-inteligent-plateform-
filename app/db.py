"""Database engine/session. Tables are created on startup (no Alembic; see README)."""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Iterator

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import get_settings


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    cleaned_text: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(16), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    results: Mapped[list["AnalysisResult"]] = relationship(back_populates="feedback")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feedback_id: Mapped[int] = mapped_column(ForeignKey("feedback.id", ondelete="CASCADE"), index=True)
    sentiment: Mapped[str] = mapped_column(String(32))
    sentiment_score: Mapped[float] = mapped_column(Float)
    emotion: Mapped[str] = mapped_column(String(32))
    emotion_score: Mapped[float] = mapped_column(Float)
    topics: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    feedback: Mapped[Feedback] = relationship(back_populates="results")


def normalize_db_url(url: str) -> str:
    """Render gives postgres:// or postgresql:// URLs; use the psycopg (v3) driver."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def make_engine(url: str) -> Engine:
    url = normalize_db_url(url)
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs.update(pool_size=3, max_overflow=2)  # small pool for the free tier
    return create_engine(url, **kwargs)


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_settings().database_url)


@lru_cache
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def init_db(engine: Engine | None = None) -> None:
    Base.metadata.create_all(engine or get_engine())


def get_db() -> Iterator[Session]:
    db = _session_factory()()
    try:
        yield db
    finally:
        db.close()

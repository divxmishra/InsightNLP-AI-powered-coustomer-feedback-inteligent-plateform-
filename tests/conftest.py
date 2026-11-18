"""Point the app at a throwaway SQLite test DB before anything imports app settings."""

import os
import shutil
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="insightnlp-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"


@pytest.fixture(scope="session", autouse=True)
def _test_db():
    from app.db import get_engine, init_db

    init_db()
    yield
    get_engine().dispose()
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture
def clean_db():
    """Empty both tables before a test that counts rows."""
    from app.db import AnalysisResult, Feedback, get_engine
    from sqlalchemy.orm import Session

    with Session(get_engine()) as s:
        s.query(AnalysisResult).delete()
        s.query(Feedback).delete()
        s.commit()

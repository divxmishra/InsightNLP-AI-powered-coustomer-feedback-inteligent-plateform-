from fastapi.testclient import TestClient

from app.db import normalize_db_url
from app.main import app
from tests.fixtures.sample_reviews import REVIEWS

client = TestClient(app)


def test_normalize_db_url():
    assert normalize_db_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_db_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_db_url("sqlite:///x.db") == "sqlite:///x.db"


def test_pipeline_run_saves_and_lists(clean_db):
    r = client.post("/pipeline/run", json={"texts": REVIEWS[:5]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 5
    assert body["batch_summary"]
    first = body["items"][0]
    assert first["feedback_id"] > 0
    assert first["sentiment"]["label"] == "negative"  # "Delivery was very late..."
    assert "delivery" in {t["topic"] for t in first["topics"]}
    assert first["summary"]
    assert body["items"][4]["language"] == "hinglish"

    r = client.get("/results?limit=3")
    assert r.status_code == 200
    listed = r.json()
    assert listed["total"] == 5
    assert len(listed["results"]) == 3
    newest = listed["results"][0]
    assert newest["text"] == REVIEWS[4]  # newest first
    assert newest["sentiment"]["label"] == "positive"
    assert newest["created_at"]


def test_pipeline_single_text_and_validation(clean_db):
    r = client.post("/pipeline/run", json={"text": "Refund never came. Terrible support."})
    assert r.status_code == 200
    assert r.json()["items"][0]["sentiment"]["label"] == "negative"
    assert client.post("/pipeline/run", json={"texts": []}).status_code == 422
    assert client.get("/results").json()["total"] == 1


def test_results_pagination(clean_db):
    client.post("/pipeline/run", json={"texts": ["good", "bad", "okay product"]})
    page = client.get("/results?limit=2&offset=2").json()
    assert page["total"] == 3
    assert [r["text"] for r in page["results"]] == ["good"]
    assert client.get("/results?limit=0").status_code == 422

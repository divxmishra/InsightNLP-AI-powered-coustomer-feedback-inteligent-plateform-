from fastapi.testclient import TestClient

from app.main import app
from tests.fixtures.sample_reviews import REVIEWS

client = TestClient(app)


def test_dashboard_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "InsightNLP" in r.text
    assert "/results/stats" in r.text and "/pipeline/run" in r.text
    assert "chart.js" in r.text.lower()


def test_stats_empty(clean_db):
    s = client.get("/results/stats").json()
    assert s == {
        "total": 0,
        "sentiment_counts": {},
        "emotion_counts": {},
        "top_topics": [],
        "recent": [],
        "summary": "",
    }


def test_stats_after_pipeline_run(clean_db):
    assert client.post("/pipeline/run", json={"texts": REVIEWS}).status_code == 200
    s = client.get("/results/stats").json()
    assert s["total"] == 25
    assert sum(s["sentiment_counts"].values()) == 25
    assert set(s["sentiment_counts"]) <= {"positive", "negative"}
    assert s["sentiment_counts"]["negative"] > 5 and s["sentiment_counts"]["positive"] > 5
    assert sum(s["emotion_counts"].values()) == 25
    assert s["top_topics"][0]["topic"] == "delivery"
    assert len(s["recent"]) == 10
    assert s["recent"][0]["text"] == REVIEWS[-1]
    assert s["summary"]

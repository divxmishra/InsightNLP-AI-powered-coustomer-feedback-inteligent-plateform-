from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "0.1.0"}


def test_feedback_ok():
    r = client.post("/feedback", json={"text": "  Great product!  "})
    assert r.status_code == 200
    assert r.json() == {"text": "Great product!", "length": 14}


def test_feedback_empty_rejected():
    assert client.post("/feedback", json={"text": ""}).status_code == 422
    assert client.post("/feedback", json={"text": "   "}).status_code == 422


def test_feedback_too_long_rejected():
    assert client.post("/feedback", json={"text": "a" * 2001}).status_code == 422
    assert client.post("/feedback", json={"text": "a" * 2000}).status_code == 200


def _upload(content: str):
    return client.post(
        "/feedback/upload", files={"file": ("f.csv", content.encode(), "text/csv")}
    )


def test_upload_with_id_and_empty_rows():
    r = _upload("id,text\n1,Good service\n2,\n3,   \n4,Bahut achha\n")
    assert r.status_code == 200
    body = r.json()
    assert body["total_rows"] == 4
    assert body["accepted"] == 2
    assert body["skipped_empty"] == 2
    assert body["rows"][0] == {"id": "1", "text": "Good service", "cleaned_text": "good service"}


def test_upload_text_only():
    r = _upload("text\nNice\n")
    assert r.status_code == 200
    assert r.json()["rows"] == [{"id": None, "text": "Nice", "cleaned_text": "nice"}]


def test_upload_missing_text_column():
    r = _upload("id,review\n1,hello\n")
    assert r.status_code == 400
    assert "text" in r.json()["detail"]


def test_upload_empty_file():
    assert _upload("").status_code == 400

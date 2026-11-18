from fastapi.testclient import TestClient

from app.main import app
from app.nlp.topics import extract_topics, extract_topics_each, topic_names

client = TestClient(app)


def _by_topic(matches):
    return {m.topic: m for m in matches}


def test_topic_list_loaded():
    assert {"delivery", "pricing", "quality", "support"} <= set(topic_names())


def test_single_review_topics():
    found = _by_topic(extract_topics(["Delivery was late and the price is too expensive"]))
    assert found["delivery"].count == 2  # delivery, late
    assert found["pricing"].count == 2  # price, expensive
    assert "quality" not in found


def test_lemmas_and_phrases_match():
    found = _by_topic(extract_topics(["Packages were delivered damaged. Customer service never responded."]))
    assert "deliver" in found["delivery"].keywords
    assert "package" in found["packaging"].keywords
    assert "damaged" in found["quality"].keywords
    assert "customer service" in found["support"].keywords


def test_hinglish_phrase_maps_to_pricing():
    found = _by_topic(extract_topics(["Product paisa vasool hai"]))
    assert "value for money" in found["pricing"].keywords


def test_aggregate_counts_and_sorting():
    texts = ["late delivery", "delivery delayed again", "good quality"]
    matches = extract_topics(texts)
    assert matches[0].topic == "delivery"
    assert matches[0].count == 4
    per_text = extract_topics_each(texts)
    assert [m.topic for m in per_text[2]] == ["quality"]


def test_no_topics():
    assert extract_topics(["hello there"]) == []


def test_topics_endpoint_single_and_batch():
    r = client.post("/analyze/topics", json={"text": "Support team was rude, refund still pending"})
    assert r.status_code == 200
    topics = {t["topic"]: t for t in r.json()["topics"]}
    assert {"support", "returns_refunds"} <= set(topics)

    r = client.post("/analyze/topics", json={"texts": ["late delivery", "bad packaging box"]})
    assert r.status_code == 200
    body = r.json()
    assert body["num_texts"] == 2
    assert {t["topic"] for t in body["topics"]} == {"delivery", "packaging"}


def test_topics_endpoint_validation():
    assert client.post("/analyze/topics", json={}).status_code == 422
    assert client.post("/analyze/topics", json={"texts": []}).status_code == 422
    assert client.post("/analyze/topics", json={"texts": ["ok", "  "]}).status_code == 422

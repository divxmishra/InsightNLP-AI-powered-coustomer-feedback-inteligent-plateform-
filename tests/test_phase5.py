from fastapi.testclient import TestClient

from app.main import app
from app.nlp.summarize import split_sentences, summarize
from tests.fixtures.sample_reviews import REVIEWS

client = TestClient(app)


def test_split_sentences():
    assert split_sentences("Good. Bad!  Ok?\nFine") == ["Good.", "Bad!", "Ok?", "Fine"]
    assert split_sentences("") == []


def test_summary_of_batch_picks_dominant_theme():
    s = summarize(REVIEWS, max_sentences=3)
    assert s.num_texts == 25
    assert len(s.sentences) == 3
    all_sentences = {x for r in REVIEWS for x in split_sentences(r)}
    assert set(s.sentences) <= all_sentences
    # Delivery is the most discussed subject in the sample, so it must lead the summary.
    assert sum("delivery" in x.lower() for x in s.sentences) >= 2


def test_summary_keeps_original_order_and_dedupes():
    text = "The phone is great. Battery lasts long. The phone is great! Camera is great too."
    s = summarize([text], max_sentences=2)
    assert s.sentences[0] == "The phone is great."
    assert "The phone is great!" not in s.sentences


def test_short_input_falls_back():
    assert summarize(["Short one."]).sentences == ["Short one."]


def test_summary_endpoint_batch_and_single():
    r = client.post("/analyze/summary", json={"texts": REVIEWS, "max_sentences": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["num_texts"] == 25
    assert len(body["sentences"]) == 2
    assert body["summary"] == " ".join(body["sentences"])

    r = client.post("/analyze/summary", json={"text": "Delivery was late. The box was damaged. Support helped."})
    assert r.status_code == 200
    assert r.json()["num_texts"] == 1


def test_summary_endpoint_validation():
    assert client.post("/analyze/summary", json={"texts": REVIEWS, "max_sentences": 0}).status_code == 422
    assert client.post("/analyze/summary", json={}).status_code == 422

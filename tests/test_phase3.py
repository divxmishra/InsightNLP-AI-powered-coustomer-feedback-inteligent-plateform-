from fastapi.testclient import TestClient

from app.main import app
from app.nlp.classifiers import analyze_sentiment_emotion, model_input

client = TestClient(app)


def test_model_input_maps_hinglish_and_keeps_negation():
    assert model_input("Product bahut BAKWAS, not good!!!!") == "product very bad, not good!!"


def test_positive_english_review():
    r = client.post("/analyze/sentiment-emotion", json={"text": "I love this phone, it works perfectly!"})
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"]["label"] == "positive"
    assert body["emotion"]["label"] in {"joy", "love"}
    assert 0.5 <= body["sentiment"]["score"] <= 1


def test_negative_english_review():
    r = client.post(
        "/analyze/sentiment-emotion",
        json={"text": "Terrible service. The delivery was late and the support staff were rude."},
    )
    body = r.json()
    assert body["sentiment"]["label"] == "negative"
    assert body["emotion"]["label"] in {"anger", "sadness", "fear"}


def test_negation_flips_sentiment():
    (pos, _), (neg, _) = analyze_sentiment_emotion(["The quality is good", "The quality is not good"])
    assert pos.label == "positive"
    assert neg.label == "negative"


def test_hinglish_negative():
    (sent, _), = analyze_sentiment_emotion(["Product bakwas hai, paisa barbaad"])
    assert sent.label == "negative"


def test_batch_and_long_text():
    out = analyze_sentiment_emotion(["great " * 1000, "awful"])
    assert len(out) == 2


def test_rejects_empty():
    assert client.post("/analyze/sentiment-emotion", json={"text": ""}).status_code == 422

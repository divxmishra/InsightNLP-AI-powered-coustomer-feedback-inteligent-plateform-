from fastapi.testclient import TestClient

from app.main import app
from app.nlp.preprocess import basic_clean, detect_language, preprocess

client = TestClient(app)


def test_removes_urls_emails_html():
    r = preprocess("<b>Visit</b> https://shop.com or www.x.in, mail me at a.b@c.com now")
    assert "http" not in r.cleaned_text
    assert "www" not in r.cleaned_text
    assert "com" not in r.tokens
    assert "b" not in r.tokens
    assert "visit" in r.tokens


def test_collapses_repeated_chars_and_spaces():
    assert basic_clean("Sooooo    GOOOOD!!!") == "soo good!!"


def test_lemmatizes_and_drops_stopwords():
    r = preprocess("The delivery boys were delivering packages")
    assert r.tokens == ["delivery", "boy", "deliver", "package"]


def test_keeps_negations():
    r = preprocess("I don't like it. It is not good, no and never again")
    assert r.tokens.count("not") == 2
    assert "no" in r.tokens
    assert "never" in r.tokens


def test_hinglish_mapping_and_language():
    r = preprocess("Product bahut achha hai, paisa vasool!")
    assert r.language == "hinglish"
    assert r.tokens == ["product", "good", "value", "money"]
    r = preprocess("Delivery bakwas thi, kaam nahi kiya")
    assert r.language == "hinglish"
    assert "bad" in r.tokens and "not" in r.tokens


def test_english_language():
    assert preprocess("The quality is great and support was helpful").language == "en"


def test_emoji_removed():
    r = preprocess("Love it 😍😍 🔥")
    assert r.tokens == ["love"]
    assert r.language == "en"


def test_empty_and_symbol_only_input():
    assert preprocess("").tokens == []
    r = preprocess("😀 !!! https://x.com")
    assert r.tokens == [] and r.cleaned_text == "" and r.language == "unknown"
    assert detect_language("") == "unknown"


def test_preprocess_endpoint():
    r = client.post("/feedback/preprocess", json={"text": "Not worth it, bekar product"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"cleaned_text", "tokens", "language"}
    assert body["tokens"][0] == "not"
    assert "bad" in body["tokens"]


def test_preprocess_endpoint_rejects_empty():
    assert client.post("/feedback/preprocess", json={"text": "   "}).status_code == 422


def test_upload_adds_cleaned_text():
    csv = "id,text\n1,Delivery was sooo late!!\n2,\n3,bahut achha\n"
    r = client.post("/feedback/upload", files={"file": ("f.csv", csv, "text/csv")})
    assert r.status_code == 200
    rows = r.json()["rows"]
    assert rows[0]["cleaned_text"] == "delivery soo late"
    assert rows[1]["cleaned_text"] == "good"  # "very" is a spaCy stopword

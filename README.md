# InsightNLP

Analyzes English + Hinglish customer feedback (sentiment, emotion, topics, summary).
Built in phases; see `PROGRESS.md` for status.

## Phase 1: Setup and feedback input

### Endpoints
| Method | Path | Description |
|---|---|---|
| GET | `/health` | `{"status":"ok","version":"0.1.0"}` |
| POST | `/feedback` | JSON `{"text": "..."}` (1 to 2000 chars) |
| POST | `/feedback/upload` | CSV file (multipart field `file`) with required `text` column, optional `id`. Missing `text` column gives 400. Empty rows are skipped and counted. |

## Phase 2: Preprocessing (spaCy + Hinglish)

`POST /feedback/preprocess` with `{"text": "..."}` returns `cleaned_text`, `tokens`, `language` (`en`, `hinglish`, `unknown`).
`POST /feedback/upload` now also returns `cleaned_text` per row.

Pipeline (`app/nlp/preprocess.py`):
1. Lowercase; remove HTML tags, URLs, emails; collapse 3+ repeated characters to 2 (`sooooo` to `soo`); squeeze spaces.
2. Detect language: Hinglish words (from `app/nlp/hinglish_map.json` plus common function words like `hai`, `ka`) mean `hinglish`; mostly Latin words mean `en`; otherwise `unknown`.
3. Apply the Hinglish map (`bahut`→`very`, `bakwas`→`bad`, `paisa vasool`→`value for money`, ...).
4. spaCy `en_core_web_sm` (parser and NER disabled, loaded once at startup) lemmatizes; stopwords, punctuation, numbers and emoji are dropped, but negations (`not`, `no`, `never`, `n't`→`not`) are kept.

```bash
curl -X POST localhost:8000/feedback/preprocess -H "Content-Type: application/json" \
  -d '{"text":"Delivery bahut late thi, product bakwas! Not worth it"}'
# {"cleaned_text":"delivery late product bad not worth","tokens":[...],"language":"hinglish"}
```

## Phase 3: Sentiment and emotion

`POST /analyze/sentiment-emotion` with `{"text": "..."}` returns
`{"sentiment": {"label": "positive|negative", "score": 0.99}, "emotion": {"label": "joy|love|anger|fear|sadness|surprise", "score": 0.95}}`.

- Models: `distilbert-base-uncased-finetuned-sst-2-english` (sentiment) and `bhadresh-savani/distilbert-base-uncased-emotion` (emotion).
- **Memory:** PyTorch + both models needs ~700-900 MB, over Render's 512 MB free tier. So the Docker build has an
  export stage (`scripts/export_models.py`, deps in `requirements-export.txt`) that converts both models to
  **int8-quantized ONNX** (~65 MB each). The runtime image only has `onnxruntime` + `tokenizers` (no PyTorch).
- Models load once at startup. Input is lightly cleaned (lowercase, URLs/HTML removed, Hinglish words mapped to
  English) but keeps stopwords and negations, which the models need.

Local setup for the models (one time):
```bash
pip install -r requirements-export.txt
python scripts/export_models.py models   # writes models/sentiment and models/emotion (git-ignored)
```

## Phase 4: Topic extraction

`POST /analyze/topics` with `{"text": "..."}` or `{"texts": ["...", "..."]}` (up to 1000) returns
`{"num_texts": 2, "topics": [{"topic": "delivery", "count": 3, "keywords": ["delivery", "late"]}, ...]}`, sorted by count.

- Rule-based: keyword lists per topic in `app/nlp/topics.json` (delivery, pricing, quality, support, packaging,
  returns_refunds, payment, app_experience, performance, size_fit). Edit the JSON to add topics.
- Single-word keywords match the spaCy lemmas from Phase 2 (so "delivered" matches "deliver"); multi-word
  keywords ("customer service", "value for money") match the cleaned, Hinglish-mapped text.
- `count` is the number of keyword hits; a phrase and a word inside it can both count.

## Phase 5: Summarization

`POST /analyze/summary` with `{"text": "..."}` or `{"texts": [...], "max_sentences": 3}` (1 to 10) returns
`{"summary": "...", "sentences": [...], "num_texts": 25, "num_sentences": 35}`.

Extractive and lightweight (`app/nlp/summarize.py`, no LLM): texts are split into sentences, each sentence is
scored by the frequency of its lemmatized keywords across the whole batch (normalized by sqrt of its length),
the first sentence of each text gets a 25% bonus, duplicates are dropped, and the top sentences are returned in
their original order.

## Phase 6: PostgreSQL storage

- `POST /pipeline/run` with `{"text": "..."}` or `{"texts": [...]}` runs preprocess, sentiment, emotion, topics and a
  per-review summary, saves each review, and returns `{"items": [...], "batch_summary": "..."}`.
- `GET /results?limit=50&offset=0` lists saved results, newest first: `{"total": N, "results": [...]}`.
- Tables (SQLAlchemy, `app/db.py`): `feedback` (id, text, cleaned_text, language, created_at) and
  `analysis_results` (id, feedback_id FK, sentiment, sentiment_score, emotion, emotion_score, topics JSON, summary,
  created_at).
- **Schema management: create-tables-on-startup** (`Base.metadata.create_all` in the app lifespan), not Alembic.
  It is simple and enough for a two-table schema; a column change needs a manual migration or a table drop.
- `DATABASE_URL`: on Render, set it to the Postgres **internal** connection string (`postgres://` is rewritten to
  the `postgresql+psycopg://` driver). If it is not set, the app uses a local SQLite file (`insightnlp.db`), which
  on Render is wiped on every deploy/restart.
- Tests use a throwaway SQLite DB (`tests/conftest.py`).

### Run locally
```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
pytest
```

### Docker
```bash
docker build -t insightnlp .
docker run -p 8000:8000 insightnlp
curl http://localhost:8000/health
```

### Examples
```bash
curl -X POST localhost:8000/feedback -H "Content-Type: application/json" -d '{"text":"Great product"}'
curl -X POST localhost:8000/feedback/upload -F "file=@reviews.csv"
```

### Config (env vars)
- `APP_ENV` (default `development`)
- `MAX_UPLOAD_ROWS` (default `5000`)

### Deploy
Render web service, Docker runtime, free plan, health check path `/health`. Auto-deploys from `main`.

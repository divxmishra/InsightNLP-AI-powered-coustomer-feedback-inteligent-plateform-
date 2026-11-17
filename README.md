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

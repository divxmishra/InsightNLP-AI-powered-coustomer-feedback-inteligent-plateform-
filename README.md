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

import io

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.schemas import FeedbackIn, FeedbackOut, UploadOut, UploadRow

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut)
def submit_feedback(payload: FeedbackIn) -> FeedbackOut:
    text = payload.text.strip()
    return FeedbackOut(text=text, length=len(text))


@router.post("/upload", response_model=UploadOut)
async def upload_feedback(
    file: UploadFile = File(...), settings: Settings = Depends(get_settings)
) -> UploadOut:
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid or empty CSV file")

    df.columns = [str(c).strip().lower() for c in df.columns]
    if "text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'text' column")
    if len(df) > settings.max_upload_rows:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {len(df)} rows; limit is {settings.max_upload_rows}",
        )

    has_id = "id" in df.columns
    rows: list[UploadRow] = []
    skipped = 0
    for _, r in df.iterrows():
        text = str(r["text"]).strip()
        if not text:
            skipped += 1
            continue
        row_id = str(r["id"]).strip() if has_id else None
        rows.append(UploadRow(id=row_id or None, text=text[:2000]))

    return UploadOut(
        total_rows=len(df), accepted=len(rows), skipped_empty=skipped, rows=rows
    )

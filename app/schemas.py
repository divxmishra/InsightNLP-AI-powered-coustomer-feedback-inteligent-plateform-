from pydantic import BaseModel, Field, field_validator


class FeedbackIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class FeedbackOut(BaseModel):
    text: str
    length: int


class UploadRow(BaseModel):
    id: str | None = None
    text: str


class UploadOut(BaseModel):
    total_rows: int
    accepted: int
    skipped_empty: int
    rows: list[UploadRow]

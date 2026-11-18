from pydantic import BaseModel, Field, field_validator, model_validator


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
    cleaned_text: str = ""


class UploadOut(BaseModel):
    total_rows: int
    accepted: int
    skipped_empty: int
    rows: list[UploadRow]


class PreprocessOut(BaseModel):
    cleaned_text: str
    tokens: list[str]
    language: str


class LabelScore(BaseModel):
    label: str
    score: float


class SentimentEmotionOut(BaseModel):
    sentiment: LabelScore
    emotion: LabelScore


MAX_BATCH = 1000


class TextsIn(BaseModel):
    """Either a single `text` or a list of `texts` (each 1 to 2000 chars)."""

    text: str | None = Field(None, min_length=1, max_length=2000)
    texts: list[str] | None = Field(None, min_length=1, max_length=MAX_BATCH)

    @model_validator(mode="after")
    def check(self) -> "TextsIn":
        items = self.all_texts()
        if not items:
            raise ValueError("provide 'text' or a non-empty 'texts' list")
        for t in items:
            if not t.strip() or len(t) > 2000:
                raise ValueError("each text must be 1 to 2000 non-blank characters")
        return self

    def all_texts(self) -> list[str]:
        return ([self.text] if self.text is not None else []) + list(self.texts or [])


class TopicOut(BaseModel):
    topic: str
    count: int
    keywords: list[str]


class TopicsOut(BaseModel):
    num_texts: int
    topics: list[TopicOut]


class SummaryIn(TextsIn):
    max_sentences: int = Field(3, ge=1, le=10)


class SummaryOut(BaseModel):
    summary: str
    sentences: list[str]
    num_texts: int
    num_sentences: int

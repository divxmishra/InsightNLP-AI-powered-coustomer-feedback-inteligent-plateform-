"""Sentiment and emotion classification with int8 ONNX models (no PyTorch at runtime)."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app.config import get_settings
from app.nlp.preprocess import apply_hinglish_map, basic_clean

MAX_TOKENS = 256
BATCH_SIZE = 8  # small inference batches keep peak memory low on a 512 MB host


@dataclass
class Prediction:
    label: str
    score: float


class OnnxClassifier:
    def __init__(self, model_dir: Path):
        meta = json.loads((model_dir / "labels.json").read_text())
        self.labels: list[str] = meta["labels"]
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False  # lower steady-state memory on a 512 MB host
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )

    def predict(self, texts: list[str]) -> list[Prediction]:
        out: list[Prediction] = []
        for i in range(0, len(texts), BATCH_SIZE):
            out.extend(self._predict_batch(texts[i : i + BATCH_SIZE]))
        return out

    def _predict_batch(self, texts: list[str]) -> list[Prediction]:
        if not texts:
            return []
        enc = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        logits = self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        best = probs.argmax(axis=1)
        return [Prediction(self.labels[i], round(float(probs[r, i]), 4)) for r, i in enumerate(best)]


@lru_cache
def get_sentiment_model() -> OnnxClassifier:
    return OnnxClassifier(Path(get_settings().models_dir) / "sentiment")


@lru_cache
def get_emotion_model() -> OnnxClassifier:
    return OnnxClassifier(Path(get_settings().models_dir) / "emotion")


def model_input(text: str) -> str:
    """Lightly cleaned text: keeps stopwords/negations; Hinglish words mapped to English."""
    return apply_hinglish_map(basic_clean(text)) or text


def analyze_sentiment_emotion(texts: list[str]) -> list[tuple[Prediction, Prediction]]:
    inputs = [model_input(t) for t in texts]
    return list(zip(get_sentiment_model().predict(inputs), get_emotion_model().predict(inputs)))

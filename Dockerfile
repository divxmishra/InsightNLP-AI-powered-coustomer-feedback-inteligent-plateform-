# Stage 1: export Hugging Face models to int8 ONNX (PyTorch lives only in this stage).
FROM python:3.11-slim AS models
WORKDIR /build
COPY requirements-export.txt .
RUN pip install --no-cache-dir -r requirements-export.txt
COPY scripts/export_models.py .
RUN python export_models.py /models && rm -rf /root/.cache/huggingface

# Stage 2: runtime image (onnxruntime + tokenizers, no PyTorch).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 MODELS_DIR=/app/models
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=models /models ./models
COPY app ./app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

"""Export Hugging Face classifiers to int8-quantized ONNX (run at Docker build time).

Usage: python scripts/export_models.py [output_dir]
Needs requirements-export.txt (torch, transformers, onnx, onnxruntime). The runtime
image only needs onnxruntime + tokenizers, which keeps RAM well under 512 MB.
"""

import json
import sys
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS = {
    "sentiment": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "emotion": "bhadresh-savani/distilbert-base-uncased-emotion",
}


def export(name: str, repo: str, out_root: Path) -> None:
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

    sample = tokenizer(["export sample"], return_tensors="pt")
    fp32 = out / "model_fp32.onnx"
    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(str(fp32), str(out / "model.onnx"), weight_type=QuantType.QInt8)
    fp32.unlink()

    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    (out / "labels.json").write_text(json.dumps({"repo": repo, "labels": labels}, indent=2))
    print(f"{name}: {repo} -> {out} labels={labels}")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
    for model_name, repo_id in MODELS.items():
        export(model_name, repo_id, root)

"""Area D - post-training dynamic INT8 quantization of the fine-tuned BERTurk.

Every nn.Linear layer (Q/K/V/output projections, feed-forward layers, classifier head)
gets int8 weights; activations are quantized dynamically at run time. Embeddings and
LayerNorm stay FP32. No re-training is needed. Runs on CPU.

Usage:  python -m src.quantize_bert
"""
import json
import shutil

import torch

from src import config as C
from src.models import BertPredictor, dir_size_mb, quantize
from src.prepare_data import load_split
from src.utils import tune_threshold


def main():
    src_dir, out = C.MODEL_DIRS["bert_fp32"], C.MODEL_DIRS["bert_int8"]
    fp32 = BertPredictor("bert_fp32")
    q = quantize(fp32.model)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(q.state_dict(), out / "model_int8.pt")
    fp32.model.config.save_pretrained(out)
    fp32.tok.save_pretrained(out)
    shutil.copy(src_dir / "threshold.json", out / "threshold.json")   # temporary

    # the quantized model gets its own threshold, tuned on validation like every model
    int8 = BertPredictor("bert_int8")
    val = load_split("val")
    p = int8.predict_proba(val["tweet"].tolist())
    thr, f1 = tune_threshold(val["y"], p)
    (out / "threshold.json").write_text(json.dumps({"threshold": thr}))
    (out / "tuning.json").write_text(json.dumps({"method": "torch dynamic quantization, qint8, nn.Linear",
                                                 "val_macro_f1": round(f1, 4)}))
    print(f"FP32 dir: {dir_size_mb(src_dir):.1f} MB  ->  INT8 dir: {dir_size_mb(out):.1f} MB")
    print(f"saved BERTurk INT8 (thr={thr:.2f}, val macro-F1={f1:.4f})")


if __name__ == "__main__":
    main()

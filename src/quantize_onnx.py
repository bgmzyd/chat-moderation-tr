"""Area D - ONNX Runtime INT8: export the fine-tuned BERTurk to ONNX, fuse the transformer
graph (attention / LayerNorm / GELU fusion) with the ONNX Runtime transformer optimizer, then
apply dynamic INT8 quantization (int8 weights, activations quantized at run time).

Why a second quantized variant: PyTorch dynamic INT8 kept F1 but missed the 50 ms CPU
budget; ONNX Runtime executes the same INT8 idea with fused kernels on CPU.

Usage:  python -m src.quantize_onnx                                        (BERTurk)
        python -m src.quantize_onnx --src distil_fp32 --dst distil_onnx_int8  (distilled student)
"""
import argparse
import json
import shutil

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config as C
from src.models import OnnxBertPredictor, dir_size_mb
from src.prepare_data import load_split
from src.utils import tune_threshold

class _Logits(torch.nn.Module):
    """Positional inputs -> keyword call -> logits, so ONNX export works for BERT and DistilBERT."""

    def __init__(self, model, names):
        super().__init__()
        self.model, self.names = model, names

    def forward(self, *xs):
        return self.model(**dict(zip(self.names, xs))).logits


def export(src, path):
    model = AutoModelForSequenceClassification.from_pretrained(
        src, attn_implementation="eager").eval()
    tok = AutoTokenizer.from_pretrained(src)
    enc = tok(["örnek bir mesaj", "ikinci örnek mesaj biraz daha uzun"],
              return_tensors="pt", padding=True)
    import inspect
    accepted = inspect.signature(model.forward).parameters
    inputs = [k for k in ("input_ids", "attention_mask", "token_type_ids")
              if k in enc and k in accepted]
    axes = {k: {0: "batch", 1: "seq"} for k in inputs} | {"logits": {0: "batch"}}
    kw = dict(input_names=inputs, output_names=["logits"], dynamic_axes=axes, opset_version=17)
    args = tuple(enc[k] for k in inputs)
    with torch.no_grad():
        try:
            torch.onnx.export(_Logits(model, inputs), args, str(path), dynamo=False, **kw)
        except TypeError:          # older torch without the `dynamo` argument
            torch.onnx.export(_Logits(model, inputs), args, str(path), **kw)
    return model.config, tok


def main():
    from onnxruntime.quantization import QuantType, quantize_dynamic
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="bert_fp32")
    ap.add_argument("--dst", default="bert_onnx_int8")
    a = ap.parse_args()
    src, out = C.MODEL_DIRS[a.src], C.MODEL_DIRS[a.dst]
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()

    cfg, tok = export(src, tmp / "model_fp32.onnx")
    to_quantize = tmp / "model_fp32.onnx"
    try:
        from onnxruntime.transformers.optimizer import optimize_model
        opt = optimize_model(str(to_quantize), model_type="bert",
                             num_heads=cfg.num_attention_heads, hidden_size=cfg.hidden_size)
        opt.save_model_to_file(str(tmp / "model_fp32_opt.onnx"))
        to_quantize = tmp / "model_fp32_opt.onnx"
        print("graph fusion: ok")
    except Exception as e:  # fusion is an optimisation only; quantize the plain graph instead
        print("graph fusion skipped:", repr(e)[:200])
    import onnx
    quantize_dynamic(str(to_quantize), str(out / "model_int8.onnx"), weight_type=QuantType.QInt8,
                     extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
    shutil.rmtree(tmp)
    cfg.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "threshold.json").write_text(json.dumps({"threshold": 0.5}))   # temporary

    int8 = OnnxBertPredictor(a.dst)
    val = load_split("val")
    p = int8.predict_proba(val["tweet"].tolist())
    thr, f1 = tune_threshold(val["y"], p)
    (out / "threshold.json").write_text(json.dumps({"threshold": thr}))
    (out / "tuning.json").write_text(json.dumps(
        {"method": "ONNX export (opset 17) + ORT transformer fusion + ORT dynamic quantization, QInt8",
         "val_macro_f1": round(f1, 4)}))
    print(f"FP32 dir: {dir_size_mb(src):.1f} MB  ->  ONNX INT8 dir: {dir_size_mb(out):.1f} MB")
    print(f"saved {a.dst} (thr={thr:.2f}, val macro-F1={f1:.4f})")


if __name__ == "__main__":
    main()

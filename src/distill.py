"""Area D (optional technique) - knowledge distillation.

Teacher : our fine-tuned BERTurk (12 layers, models/bert_fp32)
Student : DistilBERTurk (dbmdz/distilbert-base-turkish-cased, 6 layers), fine-tuned on the
          training split with   loss = a * CE(student, gold label)
                                     + (1 - a) * T^2 * KL(student/T || teacher/T)
The student is then quantized to INT8 with ONNX Runtime (src/quantize_onnx.py), which is the
model the application runs. Why: on a 2-vCPU server the 12-layer INT8 model still took ~71 ms.

Usage:  python -m src.distill        (GPU recommended; also runs on Apple-silicon "mps" or CPU)
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src import config as C
from src.prepare_data import load_split
from src.train_bert import bert_text, predict_proba
from src.utils import model_inputs, set_seed, tune_threshold


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def teacher_logits(model, tok, texts, device, batch=128):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok([bert_text(t) for t in texts[i:i + batch]], truncation=True,
                  max_length=C.BERT_MAX_LEN, padding=True, return_tensors="pt").to(device)
        out.append(model(**model_inputs(model, enc)).logits.float().cpu())
    return torch.cat(out)


def main():
    set_seed(C.SEED)
    device = pick_device()
    print("device:", device, "| teacher:", C.MODEL_DIRS["bert_fp32"], "| student:", C.STUDENT_BASE)
    train, val = load_split("train"), load_split("val")
    texts, ys = train["tweet"].tolist(), train["y"].tolist()

    t0 = time.time()
    t_tok = AutoTokenizer.from_pretrained(C.MODEL_DIRS["bert_fp32"])
    teacher = AutoModelForSequenceClassification.from_pretrained(C.MODEL_DIRS["bert_fp32"]).to(device)
    t_logits = teacher_logits(teacher, t_tok, texts, device)
    del teacher
    print(f"teacher logits for {len(texts)} training messages in {time.time() - t0:.0f}s")

    tok = AutoTokenizer.from_pretrained(C.STUDENT_BASE)
    student = AutoModelForSequenceClassification.from_pretrained(
        C.STUDENT_BASE, num_labels=2, id2label={0: "NOT", 1: "OFF"},
        label2id={"NOT": 0, "OFF": 1}).to(device)

    items = list(zip(texts, ys, t_logits))

    def collate(batch):
        enc = tok([bert_text(t) for t, _, _ in batch], truncation=True,
                  max_length=C.BERT_MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y, _ in batch])
        enc["teacher"] = torch.stack([z for _, _, z in batch])
        return enc

    loader = DataLoader(items, batch_size=C.BERT_BATCH, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(C.SEED))
    opt = torch.optim.AdamW(student.parameters(), lr=C.DISTIL_LR, weight_decay=0.01)
    steps = len(loader) * C.DISTIL_EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    T, a = C.DISTIL_T, C.DISTIL_ALPHA

    out = C.MODEL_DIRS["distil_fp32"]
    best_f1, log = -1.0, []
    for epoch in range(1, C.DISTIL_EPOCHS + 1):
        student.train()
        t0, total = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            labels, z_t = batch.pop("labels").to(device), batch.pop("teacher").to(device)
            batch = {k: v.to(device) for k, v in model_inputs(student, batch).items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                z_s = student(**batch).logits.float()
            hard = F.cross_entropy(z_s, labels)
            soft = F.kl_div(F.log_softmax(z_s / T, -1), F.softmax(z_t / T, -1),
                            reduction="batchmean") * T * T
            loss = a * hard + (1 - a) * soft
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss={total / step:.4f}")
        p = predict_proba(student, tok, val["tweet"].tolist(), device)
        f1 = f1_score(val["y"], (p >= 0.5).astype(int), average="macro")
        log.append({"epoch": epoch, "train_loss": round(total / len(loader), 4),
                    "val_macro_f1@0.5": round(f1, 4), "minutes": round((time.time() - t0) / 60, 1)})
        print(log[-1])
        if f1 > best_f1:
            best_f1 = f1
            student.save_pretrained(out, safe_serialization=True)
            tok.save_pretrained(out)

    student = AutoModelForSequenceClassification.from_pretrained(out).to(device)
    p = predict_proba(student, tok, val["tweet"].tolist(), device)
    thr, f1_t = tune_threshold(val["y"], p)
    (out / "threshold.json").write_text(json.dumps({"threshold": thr}))
    (out / "tuning.json").write_text(json.dumps(
        {"teacher": "models/bert_fp32", "student_base": C.STUDENT_BASE, "temperature": T,
         "alpha_hard_label": a, "epochs": log, "val_macro_f1": round(f1_t, 4)}, indent=1))
    print(f"saved distilled student FP32 (thr={thr:.2f}, val macro-F1={f1_t:.4f})")


if __name__ == "__main__":
    main()

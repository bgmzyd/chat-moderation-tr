"""Area C - fine-tune BERTurk (dbmdz/bert-base-turkish-cased, WordPiece tokenizer)
with a sequence-classification head (clean / offensive). Run on a GPU (Colab T4).

Usage:  python -m src.train_bert
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src import config as C
from src.normalize import normalize
from src.prepare_data import load_split
from src.utils import set_seed, tune_threshold


def bert_text(t):
    return normalize(t, lowercase=False)   # cased model: keep case, fix noise


@torch.no_grad()
def predict_proba(model, tok, texts, device, batch=64):
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok([bert_text(t) for t in texts[i:i + batch]], truncation=True,
                  max_length=C.BERT_MAX_LEN, padding=True, return_tensors="pt").to(device)
        out.append(torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    set_seed(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| base model:", C.BERT_BASE)
    train, val = load_split("train"), load_split("val")
    tok = AutoTokenizer.from_pretrained(C.BERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        C.BERT_BASE, num_labels=2, id2label={0: "NOT", 1: "OFF"},
        label2id={"NOT": 0, "OFF": 1}).to(device)

    items = list(zip(train["tweet"].tolist(), train["y"].tolist()))

    def collate(batch):
        enc = tok([bert_text(t) for t, _ in batch], truncation=True,
                  max_length=C.BERT_MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in batch])
        return enc

    loader = DataLoader(items, batch_size=C.BERT_BATCH, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(C.SEED))
    opt = torch.optim.AdamW(model.parameters(), lr=C.BERT_LR, weight_decay=0.01)
    steps = len(loader) * C.BERT_EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = C.MODEL_DIRS["bert_fp32"]
    best_f1, log = -1.0, []
    for epoch in range(1, C.BERT_EPOCHS + 1):
        model.train()
        t0, total = time.time(), 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss={total / step:.4f}")
        p = predict_proba(model, tok, val["tweet"].tolist(), device)
        f1 = f1_score(val["y"], (p >= 0.5).astype(int), average="macro")
        log.append({"epoch": epoch, "train_loss": round(total / len(loader), 4),
                    "val_macro_f1@0.5": round(f1, 4), "minutes": round((time.time() - t0) / 60, 1)})
        print(log[-1])
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out, safe_serialization=True)
            tok.save_pretrained(out)

    # threshold of the best checkpoint, tuned on validation
    model = AutoModelForSequenceClassification.from_pretrained(out).to(device)
    p = predict_proba(model, tok, val["tweet"].tolist(), device)
    thr, f1_t = tune_threshold(val["y"], p)
    (out / "threshold.json").write_text(json.dumps({"threshold": thr}))
    (out / "tuning.json").write_text(json.dumps(
        {"base": C.BERT_BASE, "epochs": log, "val_macro_f1": round(f1_t, 4)}, indent=1))
    print(f"saved BERTurk FP32 (thr={thr:.2f}, val macro-F1={f1_t:.4f})")


if __name__ == "__main__":
    main()

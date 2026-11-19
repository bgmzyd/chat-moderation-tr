"""Area A - our own word2vec (skip-gram, trained only on the training messages)
feeding a feed-forward neural network trained with SGD + backpropagation.

message -> normalize -> tokens -> word2vec vectors -> [mean ; max] pooling (200-d)
        -> Linear(200,256) -> ReLU -> Dropout(0.3) -> Linear(256,64) -> ReLU -> Linear(64,2)

Usage:  python -m src.train_w2v_ffn
"""
import json

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec, KeyedVectors
from sklearn.metrics import f1_score

from src import config as C
from src.normalize import normalize, tokenize
from src.prepare_data import load_split
from src.utils import set_seed, tune_threshold


class FFN(nn.Module):
    def __init__(self, d_in, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


def embed(texts, kv: KeyedVectors):
    """Mean- and max-pooled word2vec vectors of each message (OOV words skipped)."""
    out = np.zeros((len(texts), 2 * kv.vector_size), dtype=np.float32)
    for i, t in enumerate(texts):
        vecs = [kv[w] for w in tokenize(normalize(t)) if w in kv]
        if vecs:
            v = np.stack(vecs)
            out[i] = np.concatenate([v.mean(0), v.max(0)])
    return out


def main():
    set_seed(C.SEED)
    train, val = load_split("train"), load_split("val")
    sents = [tokenize(normalize(t)) for t in train["tweet"]]
    print("training word2vec on", len(sents), "training messages ...")
    w2v = Word2Vec(sents, vector_size=C.W2V_DIM, window=5, min_count=2, sg=1,
                   negative=10, epochs=15, workers=1, seed=C.SEED)
    kv = w2v.wv
    print("vocabulary:", len(kv))

    xtr, xva = torch.tensor(embed(train["tweet"], kv)), torch.tensor(embed(val["tweet"], kv))
    ytr = torch.tensor(train["y"].values)
    yva = val["y"].values

    model = FFN(xtr.shape[1])
    pos = ytr.float().mean().item()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.5 / (1 - pos), 0.5 / pos]))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)

    best = (-1, None, None, 0)
    g = torch.Generator().manual_seed(C.SEED)
    for epoch in range(1, C.FFN_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        total = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()                         # backpropagation
            opt.step()                              # SGD update
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(xva), -1)[:, 1].numpy()
        f1 = f1_score(yva, (p >= 0.5).astype(int), average="macro")
        if f1 > best[0]:
            best = (f1, {k: v.clone() for k, v in model.state_dict().items()}, p, epoch)
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train loss={total / len(xtr):.4f}  val macro-F1={f1:.4f}")

    f1, state, p, epoch = best
    model.load_state_dict(state)
    thr, f1_t = tune_threshold(yva, p)
    out = C.MODEL_DIRS["w2v_ffn"]
    out.mkdir(parents=True, exist_ok=True)
    kv.save(str(out / "word2vec.kv"))
    torch.save(model.state_dict(), out / "ffn.pt")
    (out / "threshold.json").write_text(json.dumps({"threshold": thr}))
    (out / "tuning.json").write_text(json.dumps(
        {"best_epoch": epoch, "val_macro_f1": round(f1_t, 4), "d_in": int(xtr.shape[1])}))
    print(f"saved w2v+FFN (best epoch {epoch}, thr={thr:.2f}, val macro-F1={f1_t:.4f})")


if __name__ == "__main__":
    main()

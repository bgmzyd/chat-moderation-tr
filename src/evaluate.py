"""R3 + R4: evaluate ALL models on the SAME held-out test set (official OffensEval-TR 2020 test).

* task metric : macro-F1 (classes are imbalanced), plus accuracy and OFF precision/recall
* size        : MB on disk of everything the model needs (models/<key>/)
* latency     : end-to-end ms per message (normalize + features/tokenize + inference),
                batch size 1, CPU only, fixed #threads, 10 warm-up + LATENCY_RUNS timed runs
Writes results/results.json, results/results_table.md, predictions and error examples.

Usage:  python -m src.evaluate
"""
import json
import os
import platform
import statistics
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # benchmark on CPU only

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src import config as C
from src.models import load_model
from src.prepare_data import load_split


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def latency_ms(model, texts):
    for t in texts[:C.LATENCY_WARMUP]:
        model.predict_one(t)
    times = []
    for t in texts:
        t0 = time.perf_counter()
        model.predict_one(t)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    import torch
    torch.set_num_threads(C.CPU_THREADS)
    test = load_split("test")
    texts, y = test["tweet"].tolist(), test["y"].values
    rng = np.random.default_rng(C.SEED)
    lat_texts = [texts[i] for i in rng.choice(len(texts), C.LATENCY_RUNS, replace=False)]

    hardware = {"cpu": cpu_name(), "logical_cores": os.cpu_count(),
                "torch_threads": C.CPU_THREADS, "python": platform.python_version(),
                "torch": torch.__version__, "os": platform.platform()}
    print("hardware:", hardware)

    rows, preds = [], {"id": test["id"], "tweet": test["tweet"], "y": y}
    for key in C.MODEL_KEYS:
        if not (C.MODEL_DIRS[key] / "threshold.json").exists():
            print(f"!! {key} not trained - skipped")
            continue
        print(f"\n== {C.MODEL_NAMES[key]}")
        m = load_model(key)
        t0 = time.time()
        p = m.predict_proba(texts)
        yhat = (p >= m.threshold).astype(int)
        preds[f"p_{key}"], preds[f"pred_{key}"] = np.round(p, 4), yhat
        m.clear_cache()            # the latency run must not reuse cached test results
        lat = latency_ms(m, lat_texts)
        row = {
            "key": key, "model": m.name, "threshold": m.threshold,
            "macro_f1": round(f1_score(y, yhat, average="macro"), 4),
            "accuracy": round(accuracy_score(y, yhat), 4),
            "off_precision": round(precision_score(y, yhat, zero_division=0), 4),
            "off_recall": round(recall_score(y, yhat), 4),
            "confusion_matrix[[TN,FP],[FN,TP]]": confusion_matrix(y, yhat).tolist(),
            "size_mb": round(m.size_mb, 2),
            "latency_mean_ms": round(statistics.mean(lat), 2),
            "latency_median_ms": round(statistics.median(lat), 2),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "latency_runs": len(lat),
            "within_budget": statistics.mean(lat) <= C.LATENCY_BUDGET_MS,
            "test_pass_seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        print({k: v for k, v in row.items() if k != "model"})
        del m

    C.RESULTS_DIR.mkdir(exist_ok=True)
    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(C.RESULTS_DIR / "predictions_test.csv", index=False)
    extra = {}
    if "pred_bert_int8" in pred_df and "pred_bert_fp32" in pred_df:
        extra["int8_vs_fp32_prediction_agreement"] = round(
            float((pred_df["pred_bert_int8"] == pred_df["pred_bert_fp32"]).mean()), 4)
        err = pred_df[pred_df["pred_bert_int8"] != pred_df["y"]]
        err = err.assign(error=np.where(err["y"] == 1, "false negative (missed)", "false positive"))
        err[["id", "tweet", "y", "p_bert_int8", "error"]].to_csv(
            C.RESULTS_DIR / "errors_bert_int8.csv", index=False)

    result = {"test_set": "OffensEval-TR 2020 official test", "n_test": len(test),
              "off_ratio": round(float(y.mean()), 4), "latency_budget_ms": C.LATENCY_BUDGET_MS,
              "hardware": hardware, "models": rows, **extra}
    (C.RESULTS_DIR / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    md = [f"Test set: OffensEval-TR 2020 official test (n={len(test)}, OFF={y.mean():.1%}).  ",
          f"Latency: mean of {C.LATENCY_RUNS} single-message runs on CPU ({hardware['cpu']}, "
          f"{C.CPU_THREADS} threads), budget {C.LATENCY_BUDGET_MS:.0f} ms.\n",
          "| Model | Macro-F1 | Accuracy | OFF recall | Size (MB) | Latency mean (ms) | p95 (ms) | ≤ budget |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['off_recall']:.4f} | "
                  f"{r['size_mb']:.2f} | {r['latency_mean_ms']:.2f} | {r['latency_p95_ms']:.2f} | "
                  f"{'yes' if r['within_budget'] else 'no'} |")
    if extra:
        md.append(f"\nINT8 vs FP32 prediction agreement on test: {extra['int8_vs_fp32_prediction_agreement']:.2%}")
    (C.RESULTS_DIR / "results_table.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()

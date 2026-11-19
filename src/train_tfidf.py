"""Area B - classical baseline: tf-idf (word 1-2 grams + char 2-5 grams) + logistic regression.
C and the decision threshold are tuned on the validation split.

Usage:  python -m src.train_tfidf
"""
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src import config as C
from src.normalize import normalize
from src.prepare_data import load_split
from src.utils import tune_threshold


def make_pipeline(c):
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                                 sublinear_tf=True, token_pattern=r"<\w+>|[^\W\d_]+")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 max_features=150_000, sublinear_tf=True)),
    ])
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    return Pipeline([("norm_tfidf", feats), ("clf", clf)])


def main():
    train, val = load_split("train"), load_split("val")
    xtr = [normalize(t) for t in train["tweet"]]
    xva = [normalize(t) for t in val["tweet"]]
    best = None
    for c in [0.5, 1.0, 2.0, 4.0, 8.0]:
        pipe = make_pipeline(c).fit(xtr, train["y"])
        p = pipe.predict_proba(xva)[:, 1]
        thr, f1 = tune_threshold(val["y"], p)
        print(f"C={c:<4}  val macro-F1={f1:.4f} (thr={thr:.2f})")
        if best is None or f1 > best[0]:
            best = (f1, c, thr, pipe)
    f1, c, thr, pipe = best
    out = C.MODEL_DIRS["tfidf"]
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "tfidf_lr.joblib", compress=3)
    (out / "threshold.json").write_text(json.dumps({"threshold": thr}))
    (out / "tuning.json").write_text(json.dumps({"C": c, "val_macro_f1": round(f1, 4)}))
    print(f"saved tf-idf model (C={c}, thr={thr:.2f}, val macro-F1={f1:.4f})")


if __name__ == "__main__":
    main()

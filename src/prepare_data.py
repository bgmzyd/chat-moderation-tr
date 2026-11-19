"""Download OffensEval-TR 2020 and create the fixed splits used by EVERY model.

train / val : stratified split of the official training file (seed 42, 10 % val)
test        : the official OffensEval-TR 2020 test set (never used for tuning)

Usage:  python -m src.prepare_data
"""
import csv
import io
import json
import zipfile

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from src import config as C


def _find(name):
    hits = list(C.RAW_DIR.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {C.RAW_DIR}")
    return hits[0]


def download():
    if any(C.RAW_DIR.rglob("offenseval-tr-training-v1.tsv")):
        print("raw data already present:", C.RAW_DIR)
        return
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("downloading", C.DATA_URL)
    r = requests.get(C.DATA_URL, timeout=120)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(C.RAW_DIR)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
                       keep_default_na=False)


def load_official():
    train = read_tsv(_find("offenseval-tr-training-v1.tsv"))
    train = train.rename(columns={"subtask_a": "label"})[["id", "tweet", "label"]]

    test = read_tsv(_find("offenseval-tr-testset-v1.tsv"))[["id", "tweet"]]
    lab = pd.read_csv(_find("offenseval-tr-labela-v1.tsv"), header=None,
                      names=["id", "label"], dtype=str, keep_default_na=False)
    lab = lab[lab["id"].str.lower() != "id"]
    test = test.merge(lab, on="id", how="inner")
    return train, test


def main():
    download()
    train_full, test = load_official()
    for df in (train_full, test):
        df["label"] = df["label"].str.strip().str.upper()
        assert set(df["label"]) <= set(C.LABELS), set(df["label"])
        df["y"] = (df["label"] == "OFF").astype(int)
        df.drop(df[df["tweet"].str.strip() == ""].index, inplace=True)

    train, val = train_test_split(train_full, test_size=C.VAL_FRACTION,
                                  stratify=train_full["y"], random_state=C.SEED)
    C.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df = df.reset_index(drop=True)
        df.to_csv(C.SPLIT_DIR / f"{name}.csv", index=False)
        stats[name] = {"n": len(df), "off": int(df["y"].sum()),
                       "off_ratio": round(float(df["y"].mean()), 4)}
        print(f"{name:5s} n={len(df):6d}  OFF={stats[name]['off']:5d} "
              f"({stats[name]['off_ratio']:.1%})")
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2))


def load_split(name):
    return pd.read_csv(C.SPLIT_DIR / f"{name}.csv", dtype={"tweet": str},
                       keep_default_na=False)


if __name__ == "__main__":
    main()

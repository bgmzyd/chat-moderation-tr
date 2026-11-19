"""Area B - rule baseline: an offensive-word lexicon matched with MINIMUM EDIT DISTANCE.

* The lexicon is mined automatically from the TRAINING split only: words whose
  Laplace-smoothed P(OFF | word) is high. The cut-off is tuned on the validation split.
* A token is flagged when one of the lexicon words is within a small Levenshtein
  distance of the token (or of the token's prefix, because Turkish adds suffixes:
  "salaklar", "salağın").  e.g.  serefsiz ~ şerefsiz (d=1),  aptl ~ aptal (d=1).

Usage:  python -m src.lexicon          (build + tune + save)
"""
import json
from collections import Counter

from sklearn.metrics import f1_score

from src import config as C
from src.normalize import normalize, tokenize

_FOLD = str.maketrans("çğıöşü", "cgiosu")


def edit_distance(a: str, b: str, max_d: int | None = None) -> int:
    """Levenshtein distance (insert = delete = substitute = 1), dynamic programming.
    Stops early and returns max_d+1 once every cell of a row exceeds max_d."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,                 # deletion
                         cur[j - 1] + 1,              # insertion
                         prev[j - 1] + (ca != cb))    # substitution
        if max_d is not None and min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[-1]


def allowed_distance(word: str) -> int:
    n = len(word)
    return 0 if n <= 3 else (1 if n <= 6 else 2)


class LexiconMatcher:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores                       # word -> smoothed P(OFF|word)
        self.buckets: dict[str, list[str]] = {}
        for w in scores:
            self.buckets.setdefault(w[0].translate(_FOLD), []).append(w)
        self._cache: dict[str, tuple[str, int] | None] = {}

    def match_token(self, tok: str):
        """Return (lexicon_word, distance) of the best match, or None."""
        if tok in self._cache:
            return self._cache[tok]
        best = None
        for w in self.buckets.get(tok[0].translate(_FOLD), []):
            k = allowed_distance(w)
            if k == 0:
                d = 0 if tok == w else 99
            elif len(tok) < len(w) - k:
                d = 99
            else:   # compare w with prefixes of tok of length len(w)-k .. len(w)+k
                d = min(edit_distance(tok[:L], w, k)
                        for L in range(max(1, len(w) - k), min(len(tok), len(w) + k) + 1))
            if d <= k and (best is None or (d, -self.scores[w]) < (best[1], -self.scores[best[0]])):
                best = (w, d)
        self._cache[tok] = best
        return best

    def analyze(self, text: str):
        """Returns (p_off, matched_tokens). p_off = score of the strongest matched word."""
        hits = []
        for tok in tokenize(normalize(text)):
            if tok.startswith("<"):
                continue
            m = self.match_token(tok)
            if m:
                hits.append((tok, m[0], m[1]))
        p = max((self.scores[w] for _, w, _ in hits), default=0.0)
        return p, hits

    def predict_proba(self, texts):
        return [self.analyze(t)[0] for t in texts]

    # --------------------------------------------------------------- io
    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "lexicon.json").write_text(json.dumps(self.scores, ensure_ascii=False, indent=0))
        (path / "threshold.json").write_text(json.dumps({"threshold": 0.5}))

    @classmethod
    def load(cls, path):
        return cls(json.loads((path / "lexicon.json").read_text()))


def mine_scores(texts, ys, min_df=8):
    df_all, df_off = Counter(), Counter()
    for t, y in zip(texts, ys):
        toks = {w for w in tokenize(normalize(t)) if not w.startswith("<") and len(w) >= 2}
        df_all.update(toks)
        if y:
            df_off.update(toks)
    return {w: (df_off[w] + 1) / (n + 2) for w, n in df_all.items() if n >= min_df}


def build_lexicon(scores, min_prec):
    words = sorted((w for w, s in scores.items() if s >= min_prec), key=len)
    kept = []
    for w in words:      # drop inflected forms already covered by a shorter stem (prefix)
        if not any(len(k) >= 4 and w.startswith(k) for k in kept):
            kept.append(w)
    return {w: round(scores[w], 4) for w in kept}


def main():
    from src.prepare_data import load_split
    train, val = load_split("train"), load_split("val")
    scores = mine_scores(train["tweet"], train["y"])
    best = None
    for min_prec in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        m = LexiconMatcher(build_lexicon(scores, min_prec))
        pred = [int(p >= 0.5) for p in m.predict_proba(val["tweet"])]
        f1 = f1_score(val["y"], pred, average="macro")
        print(f"min_prec={min_prec:.2f}  lexicon={len(m.scores):4d} words  val macro-F1={f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, min_prec, m)
    f1, min_prec, m = best
    m.save(C.MODEL_DIRS["lexicon"])
    (C.MODEL_DIRS["lexicon"] / "tuning.json").write_text(json.dumps(
        {"min_prec": min_prec, "val_macro_f1": round(f1, 4), "n_words": len(m.scores)}))
    print(f"saved lexicon (min_prec={min_prec}, {len(m.scores)} words, val macro-F1={f1:.4f})")


if __name__ == "__main__":
    main()

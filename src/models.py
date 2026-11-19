"""One common interface for the five models, used by evaluation AND the application
(so the numbers in the results table come from exactly the code the app runs)."""
import json
import warnings
from pathlib import Path

import numpy as np

from src import config as C
from src.normalize import normalize

warnings.filterwarnings("ignore", message=".*quantiz.*")   # torch deprecation notices


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**20


class Predictor:
    key: str

    def __init__(self, key):
        self.key, self.name, self.dir = key, C.MODEL_NAMES[key], C.MODEL_DIRS[key]
        self.threshold = json.loads((self.dir / "threshold.json").read_text())["threshold"]

    @property
    def size_mb(self):
        return dir_size_mb(self.dir)

    def predict_proba(self, texts) -> np.ndarray:
        raise NotImplementedError

    def clear_cache(self):
        """Forget memoized work so latency is measured cold (only the lexicon caches)."""

    def predict_one(self, text):
        p = float(self.predict_proba([text])[0])
        return p, int(p >= self.threshold)


class LexiconPredictor(Predictor):
    def __init__(self):
        super().__init__("lexicon")
        from src.lexicon import LexiconMatcher
        self.m = LexiconMatcher.load(self.dir)

    def predict_proba(self, texts):
        return np.array(self.m.predict_proba(texts))

    def clear_cache(self):
        self.m._cache.clear()


class TfidfPredictor(Predictor):
    def __init__(self):
        super().__init__("tfidf")
        import joblib
        self.pipe = joblib.load(self.dir / "tfidf_lr.joblib")

    def predict_proba(self, texts):
        return self.pipe.predict_proba([normalize(t) for t in texts])[:, 1]


class W2VFFNPredictor(Predictor):
    def __init__(self):
        super().__init__("w2v_ffn")
        import torch
        from gensim.models import KeyedVectors
        from src.train_w2v_ffn import FFN
        self.torch = torch
        self.kv = KeyedVectors.load(str(self.dir / "word2vec.kv"))
        self.ffn = FFN(2 * self.kv.vector_size)
        self.ffn.load_state_dict(torch.load(self.dir / "ffn.pt", map_location="cpu"))
        self.ffn.eval()

    def predict_proba(self, texts):
        from src.train_w2v_ffn import embed
        with self.torch.no_grad():
            x = self.torch.tensor(embed(texts, self.kv))
            return self.torch.softmax(self.ffn(x), -1)[:, 1].numpy()


class BertPredictor(Predictor):
    """FP32 = the fine-tuned checkpoint; INT8 = the same network after dynamic quantization
    (all nn.Linear weights stored as int8, activations quantized on the fly). CPU only."""

    def __init__(self, key):
        super().__init__(key)
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(C.CPU_THREADS)
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        if key == "bert_fp32":
            self.model = AutoModelForSequenceClassification.from_pretrained(self.dir)
        else:
            cfg = AutoConfig.from_pretrained(self.dir)
            fp32 = AutoModelForSequenceClassification.from_config(cfg)
            self.model = quantize(fp32)
            state = torch.load(self.dir / "model_int8.pt", map_location="cpu", weights_only=False)
            self.model.load_state_dict(state)
        self.model.eval()

    def predict_proba(self, texts, batch=32):
        from src.train_bert import bert_text
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tok([bert_text(t) for t in texts[i:i + batch]], truncation=True,
                               max_length=C.BERT_MAX_LEN, padding=True, return_tensors="pt")
                out.append(self.torch.softmax(self.model(**enc).logits, -1)[:, 1].numpy())
        return np.concatenate(out)


def quantize(model):
    import torch
    return torch.ao.quantization.quantize_dynamic(model.eval(), {torch.nn.Linear},
                                                  dtype=torch.qint8)


def load_model(key) -> Predictor:
    return {"lexicon": LexiconPredictor, "tfidf": TfidfPredictor, "w2v_ffn": W2VFFNPredictor,
            "bert_fp32": lambda: BertPredictor("bert_fp32"),
            "bert_int8": lambda: BertPredictor("bert_int8")}[key]()

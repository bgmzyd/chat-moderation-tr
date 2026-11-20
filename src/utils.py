import random

import numpy as np
from sklearn.metrics import f1_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def tune_threshold(y_true, p_off, grid=None):
    """Pick the P(OFF) threshold that maximizes macro-F1 on the VALIDATION split."""
    grid = grid if grid is not None else np.arange(0.05, 0.96, 0.01)
    p_off = np.asarray(p_off)
    scores = [(f1_score(y_true, (p_off >= t).astype(int), average="macro"), t) for t in grid]
    f1, t = max(scores)
    return round(float(t), 2), float(f1)


def model_inputs(model, enc):
    """Keep only the tokenizer outputs the model's forward() accepts
    (DistilBERT, unlike BERT, takes no token_type_ids)."""
    import inspect
    accepted = inspect.signature(model.forward).parameters
    return {k: v for k, v in enc.items() if k in accepted}

"""Smoke test helper (NOT used for any reported number).

Creates a small synthetic dataset in the official OffensEval-TR file format and a tiny,
randomly initialised BERT so the whole pipeline can be checked quickly on a CPU:

  python tests/smoke_setup.py /tmp/smoke
  BERT_BASE=/tmp/smoke/tiny-bert BERT_EPOCHS=1 FFN_EPOCHS=3 bash run_pipeline.sh
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

clean = ["maç çok güzeldi", "bugün hava harika", "takım iyi oynadı", "oyunu kazandık",
         "yarın görüşürüz", "teşekkürler kardeşim", "bu güncelleme süper", "hadi bir el daha"]
bad = ["salak", "aptal", "şerefsiz", "gerizekalı", "mal herif", "haysiyetsiz"]


def tweet(off):
    w = random.choice(clean).split()
    if off:
        w.insert(random.randrange(len(w) + 1), random.choice(bad))
    if random.random() < 0.3:
        w.insert(0, "@USER")
    return " ".join(w)


def main(out):
    random.seed(0)
    out = Path(out)
    tr = out / "raw/offenseval2020-turkish/offenseval-tr-training-v1"
    te = out / "raw/offenseval2020-turkish/offenseval-tr-testset-v1"
    tr.mkdir(parents=True, exist_ok=True)
    te.mkdir(parents=True, exist_ok=True)
    with open(tr / "offenseval-tr-training-v1.tsv", "w") as f:
        f.write("id\ttweet\tsubtask_a\n")
        for i in range(1500):
            off = random.random() < 0.2
            f.write(f"{i}\t{tweet(off)}\t{'OFF' if off else 'NOT'}\n")
    with open(te / "offenseval-tr-testset-v1.tsv", "w") as f, \
            open(te / "offenseval-tr-labela-v1.tsv", "w") as g:
        f.write("id\ttweet\n")
        for i in range(5000, 5300):
            off = random.random() < 0.2
            f.write(f"{i}\t{tweet(off)}\n")
            g.write(f"{i},{'OFF' if off else 'NOT'}\n")

    # tiny random BERT with a WordPiece vocab trained on the synthetic text
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    tk = BertWordPieceTokenizer(lowercase=False)
    tk.train([str(tr / "offenseval-tr-training-v1.tsv")], vocab_size=500)
    d = out / "tiny-bert"
    d.mkdir(exist_ok=True)
    tk.save_model(str(d))
    BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=False).save_pretrained(d)
    cfg = BertConfig(vocab_size=tk.get_vocab_size(), hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=128)
    BertForSequenceClassification(cfg).save_pretrained(d)
    print("smoke assets in", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/smoke")

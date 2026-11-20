# Real-Time Chat Moderation (Turkish) — SWE015 Homework, Use Case 4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bgmzyd/chat-moderation-tr/blob/main/notebooks/train_colab.ipynb)

A chat-like application that labels every message **clean / offensive** live, with a confidence
score, masks offensive messages, and logs the CPU response time of every message.
The application runs a **distilled, INT8-quantized BERTurk student on ONNX Runtime** by default; the other models
of the ladder can be selected or compared side by side.

## Run the application

```bash
git clone https://github.com/bgmzyd/chat-moderation-tr.git
cd chat-moderation-tr
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7860. On the first run `app.py` downloads the trained weights
(~1.2 GB) from the Hugging Face Hub repo set in `src/config.py` (`HF_REPO_ID`).
Python 3.10+ is required; no GPU is needed.

## Model ladder (all evaluated on the same held-out test set)

| # | Model | Technique area |
|---|---|---|
| 0 | Edit-distance lexicon matcher (rule baseline) | B |
| 1 | tf-idf (word 1–2 + char 2–5 grams) + logistic regression | B |
| 2 | Own word2vec (skip-gram, trained on training messages) → feed-forward net, SGD + backprop | A |
| 3 | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuned, FP32 | C |
| 4a | Model 3 after PyTorch dynamic INT8 quantization (`torch.ao.quantization.quantize_dynamic`) | D |
| 4b | Model 3 exported to ONNX, graph-fused, ONNX Runtime dynamic INT8 | D |
| 5 | Knowledge distillation: Model 3 teaches DistilBERTurk (6 layers), FP32 | D |
| 5b | Model 5 exported to ONNX, ONNX Runtime dynamic INT8 — **used by the app** | D |

All models share the same regex normalization (`src/normalize.py`, area B): Turkish casing,
mentions/URLs, hashtags, split letters (`s.a.l.a.k`), leetspeak (`s4l4k`), repeated letters.

Results (macro-F1, size, latency on two CPUs): [`results/results_table.md`](results/results_table.md);
raw runs in `results/server_colab/` and `results/laptop_m2/`.

## Reproduce the training (Google Colab, T4 GPU)

Open `notebooks/train_colab.ipynb` in Colab and run all cells, or on any GPU machine:

```bash
pip install -r requirements.txt
bash run_pipeline.sh                                   # data → 5 models → evaluation
python -m src.hub upload --repo <hf-user>/chat-moderation-tr-models
```

| Step | Script | Output |
|---|---|---|
| Data + fixed splits (seed 42) | `src/prepare_data.py` | `data/splits/{train,val,test}.csv` |
| Lexicon rule | `src/lexicon.py` | `models/lexicon/` |
| tf-idf + LR | `src/train_tfidf.py` | `models/tfidf/` |
| word2vec + FFN | `src/train_w2v_ffn.py` | `models/w2v_ffn/` |
| BERTurk fine-tuning | `src/train_bert.py` | `models/bert_fp32/` |
| PyTorch INT8 quantization | `src/quantize_bert.py` | `models/bert_int8/` |
| ONNX Runtime INT8 | `src/quantize_onnx.py` | `models/bert_onnx_int8/` |
| Distillation (teacher → student) | `src/distill.py` | `models/distil_fp32/` |
| Student ONNX Runtime INT8 | `src/quantize_onnx.py --src distil_fp32 --dst distil_onnx_int8` | `models/distil_onnx_int8/` |
| Evaluation | `src/evaluate.py` | `results/` |

Protocol: thresholds and hyper-parameters are tuned on the validation split only; the official
test set is used once, by `src/evaluate.py`. Latency = end-to-end time per message
(normalization + features/tokenization + inference), batch size 1, CPU, fixed number of
threads, 10 warm-up + 200 timed runs. Size = all files in `models/<model>/`.

## Data

OffensEval-TR 2020 — Çağrı Çöltekin. *A Corpus of Turkish Offensive Language on Social Media.*
LREC 2020. https://coltekin.github.io/offensive-turkish/ (CC BY 2.0).
31,756 training tweets (split 90/10 into train/validation) and 3,528 official test tweets.

Pretrained models: BERTurk and DistilBERTurk, Stefan Schweter,
https://huggingface.co/dbmdz/bert-base-turkish-cased and https://huggingface.co/dbmdz/distilbert-base-turkish-cased

## Repository layout

```
app.py                 Gradio application
run_pipeline.sh        full training + evaluation
src/config.py          paths, hyper-parameters, latency budget, HF repo id
src/normalize.py       regex normalization (B)
src/lexicon.py         minimum edit distance + lexicon matcher (B)
src/train_tfidf.py     tf-idf baseline (B)
src/train_w2v_ffn.py   word2vec + feed-forward network (A)
src/train_bert.py      BERTurk fine-tuning (C)
src/quantize_bert.py   PyTorch dynamic INT8 quantization (D)
src/quantize_onnx.py   ONNX export + ONNX Runtime INT8 quantization (D)
src/distill.py         knowledge distillation BERTurk -> DistilBERTurk (D)
src/combine_results.py one results table for the server and the laptop runs
src/models.py          common predictor interface used by evaluation AND the app
src/evaluate.py        same-test-set evaluation, size, latency
src/hub.py             upload/download weights to the Hugging Face Hub
notebooks/             Colab notebook
tests/smoke_setup.py   synthetic data for a quick pipeline check (no reported numbers)
```

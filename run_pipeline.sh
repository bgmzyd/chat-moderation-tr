#!/usr/bin/env bash
# Full training + evaluation pipeline (run on Colab with a GPU runtime).
set -e
python -m src.prepare_data      # download + fixed train/val/test splits
python -m src.lexicon           # B: edit-distance lexicon rule baseline
python -m src.train_tfidf       # B: tf-idf + logistic regression baseline
python -m src.train_w2v_ffn     # A: own word2vec + feed-forward net (SGD)
python -m src.train_bert        # C: fine-tune BERTurk (GPU)
python -m src.quantize_bert     # D: PyTorch dynamic INT8 quantization
python -m src.quantize_onnx     # D: ONNX export + ONNX Runtime INT8 of BERTurk
python -m src.distill           # D: knowledge distillation into DistilBERTurk (GPU)
python -m src.quantize_onnx --src distil_fp32 --dst distil_onnx_int8   # D: INT8 student (used by the app)
python -m src.evaluate          # R3/R4: same test set, macro-F1, size, latency (CPU)

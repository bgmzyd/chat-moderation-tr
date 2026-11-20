Test set: OffensEval-TR 2020 official test (n=3528, OFF=20.3%).  
Latency: mean of 200 single-message runs on CPU (Intel(R) Xeon(R) CPU @ 2.00GHz, 2 threads), budget 50 ms.

| Model | Macro-F1 | Accuracy | OFF recall | Size (MB) | Latency mean (ms) | p95 (ms) | ≤ budget |
|---|---|---|---|---|---|---|---|
| Rule: edit-distance lexicon matcher | 0.6350 | 0.7874 | 0.3478 | 0.00 | 0.50 | 1.34 | yes |
| Baseline: tf-idf + logistic regression | 0.7700 | 0.8574 | 0.5936 | 2.58 | 2.98 | 4.09 | yes |
| NN: own word2vec + feed-forward net | 0.6711 | 0.7968 | 0.4399 | 12.19 | 1.35 | 3.58 | yes |
| Transformer: BERTurk (FP32) | 0.8081 | 0.8829 | 0.6369 | 422.72 | 131.28 | 248.77 | no |
| Quantized: BERTurk INT8 (PyTorch dynamic) | 0.8049 | 0.8778 | 0.6564 | 178.14 | 63.40 | 132.56 | no |
| Quantized: BERTurk INT8 (ONNX Runtime) | 0.8073 | 0.8841 | 0.6229 | 106.65 | 40.59 | 91.93 | yes |
| Distilled student: DistilBERTurk (FP32) | 0.7961 | 0.8776 | 0.6047 | 260.48 | 70.82 | 148.01 | no |
| Quantized student: DistilBERTurk INT8 (ONNX Runtime) | 0.7962 | 0.8747 | 0.6257 | 65.89 | 19.94 | 43.73 | yes |

bert int8 vs bert fp32 prediction agreement: 96.94%

bert onnx int8 vs bert fp32 prediction agreement: 98.75%

distil fp32 vs bert fp32 prediction agreement: 93.17%

distil onnx int8 vs distil fp32 prediction agreement: 98.07%

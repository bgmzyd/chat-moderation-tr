Test set: OffensEval-TR 2020 official test (n=3528, OFF=20.3%).  
Latency: mean of 200 single-message runs on CPU (Apple M2, 2 threads), budget 50 ms.

| Model | Macro-F1 | Accuracy | OFF recall | Size (MB) | Latency mean (ms) | p95 (ms) | ≤ budget |
|---|---|---|---|---|---|---|---|
| Rule: edit-distance lexicon matcher | 0.6350 | 0.7874 | 0.3478 | 0.00 | 0.26 | 0.69 | yes |
| Baseline: tf-idf + logistic regression | 0.7700 | 0.8574 | 0.5936 | 2.58 | 0.73 | 0.93 | yes |
| NN: own word2vec + feed-forward net | 0.6711 | 0.7968 | 0.4399 | 12.19 | 0.10 | 0.18 | yes |
| Transformer: BERTurk (FP32) | 0.8081 | 0.8829 | 0.6369 | 422.72 | 42.81 | 52.07 | yes |
| Quantized: BERTurk INT8 (PyTorch dynamic) | 0.8116 | 0.8804 | 0.6802 | 178.14 | 36.63 | 76.62 | yes |
| Quantized: BERTurk INT8 (ONNX Runtime) | 0.8107 | 0.8861 | 0.6285 | 106.65 | 10.47 | 22.77 | yes |
| Distilled student: DistilBERTurk (FP32) | 0.7961 | 0.8776 | 0.6047 | 260.48 | 18.07 | 21.12 | yes |
| Quantized student: DistilBERTurk INT8 (ONNX Runtime) | 0.7966 | 0.8747 | 0.6285 | 65.89 | 5.26 | 11.48 | yes |

bert int8 vs bert fp32 prediction agreement: 97.70%

bert onnx int8 vs bert fp32 prediction agreement: 98.50%

distil fp32 vs bert fp32 prediction agreement: 93.17%

distil onnx int8 vs distil fp32 prediction agreement: 97.79%

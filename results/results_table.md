Test set: OffensEval-TR 2020 official test (n=3528, OFF=20.3%). Latency = mean of 200 single-message runs (batch 1, 2 CPU threads); budget 50 ms.

- Server CPU: Intel(R) Xeon(R) CPU @ 2.00GHz, 2 logical cores, Linux-6.6.159+-x86_64-with-glibc2.39, torch 2.14.1+cu130
- Laptop: Apple M2, 8 logical cores, macOS-14.1.1-arm64-arm-64bit, torch 2.15.0

| Model | Macro-F1 (Server CPU) | Macro-F1 (Laptop) | Size (MB) | Server CPU latency ms (p95) | Server CPU ≤ 50 ms | Laptop latency ms (p95) | Laptop ≤ 50 ms |
|---|---|---|---|---|---|---|---|
| Rule: edit-distance lexicon matcher | 0.6350 | 0.6350 | 0.00 | 0.50 (1.34) | yes | 0.26 (0.69) | yes |
| Baseline: tf-idf + logistic regression | 0.7700 | 0.7700 | 2.58 | 2.98 (4.09) | yes | 0.73 (0.93) | yes |
| NN: own word2vec + feed-forward net | 0.6711 | 0.6711 | 12.19 | 1.35 (3.58) | yes | 0.10 (0.18) | yes |
| Transformer: BERTurk (FP32) | 0.8081 | 0.8081 | 422.72 | 131.28 (248.77) | no | 42.81 (52.07) | yes |
| Quantized: BERTurk INT8 (PyTorch dynamic) | 0.8049 | 0.8116 | 178.14 | 63.40 (132.56) | no | 36.63 (76.62) | yes |
| Quantized: BERTurk INT8 (ONNX Runtime) | 0.8073 | 0.8107 | 106.65 | 40.59 (91.93) | yes | 10.47 (22.77) | yes |
| Distilled student: DistilBERTurk (FP32) | 0.7961 | 0.7961 | 260.48 | 70.82 (148.01) | no | 18.07 (21.12) | yes |
| Quantized student: DistilBERTurk INT8 (ONNX Runtime) | 0.7962 | 0.7966 | 65.89 | 19.94 (43.73) | yes | 5.26 (11.48) | yes |

Server CPU – bert int8 vs bert fp32 prediction agreement: 96.94%

Server CPU – bert onnx int8 vs bert fp32 prediction agreement: 98.75%

Server CPU – distil fp32 vs bert fp32 prediction agreement: 93.17%

Server CPU – distil onnx int8 vs distil fp32 prediction agreement: 98.07%

Laptop – bert int8 vs bert fp32 prediction agreement: 97.70%

Laptop – bert onnx int8 vs bert fp32 prediction agreement: 98.50%

Laptop – distil fp32 vs bert fp32 prediction agreement: 93.17%

Laptop – distil onnx int8 vs distil fp32 prediction agreement: 97.79%

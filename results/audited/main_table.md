# Matched-cohort exploratory results

All rows below use identical claims and image-cluster intervals.

| Method | n | Images | Accuracy | Precision | Recall | F1 | AUROC | AUPRC | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| siglip-whole | 32 | 18 | 0.9062 | 0.9524 | 0.9091 | 0.9302 | 0.9318 | 0.9622 | 0.2585 | 0.1446 |
| clipB32-grid | 32 | 18 | 0.9375 | 0.9545 | 0.9545 | 0.9545 | 0.9545 | 0.9789 | 0.2966 | 0.1451 |
| siglip-grid | 32 | 18 | 0.6562 | 1.0000 | 0.5000 | 0.6667 | 0.9409 | 0.9710 | 0.3492 | 0.2052 |

Raw-score ECE/Brier diagnose miscalibration; scores are not validated probabilities.
Original VLM, prompt detector, authentic existing mitigation, fine-tuned detector, and full repair-system results: N/A.
Per-method inference latency, tokens, memory, API cost and correction outcomes: N/A in legacy artifacts.

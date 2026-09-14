# Completed three-seed experiment

All six AdamW/IVON training runs completed on a **Tesla T4** in Google Colab on 14 September 2026. Twelve correctness tests passed in the same runtime. Metrics were independently recomputed locally from all saved test predictions; paired initialization hashes and test IDs matched.

Qwen2.5-0.5B; ARC-Easy; 512 train / 128 validation / 256 test; seeds 21, 42, 87; 128 optimizer steps per run; effective batch 8; FP32; LoRA rank 8. This trains **540,672 adapter parameters** (about 0.109% of 494,573,440 parameters including adapters).

## Measured test results

Mean ± sample standard deviation across the three seeds. Accuracy and ECE are multiplied by 100; NLL uses natural logarithms.

| Method | Accuracy (%) ↑ | ECE (points) ↓ | NLL ↓ |
|---|---:|---:|---:|
| AdamW-LoRA | 73.18 ± 1.76 | 6.89 ± 2.30 | 0.680 ± 0.005 |
| IVON mean | 72.14 ± 0.23 | 6.99 ± 0.29 | 0.680 ± 0.003 |
| IVON, 10 samples | 69.40 ± 1.37 | 17.95 ± 1.49 | 0.884 ± 0.040 |

## Paired differences and interpretation

The differences below are computed within each seed before calculating mean and sample SD. Negative ECE/NLL differences favor IVON; positive accuracy differences favor IVON.

- **IVON mean minus AdamW:** accuracy -1.042 ± 1.966 points; ECE +0.105 ± 2.394 points; NLL -0.000165 ± 0.002448.
- **IVON, 10 samples minus AdamW:** accuracy -3.776 ± 3.133 points; ECE +11.066 ± 2.155 points; NLL +0.203 ± 0.044.

IVON mean did not reduce average ECE in this experiment. Its small ECE and NLL differences are within the observed seed variation. Ten-sample prediction had higher average ECE than AdamW. This is a small-scale result under fixed hyperparameters, not a general verdict on IVON or a controlled model-size study. Three seeds and 256 test questions do not establish statistical significance.

The hyperparameters were fixed before the test evaluation. No seed was discarded and no setting was selected using test scores. The original Llama-2 results are reference evidence and are kept separately in `docs/PAPER_REFERENCE.md`.

## Training and prediction cost

- **ADAMW:** mean training time 49.3 seconds per seed; maximum allocated training memory 2.96 GiB.
- **IVON:** mean training time 51.7 seconds per seed; maximum allocated training memory 2.96 GiB.

Training times exclude validation and test prediction. Peak allocated memory is the PyTorch allocation, not all memory occupied on the GPU. Ten-sample prediction requires ten forward passes; its actual times are in `summary/comparison.csv`. Raw training curves show minibatch cross-entropy, with IVON evaluated at a sampled adapter. They exclude KL and should not be described as the same complete objective.

## Evidence to open

- [Measured comparison](results/quick/summary/comparison.csv), [individual seeds](results/quick/summary/per_seed.csv), [paired differences](results/quick/summary/paired_deltas.csv).
- [Comparison plot](results/quick/summary/comparison.png), [training curves](results/quick/summary/training_curves.png), [reliability diagram for seed 21](results/quick/summary/reliability.png).
- [Training time and memory](results/quick/summary/training_details.csv), [environment](results/quick/environment.json), [configuration](results/quick/config.json).
- Per-example probabilities, labels and IDs, completed-run metadata, histories and execution logs are under `results/quick/`.

Recompute all test metrics with `python -m ivon_lora.report results/quick`. The compact evidence excludes adapter weights; the full Colab archive contains adapter and IVON optimizer state. The earlier two-step CPU smoke is retained only as implementation validation.

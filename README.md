# Variational Low-Rank Adaptation Using IVON

**Bayesian Learning project — completed Qwen2.5-0.5B experiment**

Team: Siddhu, Pathlavath Shiva Kumar, Gugulothu Ganesh.

- [Download the project ZIP](IVON_LoRA_Project.zip): complete training code, self-contained notebook, tests and documentation.
- [Open the executed Colab notebook](https://colab.research.google.com/drive/1xN7pUAkRs3z864_4bfyQeUFzsn6B-Q5X): GPU confirmation, test output, training logs and measured results.

## Actual results from our completed experiment

All six training runs completed on a Tesla T4. Qwen2.5-0.5B on ARC-Easy; 512 training / 128 validation / 256 held-out test questions; seeds 21, 42, 87; 128 steps per optimizer/seed; FP32; effective batch 8; rank-8 LoRA; 540,672 trainable parameters. Twelve correctness tests passed in Colab. Results were independently recomputed locally from saved per-example predictions.

Mean ± sample SD across three seeds:

| Method | Accuracy (%) ↑ | ECE (points) ↓ | NLL ↓ |
|---|---:|---:|---:|
| AdamW-LoRA | 73.18 ± 1.76 | 6.89 ± 2.30 | 0.680 ± 0.005 |
| IVON mean | 72.14 ± 0.23 | 6.99 ± 0.29 | 0.680 ± 0.003 |
| IVON, 10 samples | 69.40 ± 1.37 | 17.95 ± 1.49 | 0.884 ± 0.040 |

- **IVON mean minus AdamW:** accuracy -1.042 ± 1.966 points; ECE +0.105 ± 2.394 points; NLL -0.000165 ± 0.002448.
- **IVON, 10 samples minus AdamW:** accuracy -3.776 ± 3.133 points; ECE +11.066 ± 2.155 points; NLL +0.203 ± 0.044.

IVON mean did not reduce average ECE in this experiment. Its small ECE and NLL differences are within the observed seed variation. Ten-sample prediction had higher average ECE and NLL than AdamW. This is a small-scale result under fixed hyperparameters, not a general verdict on IVON or a controlled model-size study. Three seeds and 256 test questions do not establish statistical significance.

## Training evidence

- **AdamW:** mean training time 49.3 seconds per seed; maximum allocated training memory 2.96 GiB.
- **IVON:** mean training time 51.7 seconds per seed; maximum allocated training memory 2.96 GiB.
- The main training and evaluation pipeline completed in 720.8 seconds, about 12 minutes.

Training times exclude evaluation. Memory is peak PyTorch allocation. The notebook shows training curves, comparison and reliability plots, and saves raw predictions, data IDs, package versions, source/model revisions and completed-run metadata. IVON sampling averages probabilities over ten adapter draws and costs ten forward passes.

## What to show the professor

Show the updated slides, the executed Colab T4/test/training output, the comparison with seed variation, then the training and reliability plots. Explain LoRA, IVON's Gaussian adapter approximation, ECE and NLL. Our experiment is a small-scale adaptation with fixed starting hyperparameters, not an exact six-task reproduction or an isolated model-size study.

## Paper and code attribution

[Variational Low-Rank Adaptation Using IVON](https://arxiv.org/abs/2411.04421), Cong et al. (2024). [Authors' code](https://github.com/team-approx-bayes/ivon-lora). This project uses the [official IVON optimizer](https://github.com/team-approx-bayes/ivon) in an independently assembled Qwen experiment harness.

The paper's Llama-2 7B results and the two-step smoke checks are not our main results. All values above are from our completed three-seed Qwen test evaluation.

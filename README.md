# Variational Low-Rank Adaptation Using IVON

**A small-scale Qwen2.5-0.5B reproduction for a Bayesian Learning project**

Team: **Siddhu, Pathlavath Shiva Kumar, Gugulothu Ganesh**.

This project compares AdamW-LoRA, IVON evaluated at its posterior mean, and IVON evaluated by averaging 10 posterior-sample predictions on ARC-Easy. It includes a self-contained Colab notebook, paired-seed experiments, auditable metrics, plots, and a professor presentation guide.

## Start here

**The completed project is available as normal folders in this repository.**

- [Final PowerPoint with diagrams and measured charts](Professor_Presentation_Final.pptx)
- [Results and interpretation](RESULTS.md)
- [Executed T4 notebook](notebooks/IVON_LoRA_Executed_T4.ipynb)
- [Live saved Colab notebook](https://colab.research.google.com/drive/1xN7pUAkRs3z864_4bfyQeUFzsn6B-Q5X)
- [Professor demonstration and viva guide](SHOW_PROFESSOR.md)
- [Printable professor handout](SHOW_PROFESSOR.pdf)
- [Training source](src/ivon_lora), [settings](configs), and [measured evidence](results/quick)

For class, open the PowerPoint locally and show the saved notebook outputs. This repository is private, so GitHub links require an account with access. The PowerPoint is downloadable; GitHub does not run it as a slideshow.

### Run the project again

1. Download [the Colab notebook](notebooks/IVON_LoRA_Reproduction.ipynb).
2. Open [Google Colab](https://colab.research.google.com/), choose **Upload**, and select the notebook.
3. Select **Runtime > Change runtime type > T4 GPU**.
4. Select **Run all**. The notebook installs dependencies, runs offline correctness checks, performs a two-step smoke test, and executes the main experiment.
5. Download the result archive when the last cell finishes. Save the executed notebook too.

The notebook embeds readable project source, so a private GitHub repository does not need to be connected to Colab. No API key or paid service is required by the code. GPU availability remains subject to Colab's limits. Runtime is measured, not promised.

**Experiment status: COMPLETE.** All six main T4 runs finished. See [RESULTS.md](RESULTS.md) for measured evidence. The repository never substitutes paper numbers, toy-model checks, or smoke-test metrics for reproduction results.

## Research question

Does IVON's calibration behavior remain useful when adapting a substantially smaller, openly downloadable model on limited training data?

The nominal reduction from 7B to 0.5B is about 14-fold. Qwen also differs from Llama in architecture, tokenizer and pretraining. This is a **small-model transfer of the method**, not a controlled experiment that isolates model size.

## Method

LoRA freezes the pretrained weights and learns a low-rank update:

\[W' = W + (\alpha/r)BA.\]

AdamW learns a single set of adapter parameters. IVON maintains a diagonal Gaussian over the adapter parameters:

\[q(\theta)=\mathcal N(m,\operatorname{diag}(v)),\qquad
\min_q\ \mathbb E_q[\ell(\theta)] + \lambda^{-1}\operatorname{KL}(q\Vert p).\]

With the official `ivon-opt` implementation, the sampling variance is

\[v_i = [\mathrm{ess}\,(h_i+\mathrm{weight\_decay})]^{-1}.\]

`ess` scales the variational regularization and sampling noise. This project uses a cold posterior (`ess=1e6`), rather than claiming to target the ordinary posterior for 512 examples. The value is fixed before evaluating the test subset.

We train with cross-entropy over the four answer-token logits. IVON uses one parameter draw per optimizer step, shared across gradient-accumulation microbatches. At test time we compare a mean prediction with

\[p(y\mid x,D)\approx S^{-1}\sum_{s=1}^{S} p(y\mid x,\theta_s).\]

**Average probabilities, not logits.** These probabilities are conditional on the four listed options. They do not measure unrestricted text-generation confidence.

## Experiment protocol

- Model: `Qwen/Qwen2.5-0.5B`, base model, no instruction/chat template.
- Dataset: `allenai/ai2_arc`, `ARC-Easy`, distinct official train/validation/test splits.
- Quick profile: 512 training examples, 128 validation examples, 256 test examples.
- Seeds: 21, 42, 87. Same initial adapter hash, sample order, subset and training budget within each pair.
- Fixed subset seed: 2026. The training subset is constant across optimizer seeds.
- LoRA: rank 8, alpha 16, `q_proj` and `v_proj`, dropout 0.
- Training: 128 steps, microbatch 4, accumulation 2, effective batch 8.
- AdamW learning rate: 0.0002. IVON learning rate: 0.03, `rescale_lr=False`.
- IVON: Hessian initialization 0.001, clipping radius 0.001, beta2 0.99999, ESS 1,000,000.
- Both methods use weight decay 0.0001 and 10% warmup with linear decay.
- Prediction: AdamW point estimate, IVON mean, IVON average over 10 posterior draws.
- All training uses FP32 on T4. A four-token output projection reduces memory use and is checked against the full language-model logits.

We exclude non-four-choice items and prompts longer than 256 tokens rather than cutting away choices or the answer position. Exact excluded counts, selected IDs, labels and hashes are recorded in each run. Numeric answer keys map through the original option order.

Validation is reported separately. There is **no test-based tuning, checkpoint selection, early stopping or temperature scaling**. The final training step is evaluated. The fixed learning rates are starting settings rather than an equal-budget hyperparameter search, which limits optimizer-wide conclusions.

## Metrics and uncertainty

- **Accuracy:** fraction of questions answered correctly. Higher is better.
- **NLL:** mean negative natural log of the true-answer probability. Lower is better.
- **ECE:** weighted difference between accuracy and confidence in 15 equal-width confidence bins. Lower is better. CSV ECE is a fraction; multiply by 100 for percentage points.
- **Brier score:** mean sum of squared errors across all four probabilities. Lower is better.

`comparison.csv` reports mean and sample standard deviation across seeds. `paired_deltas.csv` reports within-seed IVON-minus-AdamW differences and their sample SD. This does not measure uncertainty over all possible test sets. Three seeds are insufficient for a strong significance claim.

ECE is sensitive to binning and small samples. The report includes 5/10/15/20-bin sensitivity and a clearly labeled reliability diagram for one seed. It never pools seeds into a different ensemble and calls it an individual run.

## Local commands

Use Python 3.10–3.12 and a CUDA-enabled PyTorch installation for real training.

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q
python -m ivon_lora.train --config configs/smoke.json --output runs/smoke
python -m ivon_lora.train --config configs/quick.json --output runs/quick
python -m ivon_lora.report runs/quick
```

The offline tests use a randomly initialized tiny Qwen and require no model download. Real Qwen training can run with `--device cpu`, but that is much slower. For a larger extension, use `configs/standard.json` with `--output runs/standard`. Change `dataset_config` to `ARC-Challenge` in a copied configuration for a second task, and use a separate output directory.

After editing source, rebuild the notebook with `python scripts/make_notebook.py`.

## Files produced by a completed run

```text
runs/quick/
  config.json                 resolved parameters and source hash
  environment.json            package versions and actual GPU
  split_manifest.json         dataset revision, selected IDs and exclusions
  seed_21/adamw/               metrics, predictions, history, adapter
  seed_21/ivon/                mean + sampled predictions and posterior state
  seed_42/...
  seed_87/...
  summary/comparison.csv      accuracy, ECE, NLL, Brier; mean and SD
  summary/paired_deltas.csv   paired optimizer differences
  summary/comparison.png      seed points, means and SD bars
  summary/reliability.png     confidence versus observed correctness
```

The base model is never committed to GitHub. IVON's saved optimizer state contains the posterior curvature. Mean adapter weights alone are insufficient to reproduce posterior sampling. Only load checkpoint files from trusted sources.

The run resumes by skipping completed optimizer/seed pairs. An interrupted pair starts again. Configuration or source mismatches require a new output folder. The report rejects missing seeds, mismatched initial adapters, different test IDs, and metrics that disagree with saved predictions.

## Original paper and attribution

[Variational Low-Rank Adaptation Using IVON](https://arxiv.org/abs/2411.04421), Bai Cong, Nico Daheim, Yuesong Shen, Daniel Cremers, Rio Yokota, Mohammad Emtiyaz Khan and Thomas Möllenhoff (2024). The authors' repository cites the NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability.

The paper reports average improvements of 2.8 accuracy percentage points and 4.6 ECE points for **IVON @ mean** over AdamW on its Llama-2 7B setup. Posterior sampling has a different accuracy/calibration tradeoff. See [the exact reference values](docs/PAPER_REFERENCE.md).

This student implementation uses the authors' [official IVON optimizer](https://github.com/team-approx-bayes/ivon). It is an independent experimental harness informed by the [official IVON-LoRA repository](https://github.com/team-approx-bayes/ivon-lora), not the paper's original code or an exact replication of all six tasks.

Data: [ARC dataset card](https://huggingface.co/datasets/allenai/ai2_arc) (CC BY-SA 4.0). Model: [Qwen2.5-0.5B card](https://huggingface.co/Qwen/Qwen2.5-0.5B) (Apache 2.0). Downloaded assets retain their original licenses. This repository stores result IDs and probabilities, not redistributed ARC question text.

See [SHOW_PROFESSOR.md](SHOW_PROFESSOR.md) for the demonstration order and viva preparation.

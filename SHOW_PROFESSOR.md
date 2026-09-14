# What to show the professor

## A five-minute demonstration

Use **Professor_Presentation_Final.pptx**. Slides 3–5 contain the method diagrams, slide 6 contains the paired experiment design, slide 9 contains your measured results, slide 11 shows all six training curves, and slide 14 states the conclusion. Speaker notes explain each slide.

1. **Research question (30 seconds).** Open the presentation title slide. Explain why an answer can be correct sometimes yet overconfident overall.
2. **Method (60 seconds).** Show LoRA's frozen model plus small trainable matrices. Explain that AdamW learns one adapter and IVON learns a Gaussian distribution over adapters.
3. **Implementation (60 seconds).** Open the Colab notebook. Show the passing correctness checks, the actual T4 device line and the experiment configuration. In `train_step`, point out `sampled_params(train=True)` around the accumulated backward passes.
4. **Evidence (90 seconds).** Show `comparison.csv`, `paired_deltas.csv`, the reliability plot and individual seed results. Explain the direction and size of the differences. State whether ECE and NLL agree.
5. **Limitations and next step (60 seconds).** Discuss the small sample, three seeds, fixed starting hyperparameters, and the change of model family. Propose a larger run or ARC-Challenge without claiming it has already happened.

## Your completed result in words

“We completed all six training runs on a Tesla T4 using Qwen2.5-0.5B. We evaluated the same 256 held-out questions over three paired seeds. IVON mean did not reduce average ECE in this experiment. Its small ECE and NLL differences are within the observed seed variation. Ten-sample prediction had higher average ECE than AdamW. This is a small-scale result under fixed hyperparameters, not a general verdict on IVON or a controlled model-size study. Three seeds and 256 test questions do not establish statistical significance.”

| Method | Accuracy (%) ↑ | ECE (points) ↓ | NLL ↓ |
|---|---:|---:|---:|
| AdamW-LoRA | 73.18 ± 1.76 | 6.89 ± 2.30 | 0.680 ± 0.005 |
| IVON mean | 72.14 ± 0.23 | 6.99 ± 0.29 | 0.680 ± 0.003 |
| IVON, 10 samples | 69.40 ± 1.37 | 17.95 ± 1.49 | 0.884 ± 0.040 |

Explain the paired gaps: -1.042 ± 1.966 accuracy points and +0.105 ± 2.394 ECE points for IVON mean minus AdamW. Show all three seeds, even when they disagree. Use `Professor_Presentation_Final.pptx`, then the executed notebook and the saved comparison and training plots.

## Opening explanation

“Our topic is Bayesian Learning. We are studying whether uncertainty-aware fine-tuning helps a small language model avoid overconfidence. We use LoRA to train a small number of parameters and compare AdamW with IVON, which maintains a probability distribution over those parameters. We evaluate correctness with accuracy, and confidence quality with ECE and negative log-likelihood. We use three paired seeds so that one lucky run does not determine the conclusion.”

## A simple calibration example

Suppose a model gives 80% confidence to ten answers. If eight are correct, that group is well calibrated. If only five are correct, the model is overconfident. Accuracy alone does not reveal the difference between 55% confidence and 99% confidence on the same wrong answer. NLL strongly penalizes confident mistakes.

This example explains the metric. It is not a measured project result.

## Likely viva questions

**What makes the method Bayesian?**
We approximate uncertainty over the trainable LoRA parameters with a Gaussian posterior. The variational objective combines expected prediction loss with regularization toward a prior.

**What does “variational” mean?**
The exact posterior is too difficult to compute. We choose a simpler family of distributions and optimize within it.

**What is low rank?**
Instead of updating a full weight matrix, LoRA represents its update as the product of two smaller matrices. Rank controls the update's capacity and parameter count.

**Does IVON make the whole language model Bayesian?**
No. Here only the adapter parameters have a learned approximate distribution. The base weights remain fixed.

**Why is the variance diagonal?**
Each parameter gets a variance but the approximation omits pairwise correlations. This reduces storage and computation but limits posterior expressiveness.

**Why are AdamW and IVON learning rates different?**
Their update scales differ. Equal numerical learning rates do not imply an equally sized or well-tuned update. Our fixed settings are an initial protocol. A stronger comparison would give both optimizers an equal validation-tuning budget.

**What is ESS?**
It sets the scale of the posterior approximation and sampling noise. Our ESS is larger than the dataset size, so it corresponds to a colder posterior. It is not the number of Monte Carlo samples.

**Why evaluate both IVON mean and posterior samples?**
They separate the effect of variational training from the effect of averaging several predictions. Sampling can improve confidence estimates while changing accuracy and increasing inference time.

**Why average probabilities rather than logits?**
Bayesian prediction averages conditional probability distributions. Softmax is nonlinear, so averaging logits would produce a different predictor.

**Is uncertainty free?**
IVON reuses curvature information to obtain a variance estimate. Sampling has a cost, and 10-sample prediction needs about 10 forward passes. We report actual training and prediction times separately.

**What is ECE's weakness?**
Its estimate depends on the bins and sample size. A model can also be calibrated while making unhelpful predictions. That is why we also show accuracy, NLL and Brier score.

**Why three seeds?**
They expose sensitivity to initialization and training order. Three is a feasible small experiment, not proof of significance. We use matching seeds and report within-seed differences.

**Why Qwen instead of Llama?**
The smaller public model makes the experiment easier to run on a T4. The architecture and pretraining also change, so the results cannot isolate size alone.

**Is this an exact reproduction?**
No. It is a small-scale adaptation with one task, a different model, fewer examples and steps, no quantization, and a slightly different prompt. The repository makes those differences explicit.

**What is your contribution?**
An independently assembled experimental harness for Qwen with paired seeds, careful label handling, tested memory-efficient logits, posterior prediction, saved provenance and independently recomputable calibration metrics. The underlying IVON algorithm belongs to the paper's authors.

**What if IVON does not improve the results?**
That is a valid outcome. Report it, discuss noise and tuning limitations, and propose a pre-specified follow-up. Do not select a better-looking seed or change the test set.

## Before leaving for class

- Keep a local copy of the slides and this guide.
- Download the executed notebook and the result archive.
- Open the private GitHub repository while signed in. A professor cannot open a private link without access.
- Read the method and explain it in your own words. Be transparent about assistance if your course requires it.
- Each team member should be ready to explain one part. Agree the speaking order yourselves; the repository does not invent individual contribution claims.


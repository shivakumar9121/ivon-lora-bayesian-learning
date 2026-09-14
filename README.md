# Variational Low-Rank Adaptation Using IVON

A small-scale Qwen2.5-0.5B reproduction for Bayesian Learning.

**Team:** Siddhu, Pathlavath Shiva Kumar, Gugulothu Ganesh.

## Current evidence

The complete project is prepared locally: source code, self-contained Colab notebook, tests, presentation and viva guide. Upload of those files is pending because browser file upload is unavailable. This repository currently contains the project landing page only.

Twelve offline correctness checks passed. A real Qwen CPU smoke run also completed, training 540,672 adapter parameters with both optimizers. The smoke run used two steps and eight test examples: it validates the pipeline and cannot establish a calibration benefit.

The main three-seed experiment has not yet completed. No Qwen calibration improvement is claimed.

## Main experiment protocol

- AdamW-LoRA, IVON at its mean, and IVON averaging 10 posterior predictions.
- ARC-Easy: 512 training, 128 validation and 256 held-out test questions.
- Paired seeds 21, 42 and 87, with matching initial adapters and training order.
- LoRA rank 8, alpha 16, q_proj and v_proj.
- Accuracy, 15-bin ECE, NLL and Brier score; mean and sample SD across seeds.
- Saved predictions, dataset IDs and immutable model/dataset revision hashes.

Qwen changes model family and pretraining as well as size. This experiment cannot isolate a model-size effect.

## References

[Paper](https://arxiv.org/abs/2411.04421) · [Official IVON-LoRA code](https://github.com/team-approx-bayes/ivon-lora) · [Official IVON optimizer](https://github.com/team-approx-bayes/ivon) · [Qwen model](https://huggingface.co/Qwen/Qwen2.5-0.5B) · [ARC data](https://huggingface.co/datasets/allenai/ai2_arc)

The published 2.8-point accuracy gain and 4.6-point ECE reduction refer to IVON evaluated at its mean on Llama-2. They are reference results, not our reproduction measurements.

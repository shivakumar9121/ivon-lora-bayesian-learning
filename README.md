# Variational Low-Rank Adaptation Using IVON

A small-scale Qwen2.5-0.5B reproduction for Bayesian Learning.

**Team:** Siddhu, Pathlavath Shiva Kumar, Gugulothu Ganesh.

## Project package and Colab

[Download the complete project ZIP](IVON_LoRA_Project.zip) · [Open the experiment in Colab](https://colab.research.google.com/drive/1xN7pUAkRs3z864_4bfyQeUFzsn6B-Q5X)

The ZIP contains the source code, self-contained notebook, tests, real-model smoke evidence, ten-slide professor presentation, and viva guide. Extract it to inspect the folders. The live Colab runner uses visible logs and checks for a CUDA GPU before setup. Select **T4 GPU**, not TPU.

## Current evidence

Twelve local correctness checks passed. A real Qwen CPU smoke run completed with both optimizers, training 540,672 adapter parameters. Two steps and eight test examples validate the pipeline; they cannot establish a calibration benefit.

**The main three-seed GPU experiment has not yet completed. No Qwen calibration improvement is claimed.** The live Colab session has verified a Tesla T4 GPU. Its completed run will generate comparison tables, individual seed results, paired differences, reliability plots and saved prediction evidence.

## Main experiment protocol

- AdamW-LoRA, IVON at its mean, and IVON averaging 10 posterior predictions.
- ARC-Easy: 512 training, 128 validation and 256 held-out test questions.
- Paired seeds 21, 42 and 87, with matching initial adapters and training order.
- LoRA rank 8, alpha 16, q_proj and v_proj; 128 steps per optimizer per seed.
- Accuracy, 15-bin ECE, NLL and Brier score; mean and sample SD across seeds.
- Saved predictions, dataset IDs and immutable model/dataset revision hashes.

Qwen changes model family and pretraining as well as size. This experiment cannot isolate a model-size effect.

## Professor demonstration

Start with `Professor_Presentation.pptx`, show the executed notebook and its checks, then show the measured table and reliability plot only after the main run finishes. `SHOW_PROFESSOR.md` provides a five-minute speaking order and viva answers.

## References

[Paper](https://arxiv.org/abs/2411.04421) · [Official IVON-LoRA code](https://github.com/team-approx-bayes/ivon-lora) · [Official IVON optimizer](https://github.com/team-approx-bayes/ivon) · [Qwen model](https://huggingface.co/Qwen/Qwen2.5-0.5B) · [ARC data](https://huggingface.co/datasets/allenai/ai2_arc)

The published 2.8-point accuracy gain and 4.6-point ECE reduction refer to IVON evaluated at its mean on Llama-2. They are reference results, not our reproduction measurements.

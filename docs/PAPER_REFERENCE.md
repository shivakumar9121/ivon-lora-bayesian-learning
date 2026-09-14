# Reference values from the original paper

Source: [Table 1, arXiv:2411.04421v2](https://arxiv.org/html/2411.04421v2).
These are **published Llama-2 7B results**, averaged over six tasks. They are not student reproduction measurements.

- AdamW: accuracy 76.1%, ECE 21.8 points, NLL 2.24.
- IVON at its mean: accuracy 78.9%, ECE 17.2 points, NLL 1.60.
- IVON with 10 posterior samples: accuracy 77.8%, ECE 10.6 points, NLL 1.11.

Subtracting the published averages gives +2.8 accuracy points and -4.6 ECE points for the mean predictor. With posterior sampling, the differences are +1.7 accuracy points and -11.2 ECE points. These are absolute percentage-point differences, not relative percent improvements.

The paper uses three runs and reports standard errors for individual task entries. This project instead reports sample standard deviations across seeds and labels that difference explicitly.

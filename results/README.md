# Evidence directory

`smoke_validation/` contains a completed real-Qwen CPU pipeline check: one seed, 16 train examples, 8 validation examples, 8 test examples and 2 optimizer steps. Its metrics are debugging evidence, not a research comparison.

`quick_split_check.json` records the separately checked 512 / 128 / 256 example subsets for the main experiment. The checks found no cross-split duplicate prompts or overlapping IDs.

The main three-seed Qwen experiment is still pending. There is no measured main-results table in this directory yet.

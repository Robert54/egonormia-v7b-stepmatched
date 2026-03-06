# Experiment Context

This export keeps the main public-facing training and sweep scripts from the EgoNormia SFT series.

## Included Runs

### v6

- natural-language-output baseline
- 5-task training mix
- `epoch = 3`

### v6b

- natural-language output plus sensibility generation
- 6-task training mix
- `epoch = 3`

### v7b-stepmatched

- name: `v7b-stepmatched`
- purpose: isolate the effect of removing generation tasks while keeping the glued CoT style
- causal comparison: `v7b-stepmatched` vs `v7 glued full6`

## Why It Matters

The earlier `v7` recipe mixed:
- 3 MCQ tasks
- 3 generation tasks
- glued external CoT

That run underperformed the historical MCQ-only baseline. The v7b ablation keeps the MCQ data
and CoT format, removes the generation tasks, and matches the v7 optimization budget.

## Core Settings

- model: `nvidia/Cosmos-Reason2-2B`
- frames: `8`
- batch per replica: `8`
- mini batch: `4`
- learning rate: `1e-5`

Per-run specifics:
- `v6`: `egonormia_llava_v6_train.json`, `epoch = 3`, `save_freq = 20`
- `v6b`: `egonormia_llava_v6b_train.json`, `epoch = 3`, `save_freq = 20`
- `v7b-stepmatched`: `egonormia_llava_v7_cot_mcq3_train.json`, `4890` samples, `epoch = 6`, `456` total steps, `save_freq = 10`

## Included Here

- portable launchers
- portable TOML templates
- a portable vLLM evaluator
- a generic checkpoint sweep wrapper
- dataset expectations

Raw data, media, checkpoints, and the rest of the research tree are intentionally excluded.

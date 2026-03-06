# Experiment Context

This export keeps the single most important training run from the EgoNormia SFT series.

## Mainline Run

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

- dataset: `egonormia_llava_v7_cot_mcq3_train.json`
- samples: `4890`
- model: `nvidia/Cosmos-Reason2-2B`
- epochs: `6`
- steps per epoch: `76`
- total steps: `456`
- save frequency: `10`
- max checkpoints kept: `50`

## Included Here

- the portable launcher
- the portable TOML template
- dataset expectations

Raw data, media, checkpoints, and the rest of the research tree are intentionally excluded.

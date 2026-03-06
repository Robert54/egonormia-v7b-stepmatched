# EgoNormia SFT Scripts

Portable export of the main EgoNormia SFT experiment scripts for Cosmos-Reason2-2B.

This package now includes three key training paths:
- `v6`: natural-language output baseline
- `v6b`: natural-language output plus sensibility generation
- `v7b-stepmatched`: MCQ-only glued-CoT run matched to v7 step budget

## Included

- `scripts/run_sft_v6.sh`
- `scripts/run_sft_v6b.sh`
- `scripts/run_sft_v7b_cot_mcq3_stepmatched.sh`
- `scripts/sweep_eval_checkpoints.sh`
- `eval/evaluate_egonormia_vllm.py`
- `eval/egonormia_prompts.py`
- `configs/egonormia_sft_v6.toml.template`
- `configs/egonormia_sft_v6b.toml.template`
- `configs/egonormia_sft_v7b_cot_mcq3_stepmatched.toml.template`
- `docs/experiment_context.md`
- `data/README.md`

## Main Runs

### v6

- 5-task natural-language output baseline
- `epoch = 3`
- dataset: `egonormia_llava_v6_train.json`

### v6b

- 6-task natural-language output plus sensibility generation
- `epoch = 3`
- dataset: `egonormia_llava_v6b_train.json`

### v7b-stepmatched

This is the main causal run from the v7b ablation:
- same glued CoT data style as v7
- generation tasks removed
- total optimization steps matched to v7

So the primary causal comparison remains `v7b-stepmatched` vs `v7`.

## Requirements

- `uv`
- a working `cosmos-reason2/examples/cosmos_rl` environment
- `HF_TOKEN` exported in your shell or stored in `~/.bashrc`
- EgoNormia media directory
- training JSONs as needed:
  `egonormia_llava_v6_train.json`
  `egonormia_llava_v6b_train.json`
  `egonormia_llava_v7_cot_mcq3_train.json`
- for evaluation and sweep:
  `egonormia_llava_test.json`
  `final_data.json`

## Run Training

### v6

```bash
bash scripts/run_sft_v6.sh \
  --cosmos-rl-dir /path/to/cosmos-reason2/examples/cosmos_rl \
  --data-json /path/to/egonormia_llava_v6_train.json \
  --media-dir /path/to/EgoNormia/video \
  --output-root /path/to/outputs \
  --seed 42
```

### v6b

```bash
bash scripts/run_sft_v6b.sh \
  --cosmos-rl-dir /path/to/cosmos-reason2/examples/cosmos_rl \
  --data-json /path/to/egonormia_llava_v6b_train.json \
  --media-dir /path/to/EgoNormia/video \
  --output-root /path/to/outputs \
  --seed 42
```

### v7b-stepmatched

```bash
bash scripts/run_sft_v7b_cot_mcq3_stepmatched.sh \
  --cosmos-rl-dir /path/to/cosmos-reason2/examples/cosmos_rl \
  --data-json /path/to/egonormia_llava_v7_cot_mcq3_train.json \
  --media-dir /path/to/EgoNormia/video \
  --output-root /path/to/outputs \
  --seed 42
```

Each launcher materializes a temporary TOML config from a template and then runs:

```bash
uv run --no-sync --directory "${COSMOS_RL_DIR}" \
  cosmos-rl \
  --config "${TMP_CONFIG}" \
  scripts/llava_sft.py
```

## Sweep Evaluation

```bash
bash scripts/sweep_eval_checkpoints.sh \
  --checkpoint-root /path/to/outputs/egonormia_sft_v6b_seed42 \
  --run-name v6b_seed42 \
  --python /path/to/cosmos_rl/.venv/bin/python \
  --test-path /path/to/egonormia_llava_test.json \
  --video-base /path/to/EgoNormia/video \
  --taxonomy-path /path/to/final_data.json
```

Add `--thinking` to evaluate with `<think>` enabled.

The sweep wrapper auto-detects either:
- `.../safetensors`
- `.../<timestamp>/safetensors`

## Shared Hyperparameters

- model: `nvidia/Cosmos-Reason2-2B`
- frames: `8`
- batch per replica: `8`
- mini batch: `4`
- learning rate: `1e-5`
- `dp_shard_size = 8`

## Per-Run Schedule

- `v6`: `epoch = 3`, `save_freq = 20`, `max_keep = 20`
- `v6b`: `epoch = 3`, `save_freq = 20`, `max_keep = 20`
- `v7b-stepmatched`: `epoch = 6`, `save_freq = 10`, `max_keep = 50`

## Notes

- Raw data, videos, checkpoints, and evaluation outputs are not included.
- The config template is path-agnostic on purpose; paths are injected by the launcher.
- The evaluator is included so the sweep script is runnable, but you still need the test JSON, taxonomy JSON, and videos.
- This repo is meant to share the main training and evaluation entrypoints cleanly, not the whole research tree.

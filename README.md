# EgoNormia v7b Step-Matched SFT

Portable export of the primary EgoNormia training run for Cosmos-Reason2-2B.

This package keeps only the main v7b training path:
- MCQ-only training
- glued CoT targets
- `epoch = 6` step-matched to v7
- 4890 training samples (`1630 x 3` MCQ tasks)

## Included

- `scripts/run_sft_v7b_cot_mcq3_stepmatched.sh`
- `configs/egonormia_sft_v7b_cot_mcq3_stepmatched.toml.template`
- `docs/experiment_context.md`
- `data/README.md`

## Why This Run

This is the main causal run from the v7b ablation:
- same glued CoT data style as v7
- generation tasks removed
- total optimization steps matched to v7

So the primary comparison is `v7b-stepmatched` vs `v7`.

## Requirements

- `uv`
- a working `cosmos-reason2/examples/cosmos_rl` environment
- `HF_TOKEN` exported in your shell or stored in `~/.bashrc`
- EgoNormia media directory
- the MCQ-only LLaVA training JSON:
  `egonormia_llava_v7_cot_mcq3_train.json`

## Run

```bash
bash scripts/run_sft_v7b_cot_mcq3_stepmatched.sh \
  --cosmos-rl-dir /path/to/cosmos-reason2/examples/cosmos_rl \
  --data-json /path/to/egonormia_llava_v7_cot_mcq3_train.json \
  --media-dir /path/to/EgoNormia/video \
  --output-root /path/to/outputs \
  --seed 42
```

The launcher materializes a temporary TOML config from the template and then runs:

```bash
uv run --no-sync --directory "${COSMOS_RL_DIR}" \
  cosmos-rl \
  --config "${TMP_CONFIG}" \
  scripts/llava_sft.py
```

## Default Hyperparameters

- model: `nvidia/Cosmos-Reason2-2B`
- frames: `8`
- batch per replica: `8`
- mini batch: `4`
- learning rate: `1e-5`
- epochs: `6`
- checkpoint save frequency: `10`
- max checkpoints kept: `50`
- `dp_shard_size = 8`

## Notes

- Raw data, videos, checkpoints, and evaluation outputs are not included.
- The config template is path-agnostic on purpose; paths are injected by the launcher.
- This repo is meant to share the main training entrypoint cleanly, not the whole research tree.

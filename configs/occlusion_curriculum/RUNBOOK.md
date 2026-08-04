# LeWM strict recurrent bottleneck runbook

Run all commands from the repository root. Set the workspace root once:

```bash
export WORLD_MODEL_ROOT="$(pwd)"
export LOCAL_DATASET_DIR="$WORLD_MODEL_ROOT/datasets"
```

The corrected architecture is `oot_leaky_bottleneck_ln_zeroh0`:

```text
s_t       = Encoder(o_t)
h_t       = LeakyRNN(h_(t-1), concat(s_t, E(a_(t-1))))
s_hat_t+1 = PointwisePredictor(concat(h_t, E(a_t)))
```

The old Transformer/visual-residual checkpoints are superseded historical
negative evidence and are not a same-architecture baseline.

## 1. Verify the migrated code

```bash
python -m py_compile \
  module_oot.py train_oot.py \
  schema_tools/build_occlusion_curriculum.py \
  schema_tools/audit_occlusion_dataset.py \
  analysis/metrics.py analysis/audit_effective_rank.py \
  analysis/select_occlusion_winner.py \
  scripts/run_occlusion_campaign.py
python -m pytest tests -q
```

## 2. Regenerate and inspect data

```bash
python scripts/run_occlusion_campaign.py prepare
```

`prepare` detects the legacy `primitive_v2` weights, writes a SHA256 archive
manifest, moves the old directory to a recoverable `_quarantine_primitive_v2_*`
directory, regenerates all four variants, and enforces the dataset audit.
Re-running it with already-current datasets skips regeneration.

Before training, inspect:

```text
results/occlusion_curriculum_20260728/dataset_audit/
datasets/primitive_v2/
```

Confirm Dfinal contains reveal indices 17/18/19, lengths 2/4/8, weight sums
20.2, reveal weights at least 2.5, and an acceptance pass. Also inspect the
rendered Dfinal sample grid manually.

## 3. Corrected smoke gate

```bash
python scripts/run_occlusion_campaign.py smoke
```

This runs corrected C0, Dshort, Dmid, and Dfinal for 1,000 steps. C0 also
exports a checkpoint and strict-loads it through a one-batch audit. Continue
only when all four return code 0, have no NaN/OOM, report no missing gradients,
and the architecture audit prints:

```text
batchnorm_modules: []
strict_recurrent_bottleneck: True
predictor_temporal_modules: []
predictor_positional_parameters: []
```

## 4. Full screen and selection

```bash
python scripts/run_occlusion_campaign.py screen
python analysis/select_occlusion_winner.py \
  --campaign-root results/occlusion_curriculum_20260728
```

`screen` serially trains C0, R1, R2, and R3 for 25 epochs. C0 is additionally
evaluated on the Dshort/Dmid/Dfinal validation splits. The selector compares
each candidate's normalized MSE with corrected C0 on exactly the same dataset
and split; reserved test data is not used for selection.

## 5. Replication

```bash
python scripts/run_occlusion_campaign.py replicate --selected <dshort|dmid|dfinal>
```

- projected full run <=28 h: run R4 and R5;
- 28--38 h: run R4, and start R5 only before 2026-08-08;
- >38 h: run R4 only and label the conclusion `two-seed tentative`.

L1 is deferred until the six core runs and application package are complete.

## 6. Status and report

```bash
python scripts/run_occlusion_campaign.py status
python scripts/run_occlusion_campaign.py report
```

`report` first freezes the validation-only selection decision, then evaluates
reserved Dshort/Dmid/Dlong/Dfinal test splits and renders the final artifacts.
Reserved-test metrics are therefore unavailable to the selector.

Training loss alone is never evidence of memory. A preliminary causal-memory
claim requires hidden-shuffle error increase >=5%, bootstrap CI lower bound >0,
reset/no-memory agreement, healthy representations, same-variant NMSE ratio to
corrected C0 <=1.10, and consistent direction in at least 2/3 seeds.

# LeWM V3 remote runbook

Run all commands from the repository root. Set the workspace root once:

```bash
export WORLD_MODEL_ROOT="$(pwd)"
export LOCAL_DATASET_DIR="$WORLD_MODEL_ROOT/datasets"
```

V3 does not overwrite or delete `primitive_v2` or the C0/R1/R2/R3 results.
It writes datasets under `datasets/primitive_v3` and results under
`results/occlusion_v3_reveal5_20260731`.

## 1. Verify the overlay

```bash
python -m py_compile \
  schema_tools/build_occlusion_v3.py \
  schema_tools/audit_occlusion_v3.py \
  analysis/audit_effective_rank.py \
  analysis/compare_v3_latent_drift.py \
  analysis/select_occlusion_v3_winner.py \
  scripts/run_occlusion_v3_campaign.py

python -m pytest tests -q
```

The model remains the strict recurrent bottleneck. `module_oot.py` and
`train_oot.py` are not replaced by this overlay. L2 uses the already existing
`oot_loss.stop_gradient_target` switch.

## 2. Preview and build V3 data

```bash
python scripts/run_occlusion_v3_campaign.py prepare --dry-run
python scripts/run_occlusion_v3_campaign.py prepare
```

`prepare` creates two paired datasets:

- `primitive_v3_h4_reveal5_original_motion.h5`
- `primitive_v3_h4_reveal5_motion15x.h5`

The builder derives both from the same eligible primitive_v1 source episodes.
It does not claim to resample new physical trajectories. The first occlusion
is fixed at onset 3 with length 4. The second occlusion has fixed length 4 and
balanced reveal indices 15/16/17/18/19.

If an interrupted or incomplete `primitive_v3` directory exists, `prepare`
moves it to a recoverable `_quarantine_primitive_v3_<UTC>` directory before
rebuilding. Do not manually delete partial HDF5 files.

Inspect:

```text
results/occlusion_v3_reveal5_20260731/dataset_audit/
datasets/primitive_v3/
```

Required audit results:

- all three JSON reports say `all_required_acceptance_pass: true`;
- each split contains all five schedules, balanced within one episode;
- random second-onset weights are zero and episode weight sums are 20.2;
- both datasets contain zero boundary-contact frames;
- paired comparison confirms the exact shared trajectory/split/schedule map;
- median nonzero step displacement ratio is at least 1.45.

Manually inspect both five-row montages. Near-boundary examples may remain,
but no object may touch or cross the renderer boundary.

## 3. Smoke gate

```bash
python scripts/run_occlusion_v3_campaign.py smoke
```

This runs N1/N2/N3 for 1,000 steps, exports a checkpoint for every run, and
strict-loads each checkpoint on both validation datasets. Continue only when:

- all return codes are zero;
- no NaN/OOM/traceback appears;
- every checkpoint saves and strict-loads;
- L1/L2 gradient tests pass;
- the architecture audit still reports no BatchNorm, attention, positional
  parameter, Transformer, or visual bypass.

## 4. Three controlled full runs

Run inside `screen` or through `nohup`:

```bash
python scripts/run_occlusion_v3_campaign.py screen
```

The serial order is:

1. N1: original motion + L1 + seed 3072;
2. N2: original motion + L2 + seed 3072;
3. N3: motion 1.5x + L2 + seed 3072.

Each epoch-25 checkpoint is evaluated on both V3 validation datasets.

## 5. Validation-only selection and replication

```bash
python scripts/run_occlusion_v3_campaign.py select
```

Selection uses macro final-reappearance NMSE. Cosine error is diagnostic only:
L2 is rejected as target co-adaptation if latent error falls while rank or
physical probes materially worsen. N4/N5 reproduce the selected data/loss
configuration with seeds 4096 and 5120.

For the completed 2026-07-31 campaign, the rebuilt selector returned
`selected=null`; therefore N4/N5 replication was not started. Do not run
`replicate` unless a future campaign first produces a valid non-null selection.

## 6. Reserved test and status

```bash
python scripts/run_occlusion_v3_campaign.py status
python scripts/run_occlusion_v3_campaign.py report
```

Reserved test results are generated only after validation selection is frozen.
Training loss alone is never reported as causal-memory evidence.

## First-occlusion decision

The first occlusion is intentionally not randomized in V3. The five-way
second onset already tests absolute-time shortcuts, while a fixed first onset
keeps the amount of pre-occlusion evidence and the first memory/recovery cycle
identical across all runs. Randomizing the first onset becomes a robustness
test only after one V3 configuration passes the prediction, probe, and hidden
intervention gates.

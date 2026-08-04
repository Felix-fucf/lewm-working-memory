# C0–R3 verified status — 2026-08-04

- Strict recurrent bottleneck code and synthetic tests are implemented.
- C0, R1, R2, and R3 each completed 25 epochs with final checkpoint SHA256 and raw audits.
- R3 produced a +6.79% shuffled-hidden increase, but matched-future direction alignment was approximately zero and position probes were negative.
- A later Dfinal identifiability audit found that 99.9875% of episodes belonged to same-current-input / different-future conflict groups.
- Therefore the original positive interpretation of R3 is superseded; the completed run remains valuable negative/mechanistic evidence.

See `results/summary.csv`, `results/manifests/dataset-audits/dfinal.audit.json`, and `docs/experiment-history.md`.

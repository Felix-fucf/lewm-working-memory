# V3 verified status — 2026-08-04

- The five-reveal dataset builder, paired audit, L1/L2 configs, selector, CKA diagnostic, and synthetic tests are implemented.
- Both original-motion and motion15x datasets passed the surviving audits.
- N1, N2, and N3 each completed 25 epochs with seed 3072 and were evaluated on both validation datasets.
- N4 and N5 were not started.
- Re-running the selector from surviving raw audits returned `selected=null`.
- N1 remains the frozen reference configuration; N2/N3 are ablations, not winners.

The current conclusion is negative: no candidate passed the combined representation, performance, physical-probe, and causal-memory gates. The visible-frame representation gate is the only frozen next step and remains `planned_not_run`.

See `results/summary.csv`, `results/manifests/selection_decision.json`, and `configs/occlusion_v3/experiment-handoff.md`.

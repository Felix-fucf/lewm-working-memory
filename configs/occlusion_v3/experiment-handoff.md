# LeWM V3 experiment handoff

## Current verified state

- N1, N2, and N3 each completed 25 epochs with seed 3072.
- Both V3 datasets passed the recorded dataset audits.
- N4 and N5 were not started.
- The selection artifact was rebuilt from surviving raw audits and returned `selected=null`.

## Interpretation

N1 is the frozen reference configuration, not a winner. Its large reset/shuffle effect coincides with prediction-rank collapse and almost no no-memory penalty. N2/N3 restore prediction rank under L2 but do not restore position semantics, matched-future direction, or a material hidden-state intervention effect.

The most defensible conclusion is therefore negative: current evidence does not support object-specific causal working memory.

## Frozen next action

Run a visible-frame representation gate for x/y, angle, and velocity before changing the recurrent model, encoder, projector, or prediction objective. This action is `planned`; no run artifact exists yet.

## Evidence

- `results/summary.csv`
- `results/manifests/selection_decision.json`
- `results/manifests/v3-selector-input/`
- `docs/experiment-history.md`

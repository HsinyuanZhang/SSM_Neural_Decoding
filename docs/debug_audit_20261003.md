# Tiny SSM debug audit

This audit is a held-in local development diagnostic, not FALCON scoring or a POSSM reproduction. It compares three inference state contracts on the same target and common `k=33` query tail: continuous state within each trial, forced state reset every 50 bins, and a fresh causal 50-bin rolling window at each scored bin. Ridge adaptation is fitted only from the prefix support rows.

The JSON artifact records official-reader array geometry, trial-change and evaluator-mask shapes, prediction/target scale, support and query error, quartile-of-trial error, source-session reset sensitivity, and learned linear-recurrence time constants. It distinguishes observed state-reset sensitivity from causal claims about why a decoder fails.

The successful APST baselines use session-adaptive association/profile conditioning and a transformer-like decoder under their own frozen protocol. This pilot uses 64-wide causal SSM/RNN comparators and a fixed local held-in temporal split; numerical results are not directly comparable across those contracts.

## Completed local diagnostic

The artifact `results/debug_audit/debug_audit.json` contains six seed-0 checkpoints. On M1, the strongest query result in this audit was oscillator full-trial plus support-only output-residual ridge (`R²=0.4624`); GRU full-trial plus ridge was `0.4423`. Resetting state every 50 bins changed those to `0.4264` and `0.4165`, respectively. Thus the tested reset contract has a measurable effect on M1 but does not establish that long state is the source of its gain.

On M2, all tested modes remained negative after prefix-only ridge. The best was GRU full-trial (`R²=-0.1525`); its 50-bin reset result was `-0.1675`. Oscillator full-trial was `-0.8698`, versus `-0.7037` after reset. This rejects neither calibration shift nor representation mismatch: it only shows that changing this bounded state-reset convention did not recover a positive local-target score.

Learned linear recurrence time constants were short: median approximately 0.17–0.18 s for diagonal models and 0.26–0.28 s for oscillator models.

## Sampling audit added after the state audit

The historical endpoint score keeps only bin 50 onward in every usable trial.  The appended `sampling_audit` uses the same 33-prefix, 50-bin-capable trial cohort and reports both that endpoint subset and every finite, evaluator-valid bin in those query trials.  It is descriptive; it does not fit query labels or select a model.

For M2, the endpoint subset contains 2,747 of 14,115 valid query bins (19.46%). Its two physical target standard deviations are 0.00331 and 0.00310, compared with 0.01164 and 0.01025 for all bins. Its relative-time median is 0.879, compared with 0.492 for all bins. The analogous M1 subset covers 31,971 of 50,591 bins (63.20%). Therefore the earlier support/query scale observation cannot be treated as evidence of session-scale change: endpoint selection is a material confound.

Without retraining, seed-0 full-trial checkpoints were re-scored after the same prefix-only ridge fit. On M2, oscillator changes from endpoint `R²=-0.8698` to all-valid-bin `0.1879`; GRU changes from `-0.1525` to `0.1309`. On M1, oscillator changes from `0.4624` to `0.5172` and GRU from `0.4423` to `0.5136`. These are different scoring populations, so they establish endpoint sensitivity rather than an improvement. The artifact also records support all-bin and support bin-50-onward coverage for the same comparison.

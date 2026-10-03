# Final independent review: Mamba-3 cross-session PEFT

**Review date:** 2026-10-03 HKT  
**Reviewer role:** independent reviewer, separate from the implementation workers  
**Requested reviewer configuration:** `gpt-5.6-sol`, reasoning effort `xhigh`  
**Final status:** **PASS for the stated local diagnostic scope**

The implementation, the 50-fit primary rank-4 matrix, and the separate 6-fit
rank-8 B/C capacity control pass the independent acceptance checks. All 56
completed fits are internally consistent and their saved performance values are
admissible as **local held-in cross-session target-prefix diagnostic results**.
They are not blind held-out benchmark results.

This pass also has a naming boundary. `bc_lora`, `bc_dt`, and
`sparse_sdt_m3` are project-specific Mamba-3 adaptations. `state_offset` and
`memba_causal` are functional custom compatibility ports whose receipts state
`upstream_reproduction=false`. Their results must not be described as faithful
reproductions of the original SDLoRA, State-offset Tuning, or Memba papers.

This document supersedes `PRELIMINARY_REVIEW.md`.

## Admitted experimental scope

Both tasks use the fixed source checkpoint
`mamba3_official_w256_l4_n32`, context 128, batch size 32, 1,000 update steps,
and the same causal fixed-window policy. For each target session, the runner
uses the first 26 usable trials for supervised prefix fitting. It uses the next
7 trials for prefix-only checkpoint selection. The query begins after trial 33.
The runner excludes query labels from the training and validation tensors. It
reads those labels only after it selects the best checkpoint.

The primary matrix contains nine configurations. `none` is deterministic and
has one fit per task. Each trained method has three adapter/training seeds per
task. The source checkpoint remains fixed at source seed 0. The capacity-control
matrix contains only rank-8 `bc_lora`, with three seeds per task. This matrix
resides in a separate root. The primary rank-4 summary excludes it.

The artifacts save two final query views:

- **legacy:** one endpoint per eligible query window under the historical
  protocol.
- **all-valid:** every eval-valid bin in the held-in query trials.

`zero` applies the selected adapted model directly. `ridge` fits an exact
centered ridge correction using only the 33-trial target support and then
applies it to query predictions. Every table below reports variance-weighted
R². Values for trained methods are mean ± population standard deviation over
three seeds. The `none` row has `n=1`, so no between-seed uncertainty is
available.

## Primary rank-4 results

The all-valid cohort is the broad query view and is the clearest primary table.
The parameter columns are the live trainable counts recorded by each method.
For gradual full fine-tuning, this is the final total after staged unfreezing.
Training begins with only decoder-root I/O and normalization parameters.

| method | n | M1 trainable | M1 zero all-valid | M1 ridge all-valid | M2 trainable | M2 zero all-valid | M2 ridge all-valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1 | 0 | 0.386210 | 0.482983 | 0 | -0.005922 | 0.120918 |
| io | 3 | 21,264 | 0.564905 ± 0.001504 | 0.566524 ± 0.002935 | 25,858 | 0.190175 ± 0.002777 | 0.207412 ± 0.001833 |
| lora | 3 | 36,672 | **0.586090 ± 0.005416** | **0.582380 ± 0.005618** | 36,744 | 0.242534 ± 0.007991 | 0.254652 ± 0.008754 |
| bc_lora | 3 | 19,776 | 0.540264 ± 0.003701 | 0.530465 ± 0.001518 | 19,848 | 0.216850 ± 0.028124 | 0.230353 ± 0.025453 |
| bc_dt | 3 | 19,808 | 0.540417 ± 0.003698 | 0.530633 ± 0.001491 | 19,880 | 0.216868 ± 0.028090 | 0.230396 ± 0.025411 |
| full | 3 | 2,750,544 | 0.554760 ± 0.004760 | 0.558693 ± 0.003942 | 2,755,138 | 0.253235 ± 0.002809 | **0.265668 ± 0.002071** |
| sparse_sdt_m3 | 3 | 15,168 | 0.556715 ± 0.003561 | 0.550192 ± 0.002569 | 15,240 | 0.250788 ± 0.017085 | 0.259635 ± 0.013785 |
| state_offset | 3 | 11,072 | 0.555343 ± 0.008170 | 0.542870 ± 0.011936 | 11,144 | **0.266005 ± 0.010592** | 0.264488 ± 0.011026 |
| memba_causal | 3 | 18,752 | 0.535356 ± 0.004918 | 0.529044 ± 0.004822 | 18,824 | 0.198983 ± 0.023406 | 0.220601 ± 0.018445 |

The corresponding legacy cohort is materially different, especially for M2,
and must not be silently substituted for the all-valid view.

| method | M1 zero legacy | M1 ridge legacy | M2 zero legacy | M2 ridge legacy |
|---|---:|---:|---:|---:|
| none | 0.296805 | 0.404706 | -0.119464 | -0.229543 |
| io | 0.499829 ± 0.001706 | 0.501549 ± 0.003570 | 0.004937 ± 0.002398 | 0.007120 ± 0.001317 |
| lora | 0.524502 ± 0.006355 | 0.520075 ± 0.006860 | -0.031220 ± 0.011072 | -0.030129 ± 0.010048 |
| bc_lora | 0.468819 ± 0.004412 | 0.457202 ± 0.001848 | -0.032094 ± 0.029894 | -0.016796 ± 0.019115 |
| bc_dt | 0.469001 ± 0.004409 | 0.457404 ± 0.001817 | -0.030785 ± 0.029420 | -0.015829 ± 0.018829 |
| full | 0.487015 ± 0.005118 | 0.491355 ± 0.004428 | 0.060361 ± 0.005667 | 0.060934 ± 0.000685 |
| sparse_sdt_m3 | 0.489544 ± 0.004126 | 0.481686 ± 0.003124 | -0.022610 ± 0.007752 | -0.019153 ± 0.012769 |
| state_offset | 0.488170 ± 0.010034 | 0.473030 ± 0.014327 | -0.039776 ± 0.007854 | -0.034667 ± 0.012581 |
| memba_causal | 0.463050 ± 0.006177 | 0.455279 ± 0.006008 | -0.015376 ± 0.004016 | -0.008700 ± 0.002314 |

Within the tested recipe, rank-4 full-projection LoRA has the highest M1 mean
under both all-valid scoring modes. On M2, the custom state-offset port has the
highest direct all-valid mean, while gradual full fine-tuning has the
highest all-valid ridge mean. The M2 ridge difference between full fine-tuning
and state offset is only about 0.00118, and three seeds do not support a strong
ranking claim. The M2 direct ranking changes with the cohort. State offset
leads only the all-valid direct mean at 0.266005, while its legacy direct mean
is -0.039776. Gradual full fine-tuning leads the M2 legacy direct mean at
0.060361. `sparse_sdt_m3` is competitive on M2 with substantially fewer
trainable parameters. The current M2 Memba port varies more across seeds and
does not match the leading configurations under this recipe.

These observations describe these configurations, data splits, and budgets.
They do not establish that one adaptation family is generally superior.

## Capacity-matched B/C control

The primary fixed-rank comparison gives compact B/C LoRA about half the
trainable parameters of full-projection LoRA. The separate rank-8 control tests
whether that count difference alone explains the gap. Rank-8 B/C has about 8%
more parameters than rank-4 full-projection LoRA:

| task/configuration | trainable | zero all-valid | ridge all-valid | zero legacy | ridge legacy |
|---|---:|---:|---:|---:|---:|
| M1 LoRA rank 4 | 36,672 | 0.586090 ± 0.005416 | 0.582380 ± 0.005618 | 0.524502 ± 0.006355 | 0.520075 ± 0.006860 |
| M1 B/C LoRA rank 4 | 19,776 | 0.540264 ± 0.003701 | 0.530465 ± 0.001518 | 0.468819 ± 0.004412 | 0.457202 ± 0.001848 |
| M1 B/C LoRA rank 8 | 39,552 | 0.537598 ± 0.005495 | 0.532350 ± 0.007309 | 0.465427 ± 0.006681 | 0.459314 ± 0.008921 |
| M2 LoRA rank 4 | 36,744 | 0.242534 ± 0.007991 | 0.254652 ± 0.008754 | -0.031220 ± 0.011072 | -0.030129 ± 0.010048 |
| M2 B/C LoRA rank 4 | 19,848 | 0.216850 ± 0.028124 | 0.230353 ± 0.025453 | -0.032094 ± 0.029894 | -0.016796 ± 0.019115 |
| M2 B/C LoRA rank 8 | 39,696 | 0.228777 ± 0.009099 | 0.236742 ± 0.007035 | -0.050191 ± 0.045632 | -0.034186 ± 0.024507 |

Rank-8 B/C does not close the full-projection LoRA gap: its all-valid zero/ridge
means remain lower by 0.04849/0.05003 on M1 and 0.01376/0.01791 on M2. The
control modestly raises the M2 B/C mean relative to rank 4, while M1 is flat to
slightly lower. This evidence does not support a capacity-only explanation
under the fixed learning rate, step count, and rank scaling. It also does not
isolate the B/C target itself from rank geometry, initialization, or optimizer
tuning, so it is not evidence that B/C adaptation is intrinsically wrong.

## DT-bias result is real but small

The near-equality of `bc_lora` and `bc_dt` is not caused by an unused optimizer
path. In every one of the six `bc_dt` fits:

- the optimizer contains the four live `dt_bias` tensors in a dedicated
  32-parameter group with learning rate `1e-5` and weight decay 0.
- all 32 head values differ from the source in both best and last checkpoints.
- the largest per-fit absolute change is between 0.004394 and 0.006413, far
  inside the source-relative ±1 clamp.
- all recorded DT values are finite and positive, and all ADT values are finite
  and negative.
- the final predictions differ from paired `bc_lora` predictions.

The resulting mean all-valid gain over `bc_lora` is only 0.000153/0.000168
(zero/ridge) on M1 and 0.000017/0.000043 on M2. This rules out this particular
small-LR constrained recipe as a material improvement here. It does not rule
out time-scale tuning with another parameterization or optimization schedule.

## Independent verification evidence

The reviewer-owned oracle imports neither the project PEFT implementation nor
the project summarizer. It checks artifact bytes and checkpoint tensors
directly. It completed both matrices with no failures:

- primary matrix: 50 expected, 50 completed, 50 verified.
- B/C rank-8 control: 6 expected, 6 completed, 6 verified.
- 224 saved R² values independently recomputed in float64, with global maximum
  absolute error `1.482469e-7`.
- 112 ridge prediction arrays independently reconstructed with a NumPy
  float64 solve, with maximum physical prediction error `4.768372e-7`.
- all support, legacy, and all-valid indices and truths identical within each
  task across methods, ranks, and seeds.
- zero support/query overlap and legacy indices a subset of all-valid indices.
- every best/last checkpoint receipt, replay configuration, source hash,
  normalizer, parameter count, final tensor hash, optimizer authorization, and
  changed-parameter allowlist consistent.
- every trained last checkpoint changes every optimizer-authorized tensor and
  no unauthorized tensor. All zero-initialized adapter factors become nonzero.
- all frozen source tensors in both best and last checkpoints are bit exact.
- step-0 is present, validation steps are complete, and every selected best
  step is the first argmax of the prefix-validation log.
- all recorded losses, gradient norms, predictions, DT probes, and resource
  fields are finite, with positive gradient norms.
- gradual full fine-tuning has the expected optimizer stages at steps 0, 200,
  and 400, with cumulative parameter membership and passed frozen-stage audits.
- sparse selection uses the independently recomputed per-layer warm-up top-k,
  and all 100 warm-up losses are finite. Selected state buffers replay exactly.
  The sparse-state code restores the original B/C biases bit-for-bit before
  sparse training.

The project summarizer independently reports `verified`, 50 fits and zero
errors for the primary matrix, and `verified`, 6 fits and zero errors for the
capacity control. Sparse layer keys are integers inside PyTorch checkpoints and
strings after JSON serialization. The summarizer compares their recursive JSON
meaning. The values and selected layers are identical. JSON serialization
causes this expected type change. The type change does not indicate an artifact
defect.

A separate exact forward replay used the formal `eval_batch_size=256` and the
recorded single-thread CPU score reduction for M1 `state_offset_seed0` and
`memba_causal_seed0`. All 22 saved arrays—including support predictions and
zero/ridge legacy/all-valid predictions—are bit exact, and all eight recorded
R² values are bit exact Python floats. This exact forward-replay claim is
limited to those two fits. The remaining 54 fits have complete tensor,
provenance, cohort, saved-array score, and calibration audits, but were not all
rerun through the GPU forward operator.

The pre-formal independent validation also includes:

- 20/20 CPU regression tests across PEFT, runner, sparse-state, research-port,
  and data/calibration behavior.
- a toy end-to-end label-perturbation test in which changing every query label
  leaves the best step and complete selected state dictionary bit exact while
  changing final query metrics.
- an M2 real-checkpoint GPU smoke covering all nine methods, with exact startup
  equality, causal prefixes after activation, intended nonzero gradients, and
  no frozen gradients.
- an M1 real-checkpoint research-port smoke with exact startup equality,
  per-layer finite nonzero adapter gradients, fixed-batch future-perturbation
  causality, cross-sample isolation, and window-reset repeatability.
- direct comparison of the custom state-offset effective readout against the
  pinned kernel's rotated query coordinates: 99.993896% bit-exact values,
  maximum absolute difference 0.00390625, and mean absolute difference
  `2.384e-7`.

The fixed-batch wording matters. The pinned fused kernel itself shows small
numerical changes when evaluation batch shape changes. Historical source
predictions used batch 32, while the formal matrix uses batch 256. The `none`
checkpoint weights, cohort indices, and truths are exact, but old and new
predictions are not expected to be bit exact across those batch shapes. Every
method in the formal comparison uses the same batch-256 path, so this does not
confound within-matrix comparisons.

## Provenance closure

The primary copied matrix SHA256 is
`b06fe74b3807e818b6fffadef8873f3bcb6ce9ab2f0942edd4ce1c52bfa8b6f8`.
The capacity-control copied matrix SHA256 is
`47432707e0a0e52067daa1c189425047fae7478421f0722778ee0d38f3b9e805`.
Both run manifests identify the same snapshotted launcher SHA256,
`65a3a2001fcb3bc3012e5c11a24c15297c2466e6b1cc0a0e5f60fd7a85ea901d`,
and both launchers exited successfully without hash drift.

Every fit records the same frozen experiment-source hash set. The runner hash
is `d28cfe432445d3e42a9affd3cebb268d05c29ab2f9146db743ebfe555eb093ae`,
the ordinary PEFT implementation hash is
`e100021dff40cb9538700f8d2c14a6e25140263c2ad4ceeebd9275ad3c1ec77b`,
and the research-port hash is
`2db1cd6356858e12d2526563376dfe745bb18536d438e37c2de81fc33fed24a6`.
The pinned official Mamba-3 expected and actual commit is
`e9594ce1c732d97440f0332fdc43170a2294dbfa`. Manifests also record the
isolated dependency path, physical GPU assignment, PyTorch/CUDA versions,
source checkpoint hash, and source normalizer hash.

The task cohorts are:

| task | support | legacy query | all-valid query | support/query overlap |
|---|---:|---:|---:|---:|
| M1 | 4,192 | 31,971 | 50,591 | 0 |
| M2 | 2,053 | 2,747 | 14,115 | 0 |

The index SHA256 values are:

| task | support | legacy | all-valid |
|---|---|---|---|
| M1 | `b72dba800f930259241d7e8681746adbf1a78ae0190a3f06b8c011c278780944` | `ef8b4e2f5540a3bc9635da95f2e610b92ae571a8292b36c373be41bd0f319884` | `5b8be6ddfbd58edc28097c500b5e52532db4a711b2f2537a3c4c6cbc92936728` |
| M2 | `471b01f254cbcaea7e8f67e98d78144aafd5eec06f80020147400307856925f6` | `7918e13f6985c6634dc8c593211891017be7f6ac72a436a453d8a67edbe9e7b3` | `3fd5abf8209729768c343ade5072217e7b5127801431520141d0288cff8551f1` |

## Limits on the conclusion

The following facts bound the pass:

1. These are local held-in cross-session development targets. The review
   accessed and scored no blind held-out split.
2. There is one fixed source checkpoint and one local target session per task.
   The three seeds vary adapter initialization and training sampling, not source
   pretraining or target-session identity.
3. Three seeds provide a variability check but do not justify confidence
   intervals or formal significance claims. Reported standard deviations are
   population SDs of these three runs.
4. Ridge and direct values answer different questions. The legacy and
   all-valid cohorts also answer different aggregation questions. Every claim
   must name the cohort and calibration mode.
5. The control changes B/C rank from 4 to 8 while holding the optimizer recipe
   fixed. It is a close parameter-budget control, not a complete study of rank,
   learning-rate, or target-location interactions.
6. The custom research-port scores support the implemented ports under these
   checks. They neither reproduce nor refute the original papers.

Within those limits, there is no remaining review blocker. The performance
values in this document and `FINAL_REVIEW.json` may be used. Values outside the
two verified roots, or claims that omit the local diagnostic and custom-port
qualifiers, are not covered by this pass.

## Review artifacts

- `FINAL_REVIEW.json`: canonical machine-readable verdict, grouped values,
  hashes, evidence, and limitations.
- `independent_oracle_main_final.json`: 50-fit primary independent audit.
- `independent_oracle_bc_rank8_final.json`: 6-fit capacity-control independent
  audit.
- `independent_artifact_oracle.py`: reviewer-owned read-only oracle.
- `exact_replay_m1_research.json`: selected two-fit exact forward replay.
- `exact_checkpoint_replay.py`: selected replay driver.
- `gpu1_smoke.json`: all-method M2 real-checkpoint GPU smoke.
- `project_summary_main/verification.json`: project summarizer result for 50
  primary fits.
- `project_summary_bc_rank8/verification.json`: project summarizer result for 6
  control fits.

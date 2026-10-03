# Independent cross-session iteration review

Date: 2026-10-03

## Review decision

The independent review accepts the formal Stage A and Stage B artifacts. Both roots are complete and internally consistent.

The review also accepts both local deployment candidates. The M1 candidate is small `none` with ten calibration trials. The M2 candidate is wide `lora` seed 0 with 33 calibration trials and an unmerged LoRA runtime.

The accepted performance values are development results on one target session per task. They are not private EvalAI scores.

This review did not change model code, runner code, prior results, or the user audit. All independent reports are in this review directory. The reviewer did not run a submission controller or send a network request.

## Formal artifact status

| Scope | Expected | Verified | Errors | Decision |
|---|---:|---:|---:|---|
| Stage A adaptation fits | 116 | 116 | 0 | accepted |
| Stage A selected evaluations | 44 | 44 | 0 | accepted |
| Stage B source fits | 4 | 4 | 0 | accepted |
| Stage B adaptation matrices | 6 | 6 | 0 | accepted |
| Stage B adaptation fits | 78 | 78 | 0 | accepted |
| Stage B selected evaluations | 78 | 78 | 0 | accepted |
| M1 legal-budget fits and evaluations | 12 | 12 | 0 | accepted |
| M2 full-supplement fits and evaluations | 6 | 6 | 0 | accepted |
| M2 legal-budget fits and scores | 4 | 4 | 0 | accepted |
| M1 and M2 public banks | 20 | 20 | 0 | accepted |
| Local candidate images | 2 | 2 | 0 | accepted |

The current pinned runtime matches the saved runtime. It uses PyTorch `2.5.1.post303`, CUDA `11.8`, and Triton `3.5.0`.

The combined analysis contains 194 adaptation fit entries and four source fits. It contains 122 selected evaluations. All profiles use identical task cohorts, and no query result selected a fit.

## Stage A result

Stage A uses support normalization, interleaved prefix validation, and trial-causal fixed windows. The primary metric is direct, variance-weighted R-squared in physical units. Values are mean plus or minus population standard deviation across three seeds. The deterministic `none` method has one run.

| Task | Method | Direct all-valid R-squared |
|---|---|---:|
| M1 | `none` | 0.673568 |
| M1 | `io` | 0.658046 +/- 0.000170 |
| M1 | `lora` | 0.650999 +/- 0.005825 |
| M1 | `affine` | 0.615414 +/- 0.008108 |
| M1 | `affine_lora` | 0.648637 +/- 0.006187 |
| M1 | `offset_rotated` | 0.643893 +/- 0.001815 |
| M1 | `offset_original` | 0.641828 +/- 0.001490 |
| M1 | `full` | 0.641207 +/- 0.001599 |
| M2 | `none` | 0.233793 |
| M2 | `io` | 0.272154 +/- 0.002232 |
| M2 | `lora` | 0.319075 +/- 0.016364 |
| M2 | `affine` | 0.299668 +/- 0.001306 |
| M2 | `affine_lora` | 0.324433 +/- 0.024952 |
| M2 | `offset_rotated` | 0.315052 +/- 0.008758 |
| M2 | `offset_original` | 0.333497 +/- 0.012924 |
| M2 | `full` | 0.345628 +/- 0.002798 |

No M1 adaptation beat `none`. M2 full tuning gave the highest Stage A score. Its mean gain over `none` was 0.111835.

These results reject a general claim that parameter adaptation always helps after support normalization. The M1 target needs a different source model, context policy, calibration budget, or selection rule.

## Normalization and method checks

The independent replay produced these source-checkpoint scores:

| Task | Source normalizer | Support z-score | Fixed causal EMA |
|---|---:|---:|---:|
| M1 | 0.386209 | 0.673568 | 0.699254 |
| M2 | -0.005922 | 0.233793 | 0.266910 |

The fixed EMA leaves the support prefix unchanged. It starts its updates at the first query bin. Each query output uses only earlier moments. The update uses the residual from the previous mean.

The float64 score oracle agreed with the runner within `6.75e-8`. Cohort indices and raw physical truth matched the formal artifacts.

The GPU method smoke tested all eight Stage A methods. Every method matched the source output before optimization. Every method passed the future-prefix causality check. Trainable methods had finite gradients and changed at least one permitted parameter. Frozen parameters did not change.

The affine export uses the effective LoRA weight before it folds the channel transform:

```text
W_effective = W_base + delta_W
W_export = W_effective diag(g)
b_export = b_effective + W_effective b
```

The state-offset coordinate review found one sign error in the original explanation. With the implemented rotation convention, the rotated offset maps to `R(-Phi) h_offset` in physical coordinates. The original-coordinate method gives a constant physical-coordinate offset. Zero-phase and two-dimensional quarter-turn tests enforce this result.

## Stage B result

Stage B uses recording-causal fixed windows. It transfers each Stage A prefix-selected learning rate without a new query search. Stage B compares the old source model with small and wide session-specific source models.

| Task | Profile | `none` | `io` | `lora` | `affine_lora` | `offset_original` |
|---|---|---:|---:|---:|---:|---:|
| M1 | old | 0.670773 | 0.642537 +/- 0.007232 | 0.636240 +/- 0.003619 | 0.634068 +/- 0.004150 | 0.628282 +/- 0.004892 |
| M1 | small | 0.692343 | 0.720758 +/- 0.000980 | 0.722944 +/- 0.001767 | 0.722502 +/- 0.001861 | 0.717219 +/- 0.000996 |
| M1 | wide | 0.691231 | 0.723933 +/- 0.001375 | **0.733201 +/- 0.000791** | 0.732981 +/- 0.000793 | 0.713568 +/- 0.002898 |
| M2 | old | 0.216524 | 0.296406 +/- 0.000277 | 0.333822 +/- 0.008853 | 0.314631 +/- 0.028978 | 0.266039 +/- 0.010806 |
| M2 | small | 0.369642 | 0.366213 +/- 0.004489 | 0.386640 +/- 0.014002 | 0.398763 +/- 0.011241 | 0.389889 +/- 0.001173 |
| M2 | wide | 0.436734 | 0.460176 +/- 0.002136 | 0.486414 +/- 0.010595 | 0.477305 +/- 0.004846 | **0.491188 +/- 0.002023** |

Wide `lora` gave the highest Stage B M1 mean. Its paired gain over old `lora` was `0.096960 +/- 0.004169`.

Wide `offset_original` gave the highest Stage B M2 mean. Its paired gain over old `offset_original` was `0.225149 +/- 0.012496`.

The small model is competitive on M1. Small `lora` scored 0.722944, and wide `lora` scored 0.733201. The wide model gave a clearer M2 gain.

Stage A and Stage B use different context policies. Therefore, the difference between Stage A and Stage B old-profile values is not a model-capacity effect. The Stage B profile comparison supports a joint effect from source training, session read-in, capacity, and recording context. It does not isolate those factors.

## Source-fit evidence and uncertainty

| Task | Profile | Parameters | Best source-validation R-squared | Best step |
|---|---|---:|---:|---:|
| M1 | small | 362,272 | 0.785499 | 1,400 |
| M1 | wide | 2,750,544 | 0.788954 | 900 |
| M2 | small | 364,562 | 0.585956 | 1,400 |
| M2 | wide | 2,755,138 | 0.611253 | 1,700 |

An independent M1 replay matched the saved source-validation scores. The float64 oracle deltas were at most `8.75e-8`.

Each task and capacity has one source-training seed. The adaptation standard deviations do not include source-training uncertainty. A capacity ranking based on these four source fits remains provisional.

## Ten-trial M1 budget

The target recording starts with a neural-only segment `[0, 262)`. The first real trial starts at bin 262. The local `support_trials=11` field includes this leading segment and the first ten real trials.

All ten real trials have at least 50 bins. The public calibration export must select raw trial indices 0 through 9 before any length filter. The query starts at raw real-trial index 32, so the query cohort stays equal to the formal cohort.

The original 21-entry M1 probe stopped before query evaluation. The old-profile `affine_lora` seed 2 fit extended to 8,000 steps. Its best step was 7,700, so it failed the declared convergence rule. The launcher rejected the method and did not start the small or wide profiles.

The replacement probe includes only `none` and `io`. It uses the authenticated Stage A IO learning rate of `0.0003`. Independent verification accepted all 12 fits and all 12 evaluations with no errors.

| Profile | `none` | IO mean plus or minus population SD |
|---|---:|---:|
| old | 0.673078 | 0.622328 +/- 0.001329 |
| small | 0.688083 | 0.663260 +/- 0.056962 |
| wide | 0.687293 | **0.696100 +/- 0.001750** |

All IO fits passed the convergence and admissibility rules. Every selection receipt states `query_metrics_read=false`. The float64 score oracle differed from saved metrics by at most `7.42e-8`.

Wide IO gave the highest M10 mean. Its gain over small `none` was 0.008017, but it used about 7.6 times as many parameters.

The project selected small `none` as the M1 candidate. It avoids seed selection and gives a stable, smaller deployment. The review accepts this resource and robustness tradeoff.

## M2 full-tuning supplement

Formal Stage B did not include source-initialized full tuning. The separate supplement covers small and wide profiles with three seeds each. It uses the Stage A prefix-selected full learning rate of `0.0001`.

The first supplement process failed before it started a child fit. The process passed integer `1` as the `CUDA_VISIBLE_DEVICES` environment value. Python rejected the environment with a `TypeError`. The failed root contains zero fit results and zero evaluations.

The launcher changed this value to string `"1"`. The restarted run used a new frozen root. Independent verification accepted all six fits and all six evaluations.

| Profile | Full direct all-valid R-squared |
|---|---:|
| small | 0.380401 +/- 0.001082 |
| wide | 0.464195 +/- 0.001057 |

All six fits used source initialization and learning rate `0.0001`. They passed convergence, admissibility, prefix selection, source, cohort, truth, and float64 score checks.

The supplement did not change the Stage B M2 ranking. Wide `offset_original` scored 0.491188, and wide `lora` scored 0.486414. Both exceeded wide full tuning.

## Thirty-three-trial M2 budget

The legal M2 probe uses raw NWB trial IDs 0 through 32. It keeps all 33 trials, including six trials shorter than 50 bins. The query starts after this calibration prefix and keeps the formal 14,115-bin cohort.

The frozen probe contains wide `none` seed 0 and wide `lora` seeds 0 through 2. LoRA uses the authenticated learning rate of `0.003`. Independent verification accepted all four fits and scores.

| Method and seed | Direct all-valid R-squared |
|---|---:|
| `none`, seed 0 | 0.430611 |
| `lora`, seed 0 | 0.505119 |
| `lora`, seed 1 | 0.485418 |
| `lora`, seed 2 | 0.455886 |
| `lora`, three-seed mean +/- population SD | 0.482141 +/- 0.020232 |

All fits completed before the first query decode. The probe did not use query values to select a learning rate or checkpoint.

The project selected wide `lora` seed 0 as the M2 deployment candidate. The selection uses the fixed deployment seed. It does not use the query ranking. The selected checkpoint has best prefix-validation step 300 and completed 2,000 steps.

## CPU and public-calibration evidence

The plain CPU Mamba-3 implementation uses materialized BF16 weights. It is a numerical approximation of the official GPU kernel. It is not a pointwise or bitwise replacement.

Full-cohort M1 source-validation replay passed the predeclared score tolerance:

| Profile | GPU R-squared | CPU R-squared | Absolute delta |
|---|---:|---:|---:|
| small | 0.779044 | 0.779064 | 0.0000204 |
| wide | 0.784092 | 0.784082 | 0.0000105 |

Both profiles produced finite values. Both passed a strict future-prefix causality check. Their plain GPU scores differ from the saved source-bank scores because this check measures the mean-fold plain export.

The public-calibration loader audit passed all 20 held-in files. It checked seven M1 files and 13 M2 files. It checked raw trial indices, NWB identities, neural bytes, masks, and model isolation. The loader retained 43 selected M2 trials shorter than 50 bins.

The packaged CPU smoke passed roster routing, payload closure, task geometry, finite execution, batch caps, and observation history. Its timing values are diagnostic.

Both final candidates passed the full-query score gate. The limit was an absolute CPU and GPU R-squared difference of `0.001`.

| Candidate | Query bins | Saved GPU R-squared | CPU R-squared | Signed CPU minus GPU delta |
|---|---:|---:|---:|---:|
| M1 small `none` | 50,591 | 0.688083249 | 0.688087938 | +0.000004689 |
| M2 wide `lora` seed 0, unmerged | 14,115 | 0.505130862 | 0.505095058 | -0.000035804 |

The M1 check used the saved official GPU prediction archive and made no CUDA calls. It bound the complete input and artifact hash chain. The source and selected states were tensor-exact. The CPU future-prefix check was bitwise equal.

The M2 check compared the unmerged GPU graph with the unmerged CPU graph. The unmerged GPU score was `0.000012140` above the saved merged score. The independent replay matched every scalar and all 14,115 cohort indices and truth rows. Its causal evidence remained bound to the exact audit script and CPU runtime hashes.

Both candidates produced finite values. These checks establish score-level CPU parity for the selected checkpoints.

## M1 public-bank export

The selected M1 candidate uses one fixed source state for all seven public calibration sessions. Each session gets its own neural normalizer from its first ten real trials.

The independent export audit passed. It verified exactly these seven session tags:

```text
20120924  20120926  20120927  20120928  20121004  20121017  20121024
```

Each bank selected raw NWB trial IDs 0 through 9 before any length filter. Each normalizer matched an independent calculation from those ten trials. Each output-label mean and standard deviation stayed equal to the frozen source values.

Every bank used `method=none`, learning rate zero, and step zero. Each unmerged and merged bank matched all 38 source state tensors exactly.

The payload contains the seven models, seven normalizers, the CPU decoder, the Falcon adapter, its package initializer, and the Mamba license. All manifest hashes match. The payload contains no raw NWB file, receipt, train log, or other diagnostic artifact.

An isolated CPU smoke loaded a real bank and produced finite `[1, 16]` output. Both `predict` and `observe` advanced the causal history. Eleven calibration, payload, and decoder tests passed.

## M2 public-bank export

The first M2 exporter tried to merge LoRA weights into BF16 base weights. It stopped at the eighth session. The `Run1_20201030` fit converged, but its maximum pointwise fold difference was `0.020363`. The exporter rejected this result and wrote no payload manifest.

The replacement exporter preserved LoRA as a separate inference path. It applied the same unmerged graph to all 13 public sessions. Each linear adapter computes one base projection and one LoRA delta projection.

The independent audit accepted all 13 banks. Each bank uses raw trial IDs 0 through 32, seed 0, learning rate `0.003`, and 2,000 completed steps. Every checkpoint passed the convergence, admissibility, frozen-state, and state-geometry checks.

Two sessions fail the old pointwise merge condition. They are `Run1_20201030` and `Run2_20201124`. Their unmerged outputs pass all deployment gates. The review preserves the failed merged root as failure evidence.

The largest absolute public-prefix CPU and GPU R-squared difference was `0.000340697`. The target bank for `Run1_20201028` is tensor-exact to the selected legal-probe state. The payload contains 13 checkpoints, 13 normalizers, three runtime files, and the Mamba license. Its manifest declares exactly 30 files.

## Candidate image evidence

The reviewer inspected both completed SDK containers and images. The checks used local, read-only Docker operations on stopped containers. They did not start a container or use a network.

| Candidate | Payload files with manifest | SDK minival sessions | CPU limit | Memory limit | Network | Result |
|---|---:|---:|---:|---:|---|---|
| M1 small `none` | 19 | 4 | 2 | 4 GiB | none | accepted |
| M2 wide unmerged `lora` | 31 | 7 | 2 | 4 GiB | none | accepted |

Each container exited with status zero and did not report an out-of-memory event. Neither container mounted `/payload`. The reviewer read `/payload` from each stopped container and matched every path and SHA-256 value to its host manifest.

Both images contain Falcon Challenge SDK `1.0.2` and CPU PyTorch `2.5.1`. Native and container predictions matched element by element for every minival session. Truth arrays, masks, and direct R-squared values also matched.

The minival runs check the SDK interface and image closure on public data. The separate full-query reports establish selected-candidate score parity.

## Known limitations

The Stage A receipt has a low-severity wording defect. For `io` and `full`, two descriptive root-bias fields state that the root input bias is frozen. The authoritative trainable paths, optimizer groups, counts, and hashes show that it is trainable. The defect does not change a fit or score.

Stage A and Stage B each use one target session per task. The results do not establish generalization across target sessions or animals.

The Stage B source comparison has one source seed per task and capacity. Its source uncertainty is unknown.

The M1 ten-trial and M2 33-trial results use the same target recordings as their development queries. They are not private benchmark results.

The M2 unmerged runtime executes a separate LoRA delta projection. It has more projection work than a merged runtime. The container checks establish correctness within the declared CPU and memory limits.

The independent reviewer did not run the EvalAI controller, push an image, send a submission, or poll a private result. This review makes no private-score or leaderboard claim. Project submission receipts do not change any result in this report.

## Evidence files

- `stage_a_final`, `stage_a_final.fits.csv`, and `stage_a_final.evaluations.csv`
- `stage_b_final`, `stage_b_final.grouped.csv`, and `stage_b_final.sources.csv`
- `normalization_replay_m1/metrics.json` and `normalization_replay_m2/metrics.json`
- `gpu_method_smoke_m1/report.json`
- `stage_a_prefix_selection_audit.json`
- `stage_b_m1_source_validation_replay.json`
- `m1_m10_raw_trial_budget_audit.json`
- `m1_m10_v2_old.json`, `m1_m10_v2_small.json`, and `m1_m10_v2_wide.json`
- `m2_full_supplement_small.json` and `m2_full_supplement_wide.json`
- `m2_m33_probe_audit.json`
- `mamba3_cpu_gpu_full_source_m1.json`
- `m1_small_none_cpu_saved_gpu_full_query.json`
- `m1_small_none_public_bank_audit.json`
- `m1_small_none_sdk_container_audit.json`
- `m2_unmerged_public_bank_audit.json`
- `m2_unmerged_full_query_independent_audit.json`
- `m2_unmerged_sdk_container_audit.json`
- `public_calibration_loader_audit.json`
- `packaged_cpu_smoke.json`

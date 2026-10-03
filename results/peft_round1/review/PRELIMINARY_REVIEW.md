# Independent PEFT review log

Status: **CORE METHODS PASS; FORMAL MATRIX BLOCKED BY ARTIFACT PROVENANCE AND
VERIFICATION** as of 2026-10-03 13:15 HKT.

This review is independent of the implementation workers. It uses static
inspection, CPU tests and external oracles, and an explicitly authorized
single-GPU smoke on GPU 0. It did not edit reviewed source or run training.

## Current corrected implementation

The nine initial implementation failures listed below have been corrected in
the current PEFT modules and runner. The combined independent CPU regression
suite (`test_mamba3_research_adapters.py`, `test_peft_experiment.py`,
`test_peft.py`, `test_debug_experiment.py`, and
`test_sparse_state_tuning.py`) passes 20/20 tests. The corrected ordinary PEFT
ports also reconstruct the two real width-256 source state dictionaries
exactly for `none`, `io`, `lora`, `bc_lora`, `bc_dt`, and `full`: 70 source
tensors compared per configuration, with zero missing, extra, or unequal
tensors at startup.

The current custom research-port source has SHA256
`2db1cd6356858e12d2526563376dfe745bb18536d438e37c2de81fc33fed24a6`.
An independent real-checkpoint M1 width-256/layer-4/state-32 GPU smoke found:

- exact source equality at adapter initialization for both `state_offset` and
  `memba_causal` (`max_abs = 0.0`);
- receipt and live trainable counts agree (11,072 and 18,752 respectively),
  with no frozen gradient;
- every research-adapter factor has a finite nonzero gradient after activating
  its zero-initialized factor;
- a future-only input perturbation leaves the earlier prefix bit-exact, changing
  the other member of a fixed-size batch leaves sample 0 bit-exact, and a
  repeated independent window is bit-exact;
- the Mamba-3 state-offset readout reconstruction agrees bit-for-bit with the
  pinned kernel's stored rotated query for 99.993896% of inspected values. The
  maximum absolute difference is 0.00390625 and mean absolute difference is
  `2.38e-7`, consistent with the declared PyTorch versus PTX approximate
  nonlinear/trigonometric path. The port now includes the official
  `tanh(angle) * pi`, causal cumulative angle, modulo, and BF16 quantization
  semantics.

Changing batch shape from two samples to one produces small differences in the
pinned base kernel itself (up to `7.75e-5` in the inspected input), so batch
isolation is assessed at fixed batch shape by changing only the other sample.
That decisive check is exactly zero for both active research ports. A separate
current-source M2 GPU smoke on GPU 1 reports all nine methods at exact startup
equality, causal after-update prefixes, and nonzero intended gradients.

The model/runner mathematics are therefore cleared for the formal matrix. Two
artifact-chain items still block launching it as a reviewable formal run:

1. `run_peft_two_gpu.py` records commands but does not copy or hash the matrix
   file and does not record its own hash. Per-fit code hashes do not include the
   matrix or launcher. These hashes must be fixed before a long run starts so a
   later file change cannot make the launch configuration ambiguous.
2. `scripts/summarize_peft.py` at SHA256
   `6a701c6ad8b665877237480945be384d81234822b6cd7e9efbbc04801e6ab73e`
   cannot verify current runner output. It requests nonexistent
   `truth_legacy_physical` and `truth_allvalid_physical` NPZ keys, and its
   best-step check labels the final best score as the step-0 score, rejecting a
   legitimate nonzero best step. It also checks only artifact existence rather
   than checkpoint replay content, uses the first 16 raw bytes instead of a
   complete cohort hash, omits the support cohort from cross-fit identity,
   checks only that the sparse receipt path exists, silently treats incomplete
   completed fits as waiting, and reads resource/count fields under names the
   runner does not emit.

No formal PEFT performance value has yet been produced or admitted.

## Confirmed blockers in the initial implementation

1. `debug_experiment._freeze_io` selects parameters by the substrings `in_proj` and `out_proj`. On an official Mamba-3 width-128/layer-2/state-16 model, it marks 210,562 parameters trainable although exact decoder-root input/output plus final normalization contains only 1,666 parameters for the inspected 8-input/2-output instance. On width-256/layer-4/state-32, the respective counts are 1,674,498 and 3,330. Historical or future Mamba-3 results produced through this helper cannot be described as decoder-I/O-only.

2. Initial `LoRALinear` parameters and row masks are allocated with default CPU/FP32 constructors even when the wrapped linear layer already has a different device or dtype. A CPU float64 oracle reproduces a forward failure (`double != float`); the submitted runner inserts the adapters after moving the model to CUDA, so the same implementation would have a device mismatch.

3. Initial B/C row LoRA stores `A` for every output row and masks unused rows in the computed delta. Those unused values are still optimizer parameters and are included in the reported trainable count. For width-256/layer-4/state-32/rank-4, this inflates the trainable count by 16,896 parameters per model. The inspected M1 receipt reports 36,672 adapter parameters although only 19,776 values affect the function; M2 reports 36,744 versus 19,848. This blocks parameter-efficiency comparisons.

4. The B/C configuration first installs a full core-input LoRA and then discards it in favor of a row-masked wrapper, while retaining both entries in the receipt. The receipt therefore does not describe the final module graph.

5. The initial toy test does not consume the B/C rows in its forward pass. It therefore cannot prove that the intended B/C parameters receive gradients or update the output. Three passing tests do not cover B/C update isolation, aliases, compact parameter counting, or a reloadable merged export.

6. The initial gradual-full-FT runner snapshots all parameters frozen at step 0, later deliberately unfreezes them, and finally rejects any of those changes as a frozen-parameter violation. The `full` method cannot complete under its own audit. Its configuration receipt also claims all parameters are initially trainable even though the runner immediately freezes them and begins with a projection subset.

7. The initial runner has no step-0 validation candidate. A run can therefore select a checkpoint that is worse than the source model even when every update hurts validation performance.

8. The initial prediction artifact omits all-valid truth and predictions, so its all-valid R2 cannot be independently recomputed. It also omits a training log, last checkpoint, complete replay arguments, complete source-code hashes, official-source commit evidence, and DT stability evidence.

9. The initial launcher does not enforce `PYTHONNOUSERSITE=1`, does not record the required interpreter and isolated dependency path, and its initial matrix points to width-128 sources. The primary matrix must use the width-256/layer-4/state-32 source checkpoints. Width-128 may be retained only as a scale comparator, with the missing M1 historical startup snapshot disclosed.

## Independent reference facts

The pinned official Mamba-3 core has the per-layer input-projection row order `[z, x, B, C, dd_dt, dd_A, trap, angles]`. For width-256/layer-4/state-32, every layer has a `(1120, 256)` input-projection weight with these half-open row ranges:

| component | rows |
|---|---:|
| z | `[0, 512)` |
| x | `[512, 1024)` |
| B | `[1024, 1056)` |
| C | `[1056, 1088)` |
| dd_dt | `[1088, 1096)` |
| dd_A | `[1096, 1104)` |
| trap | `[1104, 1112)` |
| angles | `[1112, 1120)` |

No duplicate parameter objects or exact shared-storage parameter aliases were found in either unmodified official Mamba-3 base instance (width-128/layer-2/state-16 or width-256/layer-4/state-32). Adapter insertion must preserve that property.

The historical width-256 source artifacts have real `code_hashes_start.json` files for M1 and M2. Their prediction NPZs independently reproduce every reported legacy and all-valid zero/ridge variance-weighted R2 within `2.4e-7`. Within each task, width-128 and width-256 artifacts have exactly equal query indices and truth arrays. The reference query-index SHA256 values, computed from contiguous little-endian `int64` bytes, are:

| task | legacy | all-valid |
|---|---|---|
| M1 | `ef8b4e2f5540a3bc9635da95f2e610b92ae571a8292b36c373be41bd0f319884` | `5b8be6ddfbd58edc28097c500b5e52532db4a711b2f2537a3c4c6cbc92936728` |
| M2 | `7918e13f6985c6634dc8c593211891017be7f6ac72a436a453d8a67edbe9e7b3` | `3fd5abf8209729768c343ade5072217e7b5127801431520141d0288cff8551f1` |

The support/query construction is trial-disjoint for the inspected held-in targets. M1 uses 4,192 eval-valid support bins, 31,971 legacy query bins, and 50,591 all-valid query bins. M2 uses 2,053, 2,747, and 14,115 respectively. No support index overlaps either query cohort.

## Required acceptance checks

No PEFT performance number is admissible until all applicable checks pass:

- exact initial functional equality to the loaded source checkpoint in evaluation mode;
- a live first-backward/first-step check proving nonzero LoRA gradient and a nonzero permitted update;
- compact and truthful trainable/storage/effective parameter counts;
- no duplicate parameter object or overlapping parameter storage after adapter insertion;
- B/C row ranges derived from live core attributes, checked against the full official row layout;
- startup/final per-parameter hashes and a fail-closed whitelist diff, including immutable `dd_A` and angle rows;
- a separate, smaller-LR or constrained `dt_bias` group, source-relative delta evidence, and finite observed DT statistics;
- trial-disjoint target-prefix train/validation bounds, source-only normalization, and unchanged query cutoff/cohorts;
- step-0 checkpoint selection candidate and query-label perturbation invariance of the selected adapted-state hash;
- independently recomputable legacy and all-valid R2 from saved indices, truths, and predictions;
- true run-start code/config/dependency hashes plus source checkpoint and normalizer hashes;
- exact replay arguments, train log, last/best checkpoint distinction, environment, and schedule receipts;
- explicit naming that custom B/C row LoRA is not SDLoRA;
- no state-offset or Memba compatibility claim from the old wrapper, which exposes neither decoder-level state carry nor a completed recurrent integration.

Final pass/fail and the set of admissible performance values remain pending corrected code and completed artifacts.

## Stage-4 upstream compatibility findings

The inspected upstream revisions are `6a0a7247cc8905d01c70089f620d46d565c259d2` for SSM-PEFT/SDLoRA, `2cff98578668cd02ca1729f18f9f8ada05303aa6` for State-offset Tuning, and `99f8401cf8891affaba57be85f9a27c4f728d31c` for Memba.

- Original SDLoRA is not B/C-row LoRA. Its released configuration targets `A_log`, `x_proj_B`, `x_proj_C`, and `out_proj`. It uses a dense `A_log` perturbation during a warm-up phase to rank state/channel dimensions, then trains selected sparse entries while applying LoRA to projection parameters. Official Mamba-3 has no static `A_log`; its data-dependent `dd_A` is emitted by input-projection rows. Any new selection rule for Mamba-3 is therefore a custom Mamba-3-inspired port and cannot be reported as the original SDLoRA method.

- Original State-offset Tuning adds `silu(z) * C * offset[d,n]` to the scan output (or `silu(z) * offset[d]` in its output-only variant). A generic block-output bias is not equivalent. The current official Mamba-3 fused interface and project decoder wrapper do not expose the internal `C` and gate tensors needed by that implementation, so a faithful port requires a new kernel/operator boundary and independent recurrence checks.

- The released Memba language path computes `v_updated` for each sequence chunk but never assigns it back to `current_membrane`; its model loop initializes `prev_membrane` but never receives or assigns a current membrane from a block. Thus the advertised temporal and cross-layer transfer are absent in that path. The vision path does transfer a membrane, but constructs it by averaging across the concatenated chunk/batch dimension after processing the full sequence. Feeding that summary to the next layer leaks future-token information into earlier-token outputs in a causal decoder. The released code therefore cannot be copied as evidence of a causal Mamba-3 Memba implementation. A corrected causal adapter would be a new port and needs future-perturbation and chunk-continuation oracles.

# SSM pilot model contract

This is a small causal motor-decoding pilot, not a POSSM implementation, official benchmark submission, or hardware performance claim.

`ModelConfig` and `build_model` construct `diag`, `osc`, `bank`, `selective`, and same-width `gru` models. `forward(x[B,T,C])` returns `[B,T,O]`; `step(x[B,C], state)` returns one prediction and next state. Passing state to a later chunk equals concatenated input and never reads a future bin.

`diag` has one real stable state per latent. `osc` has a damped real 2x2 rotation state, so the same nominal width has twice the state values of `diag`; a state-width-matched width-32 oscillator comparator is future work. `bank` allocates a two-value state per latent but its frozen grouped codebook mixes real entries (second value redundant) and damped oscillators. With its current four-group cycle, it has only four exact coefficients: real `.82`, real `.93`, oscillator `(.88,.43 rad)`, and oscillator `(.97,1.12 rad)`. At 20 ms bins, its two rotational frequencies are approximately 3.4 and 8.9 Hz. It is a fixed baseline, not evidence that four modes are optimal. `selective` is diagonal with one current-input-only scalar rate gate. Main SSMs are linear dynamical systems after input projection; the profile frontend is nonlinear/static only in profile construction.

Raw-channel frontends assume compatible electrode order and are a same-electrode cross-session pilot. `frontend='profile'` folds support/session-derived unit profiles into frozen static weights, pools units before recurrence, is invariant to joint channel/profile permutation, and can mask padded/missing units. It has no learned profile encoder; the input/profile map is bilinear and its profile rank is at most `profile_size`. This is a set-generalization prototype, not a validated association architecture. Profiles must not use query labels.

Stable retention is strictly below one. Power-of-two mode uses `1 - 2^-k`, `k=1..16`, a shift-subtract leak convention. Bank coefficients are frozen buffers.

Calibration uses target support only. Centered RLS is two-pass; `StreamingRidge` is one-pass sufficient statistics. Delta rule is approximate. No query labels update any model or readout.

`characterize` uses CPU single-thread step timing for at least 200 steps; optional CUDA timing synchronizes every step. Accounting separates projection, recurrence, readout, state allocation, and storage, including profile basis versus per-session folded `[C,W]` storage. W8 is per-row-scale fake quantization; S8/S16 state scales are calibrated from source or target support and frozen for query. A fixed update budget equalizes optimizer updates, not parameter count: GRU has materially more parameters and recurrence MACs. These are not integer-chip, energy, throughput, or RTL claims.

Install with `pip install -e SSM`; core dependencies are Torch and NumPy. Local FALCON/PynWB loading is optional: `pip install -e 'SSM[falcon]'`.

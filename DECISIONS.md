# Binding decisions (from Gregory, 2026-10-14) and where each lives

| # | Decision | paper_v2.tex | code (hodgemix-go) |
|---|---|---|---|
| 1 | Base = Microsoft BitNet b1.58 2B4T (ternary −1/0/+1, Champ packed: 4 trits/U8 + BF16 weight_scale); other open checkpoints converted to that format and mixed into it | Sec.1 | main.go `packTrits`, `run` |
| 2 | DEC cochain resizing: W = 1-cochain on input/output channel graph; Hodge-star-weighted (unit stars) restriction/prolongation; MRH buckets order merge/split by conversion error; channels matched first | Eq.(1)-(3),(7)-(10),(26); Sec.4 | math.go `heapKey`,`chanCoords`,`bucketOf`,`splineGrid`,`sampleGrid`; whitney.go |
| 3 | Per-tensor output index (source channels+weights, permutation, MRH bucket, R/P weights, spline over channel index) stored as its own SIGNED, invertible BNZ4 chunk next to ternary weights and scales; byte layout defined | Sec.6 table | chanmap.go (`Encode`/`DecodeChanMap`, CRC-32, bijectivity check) |
| 4 | Mixed-radix index: length factorization (2560 = 5×2⁹), MRH buckets by digits, spline coefficients in frequency domain per level, resize = keep/drop DFT bins per radix level, phase-correlation matching (as in FFT correction chip), store only low-frequency coefficients | Eq.(4)-(6),(12) | math.go `radices`,`gridSize`,`NewSpectrum`,`seedShift` |
| 5 | Two-way Laplacian: down on source side, up on target side; Kuramoto lock on harmonics with heap-weight coupling (jev_kuramoto) | Eq.(4),(14)-(17) | kuramoto.go, synth.go |
| 6 | Tuning algorithm: seed (phase corr per digit + Re-Basin/OT), anneal coarse→fine MRH-weighted, refine on h with Hodge re-projection until r converges, emit BNZ4. No risks section | Alg.1, Eq.(12)-(20) | synth.go `synth` |
| 7 | Pipeline: align → resize → mix full precision → ternarize (rotation + absmean) → distill into 2B4T/adapters; ternary-native TIES for already-ternary same-base models; teachers via top-k logits | Sec.1, Eq.(21)-(25) | rotate.go, rotplan.go; distillation NOT in code (protocol) |
| 8 | Measurements as protocol only (PPL/KL ± rotation, merge KL ± alignment, Hodge split vs merge KL, r as predictor, low-bin DFT energy after sorting, chunk size vs dense permutation). No fabricated numbers | Sec.8 | — |
| 9 | Do not trust the "Argon" (Google) paper; its ideas only as candidates verified against literature | Hadamard (QuaRot), Whitney/de Rham (AFW) kept; "ω ∝ node potential" and "A_ij = cochain inner product" not adopted (no derivation) | — |
| 10 | No developer choices: every constant derived (paper_v2 App.A); flags may override only | App.A | params.go |
| 11 | Never modify paper.tex / paper.v1.tex | untouched (md5 cca24009…) | — |

## Champ / EdgeAI compatibility (from the attached bnz-loader.tgz + champ-learn/bnz-nn zip, 2026-10-14)
hodgemix output must load in bnz-loader (js/bnz-loader.js, bnz4dec.js, bitnet-rt.js = edgellm/bitnet/model/model.go). Requirements, all now met by the writer (verified end to end in headless Chrome, see README "End-to-end"):

| # | Requirement | Source in attached code | hodgemix |
|---|---|---|---|
| C1 | Container `"BNZ4"` + u64le index length + JSON index {orig_size, sha256, segments[name, method, raw_off, raw_len, off, len, shape, sha256]}; payload base = 12 + len | bnz-loader.js `open`, bnz4dec.js | `bnz4/` = Gregory's bitnet-compress/wgsl/bnz encoder vendored unchanged except C6 |
| C2 | Methods only 0 raw, 1 zstd, 4 chunked ternary, 5 chunked bf16 (BNZ3 methods 2/3 are rejected) | bnz4dec.js `decode` | bnz4.Compress picks the smallest of 0/1/4/5 |
| C3 | Ed25519 signature, 64 raw bytes in `OUT.sig`, over the 12-byte header + JSON index (bnz4sign scheme); segment sha256s make it cover the whole model | bnz-loader.js `open`, cmd/bnz4sign | `--sign-seed SEED` (32-byte bnz4sign keygen seed) |
| C4 | Packed ternary: U8 [N/4, K], code = w+1, byte (r,k) bits 2s..2s+1 = output row r + s·N/4; N, K multiples of 4 | edgellm kernels/bitlinear.wgsl, model.go `lin` | `packTrits` (already matched) |
| C5 | `<proj>.weight_scale` BF16 shape [1], **multiplied**: y = (W_t·x_q)·xs·ws[0], i.e. the magnitude of a nonzero code (2B4T stores e.g. 1.21875 for layer-0 q_proj) | bitlinear.wgsl, champtrain probe.go | **fixed twice**: was 1/γ; then (2026-10-14) γ itself shrank every layer to ~0.37×, now mean abs(W) over nonzero codes with the base tensor's magnitude (README "Scale fix") |
| C6 | Embedding segments must be named `model.embed_tokens.weight[r0:r1]` or `plan()` never schedules them | bitnet-rt.js `plan` | vendored `Plan` always row-segments the embedding (≤40 MiB chunks as before) |
| C7 | Tensor names/dtypes: model.embed_tokens.weight BF16 [V,D] (tied head), model.norm.weight, per layer input_layernorm, post_attention_layernorm, self_attn.attn_sub_norm [H·hd], mlp.ffn_sub_norm [F] BF16; q,k,v,o,gate,up,down U8 + weight_scale. Other names are ignored by the loader (bnzmix.* index tensors are carried, signed, not loaded) | bitnet-rt.js `alloc`, model.go | checked by `champCheck` after every run (report `champ_compatible`) |
| C8 | **Tied embeddings only** (`tie_word_embeddings` false → runtime throws; an `lm_head.weight` would be ignored) | bitnet-rt.js `alloc`, model.go | conflict with Eq.(24e) rotation → OPEN, see GAPS.md; `--rotate off` output loads |
| C9 | config.json fields read: hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, num_hidden_layers, vocab_size, rms_norm_eps (1e-5), rope_theta (default 10000; 2B4T needs 500000), head_dim, tie_word_embeddings | bitnet-rt.js `create` | copied from base/target config |
| C10 | champ-learn adapters bind to a base by base_sha256 = sha256(header+index of the base .bnz); a hodgemix output is a new base, so existing adapters do not bind to it | champ-learn adapter/README | not changed |

## Default change (2026-10-14)
`--rotate` now defaults to `off` (Gregory decision): Champ requires a tied LM head (C8) and rotation unties it. Use `--rotate on` only for runtimes with untied-head support.

## dtft_lock.tex APPROVED for implementation (Gregory, 2026-10-14: "build the code")
Algorithm D is the default lock (`--lock dtft`); `--lock v2` keeps the paper_v2 ω=0 lock for comparison only. Papers untouched.
Tuning items of dtft_lock App.A are flags with the paper's provisional values: `--seq-axis auto`, `--omega-update on` (Alg.D step 5),
`--persistence on`, `--persist-budget 0`; `--band-inclusive` (default off = Eq.(H2) strict "<"). Champ C1–C10 unchanged and re-verified (e2e PASS).

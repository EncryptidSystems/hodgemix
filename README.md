# hodgemix (Go) — implements paper_v2.pdf exactly

Builds a ternary BitNet b1.58 2B4T-format model from a BitNet base plus open checkpoints with Hodge–Kuramoto
alignment. Outputs: `OUT.bnz` (BNZ4, Gregory's bitnet-compress format, loadable by bnz-loader / Champ), `OUT.bnz.sig` with `--sign-seed`, `OUT.ternary.safetensors` (kept), optional
`--out-merged-st` (full precision, post-rotation), `OUT.report.json`.

    hodgemix.exe --base model.safetensors --src "model (1).safetensors" --out 2bit-qwen.bnz

Every other term has its paper_v2 derived default (Appendix A, params.go). Optional overrides:
`--src PATH:WEIGHT`, `--base-weight`, `--target-config config.json`, `--roles`, `--rotate on|off`, `--rot-seed`,
`--base-residual on|off`, `--float-dtype`, `--g-theta`, `--g-omega`, `--n-lost`, `--no-verify`.
`hodgemix --selftest` runs every operator check (TRACE.md lists equation → function → test).

Stages (live log + ASCII progress bars on stderr): 1 MRH keys (Eq.7) · 2 channel index (Eq.8-9) ·
3 per tensor: Whitney spline onto mixed-radix index (Eq.6,10) → seed (Eq.12-13) → spectral Hodge (Eq.4-5) →
ternary TIES (Eq.21) → Kuramoto anneal/refine (Eq.14-20) → synthesis (Eq.22) → Hadamard rotation (Eq.23-24) →
absmean (Eq.25) → BNZ4 channel-map chunk (Sec.6) · 4 write safetensors · 5 BNZ compress · 6 verify round-trip.
See GAPS.md for what is still open.

## Champ / bnz-loader compatibility
Output container is BNZ4 (vendored `bnz4/` = bitnet-compress/wgsl/bnz): methods 0/1/4/5, JSON index, per-segment sha256,
optional Ed25519 `OUT.sig` (`--sign-seed key.seed`, same as `bnz4sign sign`). `weight_scale` = absmean γ (multiplied by
bitlinear.wgsl). After each run `champCheck` verifies names/shapes/dtypes/tied head against bitnet-rt.js and writes
`champ_compatible` (+ `champ_issues`) into the report. **`--rotate on` (default) unties lm_head, which Champ refuses: use
`--rotate off` for a Champ-loadable file** (GAPS.md 9). Requirements C1-C10: DECISIONS.md.

## End-to-end (e2e/)
`e2e/run_e2e.sh DIR` serves DIR/model.bnz with bnz-loader's `bnz4serve`, opens `e2e/www/e2e.html` in headless Chrome
(SwiftShader WebGPU, CPU) using bnz-loader's JS unchanged, and checks: signature verifies; every segment decoded by
BNZ4Dec (WGSL for 4/5) equals the ternary safetensors bytes; runtime weight-buffer digest and logits after the streamed
load equal those after the reference safetensors upload. `--bnz-force-rans` (test only) forces methods 4/5 on tiny files.
Result on ternmix-py/test (3 sources, --rotate off): 79/79 segments bit-exact, digest and logits identical, PASS
(with and without --bnz-force-rans). With --rotate on: runtime throws "untied LM head" as expected.

## Default change (2026-10-14)
`--rotate` now defaults to `off` (Gregory decision): Champ requires a tied LM head (C8) and rotation unties it. Use `--rotate on` only for runtimes with untied-head support.

## Selective DTFT lock (dtft_lock.pdf, Algorithm D) — default since 2026-10-14
The lock now gives every oscillator a measured natural frequency (Hann DFT peak, PSR≥10/uniqueness≥1.4 gate, Candan sub-bin,
Goertzel phase, ω = Δν·N/255 turn/tick) and merges a mode only if PSR gate ∧ |Δω| ≤ 1/128 ∧ equal winding numbers ∧ lost flag clear.
Unlocked modes stay with the base. New flags: `--lock dtft|v2`, `--band-inclusive`, `--seq-axis auto|rows|cols`, `--omega-update on|off`,
`--persistence on|off`, `--persist-budget N`. Per-tensor log lines `[D] ...` show band modes, PSR-pass, L2, winding, tick-locked, merged,
r measured vs Ott–Antonsen predicted, K/K_c. BNZ4 channel-map chunk is version 2 (adds ν̂, φ, winding, period, flags per contributor and mode).
`hodgemix --selftest` includes the dtft_lock checks; `hodgemix --dtft-tests [--band-inclusive] [--omega-update off]` runs P1–P4 on tiny synthetic grids.

## Scale fix, role fixes, built-in eval (2026-10-14)
Measured on the first `2b-qwen` run (2B4T base + Qwen3-0.6B) with the champ-learn CPU reference forward on 3 sentences: perplexity 18 → 1169,
`"The capital of France is"` → `" the 2010 4 4 4"`. Two independent causes:
- **weight_scale 0.27–0.39× the base on every BitLinear.** The stored scale was the absmean of the new tensor, but absmean of an
  already-ternary tensor is sc × (fraction nonzero), and the target magnitude averaged in the source's absmean (Qwen3's belongs to its
  own norm gains). Base codes with those scales alone: ppl 18 → 60. Fixed: a tensor the base has keeps the base's absmean as its
  magnitude, absmean stays the RoundClip threshold, and `weight_scale` = mean |W| over the nonzero codes (least squares for fixed codes;
  gives back the base's scale exactly for an unchanged tensor).
- **19 % of the base's codes changed (mostly 0 → ±1), 0.06 % sign flips.** Mixed codes with base scales: ppl 702. This is the channel
  matching, not fixed here: see GAPS.md 18.

Role patterns: fused `query_key_value` was taken for q, GPT-2 `attn.c_proj` for down, MoE router `gate` for gate_proj. Now qkv is
matched first, GPT-2 Conv1D names ([in, out], transposed) and bare `gate` are not recognized, and every source tensor that is not mixed
(biases, Qwen3 q_norm/k_norm, unrecognized 2-D) is logged as a WARNING and listed in the report (`skipped_srcN`).

Eval: after writing, hodgemix scores the base and the output (`--eval on`, default; `--eval-tokenizer`, default tokenizer.json next to
`--base`): greedy text for 2 prompts and NLL on 6 fixed sentences, logged and written to the report as `eval` with `ppl_ratio` and
`verdict`. Forward = champ-learn's probe (logits exact vs bitnet-rt.js), in `eval/`. `hodgemix --score MODEL [TOKENIZER]` scores one file.
Base 2B4T: ppl 12.0 on these sentences, " Paris. Paris is a city that is known for".

## Dynamic multi-radix heap (2026-10-14, multi_radix_heap.pdf)
`mrh.go` is MRH(r; C) as defined in the technical note: k levels of r_j FIFO buckets of width π_j, active
indices α_j and lower bounds L_j with the active-path refinement (I3), least-level placement (I4), find-bucket,
insert, decrease-key and extract-min with expand (Sec. 4), and `Validate` for I1–I4. `--mrh dynamic` (default)
takes r from each axis' own mixed-radix factorisation, finest digit first, extended by 2s until π_k ≥ C+1
(C = 4095, the 12-bit key of Eq.7): 2560 → (2⁹, 5, 2), 6912 → (2⁸, 3³), 2880 → (2⁶, 3², 5, 2), 1024 → (2¹⁰, 2, 2).
- Eq.(8) channel coordinates are the MRH extract-min order (FIFO ties = index order).
- Eq.(9) a contributor's bucket is the heap handle (level, bucket) of its median key in the target rows'
  radix tuple, so every contributor to a target is bucketed in the same geometry.
- Eq.(6) the index keeps whole radix levels of the target axis, at most `--grid-max` cells (default 256) and
  no finer than the shortest contributor axis.
`--mrh paper` keeps paper_v2 as written (16-bucket levels, top 4-bit digit, grid ≤ 256). Tests: `mrh_test.go`
(random monotone sequences against a reference with I1–I4 after every step, for binary, 16-ary, (5,2⁹),
(3,7,5,2,11) and single-level tuples; pop order of real channel counts).

## Married index (`--index married`, default)
Each contributor used to sit on its own index (its own pop-order ranks). Now every contributor of a target, base
included, is married into one index per axis (`married.go`): rows into one MRH, columns into another, with
radices r_M = the factorisation of lcm of the axis lengths (2560 and 1024 → (2¹⁰, 5); every model's digit
hierarchy embeds), and each channel's coordinate is its place in the joint extract-min order,
x = (S + (r + ½)·E/E_m)/N (S channels before its key, E with its key, E_m of its own model, r its FIFO rank).
Equal keys share one span across models; a model married with itself keeps exactly its own coordinates, so a
self-mix is still exact (`TestMarriedSelfMixIsIdentity`). The log prints the married radices per tensor.
`--index own` keeps per-model indexes.

## Topology: two models in one (`--topology base|max|married`)
`topology.go` maps each model's axis graph (`metaOf`: D, F, heads, kv heads, head_dim, layers from the tensors,
with heads from a config.json beside the file when it agrees):
- **D** the residual stream (q,k,v,gate,up columns; o,down rows; embeddings, lm_head, D norms)
- **F@l** a layer's FFN (gate,up rows; down columns; ffn_sub_norm)
- **KV@l / Q@l** a layer's attention. The unit is a kv group (one kv head and the G query heads that read it), so
  heads stay whole (RoPE pairs) and GQA grouping holds; Q takes exactly its group's span on KV.

Each axis node gets an MRH key per unit, the mean Eq.(7) key over every tensor touching it. Each target axis
node is married once over its contributing (model, layer) instances (Eq.(11) depth hat), and every tensor on
that node uses the same married coordinates. The network stays one network.
- `base` (default): the base model's sizes; the base sits at its married coordinates (self-mix exact).
- `max`: the larger size per axis. `married`: the lcm per axis, so every model's radix hierarchy embeds
  exactly (2B4T + Qwen3-0.6B → D 5120, F 27648, kv 40, 160 heads of 128, 30 layers). Heads keep the base
  head_dim; layers take the deeper model's count, extending the base's own layer pattern.
- In a new topology the target channels take the married order. Every contributor's fine detail
  (I − P R)M_i is carried onto them along the married axes and mixed by weight (Eq.22 for all models).
  Embeddings, lm_head, norm gains and biases are resampled along their axis. The magnitude is the base's
  same-role tensor at the nearest depth. The written config.json has the new sizes, and the log and report
  give the plan and parameter count before anything is written.
Tests: `topology_test.go` (married sizes; two different tiny topologies → married shapes and config through
`run()`; a married self-mix is one consistent permutation: q columns and o rows share the D permutation, o
columns follow the query heads').

## Attention after marriage (`--attn on`, default; `attention.go`)
The married index keeps heads whole and kv groups intact; three attention properties are reconciled on top:
1. **QK-norm folding.** The Champ runtime has no QK-norm, so every model with per-head q_norm/k_norm (Qwen3), the
   base included, has each head scaled to the unit rms QK-norm gives on a normalized input,
   c_h = √hd / ‖W_h diag(g_in)‖_F, with its q_norm/k_norm gain folded per dim; the output drops q_norm/k_norm.
2. **RoPE frequency marriage.** The head_dim axis is married by rotation frequency: target pair d takes the
   contributor's pair at s = d·ln θ_target / ln θ_contributor (rotate-half, both halves, the same map for q and k).
   θ comes from the model's own config.json, or `--src-rope-theta` for any model (base too) without one (Qwen3: 1000000).
   A source with unknown θ is logged and left as trained.
3. **Logit temperature.** When a layer's q and k are both mixed, q's weight_scale is patched (`STW.Patch`) so the
   mean per-head spread σ_h = √tr(Q_hQ_hᵀ·K_gK_gᵀ) (std of q·k for independent unit inputs; query head h, kv head g)
   equals the base layer's. With a ternary base (2B4T) a self-mix gives ×1. With a float base it also corrects the
   temperature drift ternarization causes. Skipped when the base has QK-norm (its folded heads are already at QK-norm's scale).
Tests: `attention_test.go` (RoPE map, QK-norm fold to unit rms, σ scaling); every run logs each layer's factor.

Interrupted runs no longer leave `.hodgemix-*.tmp` files: errors and Ctrl+C remove the run's temp files.

## Any base family, config from the tensors (2026-10-14)
- **config.json follows the output's tensors.** The config copied from beside `--base` is rewritten with the
  output's hidden/intermediate/heads/kv/head_dim/layers/vocab (and rope_theta when known). The source
  `model_type` and `architectures` stay. A missing `hidden_act` is copied from `hidden_activation`.
  A Qwen3 file sitting next to
  BitNet's config.json used to get a 2560-wide config for 1024-wide tensors (the page: "input_layernorm.weight has
  1024 values, want 2560"). No config at all: the 2B4T template with the output's sizes.
- **Champ layout for any base** (`champLayout`): a layer without BitNet's attn_sub_norm/ffn_sub_norm gets them
  at unit gain; q_norm/k_norm are folded (above) and dropped.
- **Tokenizer check.** A tokenizer.json beside the output whose vocabulary is not the embeddings' row count is
  flagged (`WARNING tokenizer.json ...`, `report.tokenizer_mismatch`); dev.ps1 warns the same way.
- **report.json** no longer comes out empty: NaN/±Inf values (a DTFT peak with zero noise floor has PSR +Inf)
  are written as the strings "NaN"/"+Inf"/"-Inf".
Tests: `TestConfigFollowsTensors`, `TestQwenBaseChampLayout`.

## Making a mixed model run (2026-10-15)
Found on 2B4T + Qwen3 (default 1:1, base topology): ppl 12.0 -> 7.6e7, greedy "solessoles…".
- **TIES weight (`ternaryTIESAll`, math.go; was `tiesLambda`).** The curl-set TIES delta is the mean over the *agreeing* deltas
  only, so with one source model its weight cancels and the base's (zero delta) never enters: `--base-weight`
  and `--src PATH:W` did nothing there and a 1:1 mix replaced the base's curl set outright (24% of every
  tensor's codes). The delta is now Σ over agreeing contributors of w·d divided by the total weight of every contributor (base and zero-delta sources included): 0.5 at 1:1 as before, and the base given again as a source dilutes toward the base (a self-mix has
  zero deltas). ppl 7.6e7 -> 27.0.
- **Seed alignment (`--seed-align auto|on|off`, auto = on).** The per-tensor seed (Eq.12-13 shift per radix
  digit, then Re-Basin assignment) keeps the deltas small (off: ppl 1.3e7; the married index alone does not
  align two models), but it permutes connected weights differently (v's rows vs o's columns, …). Its long-context
  cost is removed by the OV chart below.
- **OV chart (`--ov chart|mix`, default chart; ovchart.go).** A head's circuit is O·V and V is its chart (the
  addresses it reads). Mixing both factors adds the cross term dO·dV, the source's own circuit read through the
  base's addresses: v or o mixed alone was fine, both together broke every 334-token chat while 14-token chats
  answered. The chart keeps V_b and carries v's change into o to first order, O_t = O_mix + O_b·(dV·V_b⁺), so
  O_t·V_b = O_b·V_b + dO·V_b + O_b·dV·P (P the projection on V_b's rows): no cross term, same shapes; an hd×hd
  solve per kv group. 2B4T + Qwen3: the cross term was 0.23 of ‖O_b V_b‖; chat answers again, ppl 17.0 -> 14.4.
  `hodgemix --ov-chart BASE OUT.ternary.safetensors` applies it to an existing output.
- **Pin-mean (`--pin-mean on`, default; pinmean.go).** No mixed BitLinear may change its response to its mean
  input (the base forward's): W ← W − (dW·μ̂)·μ̂ᵀ, rank one per tensor, before ternarization (the harmonic mode
  on the RMSNorm sphere keeps the base's value). 7–16% of each tensor's mean response removed; ppl 20.9 -> 17.0.
- **Union vocabulary (`--vocab union|base`, default union; vocab.go).** The output is a new model with its own
  tokenizer.json, tokenizer_config.json and config.json: the base's tokens keep their ids, each source's other
  tokens (and added tokens) are appended, its merges follow the base's. An appended token's embedding is its source
  embedding carried into the base's residual space by the shared-vocabulary map E_base ≈ E_src·M (least squares
  over the shared strings; tied heads, so the same map at the output). 2B4T + Qwen3: 170,359 tokens (42,077 + 26
  added), 109,566 shared strings, held-out R² 0.25. Sources need their tokenizer.json beside the weights.
- **Calibration.** One reference forward of the base (calibration texts + the page's chat prompt) gives both the
  singular set (surgery) and the mean inputs (pin-mean).
- **Surgery (`--surgery on`, default; surgery.go).** The base's forward concentrates in a singular set:
  massive residual channels (2B4T: 17 of 2560 at 41–60× the median rms) and the FFN spike units that write
  them. After the mix the base's codes go back on exactly those rows/columns (rows of o/down and columns of
  q/k/v/gate/up on massive channels; rows of gate/up and columns of down on spike units). `--surgery-ratio`
  (20) sets the threshold. Needs the 2B4T-shape reference forward (skipped otherwise). ppl 27.0 -> 20.9.
- **`--roles` subsets** keep every unmixed packed tensor's `weight_scale` (it was dropped).

Diagnostics:
- `hodgemix --bisect …` (same flags as a mix): each of `--roles` mixed alone, scored against the base,
  table printed and kept in OUT.bisect.json; per-role outputs are deleted.
- `hodgemix --trace BASE OUT [tok]`: base vs output sublayer by sublayer — residual cosine, norm ratio, FFN
  hidden peak ratio, the channels carrying the difference and the base's largest channels.
  `HODGEMIX_TRACE_CHAT=1` runs it on the page's chat prompt and adds last-layer cosine by position.
- `hodgemix --chat MODEL [tok]`: the page's first answer (its system prompt, chat format, ~334 tokens) on the
  CPU reference forward — the short eval texts (~20 tokens) do not exercise long context.
- `hodgemix --surgery BASE OUT.ternary.safetensors [tok]`: surgery in place on an existing output.

## Dynamic MRH limbs and the LUT-fabric forward (2026-10-15)
- **Weights are digit strings (`--limbs N`, default 1 = ternary; `--limb-tol`, default 0.15; limbs.go).** Following the
  multi-limb digits of unified_main.pdf §10.1, a mixed weight is W ≈ Σ_l s_l (D_l − c_r): limb 0 the ternary code
  every loader reads (r = 3), each further limb the residual of the earlier ones, its radix in {2, 3, 4} (2 bits,
  the TLUT-native alphabet) the one that leaves the least residual, its scale least squares. Limbs are added
  while the residual is above `--limb-tol` of ‖W‖; a tensor the mix did not move stays one limb. Files:
  `X.weight_limb<i>` (U8, packed like limb 0, raw digits), `X.weight_limb<i>_scale` (BF16[1]),
  `X.weight_limb<i>_radix` (I32[1]). LoadDense sums the limbs; the OV chart refills o's limbs and zeroes v's;
  surgery zeroes limbs on the capped rows/columns; `--limbs 1` is the old ternary output.
- **Champ runs limbs** (examples/bitnet `model/model.go`, `kernels/bitlinear.wgsl`, and bnz-loader `js/bitnet-rt.js`):
  the same BitLinear kernel with `override CENTER` (the digit meaning 0) runs each extra limb with `ACCUM`.
  There is no limb-count cap. Checkpoints without limbs are unchanged.
- **CPU forward as a LUT fabric (eval/lut.go).** Each limb is a signed incidence operator; the input axis is cut
  into blocks of K channels on its radix levels (K the largest divisor of the axis with r^K ≤ 256: 2560 → 5,
  6912 → 4 at r = 3), a row's block is a one-byte mixed-radix numeral, and per token each block's r^K-entry table
  of signed sums is filled digit by digit; rows are gather + add. Exact (bit-identical to the old multiply-
  accumulate on ternary weights), no multiplies, no SIMD, 1 byte per 5 trits instead of 5. Prompt tokens whose
  logits are not read skip the LM head (`feed`). Chat probe 3.5 min → 75 s; calibration 270 s → 117 s.
- **Conversation suite** (`hodgemix --converse MODEL [tok]`): 12 conversations in the page's format and system
  prompt (follow-ups, memory, tool calls, code, Chinese, arithmetic, summarising, translating); transcripts in
  `<model>.converse.json` in the working directory.
- **Repair** (`--repair on`, repair.go): REPAIR / surgery-paper step 7 on every channel — RMSNorm gains set so each
  channel's spread matches the base's on the calibration tokens, clamped to [½, 2], three fixed-point passes.
- **Measured (2B4T + Qwen3, `--limbs 2`):** every mixed tensor took one radix-4 limb (residual 0.27-0.32 → ~0.11 of
  ‖W‖), and the model scored **ppl 4142** against 14.4 with one ternary limb. The limb path is exact (stock 2B4T
  moved entirely into a radix-4 limb scores ppl 12.0, `TestLimbProbe`), so the loss is the float mix itself:
  ternary rounding keeps only the changes large enough to flip a code and drops the dense sub-threshold part, which
  with a residual map at R² 0.25 is noise to the base's function. Hence the default `--limbs 1`; limbs are for
  sources the shared-vocabulary map aligns well (e.g. a Llama-3-tokenizer model).
- **Singular neck, every run.** Ternarize writes up to 4 neck limbs on the T1 singular set (`NeckLimbs`). They have their own scale, the body stays the center digit on that limb, and there is no Kuramoto lock. This is not `--limbs`. The default stays 1 body limb: extra body limbs on 2B4T + Qwen3 scored ppl 4142. A pack that was written before this appends them without re-locking: `hodgemix --config RUN.yaml --sing PACK.safetensors OUT.safetensors`. The run file's base is the float checkpoint.
- **Llama 3 body pack.** A base config with `rope_scaling.rope_type` `llama3` also writes 6 body limbs and stops at residual 0.02 of ‖W‖. Those body limbs cut the neck out and Kuramoto-lock to limb 0. The neck limbs above are still written. The packed file is `<out-without-.bnz>.neck.safetensors`, and that file is what gets compressed into the `.bnz`. Re-run the body pack without remixing: `hodgemix --config RUN.yaml --neck TERNARY.safetensors OUT.neck.safetensors`. Any other arch is refused for `--neck`. An explicit `--limbs` above 6 or `--limb-tol` below 0.02 is kept.

## Any base, files from a run file (2026-10-15)

The base is whatever the run file names; nothing assumes a model, and nothing is looked up beside one.

- **Run file (runfile.go): `hodgemix --config RUN.yaml`.** It sets every model's files and the run:
  ```yaml
  base:
    model: C:\Users\grego\Downloads\model (2).safetensors   # or a list: one model's shards
    tokenizer: C:\Users\grego\Downloads\tokenizer (1).json
    tokenizer_config: C:\Users\grego\Downloads\tokenizer_config (1).json
    config: C:\Users\grego\Downloads\config (2).json
    special_tokens_map: ...      # optional, copied beside the output
    generation_config: ...       # optional, copied beside the output
  sources:
    - model: ...model-00001-of-00002.safetensors
      weight: 1                  # optional
      tokenizer: ...             # read by the union vocabulary
      config: ...
  out: C:\Users\grego\models\hodge-1.bnz
  settings:                      # optional, any option by name
    limbs: 1
  ```
  A file not set is not read; a set file that does not exist, or an unknown key, stops the run. Relative paths
  are from the run file's folder. A run writes `OUT.yaml` with its output's files; `hodgemix --config OUT.yaml --score
  OUT.ternary.safetensors` (or `--chat`, `--converse`, `--trace`, ...) probes with them.
- **A model is the files listed for it.** A shard is one file (each shard its own source reads one shard); several
  files listed under one `model:` are one model's shards. `OpenST` used to pull every shard of the index beside a
  shard path.
- **Config: the one set, checked.** Heads, head_dim, θ, ε, activation and `rope_scaling` come from the config set for
  the model; one that does not describe the tensors is another model's and is refused. Without one (and no q_norm)
  the heads are unknown and the run stops: the q/k row counts admit several head_dims (Llama-3.2-1B's 2048 / 512
  rows are 32 / 8 heads of 64 or 16 / 4 of 128). Unset keys take the config class's defaults (BitNetConfig for
  `bitnet`, LlamaConfig otherwise). A Gemma config that names the activation `hidden_activation`
  (`gelu_pytorch_tanh` on VaultGemma) is copied to `hidden_act` before that default, so the run is not stamped silu.
- **Eval / calibration forward (eval/model.go)** runs any model: sizes from its tensors; per matrix packed U8 =
  BitLinear (int8 activations, LUT fabric, limbs), BF16/F16/F32 = float matmul; biases, BitNet sub-norms, Qwen3 q/k
  norms and an untied `lm_head` where the tensors have them. Activations relu2, silu, gelu, gelu tanh; RoPE scaling
  llama3, linear, dynamic (exact within eval's contexts); others refused. 2B4T: ppl 12.0, unchanged.
  Llama-3.2-3B-Instruct (BF16): ppl 7.8, "Paris", 17 × 23 = 391.
- **Tokenizer: the one set, checked.** BOS = what its own TemplateProcessing puts first (none for ByteLevel-only
  ones); the turn end = the set tokenizer_config's `eos_token`. A tokenizer with ids past the model's embedding rows
  is not that model's: eval refuses it, the run stops at once on the base's (was a residual-map index panic 150 s
  in), and the union vocabulary skips such a source.

## Nothing fails soft (2026-10-15)

A run either makes the model it describes or stops; no step warns and carries on.

- **Preflight (preflight.go), before anything is written:** every model's config set and describing its tensors
  (a shard's sizes from the first layer holding each projection, so a shard starting mid-stack or mid-layer
  works); the base's tokenizer, tokenizer_config and config set, the tokenizer's ids within the base's
  embedding rows and the tokenizer_config's eos_token one of its added tokens; each source's tokenizer within its
  embedding (a shard without the embedding adds no tokens); no tensor of any model left out (biases, fused qkv,
  2-D tensors with no role, q/k norms with `--attn off` stop the run); the base's head tied; `--rotate on`
  refused (it unties the head); the output config checked against Champ's runtime. Every reason is listed.
- **Then calibration and the union vocabulary**, before the mix; either failing stops the run.
- **After the mix:** config, OV chart, surgery, repair, every file write, the Champ check (before compression) and
  eval stop the run on any error. The married index, attention temperature and pin-mean fail instead of falling
  back.
- **The output config keeps the base's model.** `model_type`, `architectures`, `hidden_act`,
  `rope_scaling`, `rope_theta`, and `max_position_embeddings` stay the base's. A missing `hidden_act`
  is copied from `hidden_activation` (VaultGemma's gelu). Keys the base leaves out are written with
  its config class's defaults (they used to be overwritten with relu2 / the 2B4T template, and the
  file was stamped `model_type bitnet` / `BitNetForCausalLM`). VaultGemma's FFN pre-norm stays
  `pre_feedforward_layernorm`.
- **Champ runs it** (examples/bitnet): `norm.wgsl` `ACT` (relu², silu, gelu tanh) for the gated FFN, `rope.wgsl`
  `SCALING` (linear, llama3) for rope_scaling; `model.go` `LoadConfig` reads both and refuses what the kernels do
  not run (other activations, other rope_scaling, attention bias, a run whose `layer_types` or
  `use_sliding_window` turns on sliding-window attention). A `sliding_window` number next to a
  `layer_types` of all `full_attention` is that run's unused field, not a window. A number with
  neither key is not assumed: the run has to say which layers use it. The kernels compile
  under Dawn; `model.go` needs `compile.bat`.

## Weight volumes and the forward's cochain (2026-10-15)

- **Volumes (volume.go): every run writes `volume.safetensors` beside its output.** One 3-D matrix per BitLinear role,
  layer x output cell x input cell. Channels are grouped into cells on shared axis nodes: the residual axis D has
  one order for every tensor that reads or writes it, the FFN axis F one for gate/up rows and down columns, both the
  MRH pop order (Eq.8) of each channel's Eq.(7) key averaged over the node; attention keeps heads whole (4 cells a
  head). `volume.<role>` is the cell's mean |W| over its tensor's absmean, `balance.<role>` mean W / mean |W|,
  `cell.<axis>` every channel's cell. 2B4T: 49 s, e.g. `volume.q` [30 80 64].
- **Currents from the fabric (eval/flow.go).** With the cells given, every BitLinear's digit columns are laid out
  in cell order with a LUT block size dividing the cell size, so each gathered table entry is the current from one
  input cell into one row; outputs stay bit-identical (`TestRelayoutCurrents`).
- **The cochain complex (cochain.go): `hodgemix --config RUN.yaml --flow MODEL OUT.safetensors "text"`.** Nodes are each
  layer's residual cells (entering, after attention, after the FFN) and the q, k, v, attention-output, gate, up and
  hidden cells; edges are every voxel plus the residual carries, attention's v -> output transport and up -> hidden
  (the gate as metric). The residual flow is conserved; the metric is each (layer, class) block in units of its own
  rms (the model's norms make every sublayer input scale-free). The Hodge split solves L0 phi = d0^T f by conjugate
  gradient: gradient part d0 phi, cycle part f - d0 phi (no 2-cells, so the cycle space is harmonic). Written per
  token: `t<i>.f` (raw currents), `t<i>.scale`, `t<i>.phi`, with the complex (`edge.*`, `cell.*`).
- **Measured, 2B4T, "The capital of France is":** 15,664 nodes and 760,800 edges a token; conservation 2e-6 and
  Stokes (||d0^T cycle|| / ||d0^T f||) 9e-10 on every token; 188 CG iterations, about 1.2 s a token. The flow is
  about 7% gradient across layers against 2.06% for random currents on the same complex (rank d0 / edges).
- **Initial runs, each from its own config.** Text: `2b4t`, `vaultgemma`, `llama3`, `gpt-oss`. Image generation
  is `qwen-turbo`, a separate model, not a fifth text block. VaultGemma (`model_type` `vaultgemma`) is Gemma RMS
  `(1+weight)`, an embedding scale of `sqrt(hidden)`, `gelu_pytorch_tanh`, and `pre_feedforward_layernorm` as the
  FFN pre-norm. It is not a sandwich. Its `sliding_window` 512 sits beside a `layer_types` of all `full_attention`,
  so the window is unused. A Champ file keeps `hidden_activation`, and the runtime then applies RMS `(1+weight)`,
  scales the token by `sqrt(hidden)`, and uses `query_pre_attn_scalar` as the attention denominator. gpt-oss stays
  the eval mixture (top-k, sinks, YaRN, clamped SwiGLU). The kernels do not run a mixture, YaRN, or sinks. No
  qwen-turbo checkpoint is in the tree.
- **Four families, one cochain.** `2b4t`, `llama`, `qwen`, `gemma` (including VaultGemma), and `gpt-oss` keep their own block (norm, RoPE,
  window; gpt-oss adds top-k experts, attention sinks, and clamped SwiGLU). `hodgemix --generate MODEL "text" [n]`
  (n defaults to 8) and `--chat` print one manifold line per new token: conservation, Stokes, gradient share, the
  hottest layer, the residual cell that moved, and the experts that fired. An MXFP4 expert is a 16-entry gather
  with an E8M0 shift. It is not expanded into a dense matrix. A 120B float expansion does not fit a laptop; the
  proof is a tiny checkpoint. The mixer still writes one BitNet layout. A Gemma base keeps `hidden_activation`,
  which selects RMS `(1+weight)` and the embed scale in the Champ kernels.
  `hodgemix --update UPDATE --generate MODEL "text"` keeps UPDATE's feed-forward as a second expert on the same
  cells. A token admits it when the combined block still moves the base's residual cell the same way. Otherwise
  the token runs on the base alone, and the manifold line shows expert 0. bnz-loader still has one buffer per
  linear, so this check is the hodgemix forward.

## spm2json: a SentencePiece tokenizer.model as tokenizer.json (2026-10-16)

`spm2json -model tokenizer.model -config tokenizer_config.json -out tokenizer.json` (build: `go build -o spm2json.exe ./spm2json`).
For models that ship only a SentencePiece BPE `tokenizer.model` (VaultGemma). Both files are named; nothing is looked up.
It writes the byte-level BPE form hodgemix's eval and Champ's tokenizer read: every piece keeps its id ("▁" as a space,
`<0xNN>` as byte NN's letter unless a normal piece already is that byte), merges are every two-piece split of a piece
ranked by its score (transformers' extractor, at byte boundaries), Llama 3 Split + ByteLevel, `<bos>` first when
`add_bos_token`. Control/unknown pieces and the config's `added_tokens_decoder` ids are added tokens with its special flag.
Refused: not BPE with byte fallback, a normalizer that changes text, a tokenizer_config whose tokens are not the model's.
VaultGemma-1B: 256000 ids, 582159 merges, 249 added tokens.

## stdiff: two safetensors compared tensor by tensor (2026-10-16)

`stdiff -a BASE.safetensors -b OUT.safetensors [-json report.json]` (build: `go build -o stdiff.exe ./stdiff`). Both
files named. For every shared tensor: dtype and shape; packed ternary U8 decoded code by code (changed, sign flips,
0 -> ±1, ±1 -> 0, zeros); BF16/F16/F32 elements that differ and the largest difference, and a one-element tensor's
ratio (weight_scale); tensors in one file only, by role. hodge-2b4t-1 against Microsoft's 2B4T: 542 of 542 tensors
byte-identical, 0 of 2,084,044,800 codes changed, plus 211 bnzmix tensors (channel maps and the index).

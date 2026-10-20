# Gaps in the v1 paper and how paper_v2 closes them

Sources: his files first (RESEARCH_INDEX.md: jev_kuramoto KTE rules, DECLUT, MRH, DMT chip), then verified literature (paper_v2 bibliography; every entry checked via arXiv API / Crossref / Caltech library on 2026-10-14).

## Closed
| Old term / gap | paper_v2 | Source |
|---|---|---|
| roles | q,k,v,o,gate,up,down,qkv | BitNet b1.58 2B4T (BitLinear everywhere in attn/FFN) |
| grid | Eq.(6) whole radix levels ≤ 256 | decision 4 + two MRH levels of 16 buckets (DMT unit) |
| knots, knot-jitter, spline degree | Eq.(8),(10) degree 1 = Whitney hat; knots = MRH pop order | Arnold–Falk–Winther 2006; MRH FIFO ties |
| heap weight / heap-eps | Eq.(7) 12-bit saturating key | DMT 12-bit arrival field |
| mrh-levels, bucket-weights | Eq.(9) 16 buckets; coupling strong/weak Eq.(15) | DMT 16-bucket unit; KTE g_θ=4, g_ω=8 |
| k-weak, kmax | 2^-8, 2^-4 | KTE R-KTE-1/R-KTE-8 |
| omega | 0 (same λ per bin, co-rotating frame) | Eq.(4); Dörfler–Bullo 2014 |
| dt | one KTE tick, stable by Gershgorin with d_max normalisation | R-KTE-1 |
| steps-per-stage, window, max-steps | N_lost = 255 per stage/window; ≤ D+2 windows | KTE N_lost |
| lock-ftol, r-tol | phase window 1/16 turn; 2^-8 | KTE W_lock; BF16 resolution |
| outer-r-tol, outer-h-tol, max-outer | 2^-8, 2^-8, D+1 | BF16 resolution; one pass per digit |
| harmonic-lambda | Eq.(5) lowest bin of each digit | v1 text "low bins of each digit" |
| seed-assign | Re-Basin weight matching (= uniform OT optimum) | Ainsworth et al. 2023; Birkhoff |
| pc-eps, absmean-eps | none; exact zero rules | Eq.(12),(25) |
| depth-sigma, depth-cut | Whitney hat on depth, Eq.(11) | AFW |
| base-residual | on (two-grid complement), Eq.(22) | multigrid complement |
| float-dtype | BF16 | BitNet 2B4T |
| index-bins | all low bins |H| | decision 4 |
| rotation before ternarizing | randomized Hadamard residual rotation, norm folding, Eq.(23)-(24) | QuaRot 2024, SliceGPT 2024 |
| DEC restriction/prolongation | Whitney P0/P1, de Rham R0/R1, Eq.(26); tests d∘P=P∘d, R∘P=I, harmonic kept | AFW 2006, Hirani 2003 |
| BNZ4 signed channel-map chunk | byte layout Sec.6, CRC-32, signed bijective perms | decision 3 |
| target scale | γ̄ = weighted mean source absmean, then absmean re-derived in Eq.(25) | BitNet b1.58 |
| embeddings / norms | BF16, linear (Whitney P0) resize, rotated/folded per Eq.(24) | BitNet 2B4T, SliceGPT |

## Still open (stated honestly)
1. **Triangles**: v1 says "co-activating channel triples"; that needs calibration data. v2 uses the data-free grid squares of the product complex.
2. **Non-unit Hodge stars**: DECLUT allows per-edge digit weights w_e, but nothing in his files fixes their values for weights; v2 uses unit stars.
3. **ω = 0** makes synchronization generic; on all tiny runs every oscillator locked (⟨r⟩ = 1.0). A data-derived ω (if any) is open research.
4. **Distillation** into 2B4T/adapters with top-k teacher logits is described but not implemented.
5. **Signed permutations**: format supports sign flips; the seed never produces a flip (no sign-symmetry search is defined), so all stored signs are +.
6. **BNZ4 chunk** is stored as a U8 tensor `bnzmix.bnz4.<name>` inside the BNZ4 container (Gregory's format, methods 0/1/4/5), covered by the Ed25519 index signature; bnz-loader carries it but does not load it (no container-level chunk type exists in BNZ4).
9. **Rotation vs Champ runtime (C8 in DECISIONS.md)**: Eq.(24e) folds the final RMSNorm gain into the head, which unties lm_head; bnz-loader/bitnet-rt.js and edgellm model.go support only a tied head and refuse the file. Default `--rotate on` output therefore does not load in Champ; `--rotate off` output does (verified). Resolving this needs either a runtime change (untied head support) or a paper decision; neither was made here.
7. **Large-model results**: none; protocol only. The "Argon" tables are not used.
8. Rotation is skipped (logged) when a residual reader lacks a recognized RMSNorm (e.g. the tiny base-only test layout still applies; unusual architectures may not).

## dtft_lock implementation (2026-10-14): open / interpretation points
10. **(H2) strict "<" leaves only the k=0 bin** on the mode axis (every non-zero single-digit bin has λ ≥ λ_gap). `--band-inclusive` adds the lowest bins of the coarsest long digit. Which one the paper intends is for Gregory.
11. **Grids without a digit r≥3** (e.g. 64 = 2^6, all tiny 16×16 / 8×16 test tensors): λ_gap undefined, β1 = 0, band empty, base kept. 64×64 tests therefore have no oscillators.
12. **Period/winding cochain**: implemented as the phase 1-cochain of the (analytic extension of the) band sequence on the sequence-axis torus; periods = translate-mean holonomy, winding = loop through node 0. The paper names x_i but not which cochain of the weight grid; this is an interpretation.
13. **g_ω frequency update (Alg.D step 5) makes the tick lock everything** (type-2 loop: pair lag → 0 even at Δω = 1/64). Decision (ii) is therefore evaluated analytically from (L2) on the measured ω; decision (iv) is non-selective with the update on. `--omega-update off` reproduces the L2 boundary in the ticks.
14. **Group delay** (D1) with the Hann window does not recover large cyclic shifts (12-sample shift on N=60 measured as 2.5); the seed (phase correlation) still recovers them.
15. **Not implemented**: LOBPCG for inhomogeneous (activation) stars (needs calibration data; per-level stars keep the DFT exact), weighted Forman curvature correction (paper gives only the unweighted formula), routing unlocked modes to adapters, multi-peak oscillators.
16. **Persistence** is computed and reported, but with band modes on the essential classes it cannot change the stage order (falls back to coarse→fine). Cost ≈ 2.4 s (160×160) / 5.1 s (216×160) per tensor on the box, single-threaded.
17. **Ott–Antonsen prediction** with 2–4 oscillators per tensor is a small-n estimate; Δ = IQR/2 is noisy at this n.

## Measured on 2B4T + Qwen3-0.6B (2026-10-14)
18. **Channel matching is per tensor, so a source's contribution is not aligned with the base's computation.** Channel coordinates
    (Eq.8) come from each tensor's own heap key (Eq.7, ternarization error rank), and the seed's Re-Basin assignment runs per tensor on
    the ≤256 grid. Nothing ties the rows of q to the rows of k (heads, RoPE pairs), v's rows to o's columns, gate/up rows to down's
    columns, or the residual-stream columns of one tensor to the next. Git Re-Basin solves one permutation per hidden group, shared by
    every weight that reads or writes it. Measured: 19 % of base codes changed, only 0.06 % changed sign; mixed codes with base
    scales give ppl 702 vs 18. The source acts as noise around the base. A shared permutation would need a residual-stream matching
    across all layers, a per-layer head matching (whole heads, so RoPE pairs survive) and a per-layer FFN neuron matching, and even then
    models with unrelated pretraining (Qwen3 into 2B4T) are not known to merge by permutation; distillation (4) is the known route.
19. **Qwen3 q_norm/k_norm** (per-head RMSNorm on q and k) has no BitNet counterpart and cannot be folded into linear weights; Qwen/Qwen2
    q/k/v biases are dropped too. Both are now logged per run.

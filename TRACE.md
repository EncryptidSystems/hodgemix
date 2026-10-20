# TRACE: paper_v2 equation → Go function → test

| Eq. | Content | Function (file) | Test (`hodgemix --selftest`) |
|---|---|---|---|
| (1) | channel map M = W/γ | `run` normalisation (main.go), `absmean` (math.go) | tiny runs |
| (2)-(3) | Hodge decomposition, L1, stars | `newComplex`,`hodge` (selftest.go) | S2 `hodge(complex K)` |
| (4) | λ_k two-way Laplacian | `NewSpectrum`,`fftn` (math.go) | S1 `lambda n` |
| (5) | sets G / H / C | `NewSpectrum`,`(*Spectrum).Hodge` | S3 `spectral hodge`, T4 bin counts |
| (6) | factorisation, index length | `radices`,`gridSize` (math.go) | T4 |
| (7) | 12-bit MRH key | `heapKey` | T8 |
| (8) | channel coordinate = MRH pop rank | `chanCoords` | T8 |
| (9) | tensor bucket | `bucketOf` | T8 |
| (10) | Whitney P0 spline / de Rham sample, hat weights | `splineGrid`,`sampleGrid` (math.go), `hatWeights` (chanmap.go) | T7 |
| (11) | depth hat | `depthHat` (main.go) | tiny runs (contributor counts) |
| (12) | phase correlation per digit | `seedShift` (math.go) | `seed phase-correlation` |
| (13) | Re-Basin weight matching | `seedAssign`,`hungarian` | `seed assignment` |
| (14),(14b),(15) | Kuramoto tick, ω=0, K = 2^-4 c/d_max | `kuramotoTick`,`couplingMatrix` (kuramoto.go); coupling build in `synth` | T5 |
| (16) | lock window 1/16 turn, N_lost | `kuramotoTick` | T5 |
| (17) | r, r_m | `kuramotoTick` (rOf), `synth` merge | T5, tiny runs |
| (18) | anneal schedule | `kuramotoTick` (Anneal), `synth` (MaxWin) | T5 |
| (19) | merge | `synth` | tiny runs |
| (20) | refine stop | `synth` | tiny runs (passes) |
| (21) | ternary TIES | `ternaryTIES` | `ternary TIES` |
| (22) | synthesis + base residual | `synth` (output), `run` | tiny runs |
| (23) | randomized Hadamard Q | `NewRot`,`hadBlock`,`fwht`,`RowTimesQ` (rotate.go) | T1 |
| (24) | rotation folding | `Reader`,`Writer` (rotate.go), `planRotation`,`apply`,`nonIndex` (rotplan.go) | T2 |
| (25) | absmean RoundClip | `ternQ`,`absmean`, `run` | `pack 4 trits` |
| (26) | P0,P1,R0,R1 Whitney/de Rham | `prolong0`,`prolong1`,`restrict0`,`restrict1`,`dCycle`,`l1Cycle` (whitney.go) | T3 |
| Sec.6 | BNZ4 chunk | `ChanMap.Encode`,`DecodeChanMap`,`signedPerm` (chanmap.go) | T6 |
| App.A | constants | params.go | — |

## Champ / bnz-loader interface (no paper equation; DECISIONS.md C1-C10)
| Requirement | Function (file) | Test |
|---|---|---|
| C1-C2 BNZ4 container, methods 0/1/4/5 | `bnz4.Compress`/`Decompress` (bnz4/, vendored) | STAGE 6 round-trip; e2e segments bit-exact |
| C3 Ed25519 over header+index | `signBNZ4` (champ.go) | bnz4sign verify OK; e2e `signed:true` |
| C4 packed trits | `packTrits` (main.go) | e2e weight digest = reference upload |
| C5 weight_scale = γ (multiplied) | `run` ternarize (main.go), `LoadDense` (st.go) | e2e logits identical |
| C6 embedding row segments | `bnz4.Plan` | e2e plan groups `embed.0` |
| C7-C9 names/shapes/tied/config | `champCheck` (champ.go) | report `champ_compatible` |

## dtft_lock.tex (Algorithm D, default `--lock dtft`; approved by Gregory 2026-10-14 "build the code")
| Eq. / item | Content | Function (file) | Test |
|---|---|---|---|
| Prop.1 | uniform ω ⇒ full lock | `--lock v2` keeps the old lock for comparison | real 2B4T run r=1, 147/147 |
| (H1), Prop.2 | β1 = |E| − rk d0 − rk d1, grid squares | `betti1`,`rankF` (dtft_tests.go), `longDigits` (dtft.go) | selftest `beta1 C5xC3` 2 / 16 |
| (H2) | band λ_k < λ_gap = min_{r≥3}(2−2cos 2π/r) | `bandEdge`,`bandBins`,`conjIdx` (dtft.go) | selftest `H2 band edge` |
| Sec.2.3 (H3) | periods ∮γ_a (translate-mean holonomy of the band phase 1-cochain) | `periodsWindings`,`analytic` | P3 raw periods invariant |
| (H4) | winding numbers, must match to merge | `periodsWindings`, decision (iii) in `synthD` | P1–P4 winding counts |
| Sec.2.4 | per-level stars ⋆1^(j), c_P/c_S | `levelStars`; `synthD` (Rayleigh quotients) | logged per tensor |
| Sec.2.5 | heap-key persistence filtration, 1-LSB noise floor | `persistence`,`xorDedup`,`symDiff` | selftest: essential = β1 |
| (N1)/(N2), P6 | (K−ω²M)v=0, ω=√λ; DFT diagonalization with per-level stars | `p6Check` (dtft_tests.go) | selftest P6 3e-15 |
| (D1) | Hann DFT, Goertzel phase at ν_base, group delay | `hann`,`dft`,`goertzel`,`groupDelay` | selftest group delay |
| Sec.4 gate | PSR ≥ 10, uniqueness ≥ 1.4 | `peakOf`; consts `PSRMin`,`UniqMin` (params.go) | P2 PSR counts |
| (E1) | Candan/Jacobsen 3-sample | `candan` | selftest 7.300022 |
| Sec.4 | chirp-z zoom | `czt` (cross-check `NuCZT`) | selftest |
| (L1) | ω = Δν·N/255 turn/tick, θ0 = φ_i−φ_b | `synthD` step 3 | — |
| R-KTE-1 | tick with g_θ=4, g_ω=8 | `kteTick`,`lift` | selftest L2 |
| (L2) | |Δω| ≤ 1/128 | `PairBound` (params.go), `pairLag`, decision (ii) | selftest 1/128 lock, 1/100 not |
| Sec.5 pop. | K_c = 2Δ, Δ = IQR/2, r = √(1−K_c/K) | `popPredict` | logged r meas vs pred, K/K_c |
| Sec.5 merge | (i) PSR ∧ (ii) L2 ∧ (iii) winding ∧ (iv) lost clear; unlocked → base | `synthD` step 6 | P1–P4 |
| Alg.D | per-tensor driver | `synthD` (dtft.go) | `hodgemix --dtft-tests` |
| Sec.6 proposal | BNZ4 v2: ν̂ BF16, φ 12-bit, q int8, p BF16, flags | `DtftChunk`, `Encode`/`DecodeChanMap` (chanmap.go) | selftest v2 round-trip; e2e PASS |

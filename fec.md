# A bounded commuting projection for the married-axis grid

Gregory Disney-Leugers, Encryptid Systems. 2026-10-22.

Handoff for another model. The numbers in §§4–5 were measured on synthetic tensors in `/workspace/resample-check/` (code snapshot hm-1021, 2026-10-21). Nothing in those tables was estimated. This tree was not re-run for those tables. The production Go cited below was checked against the current `hodgemix-go` sources. The production code is unchanged. No replacement has been approved.

## 1. What this project is

`hodgemix-go` assimilates a contributor model into a base model without gradients, so the result works as one model. In this project a merge means a colony of separate cooperating models. Call the operation assimilation.

Lineage:

- `hodge-2b4t-1` is BitNet 2B4T.
- Hodge-3.5 (`hodge-vaultgemma-1`) assimilates VaultGemma into the 2B4T shape (width 2560, head dim 128).
- Hodge-4 assimilates Gemma 4 into Hodge-3.5. Its reported perplexity is 1.06× Hodge-3.5's, with no gradient training. `stdiff` against the source checkpoints found 0 byte-identical tensors in either pair, so both files are their own models.

The pipeline has no loss, no backward pass, and no optimizer.

- `synthD` is the weight-space core: seed align (phase correlation plus Re-Basin), Hodge split, TIES on the curl part, DTFT Kuramoto lock on the harmonic part (`dtft.go`).
- Ternary limbs are locked against limb 0.
- Forward-only statistic corrections when `--topology base`: surgery, pin-mean, repair.

The question. Tensors are resampled along axes with hat (Whitney \(P_0\)) interpolation. Finite Element Exterior Calculus says a discretization map should commute with \(d\) and be uniformly \(L^2\)-bounded (Arnold–Falk–Winther; Schöberl; Christiansen–Winther). Does a Schöberl-type smoothed commuting projection help?

## 2. Where resampling happens

| Function | File | Used for | Topology |
|---|---|---|---|
| `newAxisInterp` / `axisInterp.cols` | `topology.go:673` | `copyResampled` (embeddings, `lm_head`, norms, biases) and the Eq.(22) detail carry (`main.go:1071`) | only when `--topology` is `max` or `married` |
| `splineGrid` / `interp` | `math.go:445`, `math.go:252` | the index grid that `synthD` and `S.Harm` read (`main.go:991`) | every topology, including `base` |

`splineGrid` and `newAxisInterp` are the same map. On the measurement machine their outputs agreed exactly (max difference 0 in all 8 cases). Both are clamped piecewise-linear interpolation: `interp` returns the endpoint value outside \([x_0,x_{n-1}]\) and otherwise \(f_i + t(f_{i+1}-f_i)\).

The grid step is a 16:1 downsample on \(D\). `gridSizeFor` (`math.go:416`) keeps whole coarse radix levels of the axis length, capped by `GridMax` = 256. For 2560 the descending radices are \(5,2,2,\ldots\) and the product stops at 160, so `gridSizeFor(2560) = 160`. Width 2560 onto 160 cells is 16:1. That grid is what the lock reads under every topology, including `base`.

`whitney.go` is a different operator. It is the periodic Whitney pair of paper_v2 Eq.(26) on a cycle of length \(n\) prolonged to \(2n\). The resampler is not that pair.

## 3. Method

Go only. Tiny synthetic tensors, at most \(60 \times 512\), every run under a 120-second timeout. No model weights were loaded.

The current arm is the production code. A byte-identical copy of the hm-1021 package (sha256 list in `ORIGIN.sha256`) was called directly: `newAxisInterp`, `splineGrid`, `S.Harm` (`math.go:217`), `synthD` (`dtft.go:600`), and synth v2 / `kuramotoTick`. New code lived only in `_test.go` files.

Arms:

- **hat (current).** `newAxisInterp`, clamped ends.
- **smoothed 0-form.** \(\pi = (Q|_{V_h})^{-1} Q\), where \(Q\) is the canonical interpolator after mollification.
- **box 1-form.** Unsmoothed cell-integral restriction. Preserves sums.
- **smoothed 1-form.** The 1-form half of the same commuting pair.

Mollifier: the \(C^\infty\) bump \(\exp(-1/(1-t^2))\) with half-width \(\varepsilon = \delta \, h_{\mathrm{coarse}}\). The main arm uses \(\delta = 1\). \(\delta = 0.5\) and \(\delta = 2\) were swept. Ends are extended by constants, the same clamping as the hat. The axis \(d\) in the commutation check is the path coboundary, with no wrap edge.

Signals:

- smooth: low DCT modes;
- rough: iid \(N(0,1)\), white like weight rows;
- mixed: equal energy in the `S.Harm` and curl parts, plus white noise.

Reference truth is the band-limited DCT-II resample. "Aliased" means source DCT modes at or above the target size. The aliased-share figures do not depend on that choice of truth. The absolute errors do.

Lock parameters are the `main.go` CLI defaults: \(g_\theta = 4\), \(g_\omega = 8\), \(N_{\mathrm{lost}} = 255\), `RTol` \(= 2^{-8}\), \(W_{\mathrm{lock}} = 1/16\) turn, `--lock dtft`, band strict, `SeedAlign` on. Seeds: 8 per cell, and 24 for the large-ratio peaked lock.

## 4. Results

### 4.1 Harmonic error on rough rows

\[
H_{\mathrm{err}} = \frac{\|H(Rs) - H(T)\|^2}{\|H(T)\|^2}, \qquad H = \texttt{S.Harm}.
\]

Lower is better.

Three arms, as first reported:

| Ratio | Hat (current) | Cell average (box) | Smoothed 1-form |
|---|---:|---:|---:|
| 3:2 | 0.155 | 0.080 | 0.052 |
| 2:1 | 0.103 | 0.103 | 0.064 |
| 10:1 | 4.11 (82% aliased) | 0.19 | 0.11 |
| 16:1 | 8.39 (88% aliased) | 0.19 | 0.095 |

Four arms, from the full log. Sizes are fine \(\to\) coarse.

| Ratio | Hat | Smoothed 0-form | Box 1-form | Smoothed 1-form |
|---|---:|---:|---:|---:|
| 3:2 (144→96) | 0.155 | 0.075 | 0.080 | 0.052 |
| 5:4 (120→96) | 0.139 | 0.116 | 0.116 | 0.098 |
| 2:1 (192→96) | 0.103 | 0.108 | 0.103 | 0.064 |
| 10:1 (480→48) | 82.5% of harmonic energy aliased | 0.194 | — | 9.6% recorded on this row |
| 16:1 (512→32) | 88.2% aliased | 0.187 | — | 7.9% recorded on this row |

The 10:1 and 16:1 rows of the four-arm log did not label which column owns 0.194, 0.187, 9.6%, and 7.9%. The assigned large-ratio errors are the three-arm table and the sentence in §5: box about 0.19, smoothed 1-form 0.11 at 10:1, hat 4.11 and 8.39. Do not invent a fourth column from those parentheticals.

On mixed data, the `synthD` strict-band error at 16:1 is: hat 10.38, smoothed 0-form 0.287, box 0.307, smoothed 1-form 0.071.

### 4.2 Operator properties

| Ratio | Hat RMS norm | Box | Smoothed 1-form | Smoothed 0-form | Hat mean-share error |
|---|---:|---:|---:|---:|---:|
| 3:2 | 1.016 | 1.000 | 1.066 | 1.209 | 0.25 |
| 10:1 | 2.236 | 1.000 | 1.159 | 1.723 | 4.0 |
| 16:1 | 2.828 | 1.000 | 1.160 | 1.729 | 7.0 |

Every arm commutes with \(d\) and is idempotent. Measured commutation error is between 0 and \(2\times 10^{-16}\) for the hat and at most \(2\times 10^{-13}\) for the smoothed arms. Idempotency is exact or near \(10^{-12}\).

The hat's defect is unboundedness. At 10:1 it reads 2 of every 10 channels and gives each \(5\times\) its fair share. Its norm grows as \(\sqrt{\rho/2}\): \(\sqrt{5} = 2.236\) at 10:1 and \(\sqrt{8} = 2.828\) at 16:1, matching the table. Mean-share error equals \(\rho/2 - 1\) (4.0 and 7.0).

Exact norm at any rational ratio (EasyMath, 2026-10-23). At \(p:q\) in lowest terms the hat matrix on centered samples is identical \(q \times p\) blocks \(B\), so \(\|A\|_{\mathrm{RMS}} = \sqrt{p/q}\,\sigma_{\max}(B)\) at every axis length. If \(p\) and \(q\) are both odd, \(\sigma_{\max} = 1\) and the norm is \(\sqrt{p/q}\). If \(p/q \ge 2\) and they are not both odd, \(\sigma_{\max}^2 = 1 - 1/q + 1/(2q^2)\); \(q = 1\) is the \(\sqrt{\rho/2}\) above. At 3:2 the norm is exactly \(\sqrt{33/32} = 1.0155\), the table's 1.016. Ratios between 1 and 2 where \(p, q\) are not both odd have no closed form. `TestHatNormRational` (`fec_test.go`) checks the spectral norm of `hatAxis` against these values at two axis lengths per ratio. `TestFECBoundedAgainstHat` uses the Frobenius norm, which equals the spectral norm only at \(q = 1\).

The smoothed 0-form is unstable in two settings. At 1:2 upsampling with \(\delta = 1\), \(\|J_0\|\) is about \(6.3\times 10^{4}\) and the norm is 222. At \(\delta = 2\) it is unstable in every case. The smoothed 1-form stays stable at \(\delta \le 1\).

Build cost: hat and box take about 1 ms per axis. The smoothed arms take 20–40 ms in unoptimized dense code.

### 4.3 Locks against a base

These counts come from hodgemix-go's own lock, not from a reimplementation.

The `S.Harm` / v2 lock is saturated for every arm. Locked source oscillators fall between 93% and 99% for all arms, including the ideal reference. At 10:1 the hat gets 279/288, the box and the smoothed 1-form get 283, and the ideal gets 285.

`synthD` locks nothing on plain random fields. It assimilated 0 oscillators (the counter is named "merged") for every arm, because nothing clears the PSR \(\ge 10\) gate (`dtft.go`, the `Pass` conjunct of the merge decision).

With a shared off-bin tone (amplitude 0.4, 24 seeds) the hat is lowest in every row:

| Case | Hat | Smoothed 0-form | Box | Smoothed 1-form | Ideal |
|---|---:|---:|---:|---:|---:|
| 10:1, locked | 5 | 7 | 10 | 7 | 9 |
| 10:1, winding gate | 9 | 13 | 17 | 12 | 12 |
| 16:1, locked | 7 | 9 | 9 | 9 | 9 |
| 16:1, L2 gate | 10 | 14 | 17 | 14 | 14 |

The direction is consistent. 5 vs 10 out of 24 is about 1.6 standard errors, so the difference is not statistically resolved.

## 5. What the measurement supports

Large downsampling of rough content is where the hat fails. At 10:1 and 16:1 the smoothed 1-form cuts harmonic error about 37× and 88× relative to the hat (\(4.11/0.11\), \(8.39/0.095\)). The index-grid build runs at 16:1 on \(D\) under every topology, so this is the path the lock actually reads.

Small ratios (3:2, 5:4, 2:1) move harmonic error from about 0.10–0.16 down to about 0.05–0.10.

Smooth signals, upsampling, and the identity map show no benefit.

Most of the gain is plain box cell-averaging: norm exactly 1, sums preserved, about 1 ms. The mollifier plus \((Q|_{V_h})^{-1}\) correction roughly halves what remains (0.19 → 0.11 at 10:1). If a smoothed arm is used, use the 1-form. The 0-form is the arm that blew up.

Lock counts barely moved. The share of Hodge-4's remaining 0.06 perplexity gap that comes from this resampler is unknown.

## 6. The 1-form map proposed for `splineGrid`

This section is the construction. It is not in the tree. It is not approved. It is the map the measurement says is worth writing down before any A/B on a real build.

### 6.1 Spaces

One axis at a time. `splineGrid` already separates the axes: it resamples rows of \(W\) onto \(G_0\) centers, then columns onto \(G_1\) centers (`math.go:445–488`). A tensor product of one-dimensional maps leaves `synthD` reading a grid of the same shape.

Let the parameter interval be \([0,1]\), matching the clamped coordinates in `(0,1]`. Fine channels sit at coordinates \(x_0 < \cdots < x_{n-1}\). Coarse cells are

\[
I_g = \Big[\frac{g}{G},\, \frac{g+1}{G}\Big), \qquad g = 0,\ldots,G-1,
\]

with the sample point the code already uses, \((g+\tfrac12)/G\).

Write \(d_h\) for the path coboundary on \(n\) nodes, the operator used in the measurement, with no wrap:

\[
(d_h u)_e = u_{e+1} - u_e, \qquad e = 0,\ldots,n-2.
\]

This is an interval. `dCycle` in `whitney.go` is the periodic operator \((d_0 a)_e = a_{e+1}-a_e\) with \(e+1\) taken mod \(n\). They are not the same \(d\). See A3.

A 0-cochain is one value per channel. A 1-cochain is one value per consecutive pair, the increment. Weight rows are 0-cochains. The stable projection measured here acts on the increments and rebuilds the values.

### 6.2 Box, the sum-preserving cell integral

Let \(\tilde u\) be the piecewise-linear function through \((x_i, u_i)\), clamped to the endpoint values outside \([x_0, x_{n-1}]\), which is what `interp` already is. The box arm replaces point evaluation by the cell mean:

\[
(\pi^{\mathrm{box}} u)_g = \frac{1}{|I_g|} \int_{I_g} \tilde u(x)\, dx.
\]

On a uniform partition this is an average of the fine samples whose Voronoi cells meet \(I_g\), weighted by the overlap lengths. Properties that were measured, not derived: RMS norm 1.000 at 3:2, 10:1, and 16:1, and the sums are preserved. Cost is about 1 ms per axis, the same order as the hat.

Argument for the norm, uniform cells, piecewise-constant reconstruction. Each coarse value is a convex combination of the fine values that meet the cell, so \(|(\pi u)_g| \le\) the RMS of those fine values, and disjoint cells partition the mass. Hence \(\|\pi u\|_2 \le \|u\|_2\) up to the usual \(\sqrt{|I|}\) normalization of the discrete \(L^2\) product. The measurement reports the normalized norm as exactly 1.

Argument for sums. \(\sum_g |I_g| (\pi^{\mathrm{box}} u)_g = \int_0^1 \tilde u\), so the cell means reconstruct the integral of the clamped piecewise-linear function. The hat does not. At ratio \(\rho\) it samples about \(2/\rho\) of the channels and multiplies each survivor by about \(\rho/2\).

### 6.3 Smoothed 1-form

Let \(\varphi\) be the standard bump, \(\varphi(t) = \exp(-1/(1-t^2))\) for \(|t|<1\) and 0 otherwise. Mollify at scale \(\varepsilon = \delta\, h_{\mathrm{coarse}}\) with \(\delta = 1\) on the main arm:

\[
(K_\varepsilon v)(x) = \frac{1}{Z} \int \varphi\big((x-y)/\varepsilon\big)\, v(y)\, dy,
\]

with \(Z\) the integral of the bump, and with values outside \([0,1]\) replaced by the endpoint value (the same clamp as the hat).

Let \(V_h^1\) be the coarse space of 1-cochains that are constant on each \(I_g\). Let \(Q\) be the canonical interpolator onto that space: the cell integral of the mollified increment,

\[
(Q\, d_h u)_g = \int_{I_g} K_\varepsilon (\widetilde{d_h u}).
\]

The smoothed 1-form is the Schöberl correction of that moment against the coarse mass matrix,

\[
\pi_1 = \big(Q\big|_{V_h^1}\big)^{-1} Q.
\]

On uniform cells the mass matrix is a multiple of the identity, so \(\pi_1\) is a scalar multiple of \(Q\) per cell and the inverse is a division. That is why the correction is cheap once \(Q\) exists, and why the unoptimized dense code still only costs 20–40 ms.

The coarse 0-cochain is recovered from the coarse increments plus one clamped endpoint, so the diagram commutes by construction:

\[
\begin{CD}
C^0_h @>{d_h}>> C^1_h \\
@V{\pi_0}VV @VV{\pi_1}V \\
C^0_H @>>{d_H}> C^1_H
\end{CD}
\qquad
(\pi_0 u)_0 = u_0,
\quad
(\pi_0 u)_{g+1} - (\pi_0 u)_g = (\pi_1 d_h u)_g.
\]

Measured commutation error for this pair is at most \(2\times 10^{-13}\). Idempotency is about \(10^{-12}\) or exact. Measured RMS norms of the smoothed 1-form: 1.066 at 3:2, 1.159 at 10:1, 1.160 at 16:1. Stable for \(\delta \le 1\).

The 0-form twin applies the same formula to point values instead of increments. That is the arm to leave out. At 1:2 upsampling with \(\delta = 1\) its norm is 222 and \(\|J_0\| \approx 6.3\times 10^{4}\). At \(\delta = 2\) it is unstable in every case tested.

### 6.4 What would change in `splineGrid`

Today each output sample is `interp(gr[g], xs, col)`, a hat evaluation at the cell center. The box replacement is the integral of that same clamped piecewise-linear function over \(I_g\), divided by \(|I_g|\). The smoothed replacement integrates the mollified increments, applies \(\pi_1\), and sums from the clamped end to rebuild the coarse 0-cochain sampled at the same centers.

Both passes (rows, then columns) use the same 1-D map. `sampleGrid` (`math.go:492`) is the inverse direction, coarse grid back to target channels. It was not the arm under test. An A/B that changes only `splineGrid` leaves `sampleGrid` on the hat, and that mismatch is part of any later perplexity comparison, not something these synthetic runs measured.

`newAxisInterp` is the same hat on the detail carry and on embeddings when the topology is `max` or `married`. The 16:1 failure is in `splineGrid` under `base`. Changing only the detail carry would miss the path the lock reads.

### 6.5 What this does not prove

Commutation of the hat was measured at about \(10^{-16}\) and idempotency was exact. Those are not reasons to replace it. Boundedness on rough rows at 10:1 and 16:1 is the reason.

The harmonic error above uses `S.Harm`. Anti-aliasing removes energy that `S.Harm` still calls harmonic, because the finest radix-2 harmonic bin is the \((-1)^i\) alternation (A2). A drop in \(H_{\mathrm{err}}\) is agreement with the DCT-II reference after `S.Harm`, not a claim that every `S.Harm` bin was preserved.

## 7. Open decisions

These are unset. The construction in §6 assumes the interval coboundary used in the measurement. A different answer to A1–A3 changes which bins the projection is supposed to protect.

**A1. Two disjoint harmonic sets.** `S.Harm` (`NewSpectrum`, `math.go:167–206`) marks a bin only when it is not a gradient bin, every nonzero digit is \(1\) or \(r_j-1\), and each axis has exactly one nonzero digit. Gradient means all row digits are 0 or all column digits are 0. `synthD`'s default strict band (`bandBins`, `dtft.go:50` and the call at `dtft.go:669`) is the low-eigenvalue end of one mode axis. Under default flags the measurement found no bin in both sets. The strict band's surviving content is the row-mean profile, which `S.Harm` classifies as gradient.

**A2. `S.Harm` is not a low-pass along the axis.** The finest radix-2 harmonic bin alternates. Any anti-aliasing filter removes part of what the code calls harmonic. Harmonic projection coefficients in the measurement were about 0.92–0.95.

**A3. Interval versus cycle.** The resampler clamps. `whitney.go`, the Hodge split, and `synthD` treat the axis as a periodic cycle or torus (`dCycle` wraps).

**A4. The PSR gate hides the resampler on peak-free fields.** `synthD` merged 0 oscillators on plain random fields for every arm, including the ideal reference, because nothing cleared PSR \(\ge 10\).

## 8. Caveats

- Synthetic signals only. Real married axes are reordered by the MRH key, so real rows may be smoother or rougher than white noise. Real weight spectra were not measured.
- The Gemma 4 and Hodge-3.5 widths were not looked up in that study. The only real ratio used is 16:1 from `gridSizeFor(2560)`.
- The reference truth is a choice (DCT-II). Aliased-share figures do not depend on it. Absolute \(H_{\mathrm{err}}\) does.
- Samples are small: 8 seeds per cell, 24 for the peaked lock. Runtimes are rough because the machine was shared.
- The fraction of Hodge-4's remaining 0.06 perplexity gap due to resampling is unknown.

## 9. Next steps, none approved

1. Measure real row spectra along the married axes, and the actual Gemma 4 \(\to\) Hodge-3.5 axis ratios, without a full 2B forward on a small machine.
2. Decide A1–A3 before coding. §6 is the write-up those decisions would turn into a patch: box or smoothed 1-form inside `splineGrid`, 0-form left out, interval coboundary unless A3 says otherwise.
3. A/B the chosen grid builder on a real Hodge-4 build and compare perplexity with the 1.06× baseline. `sampleGrid` staying on the hat is part of that comparison.

## 10. Files on the build machine

- `/workspace/resample-check/RESULTS.md` — full tables, including every delta and the operator table.
- `/workspace/resample-check/DECISIONS.md` — each measurement choice and its reason.
- `/workspace/resample-check/ORIGIN.sha256` — the production code was copied byte-identically.
- `/workspace/resample-check/hm/rc_ops_test.go` and `hm/rc_check_test.go` — the test code.
- `run_all.sh`, `run_case7.sh`, `run_extra.sh`, and `out/` — reproduction scripts and raw outputs.

Reproduce with:

```
./run_all.sh && ./run_case7.sh && ./run_extra.sh && (cd hm && go test -run TestRCAggregate)
```

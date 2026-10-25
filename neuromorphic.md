# Neuromorphic hodgemix: design spec and plan

Encryptid Systems. 2026-10-25. Design only: no code was changed, nothing below was measured on this tree, and the plan is not approved.

## 0. The rule this spec follows

Math instead of brute force. Every step below is one of:

- **an exact identity:** a re-representation that leaves every logit bit-identical to today's forward;
- **a certified skip:** work is skipped only when a proven bound shows the skipped part cannot change the next integer code. When the bound does not hold, the dense path runs, so the output is still bit-identical;
- **a derived constant:** taken from a quantity the model already fixes, such as the int8 step, the KTE gains, or BF16 resolution (DECISIONS.md #10).

Nothing here trains a network, fits a surrogate gradient, tunes a firing threshold, or sweeps a parameter. The published spiking-LLM work does those things. It is listed in §2 only as a contrast. No new CLI flag is proposed. Where a choice seems to need one, it is listed as an open question in §7.

## 1. What is already neuromorphic

A ternary BitNet with int8 ActQuant is already most of the way to a spiking network. hodgemix already contains several of the standard neuromorphic pieces under other names:

| Neuromorphic primitive | Already in hodgemix | Where |
|---|---|---|
| Multiplier-free synapses (accumulate only) | ternary limbs run as a LUT fabric: gather and add, no multiplies | `eval/lut.go:144` `applyIn`, `eval/lut.go:89` `tables` |
| Integer, bounded activations | per-token absmax int8 ActQuant | `eval/eval.go:146` `quant` |
| Phase-oscillator network | the Kuramoto / KTE tick with shift gains \(2^{-4}, 2^{-8}\), lock window 1/16 turn, \(N_\text{lost}=255\) | `kuramoto.go:60` `kuramotoTick`, `kuramoto.go:38` |
| Event queue / timing wheel | MRH: a monotone integer priority queue over mixed-radix buckets, built from the DMT 12-bit arrival field | `mrh.go` |
| Kirchhoff current law on a graph | the cochain forward: residual conservation 2e-6, Stokes 9e-10 | `cochain.go:223`, `eval/flow.go` |
| Core partition | volume cells on the married axes | `volume.go:76` `cellLayout` |
| Winner-take-all gating | the update expert, admitted per token when it moves the hottest residual cell in the same direction | `eval/eval.go:574` `admit` |
| Homeostatic synaptic scaling | repair: each RMSNorm gain is set so the channel's spread matches the base's, clamped to [½, 2] | `repair.go:94` |
| Anti-Hebbian rank-one update | pin-mean: \(W \leftarrow W - (dW\hat\mu)\hat\mu^\top\), postsynaptic response times presynaptic mean | `pinmean.go:44` |
| Structural plasticity | surgery: the singular set is cut out and capped with the base's codes | `surgery.go:85`, `eval/eval.go:1179` |
| Resonators | RoPE: each pair rotates at a fixed angular frequency per token | `eval/eval.go:239`, `attention.go:114` |

Two things are missing:

1. **The forward is dense in time and space.** Every BitLinear touches every input value on every token, including zero codes.
2. **Every operation is synchronous and global.** No part of the forward is event-driven, and nothing records how much work an event-driven version would do.

The design below adds both through exact identities, without changing what the model computes.

## 2. Literature, and why we take a different route

| Work | What it does | Brute-force part we do not adopt | What we keep |
|---|---|---|---|
| SpikingBrain 1.0 / 2.0 (arXiv 2509.05276, 2604.22575) | integer spike counts with an adaptive threshold, expanded into sparse {−1, 0, +1} trains; linear attention | continual pre-training on about 150B tokens; a dynamic threshold fitted by optimization | ternary spike trains as the event form of an integer activation |
| LAS (AAAI 2026, arXiv 2505.09659), FAS (arXiv 2502.04405) | ANN-to-SNN conversion of LLMs; special neurons for outliers and nonlinearities | calibration stages that tune the conversion | outliers need their own treatment. We already have them exactly: the singular set |
| Kirin (arXiv 2602.08817) | W4A4/8 SNN hybrid; notes that high-bit activations need long time windows | — | the time-window cost. NAF coding (§3.1) caps it at 8 steps with at most 4 events |
| MatMul-free LM on Loihi 2 (arXiv 2406.02528; Abreu et al., arXiv 2503.18002) | ternary, MatMul-free model mapped onto Loihi 2 neurocores | a new architecture trained from scratch | confirms that ternary weights with low-precision recurrence map onto Loihi 2 |
| Sigma-delta conversion on Loihi 2 (arXiv 2505.06417, 2310.03251); delta inference (arXiv 2107.07305) | graded spikes carry activation *changes*; linear layers only process deltas | a threshold that trades accuracy for sparsity | the exact case only: threshold 0 on integer codes, which needs no trade |
| Saturated Kuramoto ONN (Delacour et al. 2023); Wien-bridge ONNs (arXiv 2512.14869) | phase-domain oscillator computing in hardware | — | supports the KTE lock as a hardware primitive |
| EGRU on SpiNNaker2 (arXiv 2312.09084) | first language model on a neuromorphic chip | trained event-based RNN | SpiNNaker2 as a second target |
| BitNet a4.8 (arXiv 2411.04965) | hybrid 4-bit activations and sparsified FFN states for ternary LLMs | retraining | confirms that FFN intermediate states are where the sparsity is |

The difference is this. Published work converts a float model, so conversion error has to be trained away or calibrated away. BitNet's ActQuant already makes every BitLinear input an exact integer, so here the spike code is a change of representation, not an approximation. Conversion error is zero by construction.

## 3. Design

### 3.1 N1: an exact spike code for a BitLinear (identity)

An int8 code \(q \in [-128, 127]\) has a unique non-adjacent form (NAF), \(q = \sum_{t=0}^{7} d_t 2^t\) with \(d_t \in \{-1, 0, +1\}\) and \(d_t d_{t+1} = 0\) (Reitwiesner 1960).

- NAF has minimal Hamming weight among signed-binary representations. Its expected density is 1/3, against 1/2 for plain binary.
- Since \(|q| \le 2^7\), eight digit positions suffice, so there are at most 8 time steps and at most 4 nonzero digits per value.
- For a limb \(D\) (digits minus centre), write \(s_t\) for the vector of digit \(t\) across the input. Then
  \[ Dq = \sum_{t=0}^{7} 2^t\, D s_t. \]
- Each \(D s_t\) is a ternary-weight accumulation of ternary spikes: an add or subtract per event per nonzero synapse, and a shift per step.
- The integer accumulator equals today's `acc` in `applyIn` exactly, so the logits are bit-identical.

There are two event forms:

- **Graded** (Loihi 2): one graded spike carries \(q_j\) when \(q_j \ne 0\). The event count is \(\#\{j: q_j \ne 0\}\).
- **Ternary NAF** (binary-spike hardware and SpiNNaker2 software): the event count is \(\sum_j w_\text{NAF}(q_j)\), at most 8 steps.

The order is MSB first. Large values fire early, which is time-to-first-spike coding, and MSB-first is what makes the bounds in §3.3 possible.

The scale is unchanged. \(y = \text{acc}\cdot x_s \cdot s_l\) is applied once per row after the accumulation. The per-token scale \(x_s\) is a broadcast gain, in effect a neuromodulator.

### 3.2 N2: exact structural sparsity (identity)

- **Zero codes are not events.** Any \(q_j = 0\) contributes nothing in either form. With absmax scaling, the massive channels (2B4T: 17 of 2560 at 41–60× the median rms) set \(\max|y|\). *Hypothesis to measure:* most other channels then quantize to small codes, often 0, so input sparsity comes for free. Whether the input norm gains damp this is unknown.
- **Compute relu² gate-first.** \(h_u = \max(g_u, 0)^2\, \text{up}_u\) is exactly 0 when \(g_u \le 0\). Compute gate first, then compute up only for rows with \(g_u > 0\). `ffn_sub_norm` is a per-channel scale and keeps zeros, and int8 maps 0 to 0, so down's input is exactly 0 on those units. down reads only the live columns. A zero unit cannot be the absmax, so the result is bit-identical.
- **silu / gelu bases** (Llama, Gemma) have no exact zeros before quantization. Only the int8 zero codes of down's input are skipped.

### 3.3 N3: certified early stops (bounded, exact output)

MSB-first accumulation gives an exact bound on the part that remains. After digits \(t \ge \tau\), let \(\text{tail}_\tau = q - \sum_{t\ge\tau} d_t 2^t\). Then

\[ \big|y_r - y_r^{(\ge\tau)}\big| \;\le\; \|W_r\|_2 \,\|\text{tail}_\tau\|_2 \]

(Cauchy–Schwarz).

- \(\|W_r\|_2\) is precomputed once per row; for limb 0 it is \(\sqrt{\text{nnz}_r}\) times the scale.
- \(\|\text{tail}_\tau\|_2\) is computed once per token per \(\tau\) and shared by every row. A row's certificate therefore costs one multiply.

Uses:

1. **relu² gate sign.** Stop gate row \(u\) early, and skip its up row and down column, when \(y_u^{(\ge\tau)} + \|W_u\|\|\text{tail}_\tau\| \le 0\). The result is exact.
2. **Greedy LM head (tied BF16 embedding).** Order D by a fixed order with precomputed per-row suffix norms \(\|E_{r,\text{rest}}\|\) at cell checkpoints. Run branch-and-bound: a row is pruned when its upper bound falls below the best lower bound. The argmax is certified, and ties fall back to dense (lower index wins, as today). This is winner-take-all with a proof attached.
3. **Attention keys.** Each cached key has an upper bound \(u_n = \|q\|\|k_n\|/\sqrt{d}\), passed through softcap where the model has it, since softcap is monotone. Take any exact score as a lower bound \(s^*\) on the maximum. The softmax weight dropped with keys where \(u_n < s^* - \Delta\) is at most \(\sum e^{u_n - s^*}\), so the output moves by at most \(2W_\text{drop}\max_n|v_{n,i}|\) per coordinate.

The tolerance is not chosen. It is the next quantizer. A perturbed vector \(\tilde y \pm e\) is certified when every int8 code \(\text{round}(127\,y_i/\max|y|)\) is the same at both ends of the interval, with the absmax itself in \([\tilde M - e_M, \tilde M + e_M]\).

- Between attention and o_proj sits `attn_sub_norm`. Its perturbation is bounded through the RMSNorm Lipschitz form: \(|\text{rms}(a+\delta) - \text{rms}(a)| \le \|\delta\|_2/\sqrt n\).
- The float-rounding error of the dense reference is added through the standard summation bound \(\gamma_n \sum|\text{terms}|\) (Higham). The certificate therefore promises equality with the dense float forward, not with a real-valued ideal.
- If any code is not certified, the dense path runs. Nothing fails soft: the output is always the dense output.

### 3.4 N4: exact temporal deltas (identity, conditional)

Linearity on integers gives \(A_t = A_{t-1} + D(q_t - q_{t-1})\) per BitLinear, with \(y_t = A_t\, x_{s,t}\, s_l\). This holds even though the absmax scale changes per token, because the scale is applied after the integer accumulation. Events are the NAF weight of \(\Delta q\).

This only pays where consecutive tokens' codes are correlated. For LM hidden states that is not known. The candidate reference is the base's mean input code, which is pin-mean's \(\hat\mu\) in code units. N4 is built only if the P0 measurement shows \(E[w(\Delta q)] < E[w(q)]\). Threshold 0 only, so the result is exact; no SDNN-style accuracy trade.

### 3.5 N5: the KTE lock as integer phase neurons (identity up to a proven stability margin)

The lock is already a Kuramoto oscillator network. A fixed-point tick would make it run on integer neuromorphic cores or on the jev_kuramoto engine itself, and would make it reproducible bit for bit across machines:

- **Phase:** a 16-bit turn counter. \(W_\text{lock} = 1/16\) turn is exact, and \(\text{wrap}\) is integer overflow.
- **sin/cos:** a quarter-wave table indexed by the top bits of the phase.
- **Coupling:** \(K = 2^{-g_\theta} C / d_\max\). Replace \(d_\max\) by \(2^{\lceil\log_2 d_\max\rceil} \ge d_\max\). Gershgorin still holds, \(\lambda_\max \le 2\cdot 2^{-g_\theta} < 2\), so the tick stays stable. The cost is coupling up to 2× weaker, which N_lost and the Ott–Antonsen \(K/K_c\) already report. All gains become shifts.
- **Event view:** with \(\omega = 0\), a locked oscillator's update goes to 0, so in an event implementation traffic decays as oscillators lock.

The acceptance test is equality of decisions, not of floats: the same locked set and merged modes as the float tick on the dtft P1–P4 grids and on real tensors.

### 3.6 N6: local plasticity forms of the forward-only corrections

Pin-mean, repair and surgery each read the base's calibration forward. Each one is a local rule over quantities a neuromorphic core can count from its own spikes:

- **Pin-mean:** \(\hat\mu\) from the presynaptic code sums, and \(dW\hat\mu\) from the postsynaptic accumulators. The update is a rank-one anti-Hebbian outer product.
- **Repair:** each channel's spread from the sum of codes and the sum of squared codes.
- **Surgery:** the massive channels and spike units from per-channel running maxima.

With exact cumulative sums (integer, no decay constant), the streaming result equals the batch result on the same tokens. That equality is the test.

The calibration length then becomes derivable rather than fixed. An empirical-Bernstein bound (Maurer–Pontil 2009) on each channel's mean code gives the number of tokens to resolve \(\hat\mu\) to half an int8 step at confidence \(1 - 2^{-8}\), and the run reports it. A worst-case Hoeffding bound is far too loose here (on the order of \(10^5\)–\(10^6\) tokens for range 255), so only the variance-aware bound is usable.

### 3.7 N7: placement onto a core mesh (closed form)

The traffic graph has (layer, cell) nodes from `volume.go`, with edge weights equal to the expected events per token from P0. Hall's quadratic placement minimizes \(\sum w_{ij}\|p_i - p_j\|^2\). Its solution is the second and third eigenvectors of the traffic Laplacian. The conjugate-gradient machinery in `cochain.go` already solves this Laplacian class.

To round to grid slots, assign nodes to slots by the existing `hungarian` (`math.go:895`) on distance cost. That is optimal for the linear assignment step and needs no search over placements.

The cochain's Hodge split also says which traffic is gradient, a DAG-like pipeline, and which is cycle, the residual loops. Cycle traffic is where buffers are needed.

### 3.8 RoPE as resonators (mapping note, not a plan item)

\((R_m q)^\top(R_n k) = q^\top R(\theta(n-m))\,k\). A bank of undamped resonators at frequencies \(\theta_p\), sampled once per token tick, therefore implements RoPE with no position index. On digital hardware this gains nothing over storing rotated keys, which is what the forward does now. It matters only on oscillator hardware (ONN, KTE), where rotation is physical. The RoPE frequency marriage in `attention.go:114` (\(s = d \ln\theta_t/\ln\theta_c\)) is then a retuning of resonator frequencies.

## 4. Interfaces

None of these adds a CLI flag (memory rule, §7).

- **`eval` event layer.** A per-BitLinear event counter fed from `s.tap` and from `lin.run` / `runFlow`. Per role and layer it counts nonzero codes, NAF weight, live relu² units, certified skips, and fallbacks. The forward is unchanged.
- **`report.json` `neuro` section.** Written by the existing post-mix eval, which already runs every time:
  - events per token, graded and NAF;
  - synaptic ops (one accumulate per event per nonzero weight), against dense;
  - skip and fallback rates for N3;
  - \(E[w(\Delta q)] / E[w(q)]\) for N4.
- **The event forward.** A second implementation of `lin` that takes events, plus the gate-first FFN and the certified attention and head. It is bit-identical by construction, and tests enforce that.
- **The integer KTE tick.** Has the same signature as `kuramotoTick` and returns the same `KResult`.
- **Units.** Energy is not estimated. Energy is synaptic ops times a target chip's datasheet figure, and only once a target is chosen (DECISIONS.md #8: no fabricated numbers).

## 5. Plan

Each phase ends on a gate. A phase that fails its gate stops there and is written up as such.

| Phase | Work | Tests | Gate |
|---|---|---|---|
| **P0 Measure** | event counters and the `neuro` report section only; no change to the forward | `TestNAFIdentity` (all 256 int8 codes: unique, non-adjacent, sums back, weight ≤ 4); report fields present on the tiny `run()` fixtures | numbers on 2B4T and on one silu base (Llama-3.2) decide which of P2–P4 are worth building |
| **P1 Event BitLinear** | graded and NAF event forms of `applyIn`, radices {2, 3, 4} and several limbs | `TestEventLinearExact` against `applyIn` on random limbs; stock 2B4T logits bit-identical on the eval sentences (the `TestLimbProbe` pattern) | bit-identical, or no merge |
| **P2 Gate-first FFN** | relu² gate-first, zero-code skipping on down | logits bit-identical on eval, `--chat`, `--converse`; skipped fraction reported | bit-identical; skip fraction > 0 measured |
| **P3 Certified stops** | MSB-first gate-sign stop; certified LM-head argmax; certified attention pruning with the quantizer certificate and the float error term | property tests that the bound holds on random inputs (it must never fail); logits identical on the full converse suite; fallback rate reported | zero mismatches; fallback rate reported, not tuned |
| **P4 Temporal deltas** | only if P0 says \(E[w(\Delta q)] < E[w(q)]\) | accumulator equals dense on long chats (334 tokens) | the inequality holds measured, otherwise P4 is dropped |
| **P5 Integer KTE** | fixed-point tick, power-of-two \(d_\max\) | same locked set and merged modes as float on dtft P1–P4 and on 2B4T + Qwen3 tensors; `TestMarriedSelfMixIsIdentity` still passes | equal decisions on every tensor, or a list of the ones that differ |
| **P6 Local plasticity** | streaming pin-mean, repair and surgery statistics; empirical-Bernstein token count | streaming equals batch on the same calibration tokens (exact) | equality; the derived token count is reported against today's calibration set |
| **P7 Placement** | traffic Laplacian placement, Hungarian rounding, a core map file | placement objective ≤ the identity layout's; deterministic | needs a chosen target (§7) |
| **P8 Hardware** | port to the chosen target | that target's own logit identity check (as e2e does for Champ) | target-dependent |

Order and dependencies:

- P0 comes first.
- P1 and P5 are independent of each other.
- P2 and P3 need P1.
- P4 needs P0's numbers.
- P6 is independent.
- P7 and P8 need a target.

Everything up to P6 is Go-only and runs on the current box.

## 6. Risks

- **The CPU will be slower.** Event forms do more bookkeeping than the LUT fabric. P1–P3 are reference implementations and counters for hardware, not a CPU speed-up. The gate-first FFN (P2) is the one that may also be faster on the CPU.
- **Global reductions remain.** Absmax per token, RMSNorm and the softmax denominator are reductions over a whole vector. They are part of BitNet's function and cannot be made local without changing the model. On a mesh they become reduction trees, and P7 has to place them.
- **The sparsity is unknown until P0.** The §3.2 hypothesis (absmax set by the massive channels drives most codes to 0) may be cancelled by the input norm gains. Nothing past P1 is committed before P0.
- **Attention dominates long context.** The KV cache is read on every token, so on chip-memory-limited hardware the cache, not the BitLinears, sets the limit. N3's certified pruning reduces the reads but does not remove the storage.
- **Integer KTE may move borderline locks.** Modes near the \(|\Delta\omega| \le 1/128\) boundary or near \(W_\text{lock}\) can flip. P5's gate is equal decisions, so any flip is reported per tensor rather than averaged away.
- **Scale.** 2B4T is about 0.5 GB of packed ternary weights, plus a BF16 embedding of about the same size. A Loihi-class deployment needs a multi-chip system (MatMul-free LM used Hala Point). Single-chip targets are only realistic for the tiny fixtures and the towers.

## 7. Open questions for Gregory

1. **Target hardware.**
   - Is there a physical KTE / DMT / FFT-correction chip that the integer tick (N5) and the MRH event queue should match bit for bit?
   - Or is the target Loihi 2 (graded spikes, through INRC), SpiNNaker2, or Champ WebGPU with event compaction?
2. **Entry point for the event forward.** It is bit-identical, so it can replace the forward by logic wherever it is faster, with no flag. If you want it selectable anyway (for example to compare speeds), that would be a new flag, which needs your approval.
3. **Integer KTE.** Should the integer tick become *the* lock, so that the papers' float tick is the reference, or stay a hardware-equivalence test only?
4. **Calibration length.** If the empirical-Bernstein count (N6) exceeds today's calibration set, should the run stop (nothing fails soft) or report?
5. **The towers.** Should Gemma 4's encoders (`assimtowers.go`, `towerwasm/`) be in scope? Audio frames are the one place where temporal deltas (N4) are known to pay in the sigma-delta literature.

## Sources

- SpikingBrain: arXiv 2509.05276; SpikingBrain2.0: arXiv 2604.22575
- LAS: arXiv 2505.09659 (AAAI 2026); FAS: arXiv 2502.04405; Kirin: arXiv 2602.08817
- Scalable MatMul-free LM: arXiv 2406.02528; Neuromorphic principles for LLMs on Loihi 2: arXiv 2503.18002; event-driven LMs on neuromorphic hardware: arXiv 2608.30439 (abstract not read)
- Sigma-delta conversion on Loihi 2: arXiv 2505.06417; video/audio on Loihi 2: arXiv 2310.03251; temporal sparsity / delta inference: arXiv 2107.07305
- Delacour et al., mixed-signal Kuramoto ONN, Neuromorph. Comput. Eng. 2023 (HAL lirmm-04290523); Wien-bridge oscillator attractors: arXiv 2512.14869
- Language modeling on SpiNNaker2 (EGRU): arXiv 2312.09084
- BitNet a4.8: arXiv 2411.04965
- Reitwiesner, binary arithmetic / NAF, 1960; Hall, r-dimensional quadratic placement, Management Sci. 1970; Maurer & Pontil, empirical Bernstein bounds, COLT 2009; Higham, *Accuracy and Stability of Numerical Algorithms* (summation error \(\gamma_n\))

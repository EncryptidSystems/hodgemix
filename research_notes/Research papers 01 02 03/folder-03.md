# Folder 03: referee review of the Hodge–Kuramoto papers

Read from the files in `C:\Users\grego\Downloads\Research\03 Hodge-Kuramoto Model Merging`. PDF text was extracted page by page. `paper_v2.tex` and `hodgemix_architecture_v2.tex` were read as source. Vendored trees (`hodgemix-go` except `paper_v2`, `ternmix-py`, `bnz-loader`, `bnz4-wgsl`) were not used as evidence.

The folder `README.md` is not a map of the papers. It is the prototype note for the BNZ entropy stream (Dieharder and NIST SP 800-22 “not done or claimed here”).

`hodge-kuramoto-align.pdf` and `hodgemix-go\hodgemix-go\paper_v2.pdf` are the same file (equal length 319322 bytes, equal SHA-256, PDF creation `D:20261014143127-04'00'`). One paper.

Dates below are the dates printed in the documents, or the PDF creation timestamp when the text has only “October 2026”.

## For each paper: title, date, claim, and the status of each headline result

### Takeaway

Fourteen distinct documents. Two byte-identical copies of the mix paper. The only results that are written out as proofs inside this folder are: elementary Hodge/Kuramoto/BNZ identities and one Gershgorin tick bound in `paper_v2`; a half-circle lock sketch and a pair-lock threshold in `dtft_lock`; three short Fourier/RMSNorm propositions in `singularity_surgery`; two holonomy propositions and one unrestricted existence proposition in `monodromy_moe`; Whitney-operator invariance and a corrected counting termination theorem in the 30–31 October surgery notes; a conditional exact-recovery proposition for the entropy stream; and, in the neckpinch note, a pre-singularity convergence argument for a reduced Ricci–DeTurck system on \(S^1\times S^3\) that still depends on an external “Track A” paper not in this folder. Everything about large-model merge quality is a protocol, a smoke-test table, or a single trace. The 41-page Dynamic Ricci PDF is a pasted chat, and its continuum-limit and monotone-energy claims are sketches that later notes in this same folder either exclude or numerically contradict.

### Cited Findings

**1. `paper.pdf` — Ternary Conversion and Mixing of Open-Weight Models via Hodge–Kuramoto Alignment. Gregory Disney-Leugers, Encryptid Systems, 14 October 2026. PDF creation `D:20261014123029-04'00'`. 3 pages.**

- Claim: a pipeline that converts open-weight checkpoints to ternary weights and mixes them onto BitNet b1.58 2B4T, using a Hodge split, a mixed-radix DFT, Kuramoto locking, and a BNZ4 chunk. — [paper.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/paper.pdf)
- Headline items in §2 are a contribution list (ternary TIES, DEC resizing, MRH scheduler, mixed-radix DFT, BNZ4, Hodge–Kuramoto lock). No theorem, no proof, no table. §5 “Measurements” names perplexity, KL, Hodge energy, the order parameter \(r\), DFT compaction, and chunk size as quantities to measure. No values. Status of every headline item: pipeline specification. — [paper.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/paper.pdf)

**2. `paper_v2.tex` / `paper_v2.pdf` (and the identical `hodge-kuramoto-align.pdf`) — same title, “October 14, 2026 (version 2)”. 8 pages. The tex is the text that was reviewed; the PDF matches it (same date line, Algorithm 1, the sentence that no real or 2B-scale model was run, and the S1–S8 table).**

- Claim, abstract: convert checkpoints to BitNet ternary and mix them; report only operator checks and tiny synthetic runs; large-model evaluation is a protocol. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- Eqs. (1)–(5), Hodge split and bin sets \(\mathcal G,\mathcal C,\mathcal H\): definitions. Orthogonality is a numerical check (S2, S3), not a written proof. The sentences that assign gradient to alignment, curl to TIES, and harmonic modes to shared structure are interpretations. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- Eq. (4) diagonalization: standard circulant fact, checked in S1 to \(\le 2\times 10^{-14}\). Status: numerical. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- Eq. (26) Whitney identities: numerical (S5: \(1.1\times 10^{-16}\), 0, 0, 0). Status: numerical check of standard identities. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- Gershgorin line after Eq. (15): \(\lambda_{\max}(L_K)\le 2\cdot 2^{-g_\theta}<2\) with \(g_\theta=4\). Status: proved in the text, elementary, for that coupling only. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- “Re-Basin or OT coincide” via Birkhoff–von Neumann: one sentence, no proof. Status: citation. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- Algorithm 1, ternary TIES Eq. (21), synthesis Eq. (22), Hadamard rotation Eqs. (23)–(24), absmean Eq. (25), BNZ4 §6: specifications. Rotation invariance is numerical (S4, \(1.9\times 10^{-7}\) in float32). — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- Full lock when \(\omega_i=0\): cited to Dörfler–Bullo and Strogatz, then illustrated by S6 and a tiny run. The text says a nonzero data-derived \(\omega\) is “an open question for the protocol below.” Status: citation plus numerical illustration; the informative case is open. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- § “Measurement protocol (future work)”: perplexity, KL, Hodge-energy vs merge KL, \(r\) as a predictor, DFT compaction, chunk size, distillation. “No large-model numbers are claimed.” Status: protocol only. Distillation is “outside the code described here.” — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- No Ricci flow, no DeTurck gauge, no surgery, no safetensors theorem. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)

**3. `dtft_lock.pdf` — DTFT Frequency Assignment for Selective Hodge–Kuramoto Lock. 14 October 2026, “proposal; companion to paper v2”. PDF creation `D:20261014153950-04'00'` (later the same day than `paper_v2.pdf`). 8 pages.**

- Claim: \(\omega_i=0\) forces a full lock, so the reported 2B4T+Qwen line \(\langle r\rangle=1\), 147/147 locked, is uninformative; replace it by normal-mode frequencies measured with a DTFT. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- Proposition 1 (uniform \(\omega\) implies full lock): gradient-flow sketch via Łojasiewicz, half-circle basin, and a linear-lift contraction using paper v2’s Gershgorin bound. Status: proof sketch. The half-circle restriction is part of the statement. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- Proposition 2 (Betti number): (a) no 2-cells; (b) all squares filled, \(\beta_1\) equals the number of digits with radix \(\ge 3\), cited to Künneth, and \(\beta_1=4\) for \(2560\times 6912\); (c) complete bipartite graph. Sanity: \(C_5\times C_3\) gives \(\beta_1=2\) with squares. Status: stated, with one tiny exact-rank check. The \(2560\times 6912\) value is an application of (b), and the protocol that would check it by exact rank is listed as not yet run. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- Lemma 1 (periods depend only on the cohomology class): one-paragraph Stokes argument. Status: sketch of a standard fact. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- Consequence after (N1): true harmonics have \(\omega=0\) for every metric, so discriminating oscillators must sit on the quasi-harmonic band. Status: argument from the eigenproblem, not a merge theorem. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- (L2) pair threshold \(|\Delta\omega|\le 1/128\) turn/tick: steady-state arithmetic, plus a three-point tick sanity (\(\Delta\omega=1/256\) locked, \(1/128\) boundary locked, \(1/100\) not locked). Status: derivation plus a three-point numerical check. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- Algorithm D: specification. §6 “Proposed changes to paper v2 (not applied).” — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- Predictions P1–P6 and §9: “Tiny files only; no numbers here are measured.” Status: protocol only. The arithmetic checks that were run are the \(C_5\times C_3\) rank, the sub-bin example (true bin 7.3, estimate 7.30002, \(N=64\)), and the three tick lags. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- Forman curvature on the square torus is \(2+2-4=0\). That is a count, not a Ricci-flow theorem. The Weitzenböck sentence is a remark. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)

**4. `singularity_surgery.pdf` — Singularity Surgery for Hodge–Kuramoto Model Merging. 15 October 2026. PDF creation `D:20261015081842-04'00'`. 5 pages.**

- Claim: a ThinkPad reference-forward trace diverges in 9 of 2560 residual channels; model those channels as necks on a weight 3-manifold and cap them with base values, “built directly on” Perelman. “No new runs are reported; the only data are the ThinkPad trace in §1.” — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf)
- Definitions 1–2: specification of the weight 3-manifold and the neck. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf)
- Proposition 1 (spectral spreading of a spike): elementary DFT fact, then the numerical application \(|B|/N\le 4/640\). The bound cites “dtft lock problem Tab. 2” for \(|\mathcal H|\in\{1,2,4\}\). `dtft_lock.pdf` has no table of those cardinalities. Status: the Fourier identity is standard; the \(4/640\) figure depends on a table that is not in the cited companion. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf)
- Proposition 2 (sub-norm share \(\le g_u\sqrt{F}=83.1\,g_u\)) and Proposition 3 (share conservation on the RMSNorm sphere): short algebra. Status: proved as algebra, given the 2B4T norm description. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf)
- T1–T5 and Algorithm S: specifications. The finite-surgery sentence is an analogy with Perelman (“surgery is applied finitely often”), with no index count and no proof. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf)
- F1–F5 and §10: “Validation protocol (not run).” Open items (1)–(5), including “data-free proxy from weights alone not derived.” Status: protocol and open. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf)
- §1 ThinkPad numbers are an observation of one trace, not a theorem. Hypothesis H (one early FFN unit) is labeled a hypothesis. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf)

**5. `hodgemix_architecture.pdf` (version 1) and `hodgemix_architecture_v2.tex` (version 2). The tex is later.**

- PDF title in the extract: The Hodgemix Ternary Transformer, 15 October 2026, creation `D:20261015165308-04'00'`, 11 pages. It describes the code copied on 15 October. — [hodgemix_architecture.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture.pdf)
- Tex title: the same, “Version 2: Model Families, Expert Gates and the Forward Cochain”, date 18 October 2026. File mtime is later than the PDF. No version-2 PDF is in the folder. — [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)
- Both are architecture descriptions. The abstract of v2 says quality numbers are smoke tests. § “Implemented versus specified” is a table of gaps between Algorithm S and the code. Status of the pipeline equations (pin-mean, OV chart, repair, limbs, TIES): specifications of code. Status of the smoke-test table: numerical, with the paper’s own caveat that calibration overlaps the evaluation sentences and that no two runs share a configuration. — [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)
- v2 adds, relative to the 15 October PDF: family-general block, three MoE gates, neck limbs, the revised TIES denominator, weight volumes, the forward cochain, and a run-file/preflight. It records that no perplexity exists for neck limbs, the limb lock, or the new TIES rule. v1’s open item that \(W_k\) is applied twice is listed in the v2 changelog as fixed. — [hodgemix_architecture.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture.pdf); [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)
- v2’s monodromy table: holonomy, depth monodromy, defect \(\eta\), and \(\Delta_{\rm RH}\) are “not computed.” Status: the geometric MoE claim is unimplemented in the code the paper describes. — [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)

**6. `monodromy_moe.pdf` — Monodromy, the Riemann–Hilbert Problem and Mixture-of-Experts on the Hodgemix Weight 3-Manifold. 15 October 2026. Creation `D:20261015225811-04'00'`. 7 pages.**

- Claim: the merge is a discrete connection; trivial holonomy is necessary and sufficient for a global averaging gauge; MoE is the alternative when holonomy is nontrivial. “No runs are reported.” — [monodromy_moe.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/monodromy_moe.pdf)
- Proposition 1: proved in the text by a spanning-tree gauge. Status: proved, as a statement about frames the merge assigns, not as a measured quality theorem. The ppl \(1.3\cdot 10^7\) sentence is explicitly “smoke-test observations, not tests of the proposition.” — [monodromy_moe.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/monodromy_moe.pdf)
- Proposition 2 (\(\pi_1\) of the torus minus a slab): stated. The van Kampen write-up for the interval rows is in the later AnyL note, §3.1, which cites this proposition for the periodic rows. Status in this PDF: asserted. — [monodromy_moe.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/monodromy_moe.pdf); [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- Proposition 3 (unrestricted existence of frames for every \(\rho\)): proved by putting \(\rho\) on the chords. Status: proved. — [monodromy_moe.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/monodromy_moe.pdf)
- Conjecture 1 (Bolibruch-type obstruction for local frames): labeled a conjecture. “(c) is stated as a conjecture by analogy; it is not derived.” Status: open. — [monodromy_moe.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/monodromy_moe.pdf)
- Predictions P1–P7 and §7: protocol on tiny synthetic models, not run. Eight open items, including whether the layer axis is periodic. — [monodromy_moe.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/monodromy_moe.pdf)

**7. `bnz_anti_entropy.md` and `bnz_anti_entropy.pdf` — Audited Entropy for Generative-Model Test and Evaluation. 26 October 2026. PDF creation `D:20261026090101-04'00'`. 7 pages. The markdown is the paper source; the PDF matches it on Proposition 1, the “no pass rates” sentence, the Hodge energies, and the 4.52 / 7.59 bit/byte diagnostic.**

- Claim: an invertible entropy stream from a DEC cochain through BNZ4 maps, an integer Walsh–Hadamard, and keyed whitening. — [bnz_anti_entropy.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/bnz_anti_entropy.md)
- Proposition 1 (exact recovery if coefficients fit in \(w\) bytes): proved by inverting each bijection. Status: proved, conditional on the word-width hypothesis, which the paper lists as open (DECISIONS A3). — [bnz_anti_entropy.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/bnz_anti_entropy.md)
- Indistinguishability (Definition 1, item 2) and the Dieharder / NIST SP 800-22 acceptance criterion: “No pass rates are reported.” Status: protocol only. The 4.52 / 7.59 figures are “a diagnostic, not a battery result.” — [bnz_anti_entropy.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/bnz_anti_entropy.md)
- Go checks C1–C4: numerical on one \(16\times 16\) cochain, with HMAC-SHA256 named as a placeholder. — [bnz_anti_entropy.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/bnz_anti_entropy.md)
- The folder README repeats the same chain and the same energies and adds that a Python port matched the Go bytes (check C5). It states no cryptographic-security claim and no battery-pass claim. It does not strengthen Proposition 1. — [README.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/README.md)

**8. `Dynamic Ricci Operator.md.pdf` — title metadata “Dynamic Ricci Operator.md”. 41 pages. No creation date in the metadata. The body is a concatenation of chat drafts (“Absolutely. Here's a clean, plain-text academic write-up”, “Would you like me to add figures”, “Yes—‘Discrete Ricci Operator’ is the precise… name”). One internal date line reads “Date: 2025” under the Relaxation–Frustration heading, author “Gregory Disney”.**

- The monotone-energy proposition (§6 of the axiomatic draft): “Under sufficiently small \(\Delta t\) and \(\gamma\), the total energy \(E=E_{\rm el}+E_K\) decreases.” The text that follows is a four-bullet sketch. Status: proof sketch. — [Dynamic Ricci Operator.md.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Dynamic%20Ricci%20Operator.md.pdf)
- The convergence “theorem” on the next page: if the mesh stays non-degenerate and the step is small enough, the scheme converges to a critical point. No estimate, no threshold. Status: sketch. The summary then calls the model “fully rigorous.” That sentence is not backed by a proof in the file. — [Dynamic Ricci Operator.md.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Dynamic%20Ricci%20Operator.md.pdf)
- “Theorem: Continuum Ricci Flow as the Limit of Discrete Ricci Operator Geometry”: five-stage “Proof Sketch” citing Dodziuk (1976) and Cheeger–Müller–Schrader (1984), with the inequalities left unwritten. Status: sketch. — [Dynamic Ricci Operator.md.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Dynamic%20Ricci%20Operator.md.pdf)
- Relaxation–Frustration Principle and Ricci–Frustration Field Law: stated with “proof sketch” language and chat prompts about where to place them in a monograph. Status: sketch. — [Dynamic Ricci Operator.md.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Dynamic%20Ricci%20Operator.md.pdf)
- Later pages identify the Riemann hypothesis with “spectral flow without tension” and the prime-number theorem with “unobstructed Ricci expansion.” Status: analogy. No proof. — [Dynamic Ricci Operator.md.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Dynamic%20Ricci%20Operator.md.pdf)
- Surgery in this file is “at prescribed times,” with a three-line reinitialization list. No termination bound. — [Dynamic Ricci Operator.md.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Dynamic%20Ricci%20Operator.md.pdf)

**9. `AnyL_Dynamic_Ricci_Singularity_Surgery.md` — Dynamic Ricci and Singularity Surgery on the Weight Manifold for Any Number of Layers. 30 October 2026.**

- Claim: restate singularity surgery and the Dynamic Ricci operator for every \(L\ge 1\), and bound the surgery loop by \(N+LM\). “No runs are reported. The only computations are the threshold and cardinality evaluations.” The note says it does not touch smooth Ricci flow on 4-manifolds, and it excludes the continuum-limit theorem, the frustration principle, and the number-theory pages of the Dynamic Ricci PDF. — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- Lemma 2.1 and Corollary 2.2 (layer locality, bandwidth \(\le 2\)): proved from triangulation condition (T). Status: proved, combinatorial. — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- Lemma 3.2 (the field off the neck ignores the singular coefficients): one sentence from the hat supports. Status: proved. — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- §3.1 interval-row fundamental-group calculation: van Kampen argument written out. Periodic rows are cited to the monodromy paper. Status: the interval cases are proved in this note; the periodic cases are citations. — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- Theorem 6.2 (Whitney system preserved): the operators depend only on the cell complex, which the cap glues back. Status: proved, and the content is that cochain values do not enter those identities. — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- Theorem 7.1 (at most \(N+LM\) surgeries): the written proof says each surgery strictly enlarges \((C^\star,U^\star)\subset[0,N)\times([0,L)\times[0,M))\) and therefore there are at most \(N+LM\) enlargements. The product set has cardinality \(N\cdot LM\), and the strict-growth step for an E3 vertex is one sentence (“add the channel and unit indices that carry \(i\)”). Status in this file: proof sketch, with the counting gap the 31 October note later names. The note itself says the Dynamic Ricci convergence claim is a sketch and is not used. — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- Eighteen “open inputs the author must set,” including stiffness, masses, \(\Delta t\), the triangulation, and whether the flow slot is the lock, Dynamic Ricci, or both. Status: the evolution is unspecified until those are chosen. — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- Threshold arithmetic for 2B4T (\(\tau_N=7.673219\), \(\tau_M=7.952663\), \(\tau_{NM}=9.886624\)) and one cardinality example with \(|C^\star|=9\). Status: numerical evaluation of formulas. \(|U^\star|=1\) is “illustrative only.” — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)

**10. `MRWT_Dynamic_Ricci_Surgery.pdf` — Multi-Radix Whitney Theory with Dynamic Ricci Evolution and Layered Surgery. “October 2026”. Creation `D:20261030222620+00'00'`. 2 pages.**

- Same claims as AnyL, compressed: Lemma 3.1 (layer locality), Theorem 5.2 (Whitney preserved), Theorem 6.1 (at most \(N+LM\) surgeries). Each proof is one to three sentences. Theorem 6.1’s entire proof is “Each surgery strictly enlarges a subset of the finite index set \([0,N)\times([0,L)\times[0,M))\).” Status: the same sketch, shorter. The “closed package” paragraph asserts the enlarged theory and then lists continuum Ricci flow, the classical positive-mass theorem, mesh-independent Lichnerowicz bounds, Penrose weak cosmic censorship, and the Kontsevich–Zagier period conjecture as outside. — [MRWT_Dynamic_Ricci_Surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/MRWT_Dynamic_Ricci_Surgery.pdf)

**11. `Spectral Detection, Dynamic Ricci Operator Flow, and 4D Topological Surgery.pdf` — printed title “Spectral Detection, Dynamic Ricci Operator Flow, and Topological Surgery on 4D Pinched Tensor Manifolds”. “October 2026”. Creation `D:20261031143154+00'00'`. 5 pages.**

- Abstract: “we prove” a logarithm-free max-norm Ritz estimate and gradient supercloseness for Ricci–DeTurck on the Kuhn 4-torus, and termination after at most \(N+LM\) surgeries. — [Spectral Detection PDF](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Spectral%20Detection,%20Dynamic%20Ricci%20Operator%20Flow,%20and%204D%20Topological%20Surgery.pdf)
- Lemma 4.1: same layer-locality statement as AnyL, with a short proof. Status: proved, combinatorial. — same file.
- Theorems 5.1 and 5.2: the estimates are stated. No proof, no constant, no dependence on the external Track A paper, appears in these five pages. Status: asserted. — same file.
- Theorem 6.1 (Whitney identities): stated, no proof in this file. — same file.
- Theorem 7.1: “Proof Sketch.” It writes the pathway set as a subset of the product \([0,N)\times([0,L)\times[0,M))\) and says the cardinality is \(N+LM\). Exit claim: “no channel or unit outside \(S\) exceeds” \(\tau_n\). Status: sketch, and the product-versus-sum slip is the one the next note corrects. — same file.
- Dimension: this PDF allows \(d_{\mathcal M}\in\{0,1,2,3,4\}\) and states theorems for \(d_{\mathcal M}=4\) on \(\mathbb T^4\). AnyL defines \(d_{\mathcal M}\) as the number of axes of size \(\ge 2\) among three axes, so \(d_{\mathcal M}\le 3\), and says 4-manifold Ricci flow is outside the note. — both files.

**12. `Surgery_termination_and_neckpinch.md` — “Theorem 7.1 closed, its neckpinch instance, and the Dynamic Ricci energy proof (B)”. Thread date 31 October 2026.**

- This is the strongest write-up of the termination argument. It lists four gaps in the sketch and supplies Lemma 1.2 (every unpinned vertex carries a new index, by a role-by-role table), Lemma 1.3 (strict monotonicity of \(S\)), and Theorem 1.1: at most \(N+LM-|C_0|-|U_0|\) surgeries after the initial one, at most \(N+LM\) in total, and at most \((N+LM+1)\,n_{\max}\) flow steps. The index universe is the disjoint union \([0,N)\sqcup([0,L)\times[0,M))\), cardinality \(N+LM\). Status: the counting proof is written in this file. It does not use a convergence theorem for the flow. — [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)
- Theorem 2.2 (convergence of the semi-discrete scheme on \(S^1\times S^3\) up to any \(T'<T_{\rm sing}\)): “transplant of Track A; only the differences are listed,” with a 1D max-norm argument written out. Status: a proof relative to a paper that is not in this folder (`ricci_deturck/paper/ricci_deturck_kuhn_T4.tex`). The rates are also given as numerical tables. — same file.
- Lemma 2.4 and Corollary 2.5 (surgery stability): the comparisons (i)–(iii) are written; (iv)–(v) are a Lipschitz composition. Status: proved under the stated nondegeneracy assumptions, inside this file. — same file.
- Theorem 2.6 (discrete loop, at most \(N_{\rm el}\) surgeries): counting, with “element” in place of “index.” Status: proved as a counting statement about the pinned loop. — same file.
- Theorem 3.1 (corrected energy \(H\)): a damped-leapfrog descent argument plus Łojasiewicz, written out. Status: proved in this note under the stated Lipschitz and nondegeneracy hypotheses. The note also records a numerical counterexample to the uncorrected force. — same file.
- §4 equivalence of this operator with Track A Ricci–DeTurck: marked open. Two candidate statements, (E-lim) and (E-flow), are proposed and not proved. — same file.

### Inferences

- Chronological order of the distinct texts: `paper.pdf` (14 Oct, morning) → `paper_v2` (14 Oct, afternoon) → `dtft_lock` (14 Oct, later) → `singularity_surgery` (15 Oct, morning) → architecture PDF (15 Oct, afternoon) → `monodromy_moe` (15 Oct, night) → architecture v2 tex (18 Oct) → `bnz_anti_entropy` (26 Oct) → AnyL markdown (30 Oct) → MRWT PDF (30 Oct) → the spectral monograph and the neckpinch note (31 Oct). The Dynamic Ricci PDF has no reliable date; the neckpinch note treats it as the earlier source [DR].
- Where the same surgery theorem is copied, the 31 October note is the one a referee can check. AnyL is the fullest specification of the weight-manifold algorithm and is honest that eighteen inputs are unset and that no model run was done. MRWT and the spectral PDF repeat the unfinished count.

### Gaps

- Track A (`ricci_deturck_kuhn_T4.tex`) is cited by the neckpinch note and is not in this folder, so Theorems 5.1–5.2 of the spectral PDF and the transplanted steps of Theorem 2.2 cannot be audited from this folder alone.
- The Dynamic Ricci PDF’s internal “2025” date is a line inside a chat paste. The file’s own date is not established.

## The training/merge claim: Hodge–Kuramoto alignment, ternary TIES, DTFT lock, one Ricci–DeTurck step

### Takeaway

The papers specify a merge pipeline whose core, in `paper_v2`, is Hodge–Kuramoto alignment plus ternary TIES. The DTFT lock is a same-day proposal whose changes to `paper_v2` are marked “not applied,” while the 18 October architecture tex says the code’s default lock is Algorithm D. No paper in this folder states, let alone proves, that training or merging consists of one Ricci–DeTurck step. The Ricci–DeTurck material is a separate geometric note about a reduced flow on \(S^1\times S^3\), and that note leaves the identification of the Dynamic Ricci operator with Ricci–DeTurck open.

### Cited Findings

- `paper_v2` fixes the pipeline order: align, resize, mix in full precision, ternarize, distill. Distillation is outside the code. Ternary models on the same base skip alignment and use ternary TIES. The lock uses \(\omega_i=0\), Eq. (14b). TIES is Eq. (21): the denominator is the weight of contributors whose ternary code matches the elected sign. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- What that paper proves about the lock is the Gershgorin stability of one explicit tick, plus a citation that identical frequencies synchronize. What it specifies is Algorithm 1. What it measures is S1–S8 and a tiny synthetic run with \(\langle r\rangle=1\). What it leaves open is any nonzero \(\omega\), and every large-model quality number. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- `dtft_lock` Proposition 1 is the argument that the \(\omega=0\) lock is automatic on a connected graph inside a half circle, so \(r=1\) does not measure agreement. Algorithm D is the proposed replacement. §6 says the edits to paper v2 are not applied. The 2B-scale check P5 is a prediction, not a result. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- Architecture v2, § “Hodge split and DTFT lock”, says the code default `--lock dtft` runs Algorithm D, with band edge \(2-2\cos(2\pi/r)\), Candan/Jacobsen estimates, and a PSR/uniqueness gate. The TIES formula in that tex, Eq. (the agreeing sum over the weight of every contributor), differs from paper v2 Eq. (21): zero-delta contributors dilute the update. The paper says the old routine remains in the source and is unused by the merge, and that no perplexity is recorded for the new rule. — [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)
- Architecture v1 and v2 both say the OV chart, pin-mean, union vocabulary, and limbs have “no equation in” paper v2, the DTFT note, or the surgery note. They are code, described here. — [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)
- “One Ricci–DeTurck step” does not occur in `paper_v2`, `dtft_lock`, the architecture tex, `singularity_surgery`, or `monodromy_moe`. The spectral PDF states semi-discrete convergence theorems for a Ricci–DeTurck equation on \(\mathbb T^4\) and does not prove them in the file. The neckpinch note writes a pre-singularity convergence theorem for the reduced system \(\varphi_t,\psi_t\) on \(S^1\times S^3\), and §4 says a theorem that the Dynamic Ricci operator and Track A’s Lagrange Ricci–DeTurck flow agree is still missing a metric dictionary and a force-consistency argument. — [Spectral Detection PDF](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Spectral%20Detection,%20Dynamic%20Ricci%20Operator%20Flow,%20and%204D%20Topological%20Surgery.pdf); [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)
- AnyL offers the flow slot \(\mathfrak F\) as an input: the Hodge–Kuramoto lock, or Dynamic Ricci leapfrog, or the lock followed by leapfrog. That is a specification with the choice unset. It is not a theorem that one Ricci–DeTurck step trains or merges a model. — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)

### Inferences

- A referee can grant: a specified Algorithm 1; a specified Algorithm D that paper v2 does not adopt and that the 18 October architecture says the code runs; a TIES formula that changes between paper v2 and architecture v2; and a geometric Ricci–DeTurck analysis that the neckpinch note keeps separate from the merge and whose equivalence to the Dynamic Ricci operator is open.
- The phrase “training” in the sense of optimizing a model from data is not a theorem in any of these papers. paper v2’s distillation sentence places that step outside the described code and inside a future measurement protocol.

### Gaps

- No document derives a quality bound (perplexity, KL, or excess loss) for the composition “alignment + TIES + DTFT lock,” with or without a Ricci–DeTurck step.
- Whether the code path named in architecture v2 actually matches Algorithm D line for line was not re-checked against source, because the vendored tree was out of scope. The architecture tex is the statement on the record.

## Surgery and neckpinch: termination, topology, and what stays open

### Takeaway

The termination result that is actually proved, in the 31 October note, is a counting bound: each event adds at least one new channel, unit, or mesh element, so a discrete loop stops after at most \(N+LM\) surgeries on the weight complex, or at most \(N_{\rm el}\) surgeries on the neckpinch mesh. The topology that is shown is a construction: one cut-and-cap on a rotationally symmetric \(S^1\times S^3\) produces a metric the note calls smooth on \(S^4\), and the first discrete surgery is described as that change. Positive isotropic curvature, the Chen–Zhu canonical-neighborhood theorem, and a round-neck classification do not appear in these files. The note leaves the \(S^4\) neckpinch with poles, evolving caps, and the Dynamic Ricci / Ricci–DeTurck equivalence open, and it says the later surgeries in the numerical run are the free region shrinking against frozen Dirichlet data.

### Cited Findings

- Duplicate statements of the same counting idea, weakest first: MRWT Theorem 6.1 (one sentence, product notation); spectral Theorem 7.1 (one-paragraph sketch, product notation, exit claim that no unit outside \(S\) exceeds \(\tau_n\)); AnyL Theorem 7.1 (longer sketch, same product notation, and an honest refusal to use the Dynamic Ricci convergence sketch); the 31 October Theorem 1.1 (role-by-role Lemma 1.2, disjoint-union count, exit predicates matched to E1–E4). Review the 31 October note. — [MRWT_Dynamic_Ricci_Surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/MRWT_Dynamic_Ricci_Surgery.pdf); [Spectral Detection PDF](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Spectral%20Detection,%20Dynamic%20Ricci%20Operator%20Flow,%20and%204D%20Topological%20Surgery.pdf); [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md); [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)
- The 15 October surgery paper has Algorithm S and the sentence that surgery is applied finitely often, by analogy. It has no \(N+LM\) bound. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf)
- Remark 1.4 of the neckpinch note: the exit claim “no unit exceeds \(\tau\)” needs T1 or T3 on units inside the event predicate; as written, units are seen only through T4. Free mode needs a degeneracy event E4 or the curvature is undefined. Vertical mode does not degenerate. These caveats correct the spectral PDF’s exit sentence. — [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)
- Topology written down: monodromy Proposition 2 and AnyL §3.1. A full-wall neck (\(\ell_w=0\)) kills the channel-axis generator (\(\pi_1\cong\mathbb Z^2\) on the torus, or \(\mathbb Z\) for the interval). A partial wall keeps the generator by routing it through earlier layers. That is algebraic topology of the complement of a slab in a product of cycles or intervals. It is not a Ricci-flow singularity model. — [monodromy_moe.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/monodromy_moe.pdf); [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- Neckpinch geometry, note §2: metric \(\varphi(x)^2 dx^2+\psi(x)^2 g_{S^3}\) on \(S^1_L\times S^3\). Proposition 2.1: the DeTurck gauge exists on every \([0,T']\) with \(T'<T_{\rm sing}\), using Hamilton for short-time existence and uniqueness of the symmetric Ricci flow, and a linear parabolic equation for the harmonic-map coordinate. Status: the gauge argument is written; the Ricci-flow existence is cited. — [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)
- The cap is defined so that \(F=\sin\) near the poles and \(F\equiv 1\) to infinite order at the junction, and the note calls the result a smooth metric on \(S^4\). The first numerical surgery ( \(t=0.02328\), \(r=0.0499\), 46 elements pinned) is labeled “\(S^1\times S^3\to S^4\).” Theorem 2.6’s run then does 31 further surgeries and ends with 400/400 elements pinned. The caveat in the same section: frozen caps do not evolve; later events are the free region shrinking against fixed Dirichlet data; “this is not smooth Ricci flow with surgery inside the caps.” Post-surgery flow with poles, and convergence of the post-surgery Dirichlet problem, are marked open. — [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)
- Why \(S^1\times S^3\) and not \(S^4\): no poles, so a periodic mesh inherits Track A’s difference-quotient argument. “The rotationally symmetric \(S^4\) neckpinch needs an analysis at the poles (weighted spaces), which is open.” A round-\(S^4\) background is a symbolic residual check, not a theorem about round necks. — [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)
- Explicit \(T'\)-dependence of the constant in Theorem 2.2 is open (Remark 2.3 gives a heuristic Grönwall factor and says a proof with explicit exponents is open). Fully discrete BDF2 analysis is “not written out.” — [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)
- Searches of the markdown, the tex, and the extracted PDF text find no theorem, lemma, or conjecture named for positive isotropic curvature, for Chen–Zhu, or for round necks. Hamilton’s 1997 paper on four-manifolds with positive isotropic curvature is reference [9] of `singularity_surgery.pdf` and is not used as a proved hypothesis in any argument in that file. AnyL and MRWT list the classical positive-mass theorem, Lemma G, and the \(S^4\) Gaffney transfer as not claimed. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf); [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md); [MRWT_Dynamic_Ricci_Surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/MRWT_Dynamic_Ricci_Surgery.pdf)
- The spectral PDF’s continuum cap \(f(r)=\psi(r)\,r+(1-\psi(r))\,r_0\) is a displayed formula with no theorem attaching it to the discrete pathway set. — [Spectral Detection PDF](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Spectral%20Detection,%20Dynamic%20Ricci%20Operator%20Flow,%20and%204D%20Topological%20Surgery.pdf)
- On the model-merge side, architecture v2 § “Implemented versus specified” says the code’s surgery cap uses a flat ratio \(\rho=20\) (17 channels), does not implement pin-seed, cut-and-fill, or verify, and does not drop oscillators on the pathway. T1 is used for neck-limb entries, with \(n=R\cdot C\), not for residual channels. Status: Algorithm S is not what that paper says the code runs. — [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)

### Inferences

- “Finite termination” here is a pigeonhole on a finite index set. It is independent of curvature pinching, canonical neighborhoods, and extinction.
- The topological change \(S^1\times S^3\to S^4\) is a description of one designed cap, confirmed as the first event of a numerical pinned loop. It is not a classification of Ricci-flow singularities.

### Gaps

- No proof in this folder of a canonical-neighborhood theorem, a PIC pinching estimate, a Chen–Zhu-type compactness theorem, or a statement that neckpinch necks are round.
- The \(S^4\) pole analysis, evolving caps, explicit constants up to the singular time, fully discrete BDF2, and DRO/Ricci–DeTurck equivalence are open in the note that contains the strongest argument.
- Theorem 2.2’s dependence on Track A is unaudited from this folder.

## Measured numbers the papers themselves report

### Takeaway

Perplexity appears only in the two architecture documents, as 81-token smoke tests, and the 18 October tex says none of those perplexities cover neck limbs, the limb lock, or the revised TIES rule. Lock rates that are measurements of the \(\omega=0\) oscillator are in `paper_v2` (24/24 and a tiny run at \(r=1\)). The 147/147 line is in `dtft_lock`, attributed to a 2B4T+Qwen run that `paper_v2` says was not run. Surgery counts that are measurements are in the neckpinch note: 32 surgeries on 400 elements. No paper reports a Dieharder or NIST pass rate.

### Cited Findings

**`paper_v2` operator table (tiny only; “no real or 2B-scale model was run”):**

- S1: \(\lambda_k\) error \(\le 2\times 10^{-14}\) for \(n\in\{12,16,30,60,64\}\).
- S2: \(\|L_1 h\|=5.7\times 10^{-14}\), cross terms \(\le 2\times 10^{-17}\).
- S3: reconstruction \(1.0\times 10^{-15}\), cross terms \(\le 9\times 10^{-18}\) on a \(32\times 30\) array.
- S4: \(\max|QQ^\top-I|\le 1.1\times 10^{-16}\) for \(H\in\{32,40,48,2560\}\); RMSNorm invariance \(1.9\times 10^{-7}\).
- S5: \(|d_0P_0-P_1d_0|=1.1\times 10^{-16}\), \(|R_0P_0-I|=0\), \(|R_1P_1-I|=0\), \(|L_1P_1 h|=0\).
- S6: 24 oscillators, \(\langle r\rangle=0.99993\) at spread \(\pm\pi/4\) and \(0.99965\) at \(\pm 0.95\pi\); 24/24 locked.
- S7: BNZ4 round trip exact; non-bijective permutation rejected.
- S8: phase correlation and Hungarian seed exact on a known roll.
- Tiny run, 3 sources, 21 tensors: all lock passes converged, \(\langle r\rangle=1.0\), locked \(=n_{\rm osc}\); BNZ round trip byte-identical.

No perplexity. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)

**`dtft_lock`:**

- Cited log line, not in its own protocol table: \(\langle r\rangle=1\), locked 147/147, merged 49/49, for a “real 2B4T+Qwen run.”
- Arithmetic: \(C_5\times C_3\) has \(\beta_1=2\) with squares and 16 without; sub-bin 7.3 estimated as 7.30002 at \(N=64\); tick lags 0.031, 0.0625, 0.08 for \(\Delta\omega=1/256\), \(1/128\), \(1/100\).
- §9: “no numbers here are measured” for P1–P4 and P6. — [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)

**`singularity_surgery` §1, ThinkPad reference forward:**

- \(C^\star=\{255,417,550,700,751,865,869,1350,1834\}\), 9 of 2560 channels.
- relu² hidden peak, output/base: \(0.68\times\) (layer 0), \(0.20\times\) (layer 1), \(0.36\times\) (layer 2), then \(2\times\) at layer 3, rising to \(39\times\).
- Attention \(q\cdot k\) logit spread \(0.66\)–\(1.7\times\) base.
- Derived thresholds printed as \(\tau_{2560}=7.67\), \(\tau_{6912}=7.95\), \(\tau_{2560\cdot 6912}=9.89\); T2 threshold 254; pathway share \(9/2560=0.35\%\); absmean pool \(D\cdot F=1.77\times 10^7\); sub-norm bound \(83.1\,g_u\).
- No perplexity, no lock rate, no surgery count from a run. §10 is not run. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf)

**Architecture PDF (15 October) and v2 tex (18 October), smoke tests, 81 teacher-forced tokens:**

- 2B4T base, perplexity 12.0.
- 2B4T+Qwen3 with the then-current defaults (OV chart, surgery, pin-mean, repair, union vocab): 14.4, and the row notes 334-token chat answers.
- Same with seed alignment off: \(1.3\cdot 10^7\).
- Same with `--limbs 2`: 4142. Residual “0.27–0.32 \(\to\approx 0.11\)”.
- Stock 2B4T with every weight in a radix-4 limb: 12.0.
- v2 adds: 2B4T through the family-general forward, 12.0; Llama-3.2-3B-Instruct BF16 reference forward, 7.8, with the parenthetical “Paris” and \(17\times 23=391\); `hodge-2b4t-1` versus 2B4T, 542/542 tensors byte-identical, 0 codes changed.
- v2 cochain diagnostic on 2B4T, one prompt: 15,664 nodes and 760,800 edges per token; conservation \(2\cdot 10^{-6}\); Stokes ratio \(9\cdot 10^{-10}\); 188 conjugate-gradient iterations (about 1.2 s per token); gradient share about 7% against 2.06% for random currents.
- v2 states that no perplexity is recorded for neck limbs, the limb lock, or the new TIES rule. Both versions retract grafted runs and a “Llama without repair” figure.
- v1 open item 4 and v2 open item 6: detection returns 17 channels under a flat \(\rho=20\) rule, against the 9 of the trace. v2 also says T1 on weight-gap entries gives \(\tau\approx 9.6\)–\(9.9\). — [hodgemix_architecture.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture.pdf); [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)

**AnyL formula evaluations (not model quality):**

- \(\tau_N=7.673219\), \(\tau_M=7.952663\), \(\tau_{NM}=9.886624\); \(Q=127\), \(2Q=254\); \(g_u\sqrt{M}=83.138\,g_u\).
- With \(|C^\star|=9\): share \(9/2560=0.3516\%\); vectors \(=1899+3|U^\star|\); gains \(=549+|U^\star|\); entries \(=8{,}481{,}024+7653\,|U^\star|\).
- Toy \(L=1\), \(N=64\), \(M=256\): \(\tau_N=6.531552\), \(\tau_M=6.982523\), \(\tau_{NM}=8.187734\). — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)

**Neckpinch note, numerical sections marked [D-num]:**

- Pinch time \(T_{\rm sing}\approx 0.02388\). \(d(\psi_{\min}^2)/dt\) goes \(-3.74\to -3.79\to -3.83\) as \(\psi_{\min}\) goes \(0.3\to 0.1\to 0.03\), toward the cylinder rate \(-4\).
- Convergence table, finest-pair rates. At \(T'=0.0132\) (\(\psi_{\min}=0.20\)), \(q=1\): rates 2.00, 1.00, 2.00, 1.00 and \(L^\infty\) error \(5.0\times 10^{-7}\) at \(n=3200\); \(q=2\): 2.99, 2.00, 2.98, 2.00 and \(1.3\times 10^{-9}\) at \(n=1600\). At \(T'=0.0212\): \(q=1\) rates 2, 1, 2, 1 and error \(9.7\times 10^{-7}\); \(q=2\) rates 3.00, 2.00, 3.00, 2.01 and error \(1.0\times 10^{-8}\). At \(T'=0.0232\): \(q=1\) error \(3.0\times 10^{-6}\); \(q=2\) error \(5.8\times 10^{-8}\).
- Manufactured solution: \(q=1\) rates 2, 1, 2, 1, and 2 in \(\|U-R_h u\|_{H^1}\); \(q=2\) rates 3.00, 2.00, 3.01, 2.00, 3.00.
- Surgery error at \(\psi_{\min}=0.053\), \(\Lambda=3\): \(q=1\), rates about 2 for the value quantities and 1.00 for the slope; at \(n=3200\), \(\sup|\Delta\hat\psi|=1.7\times 10^{-6}\). \(q=2\), errors reach \(10^{-9}\) by \(n=1600\), slope rate 2.00.
- Loop (\(q=2\), \(n=400\), \(r_s=0.05\), \(\Lambda=3\)): first surgery at \(t=0.02328\), 46 elements, then 31 further surgeries, end at \(t=0.5356\) with 400 of 400 elements pinned, **32 surgeries**, bound \(N_{\rm el}=400\).
- Uncorrected Dynamic Ricci force, 200 random height fields, seed 0: largest cosine of \(\nabla E_K\) with the written force \(+0.922\) ( \(+0.933\) on vertical components); energy increase \(+3.0\times 10^{-3}\) at step \(10^{-3}\).
- Corrected scheme, \(6\times 6\) grid, 16 free heights: \(\max(H^{m+1}-H^m)=-4.1\times 10^{-9}\) over 3000 steps; \(|\nabla E|\) to \(4\times 10^{-4}\); distance to the final state 0.36, 0.18, 0.10, 0.036 at steps 10, 100, 300, 1000.
- Gauge diagnostic in that run: \(\varphi\) at the neck stays between 1.000 and 1.015 up to \(\psi_{\min}=0.03\). — [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)

**Entropy paper and its README:**

- Hodge energies \((1266.3125,\ 7910.7467,\ 96.9408)\) on the source and on the decode.
- Byte entropy 4.52 bit/byte after the linear stages, 7.59 after whitening, on the tiny sample.
- C1: a signed permutation accepted; duplicate, out-of-range, zero, and incomplete cover rejected. C2: exact integer round trip. Wrong key: source not recovered.
- README repeats \((G,C,H)=(1266.312500,\ 7910.746700,\ 96.940800)\) and says the Python encode is byte-identical to the Go chain. No battery pass rate in either file. — [bnz_anti_entropy.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/bnz_anti_entropy.md); [README.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/README.md)

**Monodromy** reports no run of its own. It quotes the architecture smoke-test perplexity \(1.3\cdot 10^7\) and says it is not a test of Proposition 1. It also mentions “41–60\(\times\) the median at one channel” as the detected surgery singularities, without a table. — [monodromy_moe.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/monodromy_moe.pdf)

**Spectral PDF** prints the same three thresholds 7.673219, 7.952663, 9.886624. No perplexity, no lock rate, no surgery count from a run. — [Spectral Detection PDF](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Spectral%20Detection,%20Dynamic%20Ricci%20Operator%20Flow,%20and%204D%20Topological%20Surgery.pdf)

**Dynamic Ricci PDF:** no measured table. The phrase “simulation methodology and results” appears as a list of intended sections inside the chat paste. — [Dynamic Ricci Operator.md.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Dynamic%20Ricci%20Operator.md.pdf)

**`paper.pdf`:** no numbers.

### Inferences

- The only perplexities a referee can attribute to a model merge are 12.0, 14.4, \(1.3\cdot 10^7\), and 4142, all smoke tests on 81 tokens, with overlapping calibration, and all from before the neck-limb / new-TIES configuration that v2 says is unmeasured. Llama 7.8 is a reference forward of an unmixed BF16 model, not a merge.
- 32 is a surgery count for a 400-element geometric mesh, not for a transformer. The model-merge papers report no surgery count from a training or merge run.

### Gaps

- No file reports perplexity, KL, or a lock rate for the composition that architecture v2 says is the current default (DTFT lock, new TIES denominator, neck limbs).
- The 147/147 figure has no protocol, no seed, and no table inside `dtft_lock`. Its conflict with `paper_v2` is recorded in the next section.

## Contradictions between paper_v2, dtft_lock, and the surgery notes

### Takeaway

`paper_v2` says no 2B-scale model was run and treats \(\langle r\rangle=1\) as the expected tiny-run outcome of \(\omega_i=0\), leaving a data-derived frequency open. `dtft_lock` quotes a 2B4T+Qwen lock of 147/147 as the reason that outcome is uninformative, and it says its proposed edits were not applied to paper v2. The surgery notes do not use either lock number. They do contradict the Dynamic Ricci monotone-energy sketch and the spectral monograph’s \(N+LM\) write-up, both of which sit downstream of the same project as paper v2’s merge, and they show that a Ricci–DeTurck step is not a proved piece of that merge.

### Cited Findings

- Scale of the experiments. `paper_v2`: “no real or 2B-scale model was run” and “No large-model numbers are claimed.” `dtft_lock` abstract and §1: the real 2B4T+Qwen run reported \(\langle r\rangle=1\), 147/147 locked, merged 49/49, and “\(r=1\) with \(n/n\) locked carries no information.” Both statements are in documents dated 14 October 2026; the DTFT PDF was created later the same afternoon. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex); [dtft_lock.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/dtft_lock.pdf)
- What \(r=1\) means. `paper_v2` presents the tiny-run \(\langle r\rangle=1.0\), locked \(=n_{\rm osc}\), as the measured behavior, and cites Dörfler–Bullo for why \(\omega_i=0\) synchronizes. It then asks whether a nonzero \(\omega\) improves merges. `dtft_lock` Proposition 1 is the claim that this synchronization is forced, so the statistic does not measure model agreement, and §6 says the replacement of Eq. (14b) and of Eq. (5) is not applied to paper v2. — both files.
- What “harmonic” denotes. `paper_v2` Eq. (5) defines \(\mathcal H\) as the lowest nonzero bin of one digit on each side and calls it the harmonic part, \(\ker L_1\) in the surrounding prose. `dtft_lock` Proposition 2 says that with the squares filled the true Betti number is 4 for the 2B4T grid, and that Eq. (5) is a quasi-harmonic band, not \(\ker L_1\). — both files.
- TIES, as later written. `paper_v2` Eq. (21) normalizes by the agreeing contributors only. Architecture v2 normalizes by every contributor, including the base, and says this differs once there are several sources. `dtft_lock` does not amend TIES. The contradiction is between paper v2 and the later architecture description, not inside the DTFT PDF. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex); [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)
- Large-model traces that paper v2’s protocol says are future work. `singularity_surgery` §1 reports the ThinkPad 9-channel trace and the \(0.68\times\) through \(39\times\) relu² ratios as data. `paper_v2` lists the corresponding quality measurements as not yet claimed. The surgery notes do not repeat the ThinkPad trace. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf); [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- Ricci–DeTurck versus the merge. `paper_v2` and `dtft_lock` contain no Ricci–DeTurck step. The surgery note §3 says the Dynamic Ricci energy claim, as written in the long PDF, is false: the force can have cosine \(+0.922\) with \(\nabla E_K\), and energy increases on the first step. Section 4 says equivalence with Track A Ricci–DeTurck is open. The spectral PDF, which the note cites as the sketch being closed, says the 4D inverse-estimate obstacle is overcome for \(q\ge 2\) and, by supercloseness, for \(q=1\). The note says the 4D \(q=1\) estimate lacked the margin that the 1D argument has, and that the \(S^4\) problem with poles is open. — [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md); [Spectral Detection PDF](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Spectral%20Detection,%20Dynamic%20Ricci%20Operator%20Flow,%20and%204D%20Topological%20Surgery.pdf)
- Termination arithmetic. The spectral sketch and AnyL write a product index set and a sum bound. The note replaces the product by a disjoint union and adds the missing strict-growth lemma. `paper_v2` and `dtft_lock` have neither statement. — [Surgery_termination_and_neckpinch.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Surgery_termination_and_neckpinch.md)
- Which channels are singular. Surgery paper: 9 channels from the trace. Architecture v2: the implemented detector returns 17 channels at \(\rho=20\), and T1 on entries uses a higher threshold. AnyL’s \(|C^\star|=9\) example is “the ThinkPad set,” with \(|U^\star|\) illustrative. These are three different detections, and only the 9-channel list is tied to a named trace. — [singularity_surgery.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/singularity_surgery.pdf); [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex); [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md)
- Dimension of the manifold. AnyL: at most 3. Spectral PDF: up to 4, with the new theorems stated in dimension 4. The neckpinch note treats the 4-torus theory as Track A and the neckpinch as a 1D reduction, and does not identify them. — [AnyL_Dynamic_Ricci_Singularity_Surgery.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/AnyL_Dynamic_Ricci_Singularity_Surgery.md); [Spectral Detection PDF](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/Spectral%20Detection,%20Dynamic%20Ricci%20Operator%20Flow,%20and%204D%20Topological%20Surgery.pdf)

### Inferences

- paper v2 and the DTFT note disagree on a fact of the project (whether a 2B run existed on 14 October) and on a definition (whether Eq. (5) is \(\ker L_1\)). They agree on the mathematics that identical frequencies on a connected graph lock. The DTFT note is explicit that paper v2 was not revised to match it. The architecture tex later says the code followed the DTFT note anyway, and it records no new lock-rate measurement.
- The surgery notes are consistent with paper v2 in one important way: neither makes Ricci–DeTurck part of the proved merge. They are inconsistent with the spectral monograph’s claim to have proved the 4D estimates and the uncorrected termination sketch.

### Gaps

- Nothing in the folder shows the log line that `dtft_lock` quotes. It cannot be confirmed or reconstructed from `paper_v2`.
- The architecture tex was not diffed against the Go sources.

## Is “training from safetensors alone” a theorem?

### Takeaway

It is not a theorem in any of these papers. It is a description, in the architecture documents, of a merge program that reads safetensors checkpoints and writes a ternary model. `paper_v2` specifies conversion and mixing of published checkpoints and puts distillation outside the code, as future measurement.

### Cited Findings

- `paper_v2` abstract and §1: published open-weight checkpoints are converted and mixed into BitNet b1.58 2B4T. The pipeline ends in “distill into 2B4T and its adapters,” and “Distillation is outside the code described here.” The measurement section lists distillation as something “to be measured.” No theorem states that the safetensors file determines a trained model. — [paper_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix-go/hodgemix-go/paper_v2.tex)
- Architecture v2 abstract: “hodgemix merges open-weight safetensors checkpoints into a BitNet b1.58 base and writes a ternary model in the BNZ4 container.” The pipeline figure starts at “load base + sources” and ends at “write ternary safetensors,” then OV chart, surgery cap, repair, BNZ4, eval. That is an implementation description. The same section separates it from what the companion papers specify, and records the unimplemented steps. — [hodgemix_architecture_v2.tex](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture_v2.tex)
- The 15 October architecture PDF uses the same safetensors wording for the 15 October code. — [hodgemix_architecture.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/hodgemix_architecture.pdf)
- `paper.pdf` §4 is the same pipeline in one paragraph, with no file format and no theorem. — [paper.pdf](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/paper.pdf)
- No occurrence of a theorem, proposition, or lemma whose conclusion is that a model can be trained from safetensors alone was found in the extracted texts, the two tex sources, or the markdown notes. The entropy paper uses safetensors only as a bibliographic context for generative models; its proved statement is Proposition 1 about inverting a byte stream. — [bnz_anti_entropy.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/bnz_anti_entropy.md)
- The folder README describes an entropy prototype, not a training theorem. — [README.md](C:/Users/grego/Downloads/Research/03%20Hodge-Kuramoto%20Model%20Merging/README.md)

### Inferences

- “From safetensors alone” describes an input format of the merge tool the architecture paper documents. A theorem would need a stated hypothesis and a proved conclusion about training. None of the headline results in the previous sections has that conclusion.

### Gaps

- The vendored `hodgemix-go` and `ternmix-py` READMEs also talk about safetensors paths. They were excluded by the assignment, and they are program notes rather than theorems in any case. Nothing in the papers fills the gap.

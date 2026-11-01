# Folder 02 results as tensor-training steps

A purely tensor training step, for this note, reads a weight tensor and writes a weight tensor. It is a closed-form map on arrays. It uses no loss, no labels, no token strings, no calibration text, and no backward pass. A stability theorem, a spectral lower bound, and an existence statement are not update rules. Folder 02’s proved statements were checked against that test. The living ledger is the Whitney-folder foundation through §5.6.109 (30 October 2026). The Kuhn draft is dated 31 October 2026 and says its proofs have not been refereed.

## Theorem A (integral Künneth realisation)

### Takeaway

Theorem A classifies integral Künneth \((p,p)\) classes on the digit torus. It does not update a weight tensor. The array it talks about is a cochain, or an integer class in the exterior algebra on \(H^1\), not a real weight matrix.

### Cited Findings

- The ambient object is \(X=(K,W,R,\mathrm{Hol})\) with \(K=T=\prod_{j=1}^k C_{r_j}\), the MRH digit torus. Classes on \(X\) are \(H^\bullet(T)\) together with holonomy data. A wall-intersection cochain \(\omega_I^t\) is the indicator of the \(p\)-cells whose direction mask is \(I\) and whose coordinates on \(I\) equal \(t\). — [foundation §0](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md)
- Theorem A, tagged proved in §1 and in the §6 row “**Proved** (integrally)”: for the Künneth bigrading, \(H^{p,p}_K\) is spanned over \(\mathbb{Z}\) by intersections of \(p\) row walls and \(p\) column walls. Equivalently, by products of first Chern classes of admissible non-flat \(U(1)\) connections of Künneth type \((1,1)\). Equivalently, by top Chern classes of diagonal \(U(p)\) holonomy. — [foundation §1 and §6](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md)
- The proof identifies that basis with the exterior algebra: \(H^\bullet(T;\mathbb{Z})=\Lambda_{\mathbb{Z}}[e_1,\ldots,e_k]\), and the basis of \(H^{p,p}_K\) is \(\{e_I:|I\cap\mathcal{R}|=|I\cap\mathcal{C}|=p\}\). Each such \(e_I\) is already a wall intersection, up to the MRH sign. — [foundation T1](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md)
- Conjecture B is a different sentence, tagged “**Open**” and “equivalent to the Hodge conjecture for \(A_J\)”. — [foundation §1 and §6](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md)

### Inferences

- If someone forced Theorem A to take an array, the input would be a cocycle or a cohomology class already of Künneth type \((p,p)\): an integer vector on the wall-intersection basis, or the Chern cochain of an admissible bi-flat \(U(1)\) connection. The output of the theorem is the statement that this class equals an integer combination of those wall indicators. On the basis the foundation uses, that combination is the class itself.
- A real weight matrix, with no cochain differential and no row/column digit splitting, is not an input the theorem evaluates. There is no displayed map \(W\mapsto W'\).

### Gaps

- The proof text of T1–T4 was not re-derived line by line in this pass. The statement, the spanning argument, and the §6 tag were read. Certificate logs were not re-run.

## Metric invariance, and Ricci–DeTurck as a trainer

### Takeaway

The proved sentence says a flow of the Kähler metric leaves Hodge classes and algebraic classes unchanged. Ricci–DeTurck, in the folder 02 draft, is a convergence theorem for a metric evolving near flat \(T^4\). That theorem does not supply a rule that trains a weight tensor, and it does not say that one Ricci step trains a language model.

### Cited Findings

- Revised Hodge–Ricci note, abstract: “No metric flow can produce algebraic cycles, because both Hodge classes and algebraic classes are independent of the Kähler metric.” The same note, tagged [P]: Hodge classes \(H^{2,2}(X)\cap H^4(X,\mathbb{Q})\) depend only on the complex and rational structures, so every Kähler metric gives the same set; algebraic classes depend only on the complex structure; a flow that moves only the Kähler metric leaves both sets unchanged. A Riemannian Ricci flow that leaves the Kähler cone is described as worse, because the \((p,q)\) splitting relative to the evolving metric is not defined. The note says this concerns a fixed frame when it discusses \(\mathrm{Ric}=3g\), and that the frame gap “concerns a fixed frame metric, not a flow.” — [Hodge_Ricci_Flow_Connection_revised.pdf](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/Hodge_Ricci_Flow_Connection_revised.pdf)
- Revised dual-presentation note, tagged [P]: “No metric flow can produce algebraic cycles, because Hodge classes and algebraic classes are both independent of the Kähler metric.” Existence (R4) is tagged [O] in the same passage. — [Dual_Presentation_Theory_revised.pdf](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/Dual_Presentation_Theory_revised.pdf)
- The Kuhn draft’s equation in harmonic-map gauge relative to \(\delta\) is \(\partial_t g_{ij}=g^{ab}\partial_a\partial_b g_{ij}+\mathcal{Q}_{ij}(g)[\partial g,\partial g]\). The unknown is the metric. The draft does not mention the Hodge conjecture, language models, or safetensors. Its “what is not covered” list includes the pullback from the DeTurck gauge to Ricci flow. — [ricci_deturck_kuhn_T4.tex](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/ricci_deturck_kuhn_T4.tex)
- The revised Hodge–Ricci note says the first open Hodge case is complex dimension 4, and that complex 2-tori, the near-flat \(T^4\) test bed, cannot probe it. The review of the unrevised notes records the same limit: flat \(T^4\) is a fixed point of Ricci flow, and the Hodge conjecture on a complex 2-torus is the Lefschetz \((1,1)\) theorem. — [Hodge_Ricci_Flow_Connection_revised.pdf](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/Hodge_Ricci_Flow_Connection_revised.pdf); [review_Hodge_Ricci_Flow_Connection.md](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/review_Hodge_Ricci_Flow_Connection.md)

### Inferences

- Using Ricci–DeTurck as a trainer of a weight tensor is outside what the proved sentence allows the flow to do. If the tensor is a Kähler metric, the [P] sentence says the flow does not create algebraic cycles. If the tensor is a model weight, it is not the object whose classes the sentence discusses.
- The Kuhn theorem assumes a smooth solution \(g\) is already given on \([0,T]\) and proves that the semi-discrete \(g_h\) stays close to it. Closeness to a prescribed solution is a stability theorem. A stability theorem is not an update rule.
- The draft therefore does not prove that one Ricci step trains a language model. It does not state that claim.

### Gaps

- The revised Yang–Mills PDF’s Stage 0 wording was not re-extracted in this pass. The dual-presentation and Hodge–Ricci PDFs were. Both state the metric-invariance sentence and tag it [P].

## Kuhn-mesh theorem against a weight matrix

### Takeaway

The proved statement is semi-discrete convergence of Lagrange Ricci–DeTurck flow on the Kuhn mesh of flat \(T^4\), for degree \(q\), on a fixed interval \([0,T]\), with small \(h\) when \(q\ge 2\) and with a near-flat hypothesis when \(q=1\). A weight matrix on a product of cycles meets the “product of cycles” shape and meets none of the metric, degree, mesh-size, or near-flat hypotheses. The paper contains the scalar principal part inside the PDE. It does not contain one explicit training step. The fully discrete scheme is a sketched linear solve.

### Cited Findings

- Setting: flat torus \(\mathbb{T}^4=\mathbb{R}^4/\mathbb{Z}^4\); Kuhn triangulation, \(h=1/n\), 24 Kuhn simplices per unit cube, translation invariance and point symmetry; \(V_h\) the continuous \(P_q\) symmetric \(4\times 4\) matrix fields. Hypothesis H: \(T>0\), \(M\ge 1\), \(M^{-1}I\le g\le MI\), and the sum over \(j\le 2\) of the suprema of \(\|\partial_t^j g\|_{W^{q+1,\infty}}\), \(\|\partial_t^j g\|_{H^{q+4}}\), and \(\|\partial_t^j g^{-1}\|_{W^{4,\infty}}\) is at most \(M\). — [ricci_deturck_kuhn_T4.tex, Hypothesis H and the mesh](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/ricci_deturck_kuhn_T4.tex)
- Theorem (Semi-discrete convergence). Assume \(g\) solves the weak reduced Ricci–DeTurck equation on \([0,T]\) and satisfies Hypothesis H. (a) If \(q\ge 2\), there exist \(h_0>0\) and \(C\) such that for \(h\le h_0\), \(g_h\) exists on \([0,T]\) and \(\max(\|g_h-g\|+h\|\nabla(g_h-g)\|)\le C h^{q+1}\), with \(H^1\)-supercloseness \(O(h^{q+1})\) to the weighted Ritz projection and an \(L^\infty\) rate \(O(h^q)\). (b) If \(q=1\), there exist \(\varepsilon_0>0\) and \(C\) depending only on \(T\) and \(M\) such that if \(\varepsilon:=\|g-\delta\|_{X_1}\le\varepsilon_0\), with \(X_1=C^2([0,T];W^{2,\infty}\cap H^5)\), then for every \(h\) the solution exists on \([0,T]\) and the rates hold with \(C\) replaced by \(C\varepsilon\), and \(\max\|\nabla(g_h-g)\|_{L^\infty}\le C\varepsilon\) with no positive power of \(h\). Author’s status line: proofs through gradient supercloseness are complete as written and have not been refereed. \(C\), \(h_0\), and \(\varepsilon_0\) are not numerical. — [ricci_deturck_kuhn_T4.tex, Theorem (Semi-discrete convergence)](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/ricci_deturck_kuhn_T4.tex)
- Scalar principal part, displayed: \(\partial_t g_{ij}=g^{ab}\partial_a\partial_b g_{ij}+\mathcal{Q}_{ij}(g)[\partial g,\partial g]\). The remainder \(\mathcal{Q}\) is quadratic in first derivatives and contains no second derivatives (identity (S), checked on random second jets to \(2\times 10^{-14}\)). The weak form is \((\partial_t g,\phi)+a(g;g,\phi)=(\widetilde{\mathcal{Q}},\phi)\) with \(a(m;u,\phi)=(m^{ab}\partial_b u,\partial_a\phi)\). The semi-discrete scheme is the same identity for \(g_h(t)\in V_h\), an ODE with an invertible mass matrix. — [ricci_deturck_kuhn_T4.tex, equations (RDT), (S), (SD)](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/ricci_deturck_kuhn_T4.tex)
- Fully discrete scheme, tagged “proof sketch only”: linearly implicit BDF\(p\) for \(1\le p\le 5\), \(q\ge 2\), under \(k^p\le c_0 h^2\). Each step is one symmetric positive definite linear solve. The text says the details have not been written to the standard of the Ritz and supercloseness sections. Numerics use BDF2 on a manufactured solution \(g^*=\delta+\varepsilon e^{-t}\varphi\) with forcing \(f=\partial_t g^*-\mathrm{RDT}(g^*)\). Starting values are \(L^2\) projections, which the paper says are not covered by the BDF proposition. At \(\varepsilon=0.15\) the \(q=1\) scheme blows up on the mesh \(n=4\). — [ricci_deturck_kuhn_T4.tex, §BDF and §num](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/ricci_deturck_kuhn_T4.tex)
- Not covered, in the draft’s list: general meshes (then \(q=1\) is open); \(q=1\) data that are \(O(1)\) away from flat; a uniform-in-time result; curvature localisation from a cited domain theorem onto \(T^4\); piecewise-flat Regge metrics; the pullback from the DeTurck gauge to Ricci flow, which loses a derivative and would require \(q\ge 3\). — [ricci_deturck_kuhn_T4.tex, “What is not covered”](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/ricci_deturck_kuhn_T4.tex)
- The code in `train/ricci.go` documents “one Ricci–DeTurck step” as a single explicit step of a scalar principal part, time step \(1/8\), plus the Hardy potential \(c_H\Omega^2\), and a neck cap where the discrete Laplacian meets \(6/\rho^2\). The Laplacian it applies is the 5-point periodic Laplacian on a row–column grid. The Hardy constant in `train/hardy.go` is \(c_H=2-4/\rho^2\), with \(\rho=531/40\). The file header cites `Hardy_lattice_conformal`, and the Ricci header cites `discrete_ricci_flow_M4`. — [train/ricci.go](file:///C:/Users/grego/hodgemix-go/train/ricci.go); [train/hardy.go](file:///C:/Users/grego/hodgemix-go/train/hardy.go)
- Outside folder 02, `discrete_ricci_flow_M4.md` writes the principal part as \(g^{ab}\bar\nabla_a\bar\nabla_b g_{ij}\) plus lower-order terms, and says the principal symbol is \(g^{ab}\xi_a\xi_b\,\mathrm{Id}\). In §8 the surgery scale \(\varrho\) has cut level proportional to \(6/\varrho^2\), “the scalar curvature” of the round cylinder \(S^3(\varrho)\times\mathbb{R}\). That sentence is a curvature value used as a cut level on a 4-manifold, not a threshold on a 2-dimensional weight-grid Laplacian. The file is not in folder 02. — [discrete_ricci_flow_M4.md §2 and §8](file:///C:/Users/grego/Downloads/discrete_ricci_flow_M4.md)
- Folder 02’s own symbol \(c_H\) is the 4-dimensional Hardy constant of Kapitanski–Laptev: \(c_H=((d-2)/2)^2=1\). A byte search of the `.pdf`, `.md`, `.tex`, and `.txt` files in folder 02 found no string `2 - 4` and no string `4/rho`. `Hardy_lattice_conformal.pdf` is outside the three folders. The constant \(c_H=2-4/\rho^2\) is not a theorem stated in folder 02. — [foundation §5.6, discrete Hardy](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md)

### Inferences

- What a weight matrix on a product of cycles does satisfy is the index geometry of the foundation’s digit torus, when the matrix is read as a scalar array on \(C_{\mathrm{rows}}\times C_{\mathrm{cols}}\). That is two cycles, or more if each axis is factored into digits. It is the domain of Theorem A and of the cycle Laplacian. It is not \(V_h\).
- What it does not satisfy, for the Kuhn theorem: the unknown is not a symmetric \(4\times 4\) positive definite matrix field; there is no polynomial degree \(q\); there is no mesh size \(h=1/n\) tending to zero; Hypothesis H asks for a smooth solution already defined on a fixed \([0,T]\); part (b) asks for that solution to be near the flat metric \(\delta\), with a smallness constant that is not evaluated. A general weight matrix is \(O(1)\) away from being a metric at all.
- One explicit step of the form \(g\mapsto g+\Delta t\,(g^{ab}\partial_a\partial_b g)\) is not a displayed scheme in the paper. The analysed scheme is the semi-discrete ODE. The time discretisation is a sketch of a linearly implicit BDF solve, and the numerical runs add a forcing term so that a manufactured \(g^*\) is the solution. The scalar principal part is the second-order piece of the PDE, with variable coefficients \(g^{ab}\) and with the quadratic remainder retained.
- The map in `train/ricci.go` is a closed-form array map, and it is not the Kuhn scheme. Its Hardy potential and its \(6/\rho^2\) cap are not identities of the Kuhn draft. The cylinder scale it cites sits in a file outside folder 02, and there it is the scalar curvature of a round 3-cylinder.
- The convergence theorem remains a stability theorem: \(g_h\) tracks a given \(g\). A stability theorem is not an update rule. Nothing in the draft says that one such step trains a language model.

### Gaps

- The Kuhn proofs were not line-checked. The draft says they have not been refereed.
- No numerical value of \(\varepsilon_0\), \(h_0\), \(C\), or \(c_0\) is printed.
- `Hardy_lattice_conformal.pdf` was not opened. The claim about it is only that folder 02 does not state \(c_H=2-4/\rho^2\).

## What must not be a step

### Takeaway

Conjecture B, Lemma G, the gap of 6, the Hilbert–Smith notes, the energy minimiser, Markman existence, and D4′ each fail the tensor-step test. They are an open existence claim, an unproved inequality, a spectral lower bound, an unresolved conjecture, a minimiser the ledger refuses to assert, an external existence citation, and an equivalence. None of them is a closed-form map from a weight tensor to a weight tensor.

### Cited Findings

- Conjecture B, foundation §1, tagged open in §6: for a polarizable complex structure \(J\), rational \((p,p)_J\) classes are spanned over \(\mathbb{Q}\) by \(\mathrm{ch}_p\) of admissible \(U(n)\) holonomy on refinements. The revised Hodge–Ricci status list tags “existence (Conjecture B)” [O]. The revised Yang–Mills note tags Stage 3 [O]: show \(\inf_A\|F^{0,2}_A\|=0\) with \(\|\Lambda F\|\) bounded. It calls that energy-only Conjecture B. — [foundation §1 and §6](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md); [Hodge_Ricci_Flow_Connection_revised.pdf](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/Hodge_Ricci_Flow_Connection_revised.pdf); [Ricci_Flow_Yang_Mills_revised.pdf](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/Ricci_Flow_Yang_Mills_revised.pdf)
- Lemma G, first lines of the plain note: “Lemma G is not proved.” Copies `(1)` and `(2)` keep that status line and then say the cubewise hypothesis is false at \(\rho=13.275\), so Lemma G does not reduce to three cube constants. The target inequality is a coercivity bound, not a substitution rule for an array. — [LemmaG_matched_sign.md](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/LemmaG_matched_sign.md); [LemmaG_matched_sign (2).md](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/LemmaG_matched_sign%20(2).md)
- Gap of 6. The revised Hodge–Ricci note tags [D], “verified two ways in the Lemma G notes,” the inequality \(2\|d_A^+ a\|^2\ge 6\|a\|^2\) on the Coulomb slice orthogonal to eight moduli, from \(\mathrm{Ric}=3g\) in the conformal \(S^4\) frame, and says it concerns a fixed frame metric, not a flow. The status line lists “Lemma G gap of 6” under [D], not [P]. Lemma G copy `(2)` §7 derives the eigenvalue 6 on unit \(S^4\), tagged derived and not machine-checked. The plain Lemma G file contains no gap of 6. A lower bound on a quadratic form is a stability statement. — [Hodge_Ricci_Flow_Connection_revised.pdf](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/Hodge_Ricci_Flow_Connection_revised.pdf); [LemmaG_matched_sign (2).md](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/LemmaG_matched_sign%20(2).md)
- Hilbert–Smith. The PDF describes a four-step path and says the path does not resolve the Hilbert–Smith conjecture. There is no theorem environment. The bridge note tags the existence goal [O] in the first paragraph, and HS-2 says the \(\mathbb{Z}_p\) action produced by the MRH tower lives on a \(p\)-adic solenoid, so Hilbert–Smith is neither contradicted nor invoked. — [Hilbert_Smith_Multi_Radix_Attack.pdf](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/Hilbert_Smith_Multi_Radix_Attack.pdf); [HilbertSmith_bridge_construction.md](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/HilbertSmith_bridge_construction.md)
- Energy minimiser. From §5.6.40 onward the foundation’s repeated terminal sentence is that \(\mathcal{C}_{\min}=\emptyset\) is not claimed. Sections §5.6.106–109 keep the unconditional (PD) floor at 7.37799 and list F250, piece 4, \(C_{\ge 22}\), \(\kappa\)-LL on the far set, and (O1) as still open. The xAI brief says never to claim \(\mathcal{C}_{\min}=\emptyset\). — [foundation §5.6.109](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md); [xAI brief](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/xAI_supercomputer_brief_2026-10-30.md)
- Markman existence. The secant note tags “[D, given Markman]” the sentence that existence holds on every abelian fourfold in codimension 2, using Markman’s Corollary 1.6.1, which the note calls a preprint, together with foundation items J4, J5, and J2\(^\infty\). It says a lattice-native secant construction is not built. The scaling identity is not the existence proof. The foundation’s Corollary D5 is “proved given Markman’s theorem” and says a lattice-native Markman secant bundle is not built. — [SecantSheaf_route_and_HS4.md](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/SecantSheaf_route_and_HS4.md); [foundation §5.6.7](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md)
- D4′. The foundation says the dictionary makes Conjecture B in the J2\(^\infty\) limit form equivalent to the Hodge conjecture for \(A_J\), and the §6 row tags D4′ proved as that equivalence while the next row tags Conjecture B open. The right-to-left direction applies Koszul–Malgrange and GAGA to a J2\(^\infty\) limit. An earlier D4 write-up contains the sentence “Let \(x\) be a Hodge class. By the Hodge conjecture \(x\) is algebraic.” — [foundation §5.4, §5.6.7, and §6](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md)

### Inferences

- An equivalence with the Hodge conjecture is not a proof of the Hodge conjecture. D4′ stays an equivalence. The §6 row is the citation that keeps Conjecture B open.
- The gap of 6 is a lower bound on a continuum quadratic form on a fixed \(S^4\) frame. A stability theorem is not an update rule. The folder’s own tags do not make it a proved lattice update: the revised note marks it [D], and the Lemma G file that writes the eigenvalue says it is not machine-checked.
- None of these seven objects reads a weight array and writes a weight array.

### Gaps

- Markman’s Corollary 1.6.1 was not re-read from arXiv:2502.03415. The secant note’s summary is reported as that note’s summary.
- The proof text of D4′ was not re-derived. The status sentences and the “by the Hodge conjecture” line in the earlier D4 write-up were read as quoted in the foundation.

## Secant scaling \(S_\star([p]^*U)=p^4 S_\star(U)\)

### Takeaway

The identity multiplies a scalar energy by \(p^4\) when a uniform radix-\(p\) pullback is applied to a lattice connection. It does not multiply tensor entries by \(p^4\). The pullback copies plaquette holonomies onto a finer grid. The note that proves the identity says the identity is not an existence proof.

### Cited Findings

- Setting, cited to foundation §5.6.2: plaquette weight \(w_p=\prod_{\lambda\neq\mu,\nu}h_\lambda/(h_\mu h_\nu)\), and \(S_\star(U)=\sum_p w_p\|\Phi_p\|^2_{\mathrm{HS}}\) with \(\Phi_p=\mathrm{Log}(U_p)/(2\pi i)\). For \(D=\mathrm{diag}(r_0,\ldots,r_{2g-1})\), the fine-grid connection \(D^*U\) has plaquette holonomies equal to those of \(U\) on the image plaquettes, exactly. The identity, tagged “[D, proved here]”: \(S_\star(D^*U)=\sum_{\mu<\nu} r_\mu^2 r_\nu^2\, S_\star^{(\mu\nu)}(U)\). Uniform radix \(p\): \(S_\star([p]^*U)=p^4 S_\star(U)\), in every real dimension \(2g\). Mixed radix: no single power. \(D\) preserves type \((1,1)_J\) only when \(D\) commutes with \(J\). — [SecantSheaf_route_and_HS4.md §1](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/SecantSheaf_route_and_HS4.md)
- A script check on a \(U(2)\) field on \((\mathbb{Z}/3\times\mathbb{Z}/2)^2\): radix \((2,2,2,2)\) gives both sides \(435.8791015194\) and ratio \(16=2^4\); radix \((2,3,1,2)\) gives both sides \(342.9749447699\) and ratio \(12.5897\), with no power law. — [SecantSheaf_route_and_HS4.md §1](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/SecantSheaf_route_and_HS4.md)
- Companions under \([p]\): \(\mathrm{ch}_2\mapsto p^4\mathrm{ch}_2\), and \(|F^{0,2}|\) scales by \(p^2\). The ratio \(S_\star/\mathrm{ch}_2\) is invariant. The existence paragraph is a separate tag, “[D, given Markman]”, and the note says the scaling identity is a consistency tool for refinement, not the existence proof. — [SecantSheaf_route_and_HS4.md §1 and §3](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/SecantSheaf_route_and_HS4.md)

### Inferences

- The pullback \([p]^*\) is a closed-form resampling of a connection: link holonomies on the coarse lattice are copied onto the preimage plaquettes of the fine lattice. The object being resampled is the link array \(U\), not a model weight matrix. The proved formula then returns a number, the energy, equal to \(p^4\) times the coarse energy.
- Writing \(p^4 U\), or \(p^4\) times a weight matrix, is a different map. The identity does not state it. For a mixed radix the note states that no single power exists.
- Because the output of the identity is a scalar, the identity itself is not a weight-tensor update. The resampling it quantifies changes the grid of a connection and is not offered as a training step.

### Gaps

- The foundation’s §5.6.2 definition of the plaquette weight was used as cited by the secant note. That subsection was not re-read in full in this pass.
- The script `hs4.py` was not re-run. The printed ten-decimal agreement is the note’s reported check.

## Bi-flatness and the cycle gap \(2-2\cos(2\pi/r)\)

### Takeaway

Bi-flatness is a curvature constraint that cuts out the Künneth \((1,1)\) summand of a lattice connection. The cycle gap is the first positive eigenvalue of the graph Laplacian on a cycle, and on a product the eigenvalues add. Both split a cochain or a connection on the digit torus. Neither is a rule that replaces a weight grid by an updated weight grid.

### Cited Findings

- Bi-flatness, foundation §5.5, tagged proved as Theorem D1: the \((1,1)_K\) condition is \(\Phi_p=0\) on every pure row plaquette and every pure column plaquette. An admissible \(U(1)\) connection has \(c_1\in H^{1,1}_K\) if and only if it is bi-flat, and every integral class there is \(c_1\) of a bi-flat Landau connection. For \(U(n)\), \(\mathrm{ch}_p\) of a bi-flat field lies in \(H^{p,p}_K\). A remark: on each column slice a bi-flat field is flat, hence gauge-equivalent to commuting holonomies, and simultaneously diagonalising them gives \(n\) points of the dual column torus, “a discrete spectral cover.” Go check reported in the foundation: 729/729 Landau flux vectors, and 81/81 diagonal \(U(2)\) fields. — [foundation §5.5, Theorem D1](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md)
- Cycle gap, foundation T2(d), tagged fully proved: on \(C_r\), the character \(\chi_k\) has \(L_0\)-eigenvalue \(2-2\cos(2\pi k/r)\), positive for \(k\neq 0\). Eigenvalues add over the product. Every S.Harm bin has \(\lambda=(2-2\cos 2\pi/r_i)+(2-2\cos 2\pi/r_j)>0\), so those bins miss \(\ker L_0\). A sweep over 3481 grids reports minimum \(\lambda=0.022661>0\). The proof of positivity is the cosine formula. — [foundation T2](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/DLHC_Whitney_Hol/Discrete_Layered_Hodge_Conjecture_foundation.md)
- Revised Hodge–Ricci note, tagged [P]: on a single cycle of length \(r\), the first positive eigenvalue of the graph Laplacian is \(\lambda_1=2-2\cos(2\pi/r)\); on a product it is the minimum over the factors; “exact on finite cochain data.” Forman curvature of an edge on the product of cycles of length \(r\ge 3\) with unit weights is 0, also tagged [P] in the status list. The same PDF lists the cycle spectral gap under [P] and the Lemma G gap of 6 under [D]. — [Hodge_Ricci_Flow_Connection_revised.pdf](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/Hodge_Ricci_Flow_Connection_revised.pdf)
- The review of the unrevised note records the same cycle formula as checked for the graph Laplacian on 0-cochains, with the product eigenvalue equal to the minimum over factors, and records Forman vanishing for radix at least 3. — [review_Hodge_Ricci_Flow_Connection.md](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/review_Hodge_Ricci_Flow_Connection.md)
- `twistor_dec_rope.tex`, inside folder 02, proves the twisted formula \(\lambda_k(\beta)=2-2\cos(2\pi k/r-\beta)\) for the connection Laplacian on a cycle, and says the eigenvalues add over levels of the digit torus. Its open list includes whether any of this enters a seed or a runtime, and it says the sanity checks support no claim about merging. — [twistor_dec_rope.tex](file:///C:/Users/grego/Downloads/Research/02%20Discrete%20Layered%20Hodge%20Conjecture/twistor_dec_rope.tex)

### Inferences

- On a scalar array indexed by a product of cycles, the graph Laplacian is diagonal in the Fourier basis of those cycles, with eigenvalues summed from \(2-2\cos(2\pi k_j/r_j)\). The kernel is the zero-frequency mode. The first positive value is the minimum of the factor gaps. That is a spectral split of scalar cochains on the grid. It names eigenvalues. It does not write a new array of weights. A spectral theorem is not an update rule.
- Bi-flatness splits a connection, not a weight matrix: pure-row and pure-column plaquettes are the ones whose curvature is set to zero, and the surviving curvature sits on mixed planes. The “spectral cover” in the foundation is the spectrum of the slice holonomies, \(n\) points on the dual torus. That is a different split from the Laplacian eigenvalues of a scalar grid.
- Zeroing the pure plaquettes of a connection, or multiplying Fourier modes of a grid by these eigenvalues, would be further maps. Folder 02 states the dictionary and the eigenvalues. It does not state those maps as training steps.

### Gaps

- The Go counts 729/729 and 81/81 were not re-executed. They are the foundation’s reported checks.
- Forman curvature 0 is a count on the cell complex. This pass used the revised PDF’s [P] tag and the review’s edge count. The Forman 2003 paper was not re-read.

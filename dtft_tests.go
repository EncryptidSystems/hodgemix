package main

// dtft_lock selftests (E1, L2, H1/Prop.2, P6) and the tiny-scale falsifiability protocol (Sec. "Tiny-scale validation":
// P1-P4 on synthetic 64x64 and 60x90 index grids, 10 permutation seeds). Synthetic only; no model files.

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"os"
	"sort"
)

// Prop.2 / (H1): beta1 = |E| - rank d0 - rank d1 on the digit torus prod C_r (C_2 = single edge), with or without squares
func betti1(rad []int, squares bool) int {
	n := 1
	for _, r := range rad {
		n *= r
	}
	strides := make([]int, len(rad))
	for j := range rad {
		strides[j] = 1
		for k := len(rad) - 1; k > j; k-- {
			strides[j] *= rad[k]
		}
	}
	nxt := func(u, j int) (int, bool) {
		r := rad[j]
		d := (u / strides[j]) % r
		if r < 2 || (r == 2 && d == 1) {
			return 0, false
		}
		return u + (((d+1)%r)-d)*strides[j], true
	}
	eid := map[[2]int]int{}
	var d0 [][]float64
	for j := range rad {
		for u := 0; u < n; u++ {
			if v, ok := nxt(u, j); ok {
				eid[[2]int{j, u}] = len(d0)
				row := make([]float64, n)
				row[u] -= 1
				row[v] += 1
				d0 = append(d0, row)
			}
		}
	}
	E := len(d0)
	var d1 [][]float64
	if squares {
		for j := range rad {
			for l := j + 1; l < len(rad); l++ {
				for u := 0; u < n; u++ {
					uj, a := nxt(u, j)
					ul, b := nxt(u, l)
					if !a || !b {
						continue
					}
					row := make([]float64, E)
					row[eid[[2]int{j, u}]] += 1
					row[eid[[2]int{l, uj}]] += 1
					row[eid[[2]int{j, ul}]] -= 1
					row[eid[[2]int{l, u}]] -= 1
					d1 = append(d1, row)
				}
			}
		}
	}
	return E - rankF(d0) - rankF(d1)
}

// exact-on-integers Gaussian rank (partial pivoting, tolerance far below 1 for these 0/+-1 matrices)
func rankF(M [][]float64) int {
	if len(M) == 0 {
		return 0
	}
	A := make([][]float64, len(M))
	for i := range M {
		A[i] = append([]float64(nil), M[i]...)
	}
	r := 0
	for c := 0; c < len(A[0]) && r < len(A); c++ {
		p := r
		for i := r; i < len(A); i++ {
			if math.Abs(A[i][c]) > math.Abs(A[p][c]) {
				p = i
			}
		}
		if math.Abs(A[p][c]) < 1e-9 {
			continue
		}
		A[r], A[p] = A[p], A[r]
		for i := range A {
			if i != r && A[i][c] != 0 {
				f := A[i][c] / A[r][c]
				for k := c; k < len(A[0]); k++ {
					A[i][k] -= f * A[r][k]
				}
			}
		}
		r++
	}
	return r
}

// P6 (N1)/(N2): dense generalized eigenproblem (K - w^2 M) v = 0 on C_a x C_b with squares and per-level stars s1, s2
// (K = star1 d0 d0^T star1 + d1^T d1, M = star1) vs the DFT prediction {0,0} U {s1 l1 + s2 l2} U {l2/s1 + l1/s2}.
func p6Check(a, b int, s1, s2 float64) (float64, int) {
	rad := []int{a, b}
	n := a * b
	ed := torusEdges(rad) // direction 0 edges first (n), then direction 1 (n)
	E := len(ed)
	star := make([]float64, E)
	for e := range ed {
		if e < n {
			star[e] = s1
		} else {
			star[e] = s2
		}
	}
	d0 := make([][]float64, E)
	for e, uv := range ed {
		d0[e] = make([]float64, n)
		d0[e][uv[0]] -= 1
		d0[e][uv[1]] += 1
	}
	var d1 [][]float64
	for u := 0; u < n; u++ {
		i, j := u/b, u%b
		uj := ((i+1)%a)*b + j
		ul := i*b + (j+1)%b
		row := make([]float64, E)
		row[u] += 1    // e0(u)
		row[n+uj] += 1 // e1(u+e0)
		row[ul] -= 1   // e0(u+e1)
		row[n+u] -= 1  // e1(u)
		d1 = append(d1, row)
	}
	// symmetric form M^-1/2 K M^-1/2 = S^1/2 d0 d0^T S^1/2 + S^-1/2 d1^T d1 S^-1/2
	H := make([][]float64, E)
	for x := range H {
		H[x] = make([]float64, E)
		for y := range H[x] {
			var down, up float64
			for v := 0; v < n; v++ {
				down += d0[x][v] * d0[y][v]
			}
			for f := range d1 {
				up += d1[f][x] * d1[f][y]
			}
			H[x][y] = math.Sqrt(star[x]*star[y])*down + up/math.Sqrt(star[x]*star[y])
		}
	}
	ev := jacobiEig(H)
	var pred []float64
	for k1 := 0; k1 < a; k1++ {
		for k2 := 0; k2 < b; k2++ {
			l1 := 2 - 2*math.Cos(2*math.Pi*float64(k1)/float64(a))
			l2 := 2 - 2*math.Cos(2*math.Pi*float64(k2)/float64(b))
			if k1 == 0 && k2 == 0 {
				pred = append(pred, 0, 0)
				continue
			}
			pred = append(pred, s1*l1+s2*l2, l2/s1+l1/s2)
		}
	}
	sort.Float64s(ev)
	sort.Float64s(pred)
	e := 0.0
	nz := 0
	for i := range ev {
		e = math.Max(e, math.Abs(math.Sqrt(math.Max(ev[i], 0))-math.Sqrt(pred[i]))) // omega = sqrt(lambda)
		if math.Abs(ev[i]) < 1e-9 {
			nz++
		}
	}
	return e, nz
}

// dtft_lock selftests
func selftestDTFT() {
	fmt.Println("== dtft_lock selftests ==")
	// (E1) tone at bin 7.3, N = 64, phase 0.4
	N := 64
	s := make([]complex128, N)
	for n := range s {
		s[n] = cmplx.Exp(complex(0, 2*math.Pi*7.3/float64(N)*float64(n)+0.4))
	}
	X := dft(s, nil)
	k := 0
	for i := range X {
		if cmplx.Abs(X[i]) > cmplx.Abs(X[k]) {
			k = i
		}
	}
	est := float64(k) + candan(X, k)
	pk := peakOf(s, false)
	fmt.Printf("E1 sub-bin: true 7.3 -> Candan %.6f (paper sanity 7.30002); Hann PSR=%.1f uniq=%.2f pass=%v; chirp-z zoom %.5f bins; Goertzel phase at 7.3/64 = %.4f rad (rect)\n",
		est, pk.PSR, pk.Uniq, pk.Pass, pk.NuCZT*float64(N), cmplx.Phase(goertzel(s, nil, 7.3/float64(N)))-cmplx.Phase(cmplx.Exp(complex(0, math.Pi*7.3/float64(N)*float64(N-1)))))
	// group delay of a circular shift by 5 samples (rect window)
	sh := make([]complex128, N)
	for n := range sh {
		sh[n] = s[(n-5+N)%N]
	}
	fmt.Printf("group delay: shift 5 samples -> %.4f (rect window)\n", groupDelay(sh, nil, 7.3/float64(N))-groupDelay(s, nil, 7.3/float64(N)))
	// (L2) pair tick: phase-only (L2 derivation) and with g_omega update (R-KTE-1)
	for _, dw := range []float64{1.0 / 256, 1.0 / 128, 1.0 / 100, 1.0 / 64} {
		a := pairLag(dw, 255*8, false)
		b := pairLag(dw, 255*8, true)
		fmt.Printf("L2 pair tick dw=1/%.0f: phase-only lag=%.4f locked=%v | with g_omega update lag=%.4f locked=%v | analytic (L2) %v\n",
			1/dw, a, a <= 1.0/16+1e-12, b, b <= 1.0/16+1e-12, dw <= PairBound+1e-15)
	}
	// Prop.2 beta1
	fmt.Printf("beta1 C5xC3: squares %d (paper 2), no squares %d (paper 16)\n", betti1([]int{5, 3}, true), betti1([]int{5, 3}, false))
	fmt.Printf("beta1 [5 3 2 2] (60): squares %d (D>=3 = %d); [3 2 2 5]? n/a; [5 2] x [3] (30): squares %d (D>=3=%d)\n",
		betti1([]int{5, 3, 2, 2}, true), longDigits([]int{5, 3, 2, 2}), betti1([]int{5, 2, 3}, true), 2)
	// persistence essential classes must equal beta1
	rad := []int{5, 3, 3}
	nk := make([]int, 45)
	g := rand.New(rand.NewSource(1))
	for i := range nk {
		nk[i] = g.Intn(4096)
	}
	pr := persistence(rad, nk, 0)
	fmt.Printf("persistence [5 3 3]: born=%d killed=%d essential=%d (beta1=%d)\n", pr.Born, pr.Killed, pr.Essential, betti1(rad, true))
	// P6
	e, nz := p6Check(5, 3, 0.7, 1.9)
	fmt.Printf("P6 generalized eigenproblem vs DFT (C5xC3, stars 0.7/1.9): max|omega_dense-omega_DFT|=%.2e, zero modes=%d (beta1 2)\n", e, nz)
	// (H2) band edges
	for _, r := range [][]int{{5, 2, 2, 2, 2, 2, 2, 2, 2, 2}, {3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2}} {
		gp, ok := bandEdge(r)
		fmt.Printf("H2 band edge %v: lambda_gap=%.4f defined=%v\n", r, gp, ok)
	}
	// BNZ4 v2 round trip
	cm := &ChanMap{Name: "t", Rout: 4, Cout: 4, Gr: 2, Gc: 2, RadR: []uint8{2}, RadC: []uint8{2},
		C:    []Contrib{{Src: 0, Shift: []uint8{0, 0}, PR: []int32{1, 2}, PC: []int32{2, 1}}},
		ProR: make([]Prolong, 4), ProC: make([]Prolong, 4),
		DT: &DtftChunk{Axis: 1, LGap: 1.38, CPS: 0.9, Modes: []uint32{0}, NLoops: 1, Nu: [][]uint16{{bf16(0.1)}}, Phi: [][]uint16{{123}},
			Flags: [][]uint8{{31}}, Q: [][][]int8{{{-1}}}, P: [][][]uint16{{{bf16(-0.5)}}}}}
	b := cm.Encode()
	m2, err := DecodeChanMap(b)
	ok := err == nil && m2.DT != nil && m2.DT.Phi[0][0] == 123 && m2.DT.Q[0][0][0] == -1 && m2.DT.Flags[0][0] == 31
	fmt.Printf("BNZ4 v2 chunk round-trip: %v (%d B)\n", ok, len(b))
}

// synthetic base index grid: three off-grid 2-D tones + 10% white noise (fixed generator; MRH order is smooth-ish in
// real models, white for random permutations: the protocol contrasts the two)
func synthBase(G [2]int, g *rand.Rand) []float64 {
	X := make([]float64, G[0]*G[1])
	type tone struct{ fr, fc, a, ph float64 }
	ts := []tone{{2.3, 3.7, 1, .3}, {1.4, 6.2, .7, 1.1}, {4.6, 1.3, .5, 2.0}}
	for r := 0; r < G[0]; r++ {
		for c := 0; c < G[1]; c++ {
			v := 0.0
			for _, t := range ts {
				v += t.a * math.Cos(2*math.Pi*(t.fr*float64(r)/float64(G[0])+t.fc*float64(c)/float64(G[1]))+t.ph)
			}
			X[r*G[1]+c] = v + 0.1*g.NormFloat64()
		}
	}
	return X
}

func absMeanF(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += math.Abs(v)
	}
	return s / float64(len(x))
}

// random permutation of rows and columns (P2 "randomly permuted source")
func permuted(X []float64, G [2]int, g *rand.Rand) []float64 {
	pr, pc := g.Perm(G[0]), g.Perm(G[1])
	o := make([]float64, len(X))
	for r := 0; r < G[0]; r++ {
		for c := 0; c < G[1]; c++ {
			o[pr[r]*G[1]+pc[c]] = X[r*G[1]+c]
		}
	}
	return o
}

// ternary noise at matched absmean (P2 alternative)
func ternNoise(X []float64, g *rand.Rand) []float64 {
	a := absMeanF(X)
	o := make([]float64, len(X))
	for i := range o {
		o[i] = float64(g.Intn(3)-1) * a * 1.5
	}
	return o
}

// --dtft-tests [--band-inclusive] [--omega-update off]: P1-P4 on tiny synthetic grids
func dtftFalsify(args []string) {
	sp := SynthParams{K: KParams{GTheta: math.Ldexp(1, -GThetaShift), Weak: math.Ldexp(1, -(GOmegaShift - GThetaShift)), WLock: WLock, NLost: NLost, RTol: RefineTol},
		D: DParams{SeqAxis: "auto", OmegaUpdate: true, Persist: true}}
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--band-inclusive":
			sp.D.BandInclusive = true
		case "--omega-update":
			i++
			sp.D.OmegaUpdate = args[i] == "on"
		case "--selftest":
			selftestDTFT()
			return
		}
	}
	quiet = os.Getenv("HM_VERBOSE") == ""
	logDParams(sp)
	type agg struct {
		osc, psr, l2, wind, lock, merged, modes, nb int
		rm, rp, kkc                                 []float64
	}
	run := func(G [2]int, base, src []float64) *DRec {
		_, rec := synthD([]Entry{{W: 1, Grid: base, IsBase: true}, {W: 1, Grid: src}}, G, sp)
		return rec.D
	}
	sum := func(name string, ds []*DRec) {
		var a agg
		for _, d := range ds {
			a.osc += d.NOsc
			a.psr += d.PSRPass
			a.l2 += d.L2Pass
			a.wind += d.WindOK
			a.lock += d.TickLocked
			a.merged += d.MergedOsc
			a.modes += d.BandModes
			a.nb += d.NonBase
			a.rm = append(a.rm, d.RMeas)
			a.rp = append(a.rp, d.RPred)
			a.kkc = append(a.kkc, d.KoverKc)
		}
		fmt.Printf("%-34s runs=%d band modes/run=%d | PSR-pass %d/%d osc | non-base: L2 %d/%d, winding %d/%d, tick-locked %d/%d, MERGED %d/%d | r meas %s pred %s K/Kc %s\n",
			name, len(ds), a.modes/max(1, len(ds)), a.psr, a.osc, a.l2, a.nb, a.wind, a.nb, a.lock, a.nb, a.merged, a.nb, stat(a.rm), stat(a.rp), stat(a.kkc))
	}
	for _, G := range [][2]int{{64, 64}, {60, 90}} {
		fmt.Printf("\n=== grid %v (radices %v x %v), band %s ===\n", G, radices(G[0]), radices(G[1]), map[bool]string{true: "<= (inclusive)", false: "< (paper H2)"}[sp.D.BandInclusive])
		var p1, p2, p2n, p3 []*DRec
		for seed := int64(0); seed < 10; seed++ {
			g := rand.New(rand.NewSource(seed))
			B := synthBase(G, g)
			p1 = append(p1, run(G, B, append([]float64(nil), B...)))
			p2 = append(p2, run(G, B, permuted(B, G, g)))
			p2n = append(p2n, run(G, B, ternNoise(B, g)))
			// P3: digit-cyclic shift of the coarsest digit of each axis (torus automorphism)
			S := NewSpectrum(G, 0)
			sh := roll(B, S.dims, 0, 1)
			sh = roll(sh, S.dims, len(S.rr), 1)
			p3 = append(p3, run(G, B, sh))
		}
		sum("P1 identical", p1)
		sum("P2 random row/col permutation", p2)
		sum("P2 ternary noise (matched absmean)", p2n)
		sum("P3 coarsest-digit cyclic shift", p3)
		// P1/P3 periods invariance + group delay
		if p3[0].Axis >= 0 && len(p3[0].Periods) == 2 {
			fmt.Printf("  P3 seed0: shifts recovered by seed=%v; periods base %s shifted(after seed) %s; group-delay shift (samples, mode 0)=%s; windings %v vs %v\n",
				"see log", fmtF(p3[0].Periods[0]), fmtF(p3[0].Periods[1]), fmtF(p3[0].Shift), p3[0].Windings[0], p3[0].Windings[1])
			fmt.Printf("  P2 seed0: periods base %s permuted %s windings %v vs %v; nu base %s perm %s\n",
				fmtF(p2[0].Periods[0]), fmtF(p2[0].Periods[1]), p2[0].Windings[0], p2[0].Windings[1], fmtF(p2[0].Nu[0]), fmtF(p2[0].Nu[1]))
		}
		// P3 without the seed: periods on the raw shifted grid vs base (translation invariance of (H3))
		if G[1] == 90 {
			g := rand.New(rand.NewSource(0))
			B := synthBase(G, g)
			S := NewSpectrum(G, 0)
			sh := roll(B, S.dims, 0, 1) // coarsest row digit = cyclic shift by 12 of the row sequence (seq axis = rows)
			for _, X := range [][]float64{B, sh} {
				_ = X
			}
			pa := rawPeriods(B, G, sp)
			pb := rawPeriods(sh, G, sp)
			fmt.Printf("  P3 raw (no seed) periods: base %s, coarsest-row-digit (sequence axis) shifted %s; group delay (samples) of that raw shift on mode 0: %.3f (sequence axis = rows here, coarsest row digit shift = 60/5 = 12 samples)\n", fmtF(pa), fmtF(pb), rawGD(B, sh, G, sp))
			if s0, _, ok := rawSeq(B, G, sp); ok {
				pk := peakOf(s0, true)
				fmt.Printf("  base band sequence (mode 0, seed 0): peak bin %d PSR %.2f uniq %.2f nu %.4f\n", pk.K, pk.PSR, pk.Uniq, pk.Nu)
			}
			sh2 := roll(B, S.dims, len(S.rr), 1)
			fmt.Printf("  P3 raw (no seed) periods: coarsest-col-digit (mode axis) shifted %s; group delay (samples) of raw shift on mode 0: %.3f\n", fmtF(rawPeriods(sh2, G, sp)), rawGD(B, sh2, G, sp))
		}
		// P4 blend
		fmt.Println("  P4 blend M_a = (1-a) base + a rand (rand = ternary noise, matched absmean), 10 seeds:")
		for _, al := range []float64{0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0} {
			var ds []*DRec
			for seed := int64(0); seed < 10; seed++ {
				g := rand.New(rand.NewSource(100 + seed))
				B := synthBase(G, g)
				Rn := ternNoise(B, g)
				M := make([]float64, len(B))
				for i := range M {
					M[i] = (1-al)*B[i] + al*Rn[i]
				}
				ds = append(ds, run(G, B, M))
			}
			sum(fmt.Sprintf("    a=%.1f", al), ds)
		}
	}
}

// helpers for P3 raw (no seed): periods and group delay of the sequence on mode 0 directly from a grid
func rawSeq(X []float64, G [2]int, sp SynthParams) ([]complex128, []int, bool) {
	S := NewSpectrum(G, 0)
	gR, okR := bandEdge(S.rr)
	gC, okC := bandEdge(S.rc)
	if !okR && !okC {
		return nil, nil, false
	}
	A := 0
	switch {
	case okR && (!okC || gR < gC-1e-12):
	case okC && (!okR || gC < gR-1e-12):
		A = 1
	default:
		if longDigits(S.rc) > longDigits(S.rr) {
			A = 1
		}
	}
	radA, radS := S.rr, S.rc
	if A == 1 {
		radA, radS = S.rc, S.rr
	}
	NS, NA := G[1-A], G[A]
	s := make([]complex128, NS)
	line := make([]complex128, NA)
	for n := 0; n < NS; n++ {
		for a := 0; a < NA; a++ {
			if A == 0 {
				line[a] = complex(X[a*G[1]+n], 0)
			} else {
				line[a] = complex(X[n*G[1]+a], 0)
			}
		}
		s[n] = fftn(line, radA, false)[0]
	}
	return s, radS, true
}
func rawPeriods(X []float64, G [2]int, sp SynthParams) []float64 {
	s, radS, ok := rawSeq(X, G, sp)
	if !ok {
		return nil
	}
	p, _ := periodsWindings(analytic(s, true), radS)
	return p
}
func rawGD(B, X []float64, G [2]int, sp SynthParams) float64 {
	a, _, _ := rawSeq(B, G, sp)
	b, _, _ := rawSeq(X, G, sp)
	pk := peakOf(a, true)
	w := hann(len(a))
	return groupDelay(b, w, pk.Nu) - groupDelay(a, w, pk.Nu)
}

func stat(v []float64) string {
	if len(v) == 0 {
		return "-"
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	m := 0.0
	for _, x := range s {
		m += x
	}
	return fmt.Sprintf("%.3g[%.3g..%.3g]", m/float64(len(s)), s[0], s[len(s)-1])
}

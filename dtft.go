package main

// Algorithm D of dtft_lock.tex (selective lock; replaces steps 3-4 of paper_v2 Alg.1).
// Every function cites its dtft_lock equation. Phases and frequencies inside the KTE tick are in TURNS
// (R-KTE-1 lift(x) = x - round(x) in (-1/2, 1/2]); Kuramoto population formulas are evaluated in radians.

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"
	"time"
)

// DParams: tuning items of dtft_lock App.A ("Tuning items") exposed as flags, with the paper's provisional values.
type DParams struct {
	BandInclusive bool   // (H2) is strict "<"; "<=" adds the lowest bins of the coarsest long digit (flag, default strict)
	SeqAxis       string // tuning item "sequence axis when both axes carry long digits": auto|rows|cols
	OmegaUpdate   bool   // Alg.D step 5 "g_omega = 8 frequency update" (R-KTE-1 second line)
	Persist       bool   // Sec.2.5 persistence filtration
	PersistBudget int64  // tuning item "H1 persistence cost": max Z2 column additions (0 = unlimited)
}

// lift: R-KTE-1 lifted phase difference in turns, (-1/2, 1/2]
func lift(x float64) float64 { return x - math.Round(x) }

// (H2): lambda_gap = min_{j: r_j >= 3} (2 - 2 cos(2 pi / r_j)); ok=false if no long digit (band undefined, beta1 = 0)
func bandEdge(rad []int) (float64, bool) {
	g, ok := math.Inf(1), false
	for _, r := range rad {
		if r >= 3 {
			g, ok = math.Min(g, 2-2*math.Cos(2*math.Pi/float64(r))), true
		}
	}
	return g, ok
}

// (H1)/Prop.2(b): D_{>=3} = number of digit levels with r_j >= 3 (beta1 of the square-filled digit torus)
func longDigits(rad []int) int {
	n := 0
	for _, r := range rad {
		if r >= 3 {
			n++
		}
	}
	return n
}

// Sec.2.2 / (H2): band bins on one axis' digit torus: lambda_k < lambda_gap (or <=), one of each conjugate pair kept
func bandBins(rad []int, gap float64, inclusive bool) []int {
	n := 1
	for _, r := range rad {
		n *= r
	}
	var out []int
	for i := 0; i < n; i++ {
		d := digitsOf(i, rad)
		lam := 0.0
		cj := 0
		for j, k := range d {
			lam += 2 - 2*math.Cos(2*math.Pi*float64(k)/float64(rad[j]))
			cj = cj*rad[j] + (rad[j]-k)%rad[j]
		}
		in := lam < gap-1e-12
		if inclusive {
			in = lam <= gap+1e-12
		}
		if in && i <= cj {
			out = append(out, i)
		}
	}
	return out
}

// conjugate flat index on a digit torus (helper for (H2) real-signal symmetry)
func conjIdx(i int, rad []int) int {
	d := digitsOf(i, rad)
	c := 0
	for j, k := range d {
		c = c*rad[j] + (rad[j]-k)%rad[j]
	}
	return c
}

// (D1) coarse peak: Hann window (Sec.4, Harris) periodic; returns window
func hann(N int) []float64 {
	w := make([]float64, N)
	for n := range w {
		w[n] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(n)/float64(N))
	}
	return w
}

// (D1): S(nu) = sum_n w[n] s[n] e^{-j 2 pi nu n} at the N DFT bins (direct DFT; N <= 256 by paper_v2 Eq.6)
func dft(s []complex128, w []float64) []complex128 {
	N := len(s)
	X := make([]complex128, N)
	for k := 0; k < N; k++ {
		var a complex128
		for n, v := range s {
			if w != nil {
				v *= complex(w[n], 0)
			}
			ang := -2 * math.Pi * float64(k*n%N) / float64(N)
			a += v * complex(math.Cos(ang), math.Sin(ang))
		}
		X[k] = a
	}
	return X
}

// Sec.4 Goertzel: S(nu) at one off-grid point, O(N) second-order recurrence (complex input is linear in x)
func goertzel(s []complex128, w []float64, nu float64) complex128 {
	om := 2 * math.Pi * nu
	c := complex(2*math.Cos(om), 0)
	var s1, s2 complex128
	for n, v := range s {
		if w != nil {
			v *= complex(w[n], 0)
		}
		s0 := v + c*s1 - s2
		s2, s1 = s1, s0
	}
	N := len(s)
	y := s1 - cmplx.Exp(complex(0, -om))*s2
	return cmplx.Exp(complex(0, -om*float64(N-1))) * y
}

// Sec.4 chirp-z zoom (Rabiner-Schafer-Rader): S on M points of the arc nu0 + m*dnu (direct evaluation of the CZT contour)
func czt(s []complex128, w []float64, nu0, dnu float64, M int) []complex128 {
	o := make([]complex128, M)
	A := cmplx.Exp(complex(0, -2*math.Pi*nu0))
	W := cmplx.Exp(complex(0, -2*math.Pi*dnu))
	for m := 0; m < M; m++ {
		zm := cmplx.Pow(W, complex(float64(m), 0))
		var a complex128
		p := complex(1, 0)
		step := A * zm
		for n, v := range s {
			if w != nil {
				v *= complex(w[n], 0)
			}
			a += v * p
			p *= step
		}
		o[m] = a
	}
	return o
}

// (E1): Jacobsen/Candan three-sample sub-bin estimator on the rectangular DFT X at peak k
func candan(X []complex128, k int) float64 {
	N := len(X)
	a, b, c := X[(k-1+N)%N], X[k], X[(k+1)%N]
	den := 2*b - a - c
	if den == 0 {
		return 0
	}
	d := real((a - c) / den)
	return math.Tan(math.Pi/float64(N)) / (math.Pi / float64(N)) * d
}

// Peak: result of the (D1)/(E1) front end for one sequence
type Peak struct {
	K         int
	PSR, Uniq float64
	Pass      bool    // PSR >= 10 and uniqueness >= 1.4 (Sec.4 gate)
	Nu        float64 // (E1) cycles/sample
	NuCZT     float64 // chirp-z zoom argmax (cross-check)
}

// (D1)+(E1)+Sec.4 gate: Hann FFT coarse peak, PSR (Bolme: (peak-mean)/std of sidelobes outside the +-2-bin Hann mainlobe),
// uniqueness = PSR(best) / PSR(best sidelobe peak), then the sub-bin estimate on the rectangular DFT.
func peakOf(s []complex128, real1 bool) Peak {
	N := len(s)
	Xh := dft(s, hann(N))
	hi := N
	if real1 {
		hi = N/2 + 1
	}
	k0 := 0
	for k := 1; k < hi; k++ {
		if cmplx.Abs(Xh[k]) > cmplx.Abs(Xh[k0]) {
			k0 = k
		}
	}
	var side []float64
	p2 := 0.0
	for k := 0; k < hi; k++ {
		d := (k - k0 + N) % N
		d = min(d, N-d)
		if d > 2 {
			a := cmplx.Abs(Xh[k])
			side = append(side, a)
			p2 = math.Max(p2, a)
		}
	}
	pk := Peak{K: k0}
	if len(side) >= 2 {
		mu, sd := 0.0, 0.0
		for _, v := range side {
			mu += v
		}
		mu /= float64(len(side))
		for _, v := range side {
			sd += (v - mu) * (v - mu)
		}
		sd = math.Sqrt(sd / float64(len(side)))
		if sd > 0 {
			pk.PSR = (cmplx.Abs(Xh[k0]) - mu) / sd
			if q := (p2 - mu) / sd; q > 0 {
				pk.Uniq = pk.PSR / q
			} else {
				pk.Uniq = math.Inf(1)
			}
		}
	}
	pk.Pass = pk.PSR >= PSRMin && pk.Uniq >= UniqMin
	X := dft(s, nil)
	pk.Nu = (float64(k0) + candan(X, k0)) / float64(N)
	Z := czt(s, nil, float64(k0-1)/float64(N), 2/float64(N*N), N+1)
	m0 := 0
	for m := range Z {
		if cmplx.Abs(Z[m]) > cmplx.Abs(Z[m0]) {
			m0 = m
		}
	}
	pk.NuCZT = float64(k0-1)/float64(N) + float64(m0)*2/float64(N*N)
	return pk
}

// (D1): group delay -d arg S / d nu (samples) at nu, central difference with Goertzel at nu +- 1/(4N)
func groupDelay(s []complex128, w []float64, nu float64) float64 {
	h := 1 / (4 * float64(len(s)))
	a := cmplx.Phase(goertzel(s, w, nu+h))
	b := cmplx.Phase(goertzel(s, w, nu-h))
	return -math.Remainder(a-b, 2*math.Pi) / (2 * math.Pi * 2 * h)
}

// (H3)/(H4) phase field: the analytic extension of a real band sequence (one-sided spectrum); complex sequences as is
func analytic(s []complex128, real1 bool) []complex128 {
	if !real1 {
		return s
	}
	N := len(s)
	X := dft(s, nil)
	for k := range X {
		switch {
		case k == 0 || 2*k == N:
		case 2*k < N:
			X[k] *= 2
		default:
			X[k] = 0
		}
	}
	z := make([]complex128, N)
	for n := range z {
		var a complex128
		for k, v := range X {
			ang := 2 * math.Pi * float64(k*n%N) / float64(N)
			a += v * complex(math.Cos(ang), math.Sin(ang))
		}
		z[n] = a / complex(float64(N), 0)
	}
	return z
}

// (H3)/(H4): for each long digit a of the sequence-axis torus, holonomy of the phase 1-cochain wrap(arg z_v - arg z_u)
// around gamma_a. Winding q_a (H4) = loop through node 0 (an integer); period p_a (H3) = mean over all parallel loops
// (gauge- and translation-invariant).
func periodsWindings(z []complex128, rad []int) (p []float64, q []int) {
	N := len(z)
	for a, r := range rad {
		if r < 3 {
			continue
		}
		stride := 1
		for j := len(rad) - 1; j > a; j-- {
			stride *= rad[j]
		}
		hol := func(n0 int) float64 {
			s := 0.0
			for d := 0; d < r; d++ {
				u := n0 + d*stride
				v := n0 + ((d+1)%r)*stride
				s += math.Remainder(cmplx.Phase(z[v])-cmplx.Phase(z[u]), 2*math.Pi)
			}
			return s / (2 * math.Pi)
		}
		var sum float64
		cnt := 0
		for n0 := 0; n0 < N; n0++ {
			if (n0/stride)%r == 0 {
				sum += hol(n0)
				cnt++
			}
		}
		p = append(p, sum/float64(cnt))
		q = append(q, int(math.Round(hol(0))))
	}
	return
}

// Sec.2.4: per-level stars star1^(j) = mean |d0^(j) X_base|^2, energy-normalized (mean over levels = 1); star0 = star2 = 1
func levelStars(X []float64, dims []int) []float64 {
	n := len(X)
	st := make([]float64, len(dims))
	for j, r := range dims {
		if r < 2 {
			continue
		}
		stride := 1
		for k := len(dims) - 1; k > j; k-- {
			stride *= dims[k]
		}
		s, c := 0.0, 0
		for u := 0; u < n; u++ {
			d := (u / stride) % r
			if r == 2 && d == 1 {
				continue
			}
			v := u + (((d+1)%r)-d)*stride
			s += sq(X[v] - X[u])
			c++
		}
		st[j] = s / float64(max(c, 1))
	}
	m, c := 0.0, 0
	for j, r := range dims {
		if r >= 2 {
			m += st[j]
			c++
		}
	}
	if m > 0 {
		m /= float64(c)
		for j := range st {
			st[j] /= m
		}
	}
	return st
}

// Sec.2.5 persistence on the grid-square complex of the index torus: edges present at filtration 0, each square enters at
// the max 12-bit heap key of its edges (paper_v2 Eq.7, node key = max(row key, col key)). Standard Z2 column reduction
// (Edelsbrunner-Letscher-Zomorodian). Returns death keys of killed H1 classes (birth 0), essential count (= beta1), and
// whether the budget was exhausted.
type PersRes struct {
	Born, Killed, Noise, Persistent, Essential int
	MaxLife                                    float64
	Budget                                     bool
	Secs                                       float64
}

func persistence(dims []int, nodeKey []int, budget int64) PersRes {
	t0 := time.Now()
	n := len(nodeKey)
	strides := make([]int, len(dims))
	for j := range dims {
		strides[j] = 1
		for k := len(dims) - 1; k > j; k-- {
			strides[j] *= dims[k]
		}
	}
	eid := make([]map[int]int32, len(dims))
	var ekey []int
	nE := 0
	nxt := func(u, j int) (int, bool) {
		r := dims[j]
		d := (u / strides[j]) % r
		if r < 2 || (r == 2 && d == 1) {
			return 0, false
		}
		return u + (((d+1)%r)-d)*strides[j], true
	}
	for j := range dims {
		eid[j] = map[int]int32{}
		for u := 0; u < n; u++ {
			if v, ok := nxt(u, j); ok {
				eid[j][u] = int32(nE)
				ekey = append(ekey, max(nodeKey[u], nodeKey[v]))
				nE++
			}
		}
	}
	type sqc struct {
		key int
		e   [4]int32
	}
	var sqs []sqc
	for j := range dims {
		for l := j + 1; l < len(dims); l++ {
			for u := 0; u < n; u++ {
				uj, ok1 := nxt(u, j)
				ul, ok2 := nxt(u, l)
				if !ok1 || !ok2 {
					continue
				}
				e := [4]int32{eid[j][u], eid[l][uj], eid[j][ul], eid[l][u]}
				k := 0
				for _, x := range e {
					k = max(k, ekey[x])
				}
				sqs = append(sqs, sqc{k, e})
			}
		}
	}
	sort.SliceStable(sqs, func(a, b int) bool { return sqs[a].key < sqs[b].key })
	// rank d0 = n - components (union-find on edges)
	par := make([]int, n)
	for i := range par {
		par[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		for par[x] != x {
			par[x] = par[par[x]]
			x = par[x]
		}
		return x
	}
	comp := n
	for j := range dims {
		for u := 0; u < n; u++ {
			if v, ok := nxt(u, j); ok {
				if a, b := find(u), find(v); a != b {
					par[a] = b
					comp--
				}
			}
		}
	}
	res := PersRes{Born: nE - (n - comp)}
	piv := map[int32][]int32{}
	var ops int64
	for _, s := range sqs {
		col := append([]int32(nil), s.e[:]...)
		sort.Slice(col, func(a, b int) bool { return col[a] < col[b] })
		col = xorDedup(col)
		for len(col) > 0 {
			p := col[len(col)-1]
			o, ok := piv[p]
			if !ok {
				break
			}
			col = symDiff(col, o)
			ops++
			if budget > 0 && ops > budget {
				res.Budget = true
				res.Secs = time.Since(t0).Seconds()
				return res
			}
		}
		if len(col) > 0 {
			piv[col[len(col)-1]] = col
			res.Killed++
			life := float64(s.key) / float64(1<<KeyBits)
			if s.key < 1 { // < 1 key LSB: noise, never coupled
				res.Noise++
			} else {
				res.Persistent++
				res.MaxLife = math.Max(res.MaxLife, life)
			}
		}
	}
	res.Essential = res.Born - res.Killed
	res.Secs = time.Since(t0).Seconds()
	return res
}

// Z2 helpers for persistence (sorted int32 sets)
func xorDedup(a []int32) []int32 {
	var o []int32
	for i := 0; i < len(a); {
		j := i
		for j < len(a) && a[j] == a[i] {
			j++
		}
		if (j-i)%2 == 1 {
			o = append(o, a[i])
		}
		i = j
	}
	return o
}
func symDiff(a, b []int32) []int32 {
	o := make([]int32, 0, len(a)+len(b))
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] < b[j]:
			o = append(o, a[i])
			i++
		case a[i] > b[j]:
			o = append(o, b[j])
			j++
		default:
			i++
			j++
		}
	}
	o = append(o, a[i:]...)
	return append(o, b[j:]...)
}

// KTE: R-KTE-1 tick in turns with (L1) frequencies: theta += omega + sum_j Kth_ij lift(theta_j - theta_i),
// omega += sum_j Kom_ij lift(theta_j - theta_i) (if OmegaUpdate). act gates the anneal stage (paper_v2 Eq.18).
func kteTick(theta, omega []float64, Kth []float64, gw float64, act []float64, scale float64, omUpd bool) {
	n := len(theta)
	d := make([]float64, n)
	parFor(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			if act[i] == 0 {
				continue
			}
			s := 0.0
			for j := 0; j < n; j++ {
				if k := Kth[i*n+j]; k != 0 && act[j] != 0 {
					s += k * lift(theta[j]-theta[i])
				}
			}
			d[i] = s
		}
	})
	for i := range theta {
		if act[i] == 0 {
			continue
		}
		theta[i] += omega[i] + scale*d[i]
		if omUpd {
			omega[i] += scale * d[i] * gw // Kom = Kth * 2^-(g_omega - g_theta)
		}
	}
}

// (L2) selftest/sanity: two-node linear-lift tick with offset dw, returns final |lag| (turns)
func pairLag(dw float64, ticks int, omUpd bool) float64 {
	th := []float64{0, 0}
	om := []float64{0, dw}
	K := []float64{0, math.Ldexp(1, -GThetaShift), math.Ldexp(1, -GThetaShift), 0}
	act := []float64{1, 1}
	for t := 0; t < ticks; t++ {
		kteTick(th, om, K, math.Ldexp(1, -(GOmegaShift-GThetaShift)), act, 1, omUpd)
	}
	return math.Abs(lift(th[1] - th[0]))
}

// Ott-Antonsen / Kuramoto population (Sec.5): Delta = IQR/2 (rad/tick), K_c = 2 Delta, r = sqrt(1 - K_c/K)
func popPredict(omTurn []float64, K float64) (delta, kc, r float64) {
	if len(omTurn) < 2 {
		return 0, 0, 1
	}
	v := make([]float64, len(omTurn))
	for i, x := range omTurn {
		v[i] = 2 * math.Pi * x
	}
	q := quantiles(v, []float64{.25, .75})
	delta = (q[1] - q[0]) / 2
	kc = 2 * delta
	if K > kc {
		r = math.Sqrt(1 - kc/K)
	}
	return
}

// DRec: per-tensor record of Algorithm D
type DRec struct {
	TiesLambda   float64     `json:"ties_lambda"` // the sources' share of the weight in the curl-set TIES delta (ternaryTIESAll)
	Axis         int         `json:"mode_axis"`
	LGap         float64     `json:"lambda_gap"`
	Beta1        int         `json:"beta1_squares"`
	BandModes    int         `json:"band_modes"`
	Modes        []int       `json:"-"`
	NOsc         int         `json:"n_osc"`
	PSRPass      int         `json:"psr_pass"`
	L2Pass       int         `json:"l2_pass"`
	WindOK       int         `json:"winding_match"`
	TickLocked   int         `json:"tick_locked"`
	Merged       int         `json:"merged_modes"`
	MergedOsc    int         `json:"merged_osc"`
	NonBase      int         `json:"non_base_osc"`
	RMeas, RPred float64     `json:"-"`
	RMeasJ       float64     `json:"r_measured"`
	RPredJ       float64     `json:"r_predicted"`
	KoverKc      float64     `json:"K_over_Kc"`
	CPS          float64     `json:"cP_over_cS"`
	Stars        []float64   `json:"stars"`
	Pers         PersRes     `json:"persistence"`
	Shift        []float64   `json:"group_delay_shift"`
	Periods      [][]float64 `json:"periods"`
	Windings     [][]int     `json:"windings"`
	Nu           [][]float64 `json:"nu"`
	Phi          [][]float64 `json:"phi"`
	Flags        [][]uint8   `json:"flags"`
	CZTErrBins   float64     `json:"czt_vs_E1_bins"`
	Secs         float64     `json:"secs"`
}

// Algorithm D (dtft_lock Sec.5 box), per target tensor.
func synthD(ents []Entry, G [2]int, sp SynthParams) ([]float64, SynthRec) {
	S := NewSpectrum(G, 0)
	dims := S.dims
	n := G[0] * G[1]
	bi := 0
	for i, e := range ents {
		if e.IsBase {
			bi = i
			break
		}
	}
	rec := SynthRec{Shifts: make([][]int, len(ents)), PermR: make([][]int, len(ents)), PermC: make([][]int, len(ents))}
	t0 := time.Now()
	ref := ents[bi].Grid
	grids := make([][]float64, len(ents))
	// Step 1: seed as paper_v2 (Eq.12-13)
	setInner("seed (phase corr + gap selection)", float64(len(ents)), "")
	for i := range ents {
		if i != bi && SeedAlign {
			g3, sh, pr, pc := placeSource(ents[i].Grid, ref, dims, G)
			grids[i], rec.Shifts[i], rec.PermR[i], rec.PermC[i] = g3, sh, pr, pc
		} else {
			grids[i], rec.Shifts[i] = ents[i].Grid, make([]int, len(dims))
			rec.PermR[i], rec.PermC[i] = identity(G[0]), identity(G[1])
		}
		innerSet(float64(i+1), 0)
	}
	endInner()
	D := &DRec{}
	rec.D = D
	rr, rc := S.rr, S.rc
	gR, okR := bandEdge(rr)
	gC, okC := bandEdge(rc)
	D.Beta1 = longDigits(rr) + longDigits(rc)
	// axis choice: the axis holding the coarsest long cycle (attains lambda_gap); tie -> more long digits -> rows
	A := -1
	switch sp.D.SeqAxis {
	case "rows": // sequence along rows => modes on cols
		A = 1
	case "cols":
		A = 0
	}
	if A < 0 {
		switch {
		case okR && (!okC || gR < gC-1e-12):
			A = 0
		case okC && (!okR || gC < gR-1e-12):
			A = 1
		case okR && okC:
			A = 0
			if longDigits(rc) > longDigits(rr) {
				A = 1
			}
		}
	}
	// Sec.2.4 stars, Sec.3 P/S ratio, energy split on the base (2-D spectrum)
	D.Stars = levelStars(ref, dims)
	FB := fftn(toC(ref), dims, false)
	var Rg, Rgd, Rc, Rcd float64
	band2 := make([]bool, n)
	var bandA []int
	var radA, radS []int
	if A >= 0 {
		radA, radS = rr, rc
		if A == 1 {
			radA, radS = rc, rr
		}
		D.LGap = math.Min(gR, gC)
		bandA = bandBins(radA, D.LGap, sp.D.BandInclusive)
		inA := map[int]bool{}
		for _, b := range bandA {
			inA[b] = true
			inA[conjIdx(b, radA)] = true
		}
		for i := 0; i < n; i++ {
			r, c := i/G[1], i%G[1]
			if (A == 0 && inA[r]) || (A == 1 && inA[c]) {
				band2[i] = true
			}
		}
	}
	D.Axis, D.BandModes, D.Modes = A, len(bandA), bandA
	var eg, ec, eh float64
	for i, v := range FB {
		e := sq(cmplx.Abs(v))
		switch {
		case band2[i]:
			eh += e
		case S.Grad[i]:
			eg += e
			Rg += S.Lam[i] * e
			Rgd += e
		default:
			ec += e
			Rc += S.Lam[i] * e
			Rcd += e
		}
	}
	if Rgd > 0 && Rcd > 0 && Rc > 0 {
		D.CPS = math.Sqrt((Rg / Rgd) / (Rc / Rcd))
	}
	tot := eg + ec + eh
	rec.Energy = [][3]float64{{eg / tot, ec / tot, eh / tot}}
	// Step 2: persistence filtration (heap keys of the base index grid, paper_v2 Eq.7)
	if sp.D.Persist {
		f32 := make([]float32, n)
		for i, v := range ref {
			f32[i] = float32(v)
		}
		kr, kc := heapKey(f32, G[0], G[1], 0), heapKey(f32, G[0], G[1], 1)
		nk := make([]int, n)
		for i := range nk {
			nk[i] = max(kr[i/G[1]], kc[i%G[1]])
		}
		D.Pers = persistence(dims, nk, sp.D.PersistBudget)
	}
	logf("      [D] seed %.2fs; lambda_gap=%.4f (H2) mode axis=%d band modes=%d beta1(squares)=%d; stars=%s c_P/c_S=%.4f; E_g:E_c:E_band=%.3f:%.3f:%.3f",
		time.Since(t0).Seconds(), D.LGap, A, len(bandA), D.Beta1, fmtF(D.Stars), D.CPS, eg/tot, ec/tot, eh/tot)
	if sp.D.Persist {
		p := D.Pers
		logf("      [D] persistence: H1 born=%d killed=%d (noise<1LSB=%d, persistent=%d, max life=%.4f) essential=%d (Prop.2b: %d) budget_exceeded=%v %.2fs",
			p.Born, p.Killed, p.Noise, p.Persistent, p.MaxLife, p.Essential, D.Beta1, p.Budget, p.Secs)
	}
	// non-band synthesis: base + ternary TIES on the curl set (2-D bins that are neither gradient nor band)
	out := append([]float64(nil), ref...)
	var others []int
	for i := range ents {
		if i != bi {
			others = append(others, i)
		}
	}
	ws := make([]float64, len(ents))
	var wsum float64
	for i, e := range ents {
		ws[i] = e.W
		wsum += e.W
	}
	for i := range ws {
		ws[i] /= wsum
	}
	if len(others) > 0 {
		var dl [][]float64
		var wo []float64
		for _, i := range others {
			F := fftn(toC(grids[i]), dims, false)
			for k := range F {
				if band2[k] || S.Grad[k] {
					F[k] = 0
				} else {
					F[k] -= FB[k]
				}
			}
			dl = append(dl, realPart(fftn(F, dims, true)))
			wo = append(wo, ws[i])
		}
		tt := ternaryTIESAll(dl, wo, 1) // ws are normalised over every contributor, the base included
		for k := range out {
			out[k] += tt[k]
		}
		for _, w := range wo {
			D.TiesLambda += w // the sources' share of the weight, for the report
		}
	}
	if A < 0 || len(bandA) == 0 || len(others) == 0 {
		if len(others) > 0 && (A < 0 || len(bandA) == 0) {
			logf("      [D] no band to lock: curl residual kept for the Ricci–DeTurck step")
		} else {
			logf("      [D] no band (no digit with r>=3 on grid %v, Prop.2b beta1=0) or no sources: band stays with base, no lock", G)
		}
		rec.R = []float64{1}
		D.Secs = time.Since(t0).Seconds()
		rec.Passes = append(rec.Passes, PassRec{NModes: 0})
		out = flowGrid(out, ref, G)
		rec.DFT = fftn(toC(out), dims, false)
		return out, rec
	}
	// Step 3: per contributor, A-axis digit DFT -> band coefficient sequences s_{i,m}[n] along the sequence axis
	NS := G[1-A]
	NA := G[A]
	seqOf := func(g []float64) [][]complex128 {
		o := make([][]complex128, len(bandA))
		for m := range o {
			o[m] = make([]complex128, NS)
		}
		line := make([]complex128, NA)
		for s := 0; s < NS; s++ {
			for a := 0; a < NA; a++ {
				if A == 0 {
					line[a] = complex(g[a*G[1]+s], 0)
				} else {
					line[a] = complex(g[s*G[1]+a], 0)
				}
			}
			F := fftn(line, radA, false)
			for m, b := range bandA {
				o[m][s] = F[b]
			}
		}
		return o
	}
	allm := append([]int{bi}, others...)
	seqs := make([][][]complex128, len(ents))
	pk := make([][]Peak, len(ents))
	isReal := make([]bool, len(bandA))
	for m, b := range bandA {
		isReal[m] = conjIdx(b, radA) == b
	}
	setInner("[D] DTFT front end", float64(len(allm)), "")
	for c, i := range allm {
		seqs[i] = seqOf(grids[i])
		pk[i] = make([]Peak, len(bandA))
		for m := range bandA {
			pk[i][m] = peakOf(seqs[i][m], isReal[m])
		}
		innerSet(float64(c+1), 0)
	}
	endInner()
	w := hann(NS)
	nm := len(bandA)
	no := len(allm) * nm
	D.NOsc = no
	theta := make([]float64, no)
	omega := make([]float64, no)
	q := make([][][]int, len(ents))
	D.Shift = make([]float64, len(allm))
	D.Periods, D.Windings = make([][]float64, len(ents)), make([][]int, len(ents))
	D.Nu, D.Phi, D.Flags = make([][]float64, len(ents)), make([][]float64, len(ents)), make([][]uint8, len(ents))
	var czte float64
	for a, i := range allm {
		q[i] = make([][]int, nm)
		var pr []float64
		var nuR, phR []float64
		var fl []uint8
		var wd []int
		for m := range bandA {
			nub := pk[bi][m].Nu
			phi := cmplx.Phase(goertzel(seqs[i][m], w, nub)) / (2 * math.Pi)
			phb := cmplx.Phase(goertzel(seqs[bi][m], w, nub)) / (2 * math.Pi)
			k := a*nm + m
			omega[k] = (pk[i][m].Nu - nub) * float64(NS) / float64(NLost) // (L1), R-KTE-7
			theta[k] = lift(phi - phb)                                    // (L1) theta(0)
			if m == 0 {
				D.Shift[a] = groupDelay(seqs[i][m], w, nub) - groupDelay(seqs[bi][m], w, nub)
			}
			pp, qq := periodsWindings(analytic(seqs[i][m], isReal[m]), radS)
			q[i][m] = qq
			pr = append(pr, pp...)
			wd = append(wd, qq...)
			nuR = append(nuR, pk[i][m].Nu)
			phR = append(phR, phi)
			if pk[i][m].Pass {
				D.PSRPass++
			}
			czte = math.Max(czte, math.Abs(pk[i][m].NuCZT-pk[i][m].Nu)*float64(NS))
			fl = append(fl, 0)
		}
		D.Periods[i], D.Windings[i], D.Nu[i], D.Phi[i], D.Flags[i] = pr, wd, nuR, phR, fl
	}
	D.CZTErrBins = czte
	// Step 4: population
	var omNB []float64
	for k := nm; k < no; k++ {
		omNB = append(omNB, omega[k])
	}
	Kpop := sp.K.GTheta
	_, kc, rpred := popPredict(omega, Kpop)
	D.RPred = rpred
	if kc > 0 {
		D.KoverKc = Kpop / kc
	} else {
		D.KoverKc = math.Inf(1)
	}
	// Step 5: the lock is the 1/16-turn window on the phase already measured.
	// A sheet outside the window is not dragged in by a tick loop.
	clear := make([]bool, no)
	for a := range allm {
		for m := 0; m < nm; m++ {
			k := a*nm + m
			clear[k] = math.Abs(lift(theta[k]-theta[m])) <= 1.0/16
		}
	}
	rbar := 1.0
	if nm > 0 && len(allm) > 0 {
		var rs float64
		for m := 0; m < nm; m++ {
			var z complex128
			for a := range allm {
				z += cmplx.Exp(complex(0, 2*math.Pi*theta[a*nm+m]))
			}
			rs += cmplx.Abs(z) / float64(len(allm))
		}
		rbar = rs / float64(nm)
	}
	tk := 1
	D.RMeas = rbar
	// Step 6: merge decision (i)-(iv) and Eq.(19) merge over S_m
	delta := make([][]complex128, nm)
	for m := 0; m < nm; m++ {
		var members []int
		for a, i := range allm {
			if i == bi {
				continue
			}
			D.NonBase++
			k := a*nm + m
			g1 := pk[i][m].Pass && pk[bi][m].Pass
			g2 := math.Abs((pk[i][m].Nu-pk[bi][m].Nu)*float64(NS)/float64(NLost)) <= PairBound
			g3 := eqInts(q[i][m], q[bi][m])
			g4 := clear[k]
			var f uint8
			if g1 {
				f |= 1
			}
			if g2 {
				f |= 2
				D.L2Pass++
			}
			if g3 {
				f |= 4
				D.WindOK++
			}
			if g4 {
				f |= 8
				D.TickLocked++
			}
			if g1 && g2 && g3 && g4 {
				f |= 16
				members = append(members, i)
				D.MergedOsc++
			}
			D.Flags[i][m] = f
		}
		if len(members) == 0 {
			continue
		}
		var z complex128
		for a := range allm {
			z += cmplx.Exp(complex(0, 2*math.Pi*theta[a*nm+m]))
		}
		rm := cmplx.Abs(z) / float64(len(allm))
		den := ws[bi]
		num := make([]complex128, NS)
		for s := range num {
			num[s] = complex(ws[bi], 0) * seqs[bi][m][s]
		}
		for _, i := range members {
			den += ws[i]
			for s := range num {
				num[s] += complex(ws[i], 0) * seqs[i][m][s]
			}
		}
		delta[m] = make([]complex128, NS)
		for s := range num {
			delta[m][s] = complex(rm, 0) * (num[s]/complex(den, 0) - seqs[bi][m][s])
		}
		D.Merged++
	}
	// Eq.(19) leaves an unlocked band bin with the base. The curl residual stays.
	// flowGrid trains that curl with one Ricci–DeTurck step and writes the
	// assembled gauge and the locked harmonic back. A site whose discrete
	// Laplacian reaches 6/ρ² is capped to the base.
	// Replacing out with ref here made that step a no-op, so an unlocked Llama
	// tensor shipped as the base.
	if D.Merged == 0 {
		logf("      [D] no mode locked: band stays with the base; curl residual kept for the Ricci–DeTurck step")
	}
	// apply band deltas: inverse A-axis digit DFT of the (Hermitian-completed) band changes
	if D.Merged > 0 {
		line := make([]complex128, NA)
		for s := 0; s < NS; s++ {
			for a := range line {
				line[a] = 0
			}
			for m, b := range bandA {
				if delta[m] == nil {
					continue
				}
				line[b] += delta[m][s]
				if cb := conjIdx(b, radA); cb != b {
					line[cb] += cmplx.Conj(delta[m][s])
				}
			}
			x := fftn(line, radA, true)
			for a := 0; a < NA; a++ {
				if A == 0 {
					out[a*G[1]+s] += real(x[a])
				} else {
					out[s*G[1]+a] += real(x[a])
				}
			}
		}
	}
	D.RMeasJ, D.RPredJ = D.RMeas, D.RPred
	out = flowGrid(out, ref, G)
	rec.DFT = fftn(toC(out), dims, false)
	D.Secs = time.Since(t0).Seconds()
	rec.R = []float64{D.RMeas}
	rec.Passes = append(rec.Passes, PassRec{R: D.RMeas, Steps: tk, Converged: true, Locked: D.TickLocked, NOsc: no, NModes: nm, Merged: D.Merged, Secs: D.Secs})
	logf("      [D] lock pass: band modes=%d osc=%d PSR-pass=%d/%d L2-pass=%d winding-match=%d window-locked=%d/%d merged osc=%d modes=%d/%d r measured=%.4f predicted(OA)=%.4f K/K_c=%.3g group-delay shift=%s czt-E1=%.3g bins steps=%d omega_update=%v (%.2fs)",
		nm, no, D.PSRPass, no, D.L2Pass, D.WindOK, D.TickLocked, D.NonBase, D.MergedOsc, D.Merged, nm, D.RMeas, D.RPred, D.KoverKc, fmtF(D.Shift), czte, tk, sp.D.OmegaUpdate, D.Secs)
	return out, rec
}

// helpers
func eqInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func fmtF(v []float64) string {
	s := "["
	for i, x := range v {
		if i > 0 {
			s += " "
		}
		s += fmt.Sprintf("%.3g", x)
	}
	return s + "]"
}

// log of the Alg.D parameters (derived values + tuning items with the paper's provisional values)
func logDParams(sp SynthParams) {
	if sp.Lock == "v2" {
		logf("lock: paper_v2 omega=0 (comparison mode; dtft_lock Prop.1 predicts full lock)")
		return
	}
	logf("dtft_lock Alg.D: band (H2) %s lambda_gap; PSR>=%.0f uniq>=%.1f (App.A); Hann; E1 Candan/Jacobsen; Goertzel phase at nu_base; chirp-z zoom; omega=(dnu N)/%d turn/tick (L1, R-KTE-7); K=2^-%d, K_omega=2^-%d (R-KTE-1), W_lock=1/16, pair bound 1/128 (L2); K_c=2Delta, Delta=IQR/2 (OA)",
		map[bool]string{true: "<=", false: "<"}[sp.D.BandInclusive], PSRMin, UniqMin, NLost, GThetaShift, GOmegaShift)
	logf("dtft_lock TUNING (provisional, paper App.A 'Tuning items'): seq-axis=%s; g Lorentzian (Delta=IQR/2); stars per digit level (no activation stars: needs calibration data, LOBPCG path not used); one oscillator per mode (single peak); persistence=%v budget=%d; winding must match exactly; c_P/c_S per tensor; omega-update=%v",
		sp.D.SeqAxis, sp.D.Persist, sp.D.PersistBudget, sp.D.OmegaUpdate)
}

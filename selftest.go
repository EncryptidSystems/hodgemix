package main

// --selftest: (1) lambda_k formula vs the dense digit-torus Laplacian and mixed-radix DFT diagonalization,
// (2) exact Hodge decomposition on the explicit complex K (d0, d1, Hodge stars; CG solves),
// (3) spectral Hodge split used by the pipeline: reconstruction, orthogonality, gradient = row/col potentials.

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"
)

// Paper: Sec.3 "each digit level circulant": digit-torus edges (selftest)
func torusEdges(rad []int) [][2]int {
	n := 1
	for _, r := range rad {
		n *= r
	}
	var out [][2]int
	for ax, r := range rad {
		if r <= 1 {
			continue
		}
		stride := 1
		for j := len(rad) - 1; j > ax; j-- {
			stride *= rad[j]
		}
		for i := 0; i < n; i++ {
			d := (i / stride) % r
			nd := (d + 1) % r
			out = append(out, [2]int{i, i + (nd-d)*stride})
		}
	}
	return out
}

// Paper: Sec.3 lambda_k check: dense symmetric eigenvalues (selftest)
func jacobiEig(A [][]float64) []float64 {
	n := len(A)
	a := make([][]float64, n)
	for i := range a {
		a[i] = append([]float64(nil), A[i]...)
	}
	for sweep := 0; sweep < 100; sweep++ {
		off := 0.0
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				off += a[i][j] * a[i][j]
			}
		}
		if off == 0 {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				apq := a[p][q]
				if apq == 0 {
					continue
				}
				th := (a[q][q] - a[p][p]) / (2 * apq)
				t := 1 / (math.Abs(th) + math.Sqrt(th*th+1))
				if th < 0 {
					t = -t
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				tau := s / (1 + c)
				a[p][p] -= t * apq
				a[q][q] += t * apq
				a[p][q], a[q][p] = 0, 0
				for k := 0; k < n; k++ {
					if k == p || k == q {
						continue
					}
					akp, akq := a[k][p], a[k][q]
					a[k][p] = akp - s*(akq+tau*akp)
					a[k][q] = akq + s*(akp-tau*akq)
					a[p][k], a[q][k] = a[k][p], a[k][q]
				}
			}
		}
	}
	ev := make([]float64, n)
	for i := range ev {
		ev[i] = a[i][i]
	}
	sort.Float64s(ev)
	return ev
}

// Paper: Sec.3 diagonalization check F L F^-1 (selftest)
func cinv(A [][]complex128) [][]complex128 {
	n := len(A)
	M := make([][]complex128, n)
	for i := range M {
		M[i] = make([]complex128, 2*n)
		copy(M[i], A[i])
		M[i][n+i] = 1
	}
	for c := 0; c < n; c++ {
		p := c
		for r := c + 1; r < n; r++ {
			if cmplx.Abs(M[r][c]) > cmplx.Abs(M[p][c]) {
				p = r
			}
		}
		M[c], M[p] = M[p], M[c]
		iv := 1 / M[c][c]
		for k := range M[c] {
			M[c][k] *= iv
		}
		for r := 0; r < n; r++ {
			if r != c && M[r][c] != 0 {
				f := M[r][c]
				for k := range M[r] {
					M[r][k] -= f * M[c][k]
				}
			}
		}
	}
	out := make([][]complex128, n)
	for i := range out {
		out[i] = M[i][n:]
	}
	return out
}

// Paper: Sec.3 lambda_k = sum_j (2 - 2cos(2 pi k_j / r_j)) vs. dense L and F L F^-1 (selftest)
func verifyLambda(n int) (float64, float64, float64) {
	rad := radices(n)
	e := torusEdges(rad)
	L := make([][]float64, n)
	for i := range L {
		L[i] = make([]float64, n)
	}
	for _, ed := range e { // D^T D with D row: -1 at a, +1 at b
		a, b := ed[0], ed[1]
		L[a][a]++
		L[b][b]++
		L[a][b]--
		L[b][a]--
	}
	ev := jacobiEig(L)
	S := NewSpectrum([2]int{n, 1}, 0)
	lf := append([]float64(nil), S.Lam...)
	lfs := append([]float64(nil), lf...)
	sort.Float64s(lfs)
	e1 := 0.0
	for i := range ev {
		e1 = math.Max(e1, math.Abs(ev[i]-lfs[i]))
	}
	F := make([][]complex128, n)
	for i := 0; i < n; i++ {
		b := make([]complex128, n)
		b[i] = 1
		F[i] = fftn(b, rad, false)
	}
	Fi := cinv(F)
	e2, e3 := 0.0, 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s complex128
			for k := 0; k < n; k++ {
				var t complex128
				for l := 0; l < n; l++ {
					t += complex(L[k][l], 0) * Fi[l][j]
				}
				s += F[i][k] * t
			}
			if i == j {
				e3 = math.Max(e3, math.Abs(real(s)-lf[i]))
			} else {
				e2 = math.Max(e2, cmplx.Abs(s))
			}
		}
	}
	return e1, e2, e3
}

// ---------- explicit complex K (as hodgemix.py Complex) ----------
type sp struct {
	r, c   []int
	v      []float64
	nr, nc int
}

// Paper: no equation (helper: I/O, progress or utility)
func (s *sp) mul(x []float64) []float64 {
	o := make([]float64, s.nr)
	for k := range s.v {
		o[s.r[k]] += s.v[k] * x[s.c[k]]
	}
	return o
}

// Paper: no equation (helper: I/O, progress or utility)
func (s *sp) rmul(x []float64) []float64 {
	o := make([]float64, s.nc)
	for k := range s.v {
		o[s.c[k]] += s.v[k] * x[s.r[k]]
	}
	return o
}

type cplx struct {
	G          [2]int
	nb, nE, nT int
	d0, d1     *sp
	s0, s1, s2 []float64
}

// Paper: Sec.3 Complex K: nodes, channel-map edges, triangles; d0, d1, Hodge stars (selftest)
func newComplex(G [2]int, rb, cb []int) *cplx {
	Gr, Gc := G[0], G[1]
	er := torusEdges(radices(Gr))
	ec := torusEdges(radices(Gc))
	N := Gr + Gc
	var E [][2]int
	for r := 0; r < Gr; r++ {
		for c := 0; c < Gc; c++ {
			E = append(E, [2]int{r, Gr + c})
		}
	}
	K := &cplx{G: G, nb: len(E)}
	E = append(E, er...)
	for _, e := range ec {
		E = append(E, [2]int{e[0] + Gr, e[1] + Gr})
	}
	K.nE = len(E)
	d0 := &sp{nr: len(E), nc: N}
	for i, e := range E {
		d0.r = append(d0.r, i, i)
		d0.c = append(d0.c, e[0], e[1])
		d0.v = append(d0.v, -1, 1)
	}
	bi := func(r, c int) int { return r*Gc + (c - Gr) }
	d1 := &sp{nc: len(E)}
	t := 0
	for k, e := range er {
		a, b := e[0], e[1]
		if rb[a] != rb[b] {
			continue
		}
		ei := K.nb + k
		for c := Gr; c < N; c++ {
			d1.r = append(d1.r, t, t, t)
			d1.c = append(d1.c, ei, bi(b, c), bi(a, c))
			d1.v = append(d1.v, 1, 1, -1)
			t++
		}
	}
	for k, e := range ec {
		a, b := e[0]+Gr, e[1]+Gr
		if cb[a-Gr] != cb[b-Gr] {
			continue
		}
		ei := K.nb + len(er) + k
		for r := 0; r < Gr; r++ {
			d1.r = append(d1.r, t, t, t)
			d1.c = append(d1.c, bi(r, a), ei, bi(r, b))
			d1.v = append(d1.v, 1, 1, -1)
			t++
		}
	}
	d1.nr = t
	K.nT, K.d0, K.d1 = t, d0, d1
	K.s0, K.s1, K.s2 = ones(N), ones(K.nE), ones(t)
	return K
}

// Paper: no equation (helper: I/O, progress or utility)
func ones(n int) []float64 {
	o := make([]float64, n)
	for i := range o {
		o[i] = 1
	}
	return o
}

// Paper: no equation (helper: I/O, progress or utility)
func dot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Paper: Sec.3 Hodge-star (*1) weighted inner product (selftest)
func (K *cplx) ip(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += K.s1[i] * a[i] * b[i]
	}
	return s
}

// Paper: Sec.3 L1 = d0 d0^T + d1^T d1 weighted by Hodge stars (selftest)
func (K *cplx) L1(x []float64) []float64 {
	y := make([]float64, len(x))
	for i := range y {
		y[i] = K.s1[i] * x[i]
	}
	a := K.d0.rmul(y)
	for i := range a {
		a[i] /= K.s0[i]
	}
	o := K.d0.mul(a)
	b := K.d1.mul(x)
	for i := range b {
		b[i] *= K.s2[i]
	}
	c := K.d1.rmul(b)
	for i := range o {
		o[i] += c[i] / K.s1[i]
	}
	return o
}

// Paper: Sec.3 Hodge decomposition solves (conjugate gradient, selftest)
func cg(op func([]float64) []float64, b []float64, tol, nb float64) ([]float64, int) {
	x := make([]float64, len(b))
	r := append([]float64(nil), b...)
	p := append([]float64(nil), r...)
	rs := dot(r, r)
	it := 0
	for ; it < 20000; it++ {
		if math.Sqrt(rs) < tol*nb {
			break
		}
		Ap := op(p)
		a := rs / dot(p, Ap)
		for i := range x {
			x[i] += a * p[i]
			r[i] -= a * Ap[i]
		}
		rn := dot(r, r)
		for i := range p {
			p[i] = r[i] + rn/rs*p[i]
		}
		rs = rn
	}
	return x, it
}

// exact Hodge on K: alpha = argmin ||d0 a - M||_{*1} (normal equations, CG), curl via d1 *1^-1 d1^T CG.
// Paper: Sec.3 M = d0 alpha (+) delta1 beta (+) h exact on K, CG solves (selftest)
func (K *cplx) hodge(M []float64, tol float64) map[string]float64 {
	nM := math.Sqrt(K.ip(M, M)) + 1e-300
	sm := make([]float64, len(M))
	for i := range sm {
		sm[i] = K.s1[i] * M[i]
	}
	rhs := K.d0.rmul(sm)
	alpha, it0 := cg(func(a []float64) []float64 {
		y := K.d0.mul(a)
		for i := range y {
			y[i] *= K.s1[i]
		}
		return K.d0.rmul(y)
	}, rhs, tol, math.Sqrt(dot(rhs, rhs))+1e-300)
	g := K.d0.mul(alpha)
	b := K.d1.mul(M)
	op := func(y []float64) []float64 {
		z := K.d1.rmul(y)
		for i := range z {
			z[i] /= K.s1[i]
		}
		return K.d1.mul(z)
	}
	x, it := cg(op, b, tol, math.Max(math.Sqrt(dot(b, b)), math.Sqrt(dot(M, M)))+1e-300)
	c := K.d1.rmul(x)
	for i := range c {
		c[i] /= K.s1[i]
	}
	h := make([]float64, len(M))
	for i := range h {
		h[i] = M[i] - g[i] - c[i]
	}
	lh := K.L1(h)
	rng := rand.New(rand.NewSource(0))
	v := make([]float64, K.d0.nc)
	for i := range v {
		v[i] = rng.NormFloat64()
	}
	dd := K.d1.mul(K.d0.mul(v))
	md := 0.0
	for _, x := range dd {
		md = math.Max(md, math.Abs(x))
	}
	return map[string]float64{"L1h": math.Sqrt(dot(lh, lh)) / nM, "gh": math.Abs(K.ip(g, h)) / nM / nM,
		"ch": math.Abs(K.ip(c, h)) / nM / nM, "gc": math.Abs(K.ip(g, c)) / nM / nM, "d1d0": md,
		"cg_iters": float64(it), "alpha_cg_iters": float64(it0),
		"E_grad": K.ip(g, g) / nM / nM, "E_curl": K.ip(c, c) / nM / nM, "E_harm": K.ip(h, h) / nM / nM}
}

// Paper: checks for Sec.3 (lambda_k, L1 Hodge decomposition, Kuramoto lock), Sec.3.1 Seed, Contrib.1 ternary TIES.
func selftestExtra() {
	// Contrib.1 ternary TIES: codes {+,+,-} with weights -> elected sign +, disjoint mean of agreeing values
	t := ternaryTIES([][]float64{{1, -1}, {1, -1}, {-1, 1}}, []float64{.5, .3, .2}, 1e-5)
	fmt.Printf("ternary TIES: got %v want [1 -1]\n", t)
	// Sec.3.1 Seed: phase correlation recovers a known digit roll; assignment recovers a known permutation
	G := [2]int{12, 8}
	S := NewSpectrum(G, 10)
	g := rand.New(rand.NewSource(3))
	ref := make([]float64, 96)
	for i := range ref {
		ref[i] = g.NormFloat64()
	}
	A := roll(roll(ref, S.dims, 2, 1), S.dims, 3, 1)
	back, sh := seedShift(A, ref, S.dims, 1e-12)
	e := 0.0
	for i := range back {
		e = math.Max(e, math.Abs(back[i]-ref[i]))
	}
	fmt.Printf("seed phase-correlation: shifts=%v max|aligned-ref|=%.1e\n", sh, e)
	perm := g.Perm(12)
	P := make([]float64, 96)
	for i := 0; i < 12; i++ {
		copy(P[perm[i]*8:perm[i]*8+8], ref[i*8:i*8+8])
	}
	Q, _, _ := seedAssign(P, ref, G)
	e = 0
	for i := range Q {
		e = math.Max(e, math.Abs(Q[i]-ref[i]))
	}
	fmt.Printf("seed assignment (Hungarian): max|aligned-ref|=%.1e\n", e)
	// Sec.4 absmean ternary pack: 4 trits/U8 round trip
	q := []int8{-1, 0, 1, 1, 0, -1, 1, -1}
	b := packTrits(q, 4, 2)
	ok := true
	for s := 0; s < 4; s++ {
		for i, v := range b {
			if int8((v>>(2*s))&3)-1 != q[s*2+i] {
				ok = false
			}
		}
	}
	fmt.Printf("pack 4 trits/U8 round-trip: %v\n", ok)
}

// Paper: numeric checks of Sec.3 lambda_k, Hodge decomposition (selftest)
func selftest() {
	for _, n := range []int{12, 16, 30, 60, 64} {
		a, b, c := verifyLambda(n)
		fmt.Printf("lambda %d %v max|eig-formula|=%.3e offdiag(F L F^-1)=%.3e |diag-formula|=%.3e\n", n, radices(n), a, b, c)
	}
	g := rand.New(rand.NewSource(0))
	G := [2]int{12, 8}
	rb := make([]int, 12)
	cb := make([]int, 8)
	for i := range rb {
		rb[i] = g.Intn(3)
	}
	for i := range cb {
		cb[i] = g.Intn(2)
	}
	for _, name := range []string{"unit stars", "random stars"} {
		K := newComplex(G, rb, cb)
		if name == "random stars" {
			for i := range K.s0 {
				K.s0[i] = .5 + 1.5*g.Float64()
			}
			for i := range K.s1 {
				K.s1[i] = .5 + 1.5*g.Float64()
			}
			for i := range K.s2 {
				K.s2[i] = .5 + 1.5*g.Float64()
			}
		}
		M := make([]float64, K.nE)
		for i := 0; i < K.nb; i++ {
			M[i] = g.NormFloat64()
		}
		r := K.hodge(M, 1e-14)
		fmt.Printf("hodge(complex K) %s: L1h=%.3e gh=%.3e ch=%.3e gc=%.3e d1d0=%.0e cg_iters=%.0f energy=[%.4f %.4f %.4f]\n",
			name, r["L1h"], r["gh"], r["ch"], r["gc"], r["d1d0"], r["cg_iters"], r["E_grad"], r["E_curl"], r["E_harm"])
	}
	// spectral split used by the pipeline
	for _, G := range [][2]int{{12, 8}, {32, 30}} {
		S := NewSpectrum(G, 10.0)
		x := make([]float64, G[0]*G[1])
		for i := range x {
			x[i] = g.NormFloat64()
		}
		p := S.Hodge(x)
		gg := realPart(fftn(p.GradF, S.dims, true))
		cc := realPart(fftn(p.CurlF, S.dims, true))
		hh := realPart(fftn(p.HarmF, S.dims, true))
		nx := dot(x, x)
		// gradient part must be exactly row potential + col potential: g[r,c] = a_r + b_c - mean
		var pot float64
		mu := 0.0
		for _, v := range gg {
			mu += v
		}
		mu /= float64(len(gg))
		ar := make([]float64, G[0])
		bc := make([]float64, G[1])
		for r := 0; r < G[0]; r++ {
			for c := 0; c < G[1]; c++ {
				ar[r] += gg[r*G[1]+c] / float64(G[1])
				bc[c] += gg[r*G[1]+c] / float64(G[0])
			}
		}
		for r := 0; r < G[0]; r++ {
			for c := 0; c < G[1]; c++ {
				pot = math.Max(pot, math.Abs(gg[r*G[1]+c]-(ar[r]+bc[c]-mu)))
			}
		}
		rec := 0.0
		for i := range x {
			rec = math.Max(rec, math.Abs(x[i]-gg[i]-cc[i]-hh[i]))
		}
		fmt.Printf("spectral hodge %v (low bins Eq.5): recon=%.2e <g,c>=%.2e <g,h>=%.2e <c,h>=%.2e grad-is-potential=%.2e energy=[%.4f %.4f %.4f]\n",
			G, rec, math.Abs(dot(gg, cc))/nx, math.Abs(dot(gg, hh))/nx, math.Abs(dot(cc, hh))/nx, pot, p.Energy[0], p.Energy[1], p.Energy[2])
	}
	selftestExtra()
	selftestV2()
	selftestDTFT()
}

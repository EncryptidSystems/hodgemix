package main

// Math of paper Sec. "Hodge--Kuramoto Alignment and Mixing", computed in the mixed-radix DFT domain
// where the two-way Laplacian is diagonal: lambda_k = sum_j (2 - 2 cos(2 pi k_j / r_j)).

import (
	"math"
	"math/cmplx"
	"runtime"
	"sort"
	"sync"
)

// Paper_v2 Eq.(6): mixed-radix factorisation of a channel length, coarsest (largest prime) digit first,
// e.g. 2560 = 5 x 2^9 -> [5 2 2 2 2 2 2 2 2 2]
func radices(n int) []int {
	var out []int
	m := n
	for p := 2; p*p <= m; p++ {
		for m%p == 0 {
			out = append(out, p)
			m /= p
		}
	}
	if m > 1 {
		out = append(out, m)
	}
	if len(out) == 0 {
		return []int{1}
	}
	sort.Sort(sort.Reverse(sort.IntSlice(out)))
	return out
}

// Paper_v2 Eq.(6): index length = product of the coarsest digits of n's factorisation not exceeding
// 2^(2*BucketBits) = 256 (two MRH levels of 16 buckets); resizing keeps/drops whole radix levels.
func gridSize(n, _ int) int {
	g := 1
	for _, r := range radices(n) {
		if g*r > 1<<(2*BucketBits) {
			break
		}
		g *= r
	}
	return max(g, 1)
}

// Paper: no equation (helper: I/O, progress or utility)
func ipow(b, e int) int {
	r := 1
	for ; e > 0; e-- {
		r *= b
	}
	return r
}

// Paper: no equation (helper: I/O, progress or utility)
func parFor(n int, f func(lo, hi int)) {
	w := runtime.NumCPU()
	if n < 64 || w == 1 {
		f(0, n)
		return
	}
	var wg sync.WaitGroup
	ch := (n + w - 1) / w
	for lo := 0; lo < n; lo += ch {
		hi := min(n, lo+ch)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); f(lo, hi) }(lo, hi)
	}
	wg.Wait()
}

// fftn: N-dimensional DFT over the digit axes dims (row-major), numpy fftn convention
// (forward e^{-2 pi i jk/r}, inverse e^{+...}/N). Each digit axis is a length-r_j DFT.
// Paper: Sec.3 "the mixed-radix DFT diagonalizes L" (digit-wise DFT, each digit level circulant)
func fftn(x []complex128, dims []int, inverse bool) []complex128 {
	a := append([]complex128(nil), x...)
	n := len(a)
	stride := n
	for _, r := range dims {
		stride /= r
		if r == 1 {
			continue
		}
		tw := make([]complex128, r)
		sg := -1.0
		if inverse {
			sg = 1.0
		}
		for k := 0; k < r; k++ {
			ang := sg * 2 * math.Pi * float64(k) / float64(r)
			tw[k] = complex(math.Cos(ang), math.Sin(ang))
		}
		block := r * stride
		nl := n / r
		parFor(nl, func(lo, hi int) {
			buf := make([]complex128, r)
			for l := lo; l < hi; l++ {
				o, in := l/stride, l%stride
				base := o*block + in
				for k := 0; k < r; k++ {
					var s complex128
					for j := 0; j < r; j++ {
						s += a[base+j*stride] * tw[(j*k)%r]
					}
					buf[k] = s
				}
				for k := 0; k < r; k++ {
					a[base+k*stride] = buf[k]
				}
			}
		})
	}
	if inverse {
		inv := complex(1/float64(n), 0)
		for i := range a {
			a[i] *= inv
		}
	}
	return a
}

// Paper: no equation (helper: I/O, progress or utility)
func toC(v []float64) []complex128 {
	o := make([]complex128, len(v))
	for i, x := range v {
		o[i] = complex(x, 0)
	}
	return o
}

// Paper: no equation (helper: I/O, progress or utility)
func realPart(v []complex128) []float64 {
	o := make([]float64, len(v))
	for i, x := range v {
		o[i] = real(x)
	}
	return o
}

// digits of flat index over dims
// Paper: no equation (helper: I/O, progress or utility)
func digitsOf(i int, dims []int) []int {
	d := make([]int, len(dims))
	for j := len(dims) - 1; j >= 0; j-- {
		d[j] = i % dims[j]
		i /= dims[j]
	}
	return d
}

// Spectrum describes the bins of the two-way Laplacian on grid G = Gr x Gc (digits rr ++ rc).
type Spectrum struct {
	G      [2]int
	rr, rc []int
	dims   []int
	Lam    []float64 // lambda_k (rows/down + cols/up)
	Grad   []bool    // im d0: all row digits 0 or all col digits 0 (row/col potentials)
	Harm   []bool    // non-gradient bins with lambda_k <= harmonic-lambda
	Level  []int     // finest digit position with k_j != 0 (coarse -> fine anneal)
	NLevel int
}

// Paper_v2 Eq.(4)-(5): lambda_k of the two-way (down x up) Laplacian; gradient bins (row- or col-digits all 0);
// harmonic/low bins = exactly one nonzero digit per axis with k_j in {1, r_j - 1} (lowest bin of each digit).
func NewSpectrum(G [2]int, _ float64) *Spectrum {
	s := &Spectrum{G: G, rr: radices(G[0]), rc: radices(G[1])}
	s.dims = append(append([]int{}, s.rr...), s.rc...)
	n := G[0] * G[1]
	s.Lam = make([]float64, n)
	s.Grad = make([]bool, n)
	s.Harm = make([]bool, n)
	s.Level = make([]int, n)
	s.NLevel = max(len(s.rr), len(s.rc))
	nr := len(s.rr)
	for i := 0; i < n; i++ {
		d := digitsOf(i, s.dims)
		rz, cz := true, true
		nzr, nzc := 0, 0
		low := true
		lv := 0
		for j, k := range d {
			r := s.dims[j]
			s.Lam[i] += 2 - 2*math.Cos(2*math.Pi*float64(k)/float64(r))
			if k != 0 {
				if k != 1 && k != r-1 {
					low = false
				}
				if j < nr {
					rz = false
					nzr++
					lv = max(lv, j)
				} else {
					cz = false
					nzc++
					lv = max(lv, j-nr)
				}
			}
		}
		s.Grad[i] = rz || cz
		s.Harm[i] = !s.Grad[i] && low && nzr == 1 && nzc == 1
		s.Level[i] = lv
	}
	return s
}

// HodgeParts: spectral Hodge decomposition of the grid cochain. Returns spectra of the three parts.
type HodgeParts struct {
	F                   []complex128 // full spectrum
	GradF, CurlF, HarmF []complex128
	Energy              [3]float64 // fraction grad/curl/harm (Parseval)
	RecErr              float64
}

// Paper: Sec.3 Hodge decomposition M = d0 alpha (+) delta1 beta (+) h, computed in the DFT domain
func (s *Spectrum) Hodge(grid []float64) HodgeParts {
	F := fftn(toC(grid), s.dims, false)
	n := len(F)
	p := HodgeParts{F: F, GradF: make([]complex128, n), CurlF: make([]complex128, n), HarmF: make([]complex128, n)}
	var tot float64
	for i, v := range F {
		e := real(v)*real(v) + imag(v)*imag(v)
		tot += e
		switch {
		case s.Grad[i]:
			p.GradF[i] = v
			p.Energy[0] += e
		case s.Harm[i]:
			p.HarmF[i] = v
			p.Energy[2] += e
		default:
			p.CurlF[i] = v
			p.Energy[1] += e
		}
	}
	if tot > 0 {
		for k := range p.Energy {
			p.Energy[k] /= tot
		}
	}
	var er float64
	for i := range F {
		er = math.Max(er, cmplx.Abs(F[i]-p.GradF[i]-p.CurlF[i]-p.HarmF[i]))
	}
	p.RecErr = er
	return p
}

// ---------------- index helpers (numpy semantics) ----------------
// Paper: no equation (helper: I/O, progress or utility)
func interp(x float64, xp, fp []float64) float64 {
	n := len(xp)
	if x <= xp[0] {
		return fp[0]
	}
	if x >= xp[n-1] {
		return fp[n-1]
	}
	j := sort.SearchFloat64s(xp, x) // first xp[j] >= x
	if xp[j] == x {
		return fp[j]
	}
	i := j - 1
	t := (x - xp[i]) / (xp[j] - xp[i])
	return fp[i] + t*(fp[j]-fp[i])
}

// Paper: no equation (helper: I/O, progress or utility)
func linspace(a, b float64, n int) []float64 {
	o := make([]float64, n)
	for i := range o {
		if n == 1 {
			o[i] = a
		} else {
			o[i] = a + (b-a)*float64(i)/float64(n-1)
		}
	}
	return o
}

// quantile: numpy linear method on sorted copy.
// Paper: no equation (helper: I/O, progress or utility)
func quantiles(v []float64, q []float64) []float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	o := make([]float64, len(q))
	for i, qq := range q {
		pos := qq * float64(len(s)-1)
		lo := int(math.Floor(pos))
		hi := min(lo+1, len(s)-1)
		o[i] = s[lo] + (pos-float64(lo))*(s[hi]-s[lo])
	}
	return o
}

// Paper: no equation (helper: I/O, progress or utility)
func median(v []float64) float64 { return quantiles(v, []float64{.5})[0] }

// Paper: no equation (helper: I/O, progress or utility)
func searchsortedLeft(a []float64, x float64) int { return sort.SearchFloat64s(a, x) }

// Paper: Sec.4 absmean ternarization scale
func absmean(w []float32) float64 {
	var s float64
	for _, v := range w {
		s += math.Abs(float64(v))
	}
	return s / float64(len(w))
}

// Paper_v2 Eq.(7): MRH integer key per channel = 12-bit saturating ternarization error relative to gamma = absmean(W):
// kappa_c = min(4095, floor(4096 * mean_j |W_cj - gamma*RoundClip(W_cj/gamma)| / gamma)); gamma = 0 -> key 0.
func heapKey(W []float32, R, C, axis int) []int {
	s := absmean(W)
	n, m := R, C
	if axis == 1 {
		n, m = C, R
	}
	acc := make([]float64, n)
	for r := 0; r < R; r++ {
		for c := 0; c < C; c++ {
			w := float64(W[r*C+c])
			e := math.Abs(w - ternQ(w, s)*s)
			if axis == 0 {
				acc[r] += e
			} else {
				acc[c] += e
			}
		}
	}
	k := make([]int, n)
	for i := range acc {
		if s > 0 {
			k[i] = min(1<<KeyBits-1, int(math.Floor(float64(int(1)<<KeyBits)*acc[i]/float64(m)/s)))
		}
	}
	return k
}

// Paper_v2 Eq.(25): RoundClip(w/gamma, -1, 1); gamma = 0 -> 0
func ternQ(w, s float64) float64 {
	if s == 0 {
		return 0
	}
	return math.Max(-1, math.Min(1, math.RoundToEven(w/s)))
}

// MRHMode selects how the multi-radix heap shapes the index (--mrh):
//
//	"dynamic": MRH(r; C) with r from each axis' own mixed-radix factorisation (RadicesFor): channel
//	           coordinates are its extract-min order, a tensor's bucket is the heap handle of its
//	           median key in the target axis' radices, the index grid keeps whole radix levels up to
//	           --grid-max and no finer than the coarsest contributor.
//	"paper":   paper_v2 as written: 16-bucket levels, bucket = top 4-bit digit, grid <= 256.
var MRHMode = "dynamic"

// GridMax caps the index length per axis (--grid-max); 2^(2*BucketBits) = 256 is paper_v2 Eq.(6).
var GridMax = 1 << (2 * BucketBits)

const keyUniverse = 1<<KeyBits - 1 // C of MRH(r; C): the 12-bit saturating key of Eq.(7)

// Paper_v2 Eq.(8): channel coordinate = (MRH pop rank + 1/2)/n, pop order = (key, index) ascending (MRH FIFO ties)
func chanCoords(key []int) []float64 {
	n := len(key)
	var o []int
	if MRHMode == "dynamic" && n > 0 {
		if p, err := PopOrder(key, RadicesFor(n, keyUniverse), keyUniverse); err == nil {
			o = p
		}
	}
	if o == nil {
		o = make([]int, n)
		for i := range o {
			o[i] = i
		}
		sort.SliceStable(o, func(a, b int) bool { return key[o[a]] < key[o[b]] })
	}
	x := make([]float64, n)
	for r, i := range o {
		x[i] = (float64(r) + .5) / float64(n)
	}
	return x
}

// Paper_v2 Eq.(9): tensor MRH bucket = top 4-bit digit of the median channel key
func bucketOf(key []int) int {
	s := append([]int(nil), key...)
	sort.Ints(s)
	return s[len(s)/2] >> (KeyBits - BucketBits)
}

// Eq.(9), dynamic: the bucket is the heap handle (level j, bucket t) the median key takes in a fresh
// MRH(r; C) with the target axis' radices, numbered across levels (sum_{i<j} r_i + t) and held in
// the chunk's u8. Contributors to one target share r, so equal buckets mean equal heap cells.
func bucketMRH(key []int, r []int) int {
	if MRHMode != "dynamic" {
		return bucketOf(key)
	}
	s := append([]int(nil), key...)
	sort.Ints(s)
	h, err := NewMRH(r, keyUniverse)
	if err != nil {
		return bucketOf(key)
	}
	j, t, _ := h.find(s[len(s)/2])
	id := t
	for i := 0; i < j; i++ {
		id += r[i]
	}
	return min(id, 255)
}

// Eq.(6), dynamic: whole coarsest radix levels of n, at most min(GridMax, finest) cells, where
// finest is the shortest contributor axis (an index cell no finer than the data feeding it).
func gridSizeFor(n, finest int) int {
	limit := GridMax
	if MRHMode == "dynamic" && finest > 0 {
		limit = min(limit, finest)
	} else if MRHMode != "dynamic" {
		limit = 1 << (2 * BucketBits)
	}
	g := 1
	for _, r := range radices(n) {
		if g*r > limit {
			break
		}
		g *= r
	}
	return max(g, 1)
}

// Paper: no equation (helper: I/O, progress or utility)
func argsortStable(x []float64) []int {
	o := make([]int, len(x))
	for i := range o {
		o[i] = i
	}
	sort.SliceStable(o, func(a, b int) bool { return x[o[a]] < x[o[b]] })
	return o
}

// splineGrid: degree-1 B-spline over channel coords (rows at xr, cols at xc), sampled on grid G.
// Paper: Contrib.4 channel index: degree-1 spline of the channel map onto the index grid
func splineGrid(W []float32, R, C int, xr, xc []float64, G [2]int, jitter float64) []float64 {
	gr := make([]float64, G[0])
	for i := range gr {
		gr[i] = (float64(i) + .5) / float64(G[0])
	}
	gc := make([]float64, G[1])
	for i := range gc {
		gc[i] = (float64(i) + .5) / float64(G[1])
	}
	o := argsortStable(xr)
	xs := make([]float64, R)
	for i, k := range o {
		xs[i] = xr[k] + float64(i)*jitter
	}
	A := make([]float64, G[0]*C) // G0 x C
	parFor(C, func(lo, hi int) {
		col := make([]float64, R)
		for j := lo; j < hi; j++ {
			for i, k := range o {
				col[i] = float64(W[k*C+j])
			}
			for g := 0; g < G[0]; g++ {
				A[g*C+j] = interp(gr[g], xs, col)
			}
		}
	})
	oc := argsortStable(xc)
	xcs := make([]float64, C)
	for i, k := range oc {
		xcs[i] = xc[k] + float64(i)*jitter
	}
	out := make([]float64, G[0]*G[1])
	parFor(G[0], func(lo, hi int) {
		row := make([]float64, C)
		for g := lo; g < hi; g++ {
			for i, k := range oc {
				row[i] = A[g*C+k]
			}
			for h := 0; h < G[1]; h++ {
				out[g*G[1]+h] = interp(gc[h], xcs, row)
			}
		}
	})
	return out
}

// Paper: Contrib.4 / Sec.4 resize: sample the index spline at the target channels
func sampleGrid(S []float64, G [2]int, xr, xc []float64) []float64 {
	gr := make([]float64, G[0])
	for i := range gr {
		gr[i] = (float64(i) + .5) / float64(G[0])
	}
	gc := make([]float64, G[1])
	for i := range gc {
		gc[i] = (float64(i) + .5) / float64(G[1])
	}
	R, C := len(xr), len(xc)
	A := make([]float64, R*G[1])
	parFor(G[1], func(lo, hi int) {
		col := make([]float64, G[0])
		for j := lo; j < hi; j++ {
			for g := 0; g < G[0]; g++ {
				col[g] = S[g*G[1]+j]
			}
			for i := 0; i < R; i++ {
				A[i*G[1]+j] = interp(xr[i], gr, col)
			}
		}
	})
	out := make([]float64, R*C)
	parFor(R, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := A[i*G[1] : (i+1)*G[1]]
			for c := 0; c < C; c++ {
				out[i*C+c] = interp(xc[c], gc, row)
			}
		}
	})
	return out
}

// ---------------- Seed: phase correlation per radix digit + assignment ----------------
// Paper: no equation (helper: I/O, progress or utility)
func roll(A []float64, dims []int, ax, s int) []float64 {
	o := make([]float64, len(A))
	stride := 1
	for j := len(dims) - 1; j > ax; j-- {
		stride *= dims[j]
	}
	r := dims[ax]
	for i := range A {
		d := (i / stride) % r
		nd := (d + s) % r
		o[i+(nd-d)*stride] = A[i]
	}
	return o
}

// Paper: Sec.3.1 step 1 Seed: phase correlation per radix digit estimates shifts
func seedShift(grid, ref []float64, dims []int, pcEps float64) ([]float64, []int) {
	A := append([]float64(nil), grid...)
	FB := fftn(toC(ref), dims, false)
	sh := make([]int, len(dims))
	for ax, r := range dims {
		FA := fftn(toC(A), dims, false)
		X := make([]complex128, len(FA))
		for i := range X {
			v := FB[i] * cmplx.Conj(FA[i])
			if a := cmplx.Abs(v); a > 0 { // Eq.(12): X = v/|v|, |v| = 0 -> 0 (no information)
				X[i] = v / complex(a, 0)
			}
		}
		c := realPart(fftn(X, dims, true))
		stride := 1
		for j := len(dims) - 1; j > ax; j-- {
			stride *= dims[j]
		}
		sum := make([]float64, r)
		for i, v := range c {
			sum[(i/stride)%r] += v
		}
		s := 0
		if r > 1 {
			for k := 1; k < r; k++ {
				if sum[k] > sum[s] {
					s = k
				}
			}
		}
		A = roll(A, dims, ax, s)
		sh[ax] = s
	}
	return A, sh
}

// hungarian: min-cost assignment, cost n x n; returns p[i] = column for row i.
// Paper: Sec.3.1 step 1 Seed: Git Re-Basin / OT assignment (linear assignment solver)
func hungarian(cost [][]float64) []int {
	n := len(cost)
	INF := math.Inf(1)
	u := make([]float64, n+1)
	v := make([]float64, n+1)
	p := make([]int, n+1)
	way := make([]int, n+1)
	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, n+1)
		used := make([]bool, n+1)
		for j := range minv {
			minv[j] = INF
		}
		for {
			used[j0] = true
			i0, delta, j1 := p[j0], INF, 0
			for j := 1; j <= n; j++ {
				if !used[j] {
					cur := cost[i0-1][j-1] - u[i0] - v[j]
					if cur < minv[j] {
						minv[j], way[j] = cur, j0
					}
					if minv[j] < delta {
						delta, j1 = minv[j], j
					}
				}
			}
			for j := 0; j <= n; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}
	ans := make([]int, n)
	for j := 1; j <= n; j++ {
		ans[p[j]-1] = j - 1
	}
	return ans
}

// seedAssign: weight-matching permutation of rows then cols of A onto ref (maximize <ref, P A Q>).
// Paper: Sec.3.1 step 1 Seed: starting permutation by Re-Basin / OT weight matching
func seedAssign(A, ref []float64, G [2]int) ([]float64, []int, []int) {
	Gr, Gc := G[0], G[1]
	cost := make([][]float64, Gr)
	for i := range cost {
		cost[i] = make([]float64, Gr)
		for j := 0; j < Gr; j++ {
			var d float64
			for c := 0; c < Gc; c++ {
				d += ref[i*Gc+c] * A[j*Gc+c]
			}
			cost[i][j] = -d
		}
	}
	pr := hungarian(cost)
	B := make([]float64, len(A))
	for i := 0; i < Gr; i++ {
		copy(B[i*Gc:(i+1)*Gc], A[pr[i]*Gc:(pr[i]+1)*Gc])
	}
	cost = make([][]float64, Gc)
	for i := range cost {
		cost[i] = make([]float64, Gc)
		for j := 0; j < Gc; j++ {
			var d float64
			for r := 0; r < Gr; r++ {
				d += ref[r*Gc+i] * B[r*Gc+j]
			}
			cost[i][j] = -d
		}
	}
	pc := hungarian(cost)
	Cm := make([]float64, len(A))
	for r := 0; r < Gr; r++ {
		for i := 0; i < Gc; i++ {
			Cm[r*Gc+i] = B[r*Gc+pc[i]]
		}
	}
	return Cm, pr, pc
}

// ---------------- ternary TIES on ternary codes (absmean codes, sign election, mask union) ----------------
// Paper: Contrib.1 ternary-native TIES: sign election and mask union on ternary codes (curl part, Sec.3)
// ternaryTIESAll is the merged TIES delta normalised by every contributor's weight: per element, the
// elected sign's agreeing deltas summed with their weights, over the total weight of all contributors
// (the base and any source whose delta is zero there included),
//
//	out_i = sum_{k agreeing at i} w_k d_k,i / total.
//
// ternaryTIES divides by the agreeing weight only, so with one source its weight cancels and the
// base's never enters: --base-weight and --src PATH:W did nothing to the curl set and a 1:1 mix
// replaced the base's curl set (2B4T + Qwen3: 24% of every tensor's codes, ppl 12 -> 7.6e7). With
// one source this equals the earlier 1 - w_base/sum w scaling; a source with a zero delta (the base
// given again as a source) adds weight that votes nothing, so it dilutes toward the base: the merged
// value cannot be pulled to an exact 0 where the base is not. A self-mix has zero deltas: x1.
func ternaryTIESAll(parts [][]float64, ws []float64, total float64) []float64 {
	n := len(parts[0])
	codes := make([][]int8, len(parts))
	for k, p := range parts {
		var s float64
		for _, v := range p {
			s += math.Abs(v)
		}
		s /= float64(n)
		c := make([]int8, n)
		for i, v := range p {
			c[i] = int8(ternQ(v, s)) // Eq.(21)
		}
		codes[k] = c
	}
	out := make([]float64, n)
	if total <= 0 {
		return out
	}
	for i := 0; i < n; i++ {
		var vote float64
		for k := range parts {
			vote += ws[k] * float64(codes[k][i])
		}
		sg := int8(0)
		if vote > 0 {
			sg = 1
		} else if vote < 0 {
			sg = -1
		}
		if sg == 0 {
			continue
		}
		var num float64
		for k := range parts {
			if codes[k][i] == sg {
				num += ws[k] * parts[k][i]
			}
		}
		out[i] = num / total
	}
	return out
}

func ternaryTIES(parts [][]float64, ws []float64, eps float64) []float64 {
	n := len(parts[0])
	codes := make([][]int8, len(parts))
	for k, p := range parts {
		var s float64
		for _, v := range p {
			s += math.Abs(v)
		}
		s /= float64(n)
		c := make([]int8, n)
		for i, v := range p {
			c[i] = int8(ternQ(v, s)) // Eq.(21)
		}
		codes[k] = c
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		var vote float64
		for k := range parts {
			vote += ws[k] * float64(codes[k][i])
		}
		sg := int8(0)
		if vote > 0 {
			sg = 1
		} else if vote < 0 {
			sg = -1
		}
		var num, cnt float64
		for k := range parts {
			if codes[k][i] == sg && sg != 0 {
				num += ws[k] * parts[k][i]
				cnt += ws[k]
			}
		}
		if cnt > 0 {
			out[i] = num / cnt
		}
	}
	return out
}

// Fibre selection on the digit torus (Lattice_Hilbert21_MRH).
//
// Lemma 4.1 certifies a unique continuation when the base fibres are separated
// by more than √2 (1 + √n) δ and exactly one source fibre lies in the √2 δ
// ball. That map is the selection. An uncertified slot is the cone centre
// (Theorem 8.2 at ρ = 0): the base value. There is no assignment search.
package train

import "math"

// GapNeed is √2 (1 + √n) δ, the separation Lemma 4.1(b) requires.
func GapNeed(n int, delta float64) float64 {
	if n < 1 {
		return 0
	}
	return math.Sqrt2 * (1 + math.Sqrt(float64(n))) * delta
}

// StripDelta is Lemma 3.1: the holonomy moves by at most the sum of the
// mixed-plaquette deviations along the strip.
func StripDelta(plaquetteDev []float64) float64 {
	var s float64
	for _, p := range plaquetteDev {
		if p < 0 {
			p = -p
		}
		s += p
	}
	return s
}

// Chern2 is Proposition 6.4: ch₂ = −n (k₂ l₃ − k₃ l₂).
func Chern2(n, k2, l3, k3, l2 int) int {
	return -n * (k2*l3 - k3*l2)
}

// ConeCap is Theorem 8.2 on one scalar: c₀ + t (boundary − c₀), t = ρ/R.
func ConeCap(c0, boundary, t float64) float64 {
	return c0 + t*(boundary-c0)
}

// BranchLevel is the least-level placement of a nonnegative key difference
// against frontier 0 (§5.2). Radices are finest first. The level is one plus
// the index of the highest nonzero digit, or 0 when the difference is 0.
func BranchLevel(diff int, radices []int) int {
	if diff < 0 {
		diff = -diff
	}
	if diff == 0 {
		return 0
	}
	level := 0
	for j, r := range radices {
		if r < 2 {
			return 0
		}
		if diff%r != 0 {
			level = j + 1
		}
		diff /= r
	}
	return level
}

// CyclicDiff is the circular distance on Z/modZ.
func CyclicDiff(a, b, mod int) int {
	if mod <= 0 {
		return 0
	}
	d := a - b
	if d < 0 {
		d = -d
	}
	d %= mod
	if mod-d < d {
		return mod - d
	}
	return d
}

type matchResult struct {
	perm []int
	cert []bool
	hits int
}

// match pairs source vectors to destination vectors by Lemma 4.1.
// perm is always a bijection. cert[i] is false when slot i was cone-capped;
// perm[i] is then only a filler index so the stored map stays invertible.
func match(src, dst [][]float64) matchResult {
	n := len(dst)
	out := matchResult{perm: make([]int, n), cert: make([]bool, n)}
	if n == 0 || len(src) != n {
		return out
	}
	for i := range out.perm {
		out.perm[i] = i
	}
	delta := 0.0
	for j := 0; j < n; j++ {
		bd := math.Inf(1)
		for i := 0; i < n; i++ {
			if d := l2(src[j], dst[i]); d < bd {
				bd = d
			}
		}
		if bd > delta {
			delta = bd
		}
	}
	limit := math.Sqrt2 * delta
	need := GapNeed(n, delta)
	claim := make([]int, n)
	ok := make([]bool, n)
	for i := 0; i < n; i++ {
		g := math.Inf(1)
		if n == 1 {
			g = math.Inf(1)
		} else {
			for k := 0; k < n; k++ {
				if k == i {
					continue
				}
				if d := l2(dst[i], dst[k]); d < g {
					g = d
				}
			}
		}
		separated := g > need || (delta == 0 && g > 0) || n == 1
		if !separated {
			continue
		}
		hit, cnt := -1, 0
		for j := 0; j < n; j++ {
			if l2(src[j], dst[i]) <= limit {
				cnt++
				hit = j
			}
		}
		if cnt == 1 {
			ok[i] = true
			claim[i] = hit
		}
	}
	// A source claimed by two slots is a collision. Both slots lose the certificate.
	owner := make([][]int, n)
	for i := 0; i < n; i++ {
		if ok[i] {
			owner[claim[i]] = append(owner[claim[i]], i)
		}
	}
	for j := 0; j < n; j++ {
		if len(owner[j]) > 1 {
			for _, i := range owner[j] {
				ok[i] = false
			}
		}
	}
	used := make([]bool, n)
	for i := 0; i < n; i++ {
		if ok[i] {
			out.perm[i] = claim[i]
			out.cert[i] = true
			used[claim[i]] = true
			out.hits++
		} else {
			out.perm[i] = -1
		}
	}
	next := 0
	for i := 0; i < n; i++ {
		if out.perm[i] >= 0 {
			continue
		}
		for next < n && used[next] {
			next++
		}
		if next >= n {
			out.perm[i] = 0
			continue
		}
		out.perm[i] = next
		used[next] = true
		next++
	}
	return out
}

func l2(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}

func splitRows(A []float64, rows, cols int) [][]float64 {
	o := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		o[i] = A[i*cols : (i+1)*cols]
	}
	return o
}

func splitCols(A []float64, rows, cols int) [][]float64 {
	o := make([][]float64, cols)
	for c := 0; c < cols; c++ {
		col := make([]float64, rows)
		for r := 0; r < rows; r++ {
			col[r] = A[r*cols+c]
		}
		o[c] = col
	}
	return o
}

// Align copies a source fibre onto the base only when Lemma 4.1 certifies it.
// Every other slot is the base value. permR and permC are bijections.
// rowHits and colHits are the certified fibres. Zero on an axis means the
// cone cap replaced that axis with the base.
func Align(A, ref []float64, rows, cols int) (out []float64, permR, permC []int, rowHits, colHits int) {
	if rows < 1 || cols < 1 || len(A) != rows*cols || len(ref) != len(A) {
		out = append([]float64(nil), A...)
		return out, id(rows), id(cols), rows, cols
	}
	rm := match(splitRows(A, rows, cols), splitRows(ref, rows, cols))
	mid := make([]float64, len(A))
	for i := 0; i < rows; i++ {
		if rm.cert[i] {
			copy(mid[i*cols:(i+1)*cols], A[rm.perm[i]*cols:(rm.perm[i]+1)*cols])
		} else {
			copy(mid[i*cols:(i+1)*cols], ref[i*cols:(i+1)*cols])
		}
	}
	cm := match(splitCols(mid, rows, cols), splitCols(ref, rows, cols))
	out = make([]float64, len(A))
	for c := 0; c < cols; c++ {
		for r := 0; r < rows; r++ {
			if cm.cert[c] {
				out[r*cols+c] = mid[r*cols+cm.perm[c]]
			} else {
				out[r*cols+c] = ref[r*cols+c]
			}
		}
	}
	return out, rm.perm, cm.perm, rm.hits, cm.hits
}

// SelectAngles selects joint eigenvalues on T̂². Each point is a pair of angles.
// The distance is Euclidean in C², the metric of §2.3.
func SelectAngles(src, dst [][2]float64) (perm []int, certified int) {
	n := len(dst)
	if n == 0 || len(src) != n {
		return id(n), 0
	}
	S := make([][]float64, n)
	D := make([][]float64, n)
	for i := 0; i < n; i++ {
		S[i] = embed(src[i])
		D[i] = embed(dst[i])
	}
	m := match(S, D)
	return m.perm, m.hits
}

func embed(a [2]float64) []float64 {
	return []float64{math.Cos(a[0]), math.Sin(a[0]), math.Cos(a[1]), math.Sin(a[1])}
}

func id(n int) []int {
	if n < 0 {
		n = 0
	}
	o := make([]int, n)
	for i := range o {
		o[i] = i
	}
	return o
}

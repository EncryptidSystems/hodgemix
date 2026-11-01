// A child is made from a mother and a father and matches neither of them.
//
// Each parent is a writer and a chart. The circuit is their product. The turn
// is the orthogonal factor of C_father C_mother^T: it lines the mother's
// circuit up with the father's so the two add. The child circuit is that sum,
// carried into the halfway orientation of the same turn. Its energy is both
// parents' energies plus their aligned overlap, so the map is a third map.
// The stored writer and chart are a new splitting of that circuit, under the
// child's own labels.
//
// Nothing in this file searches, and nothing reads a token.
package turn

import (
	"fmt"
	"math"
	"sort"
)

// Parent is one source. Writer is M×K, chart is K×N.
type Parent struct {
	Writer, Chart           []float64
	WriterLabel, ChartLabel string
	M, K, N                 int
}

// Model is the child. Writer is M×K, chart is K×N, and K is the child's own.
type Model struct {
	Writer, Chart           []float64
	WriterLabel, ChartLabel string
	M, K, N                 int
}

// Circuit is the product B A. B is m×k, A is k×n, the product is m×n.
func Circuit(B, A []float64, m, k, n int) []float64 {
	return mul(B, A, m, k, n)
}

// Child builds one model from two parents. ok is false when either parent
// has no circuit, so a single source is never returned as the child.
func Child(mother, father Parent) (Model, bool) {
	if mother.M != father.M || mother.N != father.N || mother.M < 1 || mother.N < 1 {
		return Model{}, false
	}
	if mother.K < 1 || father.K < 1 {
		return Model{}, false
	}
	m, n := mother.M, mother.N
	if len(mother.Writer) != m*mother.K || len(mother.Chart) != mother.K*n {
		return Model{}, false
	}
	if len(father.Writer) != m*father.K || len(father.Chart) != father.K*n {
		return Model{}, false
	}
	Cm := Circuit(mother.Writer, mother.Chart, m, mother.K, n)
	Cf := Circuit(father.Writer, father.Chart, m, father.K, n)
	C, ok := inherit(Cm, Cf, m, n)
	if !ok {
		return Model{}, false
	}
	B, A, k, ok := factor(C, m, n)
	if !ok {
		return Model{}, false
	}
	B, A = unmatch(B, A, m, k, n,
		[][]float64{mother.Writer, father.Writer},
		[][]float64{mother.Chart, father.Chart})
	taken := map[string]bool{}
	for _, s := range []string{mother.WriterLabel, mother.ChartLabel, father.WriterLabel, father.ChartLabel} {
		if s != "" {
			taken[s] = true
		}
	}
	return Model{
		Writer:      B,
		Chart:       A,
		WriterLabel: fresh(taken, "writer"),
		ChartLabel:  fresh(taken, "chart"),
		M:           m,
		K:           k,
		N:           n,
	}, true
}

// inherit is the child circuit. The turn U lines the mother up with the
// father. The sum of the lined-up circuits is then set halfway along U.
func inherit(Cm, Cf []float64, m, n int) ([]float64, bool) {
	if m < 1 || n < 1 || len(Cm) != m*n || len(Cf) != m*n {
		return nil, false
	}
	if energy(Cm) == 0 || energy(Cf) == 0 {
		return nil, false
	}
	M := mul(Cf, transpose(Cm, m, n), m, n, m)
	U, ok := polar(M, m)
	if !ok {
		U = eye(m)
	}
	aligned := mul(U, Cm, m, m, n)
	V := halfTurn(U, m)
	sum := make([]float64, m*n)
	for i := range sum {
		sum[i] = aligned[i] + Cf[i]
	}
	return mul(transpose(V, m, m), sum, m, m, n), true
}

// halfTurn is the orthogonal factor of I+U, the halfway orientation of a turn.
// Opposite parents make I+U singular, and the child keeps the lined-up frame.
func halfTurn(U []float64, n int) []float64 {
	M := eye(n)
	for i := range U {
		M[i] += U[i]
	}
	V, ok := polar(M, n)
	if !ok {
		return eye(n)
	}
	return V
}

// factor splits C into writer and chart, C = B A, by the square root of its
// singular values. The inner width is the rank of C.
func factor(C []float64, m, n int) (B, A []float64, k int, ok bool) {
	if m < 1 || n < 1 || len(C) != m*n {
		return nil, nil, 0, false
	}
	G := mul(transpose(C, m, n), C, n, m, n)
	Q, lam := jacobi(G, n)
	hi := 0.0
	for _, v := range lam {
		if v > hi {
			hi = v
		}
	}
	if hi <= 0 {
		return nil, nil, 0, false
	}
	tol := 1e-12 * hi
	type mode struct {
		s           float64
		left, right []float64
	}
	var modes []mode
	for c, lv := range lam {
		if lv <= tol {
			continue
		}
		sigma := math.Sqrt(lv)
		right := column(Q, n, c)
		left := apply(C, right, m, n)
		for i := range left {
			left[i] /= sigma
		}
		modes = append(modes, mode{math.Sqrt(sigma), left, right})
	}
	k = len(modes)
	if k == 0 {
		return nil, nil, 0, false
	}
	B = make([]float64, m*k)
	A = make([]float64, k*n)
	for j, md := range modes {
		for r := 0; r < m; r++ {
			B[r*k+j] = md.left[r] * md.s
		}
		for c := 0; c < n; c++ {
			A[j*n+c] = md.s * md.right[c]
		}
	}
	return B, A, k, true
}

// unmatch flips one inherited mode when a factor lands on a parent tensor.
// The product is unchanged: that column of the writer and that row of the
// chart change sign together.
func unmatch(B, A []float64, m, k, n int, writers, charts [][]float64) ([]float64, []float64) {
	if !clashes(B, writers...) && !clashes(A, charts...) {
		return B, A
	}
	for mode := 0; mode < k; mode++ {
		Bp := append([]float64(nil), B...)
		Ap := append([]float64(nil), A...)
		for r := 0; r < m; r++ {
			Bp[r*k+mode] = -Bp[r*k+mode]
		}
		for c := 0; c < n; c++ {
			Ap[mode*n+c] = -Ap[mode*n+c]
		}
		if !clashes(Bp, writers...) && !clashes(Ap, charts...) {
			return Bp, Ap
		}
	}
	return B, A
}

func clashes(x []float64, parents ...[]float64) bool {
	for _, p := range parents {
		if sameTensor(x, p) {
			return true
		}
	}
	return false
}

func sameTensor(a, b []float64) bool {
	if len(a) == 0 || len(a) != len(b) {
		return false
	}
	var s, na float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
		na += a[i] * a[i]
	}
	return math.Sqrt(s) <= 1e-8*(math.Sqrt(na)+1)
}

func fresh(taken map[string]bool, kind string) string {
	base := "child/" + kind
	if !taken[base] {
		taken[base] = true
		return base
	}
	for i := 2; ; i++ {
		s := fmt.Sprintf("%s/%d", base, i)
		if !taken[s] {
			taken[s] = true
			return s
		}
	}
}

func energy(a []float64) float64 {
	var s float64
	for _, v := range a {
		s += v * v
	}
	return s
}

func apply(M, v []float64, rows, cols int) []float64 {
	o := make([]float64, rows)
	for r := 0; r < rows; r++ {
		var s float64
		for c := 0; c < cols; c++ {
			s += M[r*cols+c] * v[c]
		}
		o[r] = s
	}
	return o
}

// polar is the orthogonal factor of M nearest the identity on the kernel of M.
// The support is fixed by the singular vectors. A zero matrix is no turn.
func polar(M []float64, n int) ([]float64, bool) {
	G := mul(transpose(M, n, n), M, n, n, n)
	Q, lam := jacobi(G, n)
	ord := make([]int, n)
	for i := range ord {
		ord[i] = i
	}
	sort.Slice(ord, func(i, j int) bool { return lam[ord[i]] > lam[ord[j]] })
	hi := lam[ord[0]]
	if hi <= 0 {
		return nil, false
	}
	tol := 1e-12 * hi
	var left, right, ker [][]float64
	for _, c := range ord {
		b := column(Q, n, c)
		if lam[c] <= tol {
			ker = append(ker, b)
			continue
		}
		a := matVec(M, b, n)
		s := math.Sqrt(lam[c])
		for i := range a {
			a[i] /= s
		}
		left = append(left, a)
		right = append(right, b)
	}
	left = orthonormal(left)
	extra := extend(left, n)
	if len(ker) > 0 && detCols(append(left, extra...)) < 0 {
		flip(extra[len(extra)-1])
	}
	return mul(pack(append(left, extra...)), transpose(pack(append(right, ker...)), n, n), n, n, n), true
}

func column(q []float64, n, c int) []float64 {
	o := make([]float64, n)
	for r := 0; r < n; r++ {
		o[r] = q[r*n+c]
	}
	return o
}

func matVec(M, v []float64, n int) []float64 {
	o := make([]float64, n)
	for r := 0; r < n; r++ {
		var s float64
		for c := 0; c < n; c++ {
			s += M[r*n+c] * v[c]
		}
		o[r] = s
	}
	return o
}

func pack(cols [][]float64) []float64 {
	if len(cols) == 0 {
		return nil
	}
	n := len(cols[0])
	o := make([]float64, n*len(cols))
	for c, col := range cols {
		for r := 0; r < n; r++ {
			o[r*len(cols)+c] = col[r]
		}
	}
	return o
}

func orthonormal(vs [][]float64) [][]float64 {
	var o [][]float64
	for _, v := range vs {
		w := projectOut(v, o)
		if nrm := norm(w); nrm > 1e-10 {
			o = append(o, scale(w, 1/nrm))
		}
	}
	return o
}

// extend fills the complement. Each new vector is the standard basis vector
// closest to the complement, so an unused direction stays near itself.
func extend(span [][]float64, n int) [][]float64 {
	var extra [][]float64
	for e := 0; len(span)+len(extra) < n && e < n; e++ {
		b := make([]float64, n)
		b[e] = 1
		w := projectOut(b, append(span, extra...))
		if nrm := norm(w); nrm > 1e-8 {
			extra = append(extra, scale(w, 1/nrm))
		}
	}
	return extra
}

func projectOut(v []float64, span [][]float64) []float64 {
	o := append([]float64(nil), v...)
	for _, s := range span {
		var d float64
		for i := range o {
			d += o[i] * s[i]
		}
		for i := range o {
			o[i] -= d * s[i]
		}
	}
	return o
}

func norm(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	return math.Sqrt(s)
}

func scale(v []float64, a float64) []float64 {
	o := make([]float64, len(v))
	for i := range v {
		o[i] = v[i] * a
	}
	return o
}

func flip(v []float64) {
	for i := range v {
		v[i] = -v[i]
	}
}

func detCols(cols [][]float64) float64 {
	n := len(cols)
	a := pack(cols)
	sign := 1.0
	for k := 0; k < n; k++ {
		piv := k
		best := math.Abs(a[k*n+k])
		for i := k + 1; i < n; i++ {
			if x := math.Abs(a[i*n+k]); x > best {
				best, piv = x, i
			}
		}
		if best < 1e-15 {
			return 0
		}
		if piv != k {
			sign = -sign
			for j := 0; j < n; j++ {
				a[k*n+j], a[piv*n+j] = a[piv*n+j], a[k*n+j]
			}
		}
		for i := k + 1; i < n; i++ {
			f := a[i*n+k] / a[k*n+k]
			for j := k; j < n; j++ {
				a[i*n+j] -= f * a[k*n+j]
			}
		}
	}
	d := sign
	for k := 0; k < n; k++ {
		d *= a[k*n+k]
	}
	return d
}

func eye(n int) []float64 {
	o := make([]float64, n*n)
	for i := 0; i < n; i++ {
		o[i*n+i] = 1
	}
	return o
}

func transpose(a []float64, rows, cols int) []float64 {
	o := make([]float64, len(a))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			o[j*rows+i] = a[i*cols+j]
		}
	}
	return o
}

func mul(a, b []float64, m, k, n int) []float64 {
	o := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for t := 0; t < k; t++ {
			aik := a[i*k+t]
			if aik == 0 {
				continue
			}
			for j := 0; j < n; j++ {
				o[i*n+j] += aik * b[t*n+j]
			}
		}
	}
	return o
}

// jacobi returns eigenvectors in the columns and the eigenvalues of a symmetric matrix.
func jacobi(a0 []float64, n int) (q, lam []float64) {
	a := append([]float64(nil), a0...)
	q = eye(n)
	if n == 1 {
		return q, []float64{a[0]}
	}
	scale := 0.0
	for _, v := range a {
		if x := math.Abs(v); x > scale {
			scale = x
		}
	}
	tol := 1e-15 * (scale + 1)
	for sweep := 0; sweep < 32*n; sweep++ {
		off := 0.0
		for p := 0; p < n-1; p++ {
			for r := p + 1; r < n; r++ {
				apr := a[p*n+r]
				if x := math.Abs(apr); x > off {
					off = x
				}
				if math.Abs(apr) <= tol {
					continue
				}
				app, arr := a[p*n+p], a[r*n+r]
				tau := (arr - app) / (2 * apr)
				t := 1.0
				if tau < 0 {
					t = -1
				}
				t /= math.Abs(tau) + math.Hypot(1, tau)
				c := 1 / math.Sqrt(1+t*t)
				s := t * c
				a[p*n+p] = app - t*apr
				a[r*n+r] = arr + t*apr
				a[p*n+r], a[r*n+p] = 0, 0
				for k := 0; k < n; k++ {
					if k != p && k != r {
						akp, akr := a[k*n+p], a[k*n+r]
						a[k*n+p] = c*akp - s*akr
						a[p*n+k] = a[k*n+p]
						a[k*n+r] = s*akp + c*akr
						a[r*n+k] = a[k*n+r]
					}
					qkp, qkr := q[k*n+p], q[k*n+r]
					q[k*n+p] = c*qkp - s*qkr
					q[k*n+r] = s*qkp + c*qkr
				}
			}
		}
		if off <= tol {
			break
		}
	}
	lam = make([]float64, n)
	for i := 0; i < n; i++ {
		lam[i] = a[i*n+i]
	}
	return q, lam
}

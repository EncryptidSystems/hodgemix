package main

// Randomized Hadamard residual-stream rotation (paper_v2 Eq.(23)-(24)), QuaRot / SliceGPT computational invariance.

import (
	"math"
	"math/rand"
)

// Paper_v2 Eq.(23): block size b = largest power of two dividing H
func hadBlock(H int) int {
	b := 1
	for H%(2*b) == 0 {
		b *= 2
	}
	return b
}

// Rot holds Q = D (I_{H/b} (x) H_b/sqrt(b)), D = diag(sign), applied as x -> x Q to row vectors.
type Rot struct {
	H, B int
	Sign []float64
}

// Paper_v2 Eq.(23): Rademacher signs from seed (reproducibility only; invariance holds for any signs)
func NewRot(H int, seed int64) *Rot {
	g := rand.New(rand.NewSource(seed))
	s := make([]float64, H)
	for i := range s {
		s[i] = 1
		if g.Intn(2) == 1 {
			s[i] = -1
		}
	}
	return &Rot{H, hadBlock(H), s}
}

// fwht: in-place unnormalized Walsh-Hadamard (Sylvester) on len power of two
func fwht(a []float64) {
	for h := 1; h < len(a); h *= 2 {
		for i := 0; i < len(a); i += 2 * h {
			for j := i; j < i+h; j++ {
				x, y := a[j], a[j+h]
				a[j], a[j+h] = x+y, x-y
			}
		}
	}
}

// Paper_v2 Eq.(23): row vector v -> v Q  (v Q = (v D) blockdiag(H_b)/sqrt(b), H_b symmetric)
func (q *Rot) RowTimesQ(v []float64) {
	for i := range v {
		v[i] *= q.Sign[i]
	}
	n := 1 / math.Sqrt(float64(q.B))
	for o := 0; o < q.H; o += q.B {
		fwht(v[o : o+q.B])
		for i := o; i < o+q.B; i++ {
			v[i] *= n
		}
	}
}

// Paper_v2 Eq.(23): column vector v -> Q^T v = B D v (B = blockdiag(H_b)/sqrt(b) symmetric), same map as RowTimesQ
func (q *Rot) QTTimes(v []float64) { q.RowTimesQ(v) }

// Paper_v2 Eq.(24a): reader W[out,H] -> W diag(g) Q (each row r: r*g then r Q)
func (q *Rot) Reader(W []float32, R int, g []float32) {
	row := make([]float64, q.H)
	for r := 0; r < R; r++ {
		for c := 0; c < q.H; c++ {
			x := float64(W[r*q.H+c])
			if g != nil {
				x *= float64(g[c])
			}
			row[c] = x
		}
		q.RowTimesQ(row)
		for c := range row {
			W[r*q.H+c] = float32(row[c])
		}
	}
}

// Paper_v2 Eq.(24b): writer W[H,in] -> Q^T W (each column)
func (q *Rot) Writer(W []float32, C int) {
	col := make([]float64, q.H)
	for c := 0; c < C; c++ {
		for r := 0; r < q.H; r++ {
			col[r] = float64(W[r*C+c])
		}
		q.QTTimes(col)
		for r := range col {
			W[r*C+c] = float32(col[r])
		}
	}
}

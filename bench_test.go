package main

import (
	"math/rand"
	"testing"
	"time"
)

// TestScaling: synthetic small/medium timings used to extrapolate runtime; never touches real models.
// Paper: timing protocol of Sec.7 (no equation)
func TestScaling(t *testing.T) {
	g := rand.New(rand.NewSource(1))
	for _, n := range []int{100, 400, 1200} {
		th := make([]float64, n)
		lv, ref := make([]int, n), make([]int, n)
		C := make([]float64, n*n)
		for i := range th {
			th[i] = g.Float64() - .5
			lv[i] = i % 8
		}
		for i := range C {
			C[i] = 1
		}
		K := couplingMatrix(C, n, 1.0/16)
		t0 := time.Now()
		r := kuramotoTick(th, K, lv, ref, KParams{GTheta: 1.0 / 16, Weak: 1.0 / 16, WLock: WLock, NLost: NLost, RTol: RefineTol, MaxWin: 10, Anneal: true}, "bench")
		t.Logf("kuramoto n_osc=%d: %d ticks %.3f s (%.3f ms/tick)", n, r.Steps, time.Since(t0).Seconds(), time.Since(t0).Seconds()*1e3/float64(r.Steps))
	}
	for _, G := range [][2]int{{160, 216}, {256, 216}} {
		n := G[0] * G[1]
		A, B := make([]float64, n), make([]float64, n)
		for i := range A {
			A[i], B[i] = g.NormFloat64(), g.NormFloat64()
		}
		Sp := NewSpectrum(G, 0)
		t0 := time.Now()
		g2, _ := seedShift(A, B, Sp.dims, 0)
		t1 := time.Now()
		seedAssign(g2, B, G)
		t2 := time.Now()
		Sp.Hodge(g2)
		t.Logf("grid %v: seed shift %.2fs assign %.2fs hodge %.2fs", G, t1.Sub(t0).Seconds(), t2.Sub(t1).Seconds(), time.Since(t2).Seconds())
	}
	R, C := 2560, 6912
	W := make([]float32, R*C)
	for i := range W {
		W[i] = float32(g.NormFloat64())
	}
	t0 := time.Now()
	kr := heapKey(W, R, C, 0)
	kc := heapKey(W, R, C, 1)
	t1 := time.Now()
	S := splineGrid(W, R, C, chanCoords(kr), chanCoords(kc), [2]int{160, 216}, 0)
	t2 := time.Now()
	sampleGrid(S, [2]int{160, 216}, chanCoords(kr), chanCoords(kc))
	q := NewRot(2560, 0)
	t3 := time.Now()
	q.Writer(W, C)
	t.Logf("2560x6912 synthetic: keys %.2fs spline %.2fs sample %.2fs rotate %.2fs", t1.Sub(t0).Seconds(), t2.Sub(t1).Seconds(), t3.Sub(t2).Seconds(), time.Since(t3).Seconds())
}

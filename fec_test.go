package main

import (
	"math"
	"math/rand"
	"testing"
)

// fec.md §6 on synthetic samples. No checkpoint is opened.
// Idempotency is the nodal one: coarse nodes at 0, h, ..., 1 come back as the midpoints of those
// nodes. That is the commuting diagram. Re-applying the map to the center samples is a different
// grid and is not that identity.

func TestFECMollifierIntegral(t *testing.T) {
	for _, G := range []int{1, 8, 32} {
		h := 1 / float64(G)
		for c := 0; c < G; c++ {
			L, R := float64(c)*h, float64(c+1)*h
			got := mollRayLeft(L, R, h) + mollWeight(L, R, 0, 1, h) + mollRayRight(L, R, h)
			if math.Abs(got-h) > 1e-9 {
				t.Fatalf("G=%d cell %d: ray+interval integral %g, want %g", G, c, got, h)
			}
		}
	}
	if got := mollWeight(0.4, 0.6, 0, 1, 0.01); math.Abs(got-0.2) > 1e-9 {
		t.Fatalf("interior indicator integral %g", got)
	}
}

func TestFECConstantAndLine(t *testing.T) {
	n, G := 65, 8
	xs := linspace(0, 1, n)
	geom := newAxisGeom(xs, G)
	for g, v := range geom.project(constant(n, 2.5)) {
		if math.Abs(v-2.5) > 1e-10 {
			t.Fatalf("constant cell %d: %g", g, v)
		}
	}
	a, b := 0.3, -1.7
	checkLine := func(name string, xs []float64) {
		t.Helper()
		u := make([]float64, len(xs))
		for i, x := range xs {
			u[i] = a + b*x
		}
		for g, v := range newAxisGeom(xs, G).project(u) {
			want := a + b*(float64(g)+0.5)/float64(G)
			if math.Abs(v-want) > 1e-8 {
				t.Fatalf("%s cell %d: got %g want %g", name, g, v, want)
			}
		}
	}
	checkLine("uniform", xs)
	checkLine("irregular", []float64{0, 0.07, 0.2, 0.21, 0.55, 0.8, 1})
}

func TestFECNodalIdempotent(t *testing.T) {
	G := 32
	xs := linspace(0, 1, G+1)
	rng := rand.New(rand.NewSource(3))
	U := make([]float64, G+1)
	for i := range U {
		U[i] = rng.NormFloat64()
	}
	got := newAxisGeom(xs, G).project(U)
	var max float64
	for g := 0; g < G; g++ {
		want := 0.5 * (U[g] + U[g+1])
		max = math.Max(max, math.Abs(got[g]-want))
	}
	if max > 1e-9 {
		t.Fatalf("nodal idempotency %.3g", max)
	}
}

func TestFECSplineGridSeparable(t *testing.T) {
	R, C := 16, 12
	G := [2]int{4, 3}
	xr := linspace(0, 1, R)
	xc := linspace(0, 1, C)
	a, b, c := 0.25, -0.5, 1.25
	W := make([]float32, R*C)
	for i := 0; i < R; i++ {
		for j := 0; j < C; j++ {
			W[i*C+j] = float32(a + b*xr[i] + c*xc[j])
		}
	}
	out := splineGrid(W, R, C, xr, xc, G, 0)
	for g := 0; g < G[0]; g++ {
		for h := 0; h < G[1]; h++ {
			want := a + b*(float64(g)+0.5)/float64(G[0]) + c*(float64(h)+0.5)/float64(G[1])
			if math.Abs(out[g*G[1]+h]-want) > 1e-5 {
				t.Fatalf("cell %d,%d: got %g want %g", g, h, out[g*G[1]+h], want)
			}
		}
	}
}

// The hat's RMS norm on centered samples is √(ρ/2) at an even integer ratio. The smoothed
// 1-form stays under that. fec.md §4.2 recorded a still smaller number on the measurement
// machine; this locks the comparison the hat explanation actually defines.
func TestFECBoundedAgainstHat(t *testing.T) {
	for _, tc := range []struct {
		n, G int
		hat  float64
	}{
		{480, 48, math.Sqrt(5)},
		{512, 32, math.Sqrt(8)},
	} {
		xs := make([]float64, tc.n)
		for i := range xs {
			xs[i] = (float64(i) + 0.5) / float64(tc.n)
		}
		geom := newAxisGeom(xs, tc.G)
		boxF := discBoxNorm(tc.n, tc.G)
		sm := axisFrobenius(tc.n, func(u []float64) []float64 { return geom.project(u) })
		hat := axisFrobenius(tc.n, func(u []float64) []float64 { return hatAxis(xs, u, tc.G) })
		if math.Abs(hat/boxF-tc.hat) > 1e-6 {
			t.Fatalf("%d→%d hat norm %.6f, want %.6f", tc.n, tc.G, hat/boxF, tc.hat)
		}
		if !(sm < hat) {
			t.Fatalf("%d→%d smoothed Frobenius %g is not under the hat %g", tc.n, tc.G, sm, hat)
		}
	}
}

// At a rational ratio p:q (gcd 1) the hat matrix is identical q×p blocks B, so its RMS norm is
// √(p/q)·σmax(B) at every axis length kp→kq. Both odd: σmax = 1. p/q ≥ 2 otherwise:
// σmax² = 1 − 1/q + 1/(2q²) (q = 1 is the √(ρ/2) above). 3:2 is 33/32 squared, fec.md §4.2's 1.016.
// Between 1 and 2 with p, q not both odd there is no closed form. Ends never clamp when ρ ≥ 1.
func TestHatNormRational(t *testing.T) {
	sig2 := func(q float64) float64 { return 1 - 1/q + 1/(2*q*q) }
	for _, tc := range []struct {
		p, q  int
		norm2 float64
	}{
		{3, 2, 33.0 / 32},
		{3, 1, 3},
		{9, 1, 9},
		{5, 3, 5.0 / 3},
		{7, 5, 7.0 / 5},
		{10, 1, 10 * sig2(1)},
		{16, 1, 16 * sig2(1)},
		{5, 2, 2.5 * sig2(2)},
		{7, 3, 7.0 / 3},
		{8, 3, 8.0 / 3 * sig2(3)},
		{9, 4, 9.0 / 4 * sig2(4)},
	} {
		for _, k := range []int{4, 7} {
			n, G := k*tc.p, k*tc.q
			got := hatSpectralRMS(n, G)
			if math.Abs(got*got-tc.norm2) > 1e-9 {
				t.Fatalf("%d:%d (%d→%d) hat RMS norm² %.12f, want %.12f", tc.p, tc.q, n, G, got*got, tc.norm2)
			}
		}
	}
}

// √(n/G)·σmax of hatAxis on centered samples, by power iteration on A·Aᵀ.
func hatSpectralRMS(n, G int) float64 {
	xs := make([]float64, n)
	for i := range xs {
		xs[i] = (float64(i) + 0.5) / float64(n)
	}
	A := make([][]float64, n) // A[i] = column i
	e := make([]float64, n)
	for i := range A {
		e[i] = 1
		A[i] = hatAxis(xs, e, G)
		e[i] = 0
	}
	v := constant(G, 1)
	v[0] = 2 // off the symmetric start
	var lam float64
	for it := 0; it < 2000; it++ {
		w := make([]float64, G)
		for i := range A {
			var d float64
			for g, a := range A[i] {
				d += a * v[g]
			}
			for g, a := range A[i] {
				w[g] += a * d
			}
		}
		var s float64
		for _, x := range w {
			s += x * x
		}
		s = math.Sqrt(s)
		var vv float64
		for _, x := range v {
			vv += x * x
		}
		lam = s / math.Sqrt(vv)
		for g := range w {
			v[g] = w[g] / s
		}
	}
	return math.Sqrt(float64(n) / float64(G) * lam)
}

func constant(n int, v float64) []float64 {
	u := make([]float64, n)
	for i := range u {
		u[i] = v
	}
	return u
}

func discBoxNorm(n, G int) float64 {
	rho := float64(n) / float64(G)
	return math.Sqrt(float64(G) / rho)
}

func axisFrobenius(n int, apply func([]float64) []float64) float64 {
	e := make([]float64, n)
	var s float64
	for i := 0; i < n; i++ {
		e[i] = 1
		if i > 0 {
			e[i-1] = 0
		}
		for _, v := range apply(e) {
			s += v * v
		}
	}
	return math.Sqrt(s)
}

func hatAxis(xs, us []float64, G int) []float64 {
	out := make([]float64, G)
	for g := range out {
		out[g] = interp((float64(g)+0.5)/float64(G), xs, us)
	}
	return out
}

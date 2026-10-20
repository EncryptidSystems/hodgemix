package main

import (
	"math"
	"math/rand"
	"testing"
)

func TestMarriedRadices(t *testing.T) {
	got := MarriedRadices([]int{2560, 1024}, 4095) // lcm = 2^10 * 5 = 5120
	want := []int{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5}
	if len(got) != len(want) {
		t.Fatalf("%v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%v", got)
		}
	}
	got = MarriedRadices([]int{2560, 2880}, 4095) // lcm = 2^9 * 3^2 * 5 = 23040
	p := 1
	for _, x := range got {
		p *= x
	}
	if p != 23040 {
		t.Fatalf("lcm(2560,2880) radices %v multiply to %d", got, p)
	}
}

// One model married with itself keeps exactly its own Eq.(8) coordinates (self-mix stays exact).
func TestMarriedSelfIsOwnIndex(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, n := range []int{64, 1024, 2560} {
		k := make([]int, n)
		for i := range k {
			k[i] = rng.Intn(64) // many ties
		}
		own := chanCoords(k)
		mc, _, err := MarriedCoords([][]int{k, append([]int(nil), k...)}, keyUniverse)
		if err != nil {
			t.Fatal(err)
		}
		for m := 0; m < 2; m++ {
			for i := range k {
				if math.Abs(mc[m][i]-own[i]) > 1e-12 {
					t.Fatalf("n=%d model %d channel %d: married %.15f own %.15f", n, m, i, mc[m][i], own[i])
				}
			}
		}
	}
}

// Two models of different width on one axis: coordinates in (0,1), increasing with key in each
// model, lower keys before higher keys across models, equal keys sharing one span.
func TestMarriedTwoWidths(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	a, b := make([]int, 2560), make([]int, 1024)
	for i := range a {
		a[i] = rng.Intn(4096)
	}
	for i := range b {
		b[i] = 1024 + rng.Intn(2048) // b's keys sit in the middle of a's range
	}
	mc, r, err := MarriedCoords([][]int{a, b}, keyUniverse)
	if err != nil {
		t.Fatal(err)
	}
	if len(r) != 11 {
		t.Fatalf("radices %v", r)
	}
	for m, keys := range [][]int{a, b} {
		for i := range keys {
			x := mc[m][i]
			if x <= 0 || x >= 1 {
				t.Fatalf("model %d coord %g", m, x)
			}
			for j := range keys {
				if keys[i] < keys[j] && mc[m][i] >= mc[m][j] {
					t.Fatalf("model %d: key %d at %g not before key %d at %g", m, keys[i], mc[m][i], keys[j], mc[m][j])
				}
			}
		}
	}
	for i := range a {
		for j := range b {
			if a[i] < b[j] && mc[0][i] >= mc[1][j] {
				t.Fatalf("a key %d at %g not before b key %d at %g", a[i], mc[0][i], b[j], mc[1][j])
			}
			if a[i] > b[j] && mc[0][i] <= mc[1][j] {
				t.Fatalf("a key %d at %g not after b key %d at %g", a[i], mc[0][i], b[j], mc[1][j])
			}
		}
	}
	// b's channels occupy the middle of the married axis, not the whole of it as their own index would.
	lo, hi := 1.0, 0.0
	for _, x := range mc[1] {
		lo, hi = math.Min(lo, x), math.Max(hi, x)
	}
	if lo < 0.15 || hi > 0.85 {
		t.Fatalf("b spans [%.3f, %.3f] on the married axis", lo, hi)
	}
}

// run()'s married path for a self-mix: married coords -> splineGrid -> synth -> sampleGrid + base
// residual at the base's married coords gives back the base codes exactly.
func TestMarriedSelfMixIsIdentity(t *testing.T) {
	R, C := 320, 432
	rng := rand.New(rand.NewSource(11))
	W := make([]float32, R*C)
	for i := range W {
		W[i] = float32(rng.Intn(3)-1) * 1.25
	}
	s := absmean(W)
	norm := make([]float32, len(W))
	for i := range W {
		norm[i] = float32(float64(W[i]) / s)
	}
	kr, kc := heapKey(W, R, C, 0), heapKey(W, R, C, 1)
	mr, _, err := MarriedCoords([][]int{kr, kr}, keyUniverse)
	if err != nil {
		t.Fatal(err)
	}
	mc, _, err := MarriedCoords([][]int{kc, kc}, keyUniverse)
	if err != nil {
		t.Fatal(err)
	}
	G := [2]int{gridSizeFor(R, R), gridSizeFor(C, C)}
	sp := SynthParams{K: KParams{GTheta: math.Ldexp(1, -GThetaShift), Weak: math.Ldexp(1, -(GOmegaShift - GThetaShift)), WLock: WLock,
		NLost: NLost, RTol: RefineTol}, Lock: "dtft", D: DParams{SeqAxis: "auto", OmegaUpdate: true}}
	g0 := splineGrid(norm, R, C, mr[0], mc[0], G, 0)
	g1 := splineGrid(norm, R, C, mr[1], mc[1], G, 0)
	out, _ := synth([]Entry{{W: 1, Grid: g0, IsBase: true}, {W: 1, Grid: g1}}, G, sp)
	Wn := sampleGrid(out, G, mr[0], mc[0])
	sm := sampleGrid(g0, G, mr[0], mc[0])
	changed := 0
	var th float64
	for i := range Wn {
		Wn[i] += float64(norm[i]) - sm[i]
		th += math.Abs(Wn[i])
	}
	th /= float64(len(Wn))
	for i := range Wn {
		if ternQ(Wn[i], th) != ternQ(float64(norm[i]), 1) {
			changed++
		}
	}
	if changed != 0 {
		t.Fatalf("married self-mix changed %d of %d codes", changed, len(Wn))
	}
}

package main

import (
	"math"
	"testing"
)

// The merged TIES delta is normalised by every contributor's weight: one source at 1:1 gives half
// its delta (the earlier lambda), and the base given again as a source (zero delta) dilutes it to a
// third at equal weights.
func TestTIESAllWeights(t *testing.T) {
	d := []float64{1, -2, 1.5, -1}                           // all above half their absmean: none trimmed (Eq. 21)
	one := ternaryTIESAll([][]float64{d}, []float64{0.5}, 1) // base 0.5 + source 0.5
	zero := make([]float64, len(d))
	withBase := ternaryTIESAll([][]float64{d, zero}, []float64{1.0 / 3, 1.0 / 3}, 1) // base, source, base-as-source
	for i := range d {
		if math.Abs(one[i]-0.5*d[i]) > 1e-12 {
			t.Fatalf("one source: %v, want %v", one[i], 0.5*d[i])
		}
		if math.Abs(withBase[i]-d[i]/3) > 1e-12 {
			t.Fatalf("base as a source too: %v, want %v", withBase[i], d[i]/3)
		}
	}
	// Two sources that disagree in sign: the elected sign's deltas only, over the total weight.
	a, b := []float64{1, 1}, []float64{-1, 2}
	got := ternaryTIESAll([][]float64{a, b}, []float64{0.4, 0.2}, 1)
	if math.Abs(got[0]-0.4) > 1e-12 || math.Abs(got[1]-(0.4*1+0.2*2)) > 1e-12 {
		t.Fatalf("elected sign: %v", got)
	}
}

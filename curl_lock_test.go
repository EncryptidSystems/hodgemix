package main

import (
	"math"
	"testing"
)

// Gap selection on a source that shares no fibre with the base used to write
// the base over the source, and the lock then merged the base with itself.
func TestUncertifiedSourceIsKept(t *testing.T) {
	prevSeed, prevQuiet := SeedAlign, quiet
	SeedAlign, quiet = true, true
	t.Cleanup(func() { SeedAlign, quiet = prevSeed, prevQuiet })

	const R, C = 15, 15
	G := [2]int{R, C}
	ref := make([]float64, R*C)
	src := make([]float64, len(ref))
	for r := 0; r < R; r++ {
		for c := 0; c < C; c++ {
			src[r*C+c] = 0.01 * math.Sin(2*math.Pi*float64(r)/float64(R)) * math.Sin(2*math.Pi*float64(c)/float64(C))
		}
	}
	sp := SynthParams{
		K: KParams{GTheta: math.Ldexp(1, -GThetaShift), WLock: WLock, NLost: NLost, RTol: RefineTol},
		D: DParams{SeqAxis: "auto", OmegaUpdate: true},
	}
	out, _ := synthD([]Entry{{W: 1, Grid: ref, IsBase: true}, {W: 1, Grid: src}}, G, sp)
	var drift float64
	for i := range out {
		drift += math.Abs(out[i] - ref[i])
	}
	if drift == 0 {
		t.Fatal("uncertified source was replaced by the base")
	}
}

// A smooth source that fails the band lock used to be replaced by the base
// before the Ricci–DeTurck step, so the Hardy neck never saw the curl.
func TestUnlockedCurlReachesHardy(t *testing.T) {
	prevSeed, prevQuiet := SeedAlign, quiet
	SeedAlign, quiet = false, true
	t.Cleanup(func() { SeedAlign, quiet = prevSeed, prevQuiet })

	sp := SynthParams{
		K: KParams{GTheta: math.Ldexp(1, -GThetaShift), WLock: WLock, NLost: NLost, RTol: RefineTol},
		D: DParams{SeqAxis: "auto", OmegaUpdate: true},
	}
	wave := func(G [2]int) (ref, src []float64) {
		ref = make([]float64, G[0]*G[1])
		src = make([]float64, len(ref))
		for r := 0; r < G[0]; r++ {
			for c := 0; c < G[1]; c++ {
				src[r*G[1]+c] = 0.01 * math.Sin(2*math.Pi*float64(r)/float64(G[0])) * math.Sin(2*math.Pi*float64(c)/float64(G[1]))
			}
		}
		return ref, src
	}
	for _, G := range [][2]int{{15, 15}, {16, 16}} {
		ref, src := wave(G)
		out, rec := synthD([]Entry{{W: 1, Grid: ref, IsBase: true}, {W: 1, Grid: src}}, G, sp)
		if rec.D == nil {
			t.Fatalf("grid %v: no D record", G)
		}
		if rec.D.Merged != 0 {
			t.Fatalf("grid %v: mode locked (merged %d); this case is the unlocked path", G, rec.D.Merged)
		}
		var drift float64
		for i := range out {
			drift += math.Abs(out[i] - ref[i])
		}
		if drift == 0 {
			t.Fatalf("grid %v: unlocked curl was replaced by the base", G)
		}
	}
}

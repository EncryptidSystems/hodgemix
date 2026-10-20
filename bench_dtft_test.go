package main

import (
	"math"
	"math/rand"
	"os"
	"testing"
	"time"
)

// timing of Algorithm D on 2B4T-sized index grids (synthetic grids only; no model files)
func TestDtftScaling(t *testing.T) {
	if os.Getenv("HM_BENCH") == "" {
		t.Skip()
	}
	sp := SynthParams{K: KParams{GTheta: math.Ldexp(1, -4), Weak: math.Ldexp(1, -4), WLock: WLock, NLost: NLost, RTol: RefineTol},
		D: DParams{SeqAxis: "auto", OmegaUpdate: true, Persist: os.Getenv("HM_NOPERS") == ""}}
	for _, G := range [][2]int{{160, 160}, {216, 160}} {
		g := rand.New(rand.NewSource(1))
		B := synthBase(G, g)
		ents := []Entry{{W: 1, Grid: B, IsBase: true}, {W: 1, Grid: ternNoise(B, g)}, {W: 1, Grid: permuted(B, G, g)}}
		t0 := time.Now()
		_, rec := synthD(ents, G, sp)
		t.Logf("grid %v: %.2fs persistence %.2fs (budget hit %v) band=%d", G, time.Since(t0).Seconds(), rec.D.Pers.Secs, rec.D.Pers.Budget, rec.D.BandModes)
	}
}

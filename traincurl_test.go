package main

import (
	"math"
	"math/rand"
	"testing"

	"hodgemix/train"
)

func separatedRef(rows, cols int) []float64 {
	ref := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			ref[r*cols+c] = float64((r+1)*10000 + (c + 1))
		}
	}
	return ref
}

func add(a, b []float64, scale float64) []float64 {
	o := make([]float64, len(a))
	for i := range a {
		o[i] = a[i] + scale*b[i]
	}
	return o
}

func maxDiff(a, b []float64) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

func TestTrainCurlLeavesGauge(t *testing.T) {
	const rows, cols = 8, 8
	ref := separatedRef(rows, cols)
	g := rand.New(rand.NewSource(1))
	raw := make([]float64, rows*cols)
	for i := range raw {
		raw[i] = g.NormFloat64()
	}
	gauge := gradOf(raw, rows, cols)
	src := add(ref, gauge, 1)
	got := TrainCurl(src, ref, rows, cols)
	if d := maxDiff(got.Grid, ref); d > 1e-8 {
		t.Fatalf("gauge moved the reference by %g", d)
	}
}

func TestFlowGridKeepsHarmonic(t *testing.T) {
	const rows, cols = 8, 8
	ref := make([]float64, rows*cols)
	g := rand.New(rand.NewSource(3))
	raw := make([]float64, len(ref))
	for i := range raw {
		raw[i] = g.NormFloat64()
	}
	S := NewSpectrum([2]int{rows, cols}, 0)
	p := S.Hodge(raw)
	harm := realPart(fftn(p.HarmF, S.dims, true))
	out := flowGrid(harm, ref, [2]int{rows, cols})
	if d := maxDiff(out, harm); d > 1e-8 {
		t.Fatalf("harmonic moved by %g", d)
	}
}

func TestTrainCurlDescendsCurl(t *testing.T) {
	const rows, cols = 8, 8
	ref := separatedRef(rows, cols)
	g := rand.New(rand.NewSource(2))
	raw := make([]float64, rows*cols)
	for i := range raw {
		raw[i] = g.NormFloat64()
	}
	curl := curlOf(raw, rows, cols)
	const scale = 1e-3
	src := add(ref, curl, scale)
	before := make([]float64, len(curl))
	for i := range before {
		before[i] = scale * curl[i]
	}
	e0 := hardyEnergy(before, rows, cols)
	got := TrainCurl(src, ref, rows, cols)
	residual := make([]float64, len(ref))
	for i := range residual {
		residual[i] = got.Grid[i] - ref[i]
	}
	e1 := hardyEnergy(residual, rows, cols)
	if !(e1 < e0) {
		t.Fatalf("energy %g -> %g, neck %d", e0, e1, got.Neck)
	}
	if got.DT != train.DT {
		t.Fatalf("step %v", got.DT)
	}
}

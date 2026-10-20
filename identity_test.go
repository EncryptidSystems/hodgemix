package main

import (
	"math"
	"math/rand"
	"testing"
)

// A base mixed with an exact copy of itself must come back unchanged: same ternary codes, same
// weight_scale. Builds the entries the way run() does for one tensor and follows the same path
// (synth -> sampleGrid + base residual -> absmean codes).
func TestSelfMixIsIdentity(t *testing.T) {
	R, C := 320, 432
	rng := rand.New(rand.NewSource(1))
	sc := float32(1.5)
	W := make([]float32, R*C)
	for i := range W {
		W[i] = float32(rng.Intn(3)-1) * sc
	}
	G := [2]int{gridSize(R, 0), gridSize(C, 0)}
	s := absmean(W)
	xr, xc := chanCoords(heapKey(W, R, C, 0)), chanCoords(heapKey(W, R, C, 1))
	norm := make([]float32, len(W))
	for i := range W {
		norm[i] = float32(float64(W[i]) / s)
	}
	grid := splineGrid(norm, R, C, xr, xc, G, 0)
	g2 := append([]float64(nil), grid...)
	sp := SynthParams{K: KParams{GTheta: math.Ldexp(1, -GThetaShift), Weak: math.Ldexp(1, -(GOmegaShift - GThetaShift)), WLock: WLock,
		NLost: NLost, RTol: RefineTol}, Lock: "dtft", D: DParams{SeqAxis: "auto", OmegaUpdate: true, Persist: false}}
	out, _ := synth([]Entry{{W: 1, Grid: grid, IsBase: true}, {W: 1, Grid: g2}}, G, sp)
	var dg float64
	for i := range out {
		dg = math.Max(dg, math.Abs(out[i]-grid[i]))
	}
	t.Logf("grid %v: max |synth - base grid| = %.3g", G, dg)
	Wn := sampleGrid(out, G, xr, xc)
	sm := sampleGrid(splineGrid(norm, R, C, xr, xc, G, 0), G, xr, xc)
	var dw float64
	changed := 0
	for i := range Wn {
		Wn[i] += float64(norm[i]) - sm[i]
		dw = math.Max(dw, math.Abs(Wn[i]-float64(norm[i])))
	}
	Wf := make([]float32, len(Wn))
	for i := range Wn {
		Wf[i] = float32(Wn[i] * s)
	}
	th := absmean(Wf)
	for i, v := range Wf {
		if ternQ(float64(v), th) != ternQ(float64(W[i]), s) {
			changed++
		}
	}
	t.Logf("max |W_new - W_base| (normalized) = %.3g, codes changed %d/%d", dw, changed, len(W))
	if dg > 1e-6 || changed > 0 {
		t.Fatalf("self-mix is not the identity")
	}
}

// Same check on a real 2B4T tensor, through OpenST/LoadDense like run(). Skipped without the checkpoint.
func TestSelfMixRealTensor(t *testing.T) {
	p := `C:\Users\grego\models\microsoft\bitnet-b1.58-2B-4T\model.safetensors`
	st, err := OpenST(p)
	if err != nil {
		t.Skip(err)
	}
	name := "model.layers.0.self_attn.q_proj.weight"
	W, sh, err := st.LoadDense(name)
	if err != nil {
		t.Fatal(err)
	}
	R, C := sh[0], sh[1]
	G := [2]int{gridSize(R, 0), gridSize(C, 0)}
	s := absmean(W)
	xr, xc := chanCoords(heapKey(W, R, C, 0)), chanCoords(heapKey(W, R, C, 1))
	norm := make([]float32, len(W))
	for i := range W {
		norm[i] = float32(float64(W[i]) / s)
	}
	grid := splineGrid(norm, R, C, xr, xc, G, 0)
	g2 := splineGrid(norm, R, C, xr, xc, G, 0)
	sp := SynthParams{K: KParams{GTheta: math.Ldexp(1, -GThetaShift), Weak: math.Ldexp(1, -(GOmegaShift - GThetaShift)), WLock: WLock,
		NLost: NLost, RTol: RefineTol}, Lock: "dtft", D: DParams{SeqAxis: "auto", OmegaUpdate: true, Persist: true}}
	out, _ := synth([]Entry{{W: 1, Grid: grid, IsBase: true}, {W: 1, Grid: g2}}, G, sp)
	var dg float64
	for i := range out {
		dg = math.Max(dg, math.Abs(out[i]-grid[i]))
	}
	Wn := sampleGrid(out, G, xr, xc)
	sm := sampleGrid(splineGrid(norm, R, C, xr, xc, G, 0), G, xr, xc)
	changed := 0
	th := 0.0
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
	t.Logf("%s %v grid %v: max |synth - base grid| = %.3g, codes changed %d/%d", name, sh, G, dg, changed, len(W))
	if dg > 1e-6 || changed > 0 {
		t.Fatalf("self-mix is not the identity")
	}
}

// Equal-depth models: a target layer takes only the same layer of each source, never a neighbour.
func TestDepthHatEqualDepth(t *testing.T) {
	for _, L := range []int{2, 24, 28, 30, 32, 40} {
		for lt := 0; lt < L; lt++ {
			ut := float64(lt) / float64(L-1)
			for ls := 0; ls < L; ls++ {
				us := float64(ls) / float64(L-1)
				w := depthHat(us, ut, L)
				if ls == lt && math.Abs(w-1) > 1e-12 || ls != lt && w != 0 {
					t.Fatalf("L=%d target %d source %d: weight %g", L, lt, ls, w)
				}
			}
		}
	}
}

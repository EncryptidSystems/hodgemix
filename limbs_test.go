package main

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestLlama3NeckBudget(t *testing.T) {
	llama := map[string]any{"model_type": "llama", "rope_scaling": map[string]any{"rope_type": "llama3"}}
	bitnet := map[string]any{"model_type": "bitnet"}
	champ := map[string]any{"model_type": "bitnet", "rope_scaling": map[string]any{"rope_type": "llama3", "factor": 32.0}}
	moved := map[string]any{"model_type": "llama", "rope_parameters": map[string]any{"rope_type": "llama3", "factor": 32.0}}
	if !llama3Arch(llama) || llama3Arch(bitnet) || !llama3Arch(champ) || !llama3Arch(moved) {
		t.Fatalf("llama3 %v bitnet %v champ %v moved %v", llama3Arch(llama), llama3Arch(bitnet), llama3Arch(champ), llama3Arch(moved))
	}
	n, tol := llama3Budget(1, 0.15)
	if n != Llama3Limbs || tol != Llama3Tol {
		t.Fatalf("default llama3 budget %d %g", n, tol)
	}
	n, tol = llama3Budget(8, 0.01)
	if n != 8 || tol != 0.01 {
		t.Fatalf("explicit budget %d %g", n, tol)
	}
	n, tol = llama3Budget(1, 0.15)
	if n == 1 {
		t.Fatal("llama3 stayed at one limb")
	}
}

func TestAnnealStopsWhenRStalls(t *testing.T) {
	ticks, _ := runAnnealStage(NLost, RefineTol, 1, func(float64) {
		t.Fatal("a synced stage ticked")
	}, func() float64 { return 1 })
	if ticks != 0 {
		t.Fatalf("synced stage used %d ticks", ticks)
	}
	calls := 0
	r := 0.0
	ticks, got := runAnnealStage(NLost, RefineTol, 0.2, func(float64) { calls++ }, func() float64 {
		if calls < 5 {
			r += 0.1
		}
		return r
	})
	if ticks >= NLost || calls >= NLost {
		t.Fatalf("spent the budget: ticks %d calls %d r %.3f", ticks, calls, got)
	}
	if ticks < 5 {
		t.Fatalf("stopped before r rose: %d", ticks)
	}
	flat := 0
	ticks, _ = runAnnealStage(NLost, RefineTol, 0.4, func(float64) { flat++ }, func() float64 { return 0.4 })
	if flat != annealStall {
		t.Fatalf("a flat stage used %d ticks", flat)
	}
	climb := 0
	r = 0.2
	ticks, _ = runAnnealStage(NLost, RefineTol, 0.2, func(float64) { climb++ }, func() float64 {
		r += 1e-4
		return r
	})
	if climb != annealStall {
		t.Fatalf("a creep slower than rtol used %d ticks", climb)
	}
}

func TestWideNeckNotCoded(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	R, C := 32, 40
	W := make([]float32, R*C)
	for i := range W {
		W[i] = float32(rng.NormFloat64())
	}
	am := absmean(W)
	for i := 0; i < len(W)/2; i++ {
		W[i] = float32(50 * am)
	}
	q0, s0 := ternarize(W)
	limbs, _ := residualLimbs(W, q0, s0, 1, 0.15, R, C)
	if len(limbs) != 0 {
		t.Fatalf("a neck covering half the tensor was written: %d limbs", len(limbs))
	}
}

// A limb the residual barely notices is not written. The loader accumulates every limb.
func TestLimbDustStops(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	R, C := 16, 20
	W := make([]float32, R*C)
	for i := range W {
		W[i] = float32(rng.NormFloat64())
	}
	q0, s0 := ternarize(W)
	_, rel := residualLimbs(W, q0, s0, 8, 0, R, C)
	for i := 1; i < len(rel); i++ {
		if rel[i-1]-rel[i] < RefineTol {
			t.Fatalf("kept a limb that moved the residual by %.3g: %v", rel[i-1]-rel[i], rel)
		}
	}
}

// Each limb lowers the residual; limbs written to a file read back, and LoadDense is their sum.
func TestLimbs(t *testing.T) {
	for _, n := range []string{"a.weight_limb1", "a.weight_limb12_scale", "a.weight_limb3_radix"} {
		if !isLimbTensor(n) {
			t.Fatalf("%s not a limb tensor", n)
		}
	}
	for _, n := range []string{"a.weight", "a.weight_scale", "a.weight_limb", "a.weight_limbx"} {
		if isLimbTensor(n) {
			t.Fatalf("%s taken for a limb tensor", n)
		}
	}
	rng := rand.New(rand.NewSource(3))
	R, C := 16, 20
	W := make([]float32, R*C)
	for i := range W {
		W[i] = float32(rng.NormFloat64())
	}
	q0, s0 := ternarize(W)
	limbs, rel := residualLimbs(W, q0, s0, 4, 0, R, C)
	if len(limbs) < 3 || len(rel) != len(limbs)+1 {
		t.Fatalf("limbs %d, residuals %v", len(limbs), rel)
	}
	for i := 1; i < len(rel); i++ {
		if rel[i] >= rel[i-1] {
			t.Fatalf("residual did not fall %v", rel)
		}
	}
	dir := tempDir(t)
	p := filepath.Join(dir, "m.safetensors")
	sw, err := NewSTW(p, nil)
	if err != nil {
		t.Fatal(err)
	}
	sw.Add("x.weight", "U8", []int{R / 4, C}, packTrits(q0, R, C))
	sw.AddBF16("x.weight_scale", []int{1}, []float32{float32(s0)})
	writeLimbs(sw, "x.weight", R, C, limbs)
	if err := sw.Close(nil); err != nil {
		t.Fatal(err)
	}
	st, err := OpenST(p)
	if err != nil {
		t.Fatal(err)
	}
	got := st.limbsOf("x.weight")
	if len(got) != len(limbs) {
		t.Fatalf("read %d limbs, wrote %d", len(got), len(limbs))
	}
	D, _, err := st.LoadDense("x.weight")
	if err != nil {
		t.Fatal(err)
	}
	var e, w float64
	for i := range W {
		e += (float64(W[i]) - float64(D[i])) * (float64(W[i]) - float64(D[i]))
		w += float64(W[i]) * float64(W[i])
	}
	if r := math.Sqrt(e / w); math.Abs(r-rel[len(rel)-1]) > 0.02 { // BF16 scales round
		t.Fatalf("LoadDense residual %.4f, limbs said %.4f", r, rel[len(rel)-1])
	}
}

// T1 thresholds quoted in singularity surgery §5.
func TestT1Threshold(t *testing.T) {
	if d := math.Abs(t1Tau(2560) - 7.67); d > 0.02 {
		t.Fatalf("tau 2560 = %.4f", t1Tau(2560))
	}
	if d := math.Abs(t1Tau(6912) - 7.95); d > 0.02 {
		t.Fatalf("tau 6912 = %.4f", t1Tau(6912))
	}
}

// A spike thousands of times the body absmean is a neck. It does not set a body limb's scale.
// Body limbs stay at center on it. A later neck limb carries it, and is center on the body.
func TestSingularNeck(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	R, C := 16, 20
	body := make([]float32, R*C)
	for i := range body {
		body[i] = float32(rng.NormFloat64())
	}
	am := absmean(body)
	W := append([]float32(nil), body...)
	W[0] = float32(1000 * am)
	q0, s0 := ternarize(W)
	limbs, _ := residualLimbs(W, q0, s0, 4, 0, R, C)
	qb, sb := ternarize(body)
	limbsB, _ := residualLimbs(body, qb, sb, 4, 0, R, C)
	if len(limbs) == 0 || len(limbsB) == 0 {
		t.Fatalf("limbs %d, body limbs %d", len(limbs), len(limbsB))
	}
	if limbs[0].s > 4*limbsB[0].s {
		t.Fatalf("spike set body-limb scale %.4g, body limb %.4g, absmean %.4g", limbs[0].s, limbsB[0].s, am)
	}
	gap := make([]float32, len(W))
	for i := range W {
		gap[i] = W[i] - float32(s0*float64(q0[i]))
	}
	sing := singularNeck(gap)
	coded := false
	for i, lb := range limbs {
		cen := (lb.r - 1) / 2
		neck, body := false, false
		for j, d := range lb.d {
			if int(d) == cen {
				continue
			}
			if sing != nil && sing[j] {
				neck = true
			} else {
				body = true
			}
		}
		if neck && body {
			t.Fatalf("limb %d codes both the neck and the body", i)
		}
		if neck {
			coded = true
		}
	}
	if !coded {
		t.Fatal("no limb carries the neck")
	}
	// Limb 0 still clips the neck to one trit (surgery §7). The magnitude lives on the neck limb.
	if q0[0] != 1 && q0[0] != -1 {
		t.Fatalf("neck code %d, scale %.4g, absmean %.4g", q0[0], s0, am)
	}
}

// --limbs 1 is the default body budget. The neck is still coded, and no body limb is added.
func TestNeckAtOneLimb(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	R, C := 16, 20
	body := make([]float32, R*C)
	for i := range body {
		body[i] = float32(rng.NormFloat64())
	}
	am := absmean(body)
	W := append([]float32(nil), body...)
	W[0] = float32(1000 * am)
	q0, s0 := ternarize(W)
	limbs, rel := residualLimbs(W, q0, s0, 1, 0.15, R, C)
	if len(limbs) == 0 || len(limbs) > NeckLimbs || len(rel) != len(limbs)+1 {
		t.Fatalf("neck limbs %d rel %v", len(limbs), rel)
	}
	gap := make([]float32, len(W))
	for i := range W {
		gap[i] = W[i] - float32(s0*float64(q0[i]))
	}
	sing := singularNeck(gap)
	coded := false
	for i, lb := range limbs {
		cen := (lb.r - 1) / 2
		for j, d := range lb.d {
			if int(d) == cen {
				continue
			}
			if sing == nil || !sing[j] {
				t.Fatalf("limb %d codes a body entry at %d", i, j)
			}
			coded = true
		}
	}
	if !coded {
		t.Fatal("no neck limb carries the spike")
	}
}

// The OV chart refills o_proj in the limb slots the ternary file already has.
// A one-limb body pack stores only the neck. Refitting that weight wants body
// limbs plus another neck, and writing the extra limb used to panic.
func TestRefillLimbsStaysInSlots(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	R, C := 16, 20
	body := make([]float32, R*C)
	for i := range body {
		body[i] = float32(rng.NormFloat64())
	}
	am := absmean(body)
	W := append([]float32(nil), body...)
	W[0] = float32(1000 * am)
	q0, s0 := ternarize(W)
	neck, _ := residualLimbs(W, q0, s0, 1, 0.15, R, C)
	if len(neck) == 0 {
		t.Fatal("no neck slots")
	}
	dir := tempDir(t)
	p := filepath.Join(dir, "m.safetensors")
	sw, err := NewSTW(p, nil)
	if err != nil {
		t.Fatal(err)
	}
	sw.Add("x.weight", "U8", []int{R / 4, C}, packTrits(q0, R, C))
	sw.AddBF16("x.weight_scale", []int{1}, []float32{float32(s0)})
	writeLimbs(sw, "x.weight", R, C, neck)
	if err := sw.Close(nil); err != nil {
		t.Fatal(err)
	}
	Ot := append([]float32(nil), W...)
	for i := range Ot {
		Ot[i] += float32(rng.NormFloat64()) * 0.25
	}
	Ot[0] = float32(1200 * am)
	q, ws := ternarize(Ot)
	want, _ := residualLimbs(Ot, q, ws, len(neck)+1, 0, R, C)
	if len(want) <= len(neck) {
		t.Fatalf("refit produced %d limbs, file has %d", len(want), len(neck))
	}
	st, err := OpenST(p)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(p, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := refillLimbs(f, st, "x.weight", Ot, q, ws); err != nil {
		t.Fatal(err)
	}
	got := st.limbsOf("x.weight")
	if len(got) != len(neck) {
		t.Fatalf("slots %d after refill, file had %d", len(got), len(neck))
	}
	if st.Has("x.weight_limb" + strconv.Itoa(len(neck)+1)) {
		t.Fatal("refill wrote a limb the file did not have")
	}
	D, _, err := st.LoadDense("x.weight")
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range D {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("refilled weight is not finite")
		}
	}
}

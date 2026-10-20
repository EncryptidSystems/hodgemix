package main

import (
	"math"
	"math/rand"
	"testing"
)

// The RoPE marriage is the identity at equal theta, and maps pair d to the contributor's pair at
// d ln(tt)/ln(tc) in both rotate-half halves.
func TestRopeRemap(t *testing.T) {
	heads, hd, D := 2, 8, 3
	W := make([]float32, heads*hd*D)
	for r := 0; r < heads*hd; r++ {
		for c := 0; c < D; c++ {
			W[r*D+c] = float32(r%hd) + float32(c)/10
		}
	}
	same := ropeRemap(W, heads, hd, D, 5e5, 5e5)
	for i := range W {
		if math.Abs(float64(same[i]-W[i])) > 1e-6 {
			t.Fatalf("equal theta changed value %d", i)
		}
	}
	out := ropeRemap(W, heads, hd, D, 1e6, 5e5)
	ratio := math.Log(5e5) / math.Log(1e6)
	for h := 0; h < heads; h++ {
		for d := 0; d < hd/2; d++ {
			for _, off := range []int{0, hd / 2} {
				got := float64(out[(h*hd+off+d)*D])
				want := float64(off) + math.Min(float64(d)*ratio, float64(hd/2-1))
				if math.Abs(got-want) > 1e-5 {
					t.Fatalf("head %d pair %d half %d: %g want %g", h, d, off, got, want)
				}
			}
		}
	}
}

// Folding QK-norm leaves every head at the unit rms QK-norm gives: ||W_h||_F = sqrt(hd) (unit gains).
func TestQKNormFold(t *testing.T) {
	tp := &topology{metas: []modelMeta{{HD: 4, Theta: 5e5}, {HD: 4, Theta: 5e5, QKNorm: true}}, tgt: modelMeta{HD: 4, Theta: 5e5}}
	st := &ST{H: map[string]TInfo{}}
	rng := rand.New(rand.NewSource(1))
	heads, hd, D := 3, 4, 10
	W := make([]float32, heads*hd*D)
	for i := range W {
		W[i] = float32(rng.NormFloat64()) * float32(1+i/(hd*D)) // heads of different size
	}
	out, notes := tp.prepQK(1, st, "model.layers.0.self_attn.q_proj.weight", "q", W, []int{heads * hd, D})
	if len(notes) != 1 {
		t.Fatalf("notes %v", notes)
	}
	for h := 0; h < heads; h++ {
		var f float64
		for i := h * hd * D; i < (h+1)*hd*D; i++ {
			f += float64(out[i]) * float64(out[i])
		}
		if math.Abs(math.Sqrt(f)-math.Sqrt(float64(hd))) > 1e-4 {
			t.Fatalf("head %d: ||W_h||_F %g, want %g", h, math.Sqrt(f), math.Sqrt(float64(hd)))
		}
	}
	// The base itself (no QK-norm, same theta) is left alone.
	if same, notes := tp.prepQK(0, st, "model.layers.0.self_attn.q_proj.weight", "q", W, []int{heads * hd, D}); notes != nil || &same[0] != &W[0] {
		t.Fatalf("base q changed: %v", notes)
	}
}

// The logit spread scales with q and k, and is the same for every head-sampling step on equal heads.
func TestHeadSigma(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	H, KV, hd, D := 4, 2, 4, 16
	q := make([]float64, H*hd*D)
	k := make([]float64, KV*hd*D)
	for i := range q {
		q[i] = rng.NormFloat64()
	}
	for i := range k {
		k[i] = rng.NormFloat64()
	}
	at := func(m []float64, s float64) func(r, c int) float64 { return func(r, c int) float64 { return s * m[r*D+c] } }
	s1 := headSigma(at(q, 1), at(k, 1), H, KV, hd, D)
	s6 := headSigma(at(q, 2), at(k, 3), H, KV, hd, D)
	if s1 <= 0 || math.Abs(s6/s1-6) > 1e-9 {
		t.Fatalf("sigma %g, scaled %g (want x6)", s1, s6)
	}
}

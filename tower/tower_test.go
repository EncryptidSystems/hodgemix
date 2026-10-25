package tower

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestLAFixedPoint(t *testing.T) {
	if !Ready() {
		t.Fatal("L-A tower is not ready")
	}
	sig, lam := Sigma(), Lambda()
	// tower_check output for z* = 1.6i.
	if d := cmplx.Abs(sig - (4.533563 - 1.992497i)); d > 1e-5 {
		t.Fatalf("sigma %v, published 4.533563-1.992497i, |diff| %g", sig, d)
	}
	if d := cmplx.Abs(lam - (-0.022719 - 0.132388i)); d > 1e-5 {
		t.Fatalf("lambda %v, published -0.022719-0.132388i, |diff| %g", lam, d)
	}
	if a := cmplx.Abs(lam); math.Abs(a-0.1343) > 1e-3 {
		t.Fatalf("|lambda| %g, published 0.1343", a)
	}
	f, _, ok := Operator().Eval(ZStar)
	if !ok {
		t.Fatal("Eval at z* failed")
	}
	if cmplx.Abs(f-ZStar) > 1e-9 {
		t.Fatalf("|T(z*)-z*| %g", cmplx.Abs(f-ZStar))
	}
}

func TestKoenigsEmbed(t *testing.T) {
	r, ok := Embed(ZStar)
	if !ok {
		t.Fatal("Embed at z* failed")
	}
	if cmplx.Abs(r) > 1e-6 {
		t.Fatalf("|R(z*)| %g", cmplx.Abs(r))
	}
	z := ZStar + 1e-6
	got := Micro(z, 1)
	f, _, ok := Operator().Eval(z)
	if !ok {
		t.Fatal("Eval near z* failed")
	}
	if cmplx.Abs(got-f) > 1e-9 {
		t.Fatalf("micro at alpha 1 differs from one layer by %g", cmplx.Abs(got-f))
	}
}

func TestMicroNano(t *testing.T) {
	z := ZStar + complex(0.02, -0.01)
	a := Micro(Micro(z, 0.25), 0.5)
	b := Micro(z, 0.75)
	if cmplx.Abs(a-b) > 1e-12 {
		t.Fatalf("micro semigroup |diff| %g", cmplx.Abs(a-b))
	}
	eps := 1e-4
	step := (Nano(z, eps) - z) / complex(eps, 0)
	want := Generator() * (z - ZStar)
	if cmplx.Abs(step-want) > 1e-9 {
		t.Fatalf("nano generator |diff| %g", cmplx.Abs(step-want))
	}
	if Micro(ZStar, 0.3) != ZStar {
		t.Fatal("micro moves the fixed point")
	}
}

func TestVisionPatch(t *testing.T) {
	if Patches(VisSide, VisSide) != VisSlots {
		t.Fatalf("224 patch count %d", Patches(VisSide, VisSide))
	}
	f, phi, err := VisionRaw(probeRGB(0), Patch, Patch)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 1 || len(f[0]) != VisFeat || len(phi) != 1 {
		t.Fatalf("got %d vectors width %d", len(f), len(f[0]))
	}
	for i, v := range f[0] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("feature %d not finite", i)
		}
	}
}

func TestVisionImage(t *testing.T) {
	rgb := make([]float32, 8*8*3)
	for i := range rgb {
		rgb[i] = float32(i%7) / 7
	}
	f, phi, err := Vision(rgb, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != VisSlots || len(phi) != VisSlots {
		t.Fatalf("tokens %d", len(f))
	}
	if len(f[0]) != VisFeat || len(f[100]) != VisFeat {
		t.Fatalf("width %d", len(f[0]))
	}
}

func TestResize(t *testing.T) {
	src := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}
	dst := resizeRGB(src, 2, 2, 4, 4)
	if len(dst) != 48 {
		t.Fatalf("resized length %d", len(dst))
	}
	if dst[0] != 1 || dst[1] != 0 || dst[2] != 0 {
		t.Fatalf("origin pixel %v", dst[:3])
	}
}

func TestAudioToken(t *testing.T) {
	if Tokens(SampleRate*30) != 1500 {
		t.Fatalf("30s tokens %d, want 1500", Tokens(SampleRate*30))
	}
	pcm := make([]float32, 3200)
	for i := range pcm {
		pcm[i] = float32(0.2 * math.Sin(2*math.Pi*440*float64(i)/SampleRate))
	}
	f, phi, err := Audio(pcm)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != Tokens(len(pcm)) || len(phi) != len(f) {
		t.Fatalf("tokens %d, want %d", len(f), Tokens(len(pcm)))
	}
	if len(f[0]) != AudFeat {
		t.Fatalf("width %d", len(f[0]))
	}
	for i, v := range f[0] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("feature %d not finite", i)
		}
	}
}

func TestMelBands(t *testing.T) {
	m := melBank()
	if len(m) != MelBands || len(m[0]) != Bins {
		t.Fatalf("mel %d×%d", len(m), len(m[0]))
	}
	low, high := 0, 0
	bestL, bestH := 0.0, 0.0
	for k, w := range m[0] {
		if w > bestL {
			bestL, low = w, k
		}
	}
	for k, w := range m[MelBands-1] {
		if w > bestH {
			bestH, high = w, k
		}
	}
	if !(low < high) {
		t.Fatalf("mel peaks low band %d, high band %d", low, high)
	}
}

package eval

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestWedgeArea(t *testing.T) {
	sign, area, keep := wedgeKeep([]float32{1, -1}, []float32{1, 1}, []int32{0, 1}, 2)
	if sign != 0 || keep || area != 0 {
		t.Fatalf("cancelling cells sign %d area %g keep %v", sign, area, keep)
	}
	sign, area, keep = wedgeKeep([]float32{1, -1}, []float32{2, 0}, []int32{0, 1}, 2)
	if sign != 1 || !keep || area != 2 {
		t.Fatalf("positive sign %d area %g keep %v", sign, area, keep)
	}
	sign, area, keep = wedgeKeep([]float32{1, 0}, []float32{-3, 4}, nil, 0)
	if sign != -1 || keep || area != -3 {
		t.Fatalf("channel wedge sign %d area %g keep %v", sign, area, keep)
	}
}

func TestWedgeKeepsWrite(t *testing.T) {
	cells := Cells{
		D: []int32{0, 1}, F: []int32{0, 1}, Q: []int32{0, 1}, KV: []int32{0, 1},
		ND: 2, NF: 2, NQ: 2, NKV: 2, Per: 1,
	}
	t.Run("attention kept", func(t *testing.T) {
		m := wedgeModel(t, true, 0)
		y := wedgeFeed(t, m, cells)
		want := float32(1 + math.Sqrt(2))
		if math.Abs(float64(y[0]-want)) > 1e-4 || y[1] != 0 {
			t.Fatalf("residual %v, want [%g 0]", y, want)
		}
		L := m.rec[0]
		if L.Sign[0] != 1 || L.Keep[0] != 1 {
			t.Fatalf("attention sign %d keep %d", L.Sign[0], L.Keep[0])
		}
		if L.Sign[1] != 0 || L.Keep[1] != 0 {
			t.Fatalf("ffn sign %d keep %d", L.Sign[1], L.Keep[1])
		}
		if sumRow(L.Edge["down"], 2, 0) != 0 || sumRow(L.Edge["down"], 2, 1) != 0 {
			t.Fatalf("refused down currents %v", L.Edge["down"])
		}
	})
	t.Run("ffn kept", func(t *testing.T) {
		m := wedgeModel(t, false, 1)
		y := wedgeFeed(t, m, cells)
		if math.Abs(float64(y[0]-2)) > 1e-4 || y[1] != 0 {
			t.Fatalf("residual %v, want [2 0]", y)
		}
		L := m.rec[0]
		if L.Sign[0] != 0 || L.Keep[0] != 0 {
			t.Fatalf("attention sign %d keep %d", L.Sign[0], L.Keep[0])
		}
		if L.Sign[1] != 1 || L.Keep[1] != 1 || L.Wedge[1] <= 0 {
			t.Fatalf("ffn sign %d keep %d wedge %g", L.Sign[1], L.Keep[1], L.Wedge[1])
		}
	})
	t.Run("ffn refused", func(t *testing.T) {
		m := wedgeModel(t, false, -1)
		y := wedgeFeed(t, m, cells)
		if math.Abs(float64(y[0]-1)) > 1e-4 || y[1] != 0 {
			t.Fatalf("residual %v, want [1 0]", y)
		}
		L := m.rec[0]
		if L.Sign[1] != -1 || L.Keep[1] != 0 || L.Wedge[1] >= 0 {
			t.Fatalf("ffn sign %d keep %d wedge %g", L.Sign[1], L.Keep[1], L.Wedge[1])
		}
		if sumRow(L.Edge["down"], 2, 0) != 0 {
			t.Fatalf("refused down current %v", L.Edge["down"])
		}
	})
}

func sumRow(e []float32, nc, r int) float32 {
	var s float32
	for c := 0; c < nc; c++ {
		s += e[r*nc+c]
	}
	return s
}

// wedgeModel is one float layer. attn copies the normalised residual through v and o when on.
// downSign is the feed-forward write on channel 0: +1, -1, or 0.
func wedgeModel(t *testing.T, attn bool, downSign int) *model {
	t.Helper()
	const D = 2
	ones := []float32{1, 1}
	z := make([]float32, D*D)
	id := []float32{1, 0, 0, 1}
	eye := func(on bool) []float32 {
		if on {
			return append([]float32(nil), id...)
		}
		return append([]float32(nil), z...)
	}
	s2 := float32(1 / math.Sqrt(2))
	gate := []float32{s2, 0, 0, 0}
	up := append([]float32(nil), gate...)
	down := make([]float32, D*D)
	if downSign != 0 {
		down[0] = float32(downSign)
	}
	emb := make([]byte, D*4)
	binary.LittleEndian.PutUint32(emb[0:], math.Float32bits(1))
	m := &model{
		Shape:   Shape{D: D, F: D, L: 1, H: 1, KVH: 1, HD: D, Act: "relu2"},
		invFreq: []float64{0},
		emb:     emb,
		embDT:   "F32",
		nEmb:    1,
		ls: []layer{{
			an: ones, fn: ones,
			q: &lin{n: D, k: D, f: eye(false)},
			k: &lin{n: D, k: D, f: eye(false)},
			v: &lin{n: D, k: D, f: eye(attn)},
			o: &lin{n: D, k: D, f: eye(attn)},
			g: &lin{n: D, k: D, f: gate},
			u: &lin{n: D, k: D, f: up},
			d: &lin{n: D, k: D, f: down},
		}},
	}
	return m
}

func wedgeFeed(t *testing.T, m *model, cells Cells) []float32 {
	t.Helper()
	if err := m.instrument(cells); err != nil {
		t.Fatal(err)
	}
	m.rec = []FlowLayer{{Edge: map[string][]float32{}, Node: map[string][]float32{}}}
	return m.feed(&seq{}, 0, 0)
}

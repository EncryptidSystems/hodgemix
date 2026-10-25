package eval

import (
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"

	"hodgemix/tower"
)

func TestTowerSlotReplacesRow(t *testing.T) {
	m := &model{Shape: Shape{D: 4}, nEmb: 1, embDT: "F32"}
	m.emb = make([]byte, 16)
	binary.LittleEndian.PutUint32(m.emb[0:], math.Float32bits(1))
	m.slots = map[int]towerSlot{1: {kind: "vision", x: []float32{0, 2, 0, 0}, phi: 0.3}}
	s := &seq{}
	y := m.feed(s, 1, 0)
	if y[0] != 0 || y[1] != 2 {
		t.Fatalf("slot row %v", y)
	}
	if s.phaseAt(0) != 0.3 {
		t.Fatalf("phase %g", s.phaseAt(0))
	}
	text := m.feed(s, 0, 1)
	if text[0] != 1 || text[1] != 0 {
		t.Fatalf("text row %v", text)
	}
	if s.phaseAt(1) != 0 {
		t.Fatalf("text phase %g", s.phaseAt(1))
	}
}

func TestRopeTowerPhase(t *testing.T) {
	m := &model{Shape: Shape{HD: 128}}
	freq := make([]float64, 64)
	x := make([]float32, 128)
	for i := range x {
		x[i] = 1
	}
	m.rope(x, 0, freq, 1, math.Pi/2)
	// Planes 0..55 stay at angle 0. Planes 56..63 turn by pi/2: (1, 1) -> (-1, 1).
	if x[0] != 1 || x[64] != 1 {
		t.Fatalf("plane 0 moved to %g,%g", x[0], x[64])
	}
	if math.Abs(float64(x[56]+1)) > 1e-5 || math.Abs(float64(x[56+64]-1)) > 1e-5 {
		t.Fatalf("plane 56 is %g,%g", x[56], x[56+64])
	}
}

func TestLoadTowerEmbed(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.safetensors")
	// D=4, df=46 would be the real width. Use the vision width so the check passes,
	// with a zero projector and a mean row.
	const D, df = 4, tower.VisFeat
	mean := make([]float32, D)
	mean[0] = 3
	feat := make([]float32, df)
	writeTensors(t, path, []ten{
		{name: tower.VisEmbed + ".weight", dtype: "F32", shape: []int{D, df}, f: make([]float32, D*df)},
		{name: tower.VisEmbed + ".mean_feature", dtype: "F32", shape: []int{df}, f: feat},
		{name: tower.VisEmbed + ".mean_row", dtype: "F32", shape: []int{D}, f: mean},
		{name: tower.VisEmbed + ".rms", dtype: "F32", shape: []int{1}, f: []float32{0}},
	})
	s, err := openShards(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.close()
	tab, err := s.towerEmbed(tower.VisEmbed, tower.VisFeat)
	if err != nil {
		t.Fatal(err)
	}
	if tab == nil || tab.D != D || tab.df != df {
		t.Fatalf("tab %+v", tab)
	}
	y, err := tower.Project(tab.W, tab.D, tab.df, tab.meanF, tab.meanRow, tab.rms, make([]float32, df))
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(y[0]-3)) > 1e-5 {
		t.Fatalf("projected %v", y[:4])
	}
}

package main

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hodgemix/eval"
)

func TestVocabCochainFloat(t *testing.T) {
	D, n := 8, 2
	cell := []int32{0, 0, 0, 0, 1, 1, 1, 1}
	emb := []float32{1, 3, 5, 7, 10, 0, 0, 0, 4, 4, 4, 4, 2, 2, 2, 2}
	co := vocabCochain(emb, D, cell, n)
	if co == nil || len(co) != 4 || co[0] != 16 || co[1] != 10 || co[2] != 16 || co[3] != 8 {
		t.Fatalf("cochain %v", co)
	}
	fl := cochainFloat(co, D, cell, n)
	for j := 0; j < 4; j++ {
		if fl[j] != 4 || fl[8+j] != 4 {
			t.Fatalf("cell 0 lift %v", fl)
		}
	}
	for j := 4; j < 8; j++ {
		if fl[j] != 2.5 || fl[8+j] != 2 {
			t.Fatalf("cell 1 lift %v", fl)
		}
	}
	back := vocabCochain(fl, D, cell, n)
	for i := range co {
		if math.Abs(float64(back[i]-co[i])) > 1e-4 {
			t.Fatalf("round trip %v vs %v", back, co)
		}
	}
	// A row already constant on each cell is its own floating point.
	for j := range fl[8:16] {
		if fl[8+j] != emb[8+j] {
			t.Fatalf("constant row moved: %v", fl[8:16])
		}
	}
}

func TestVolumeUsesServedConfig(t *testing.T) {
	dir := tempDir(t)
	m := tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 1, V: 260}
	a := writeTiny(t, filepath.Join(dir, "a"), m, 31)
	neck := filepath.Join(dir, "model.neck.safetensors")
	b, err := os.ReadFile(a)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(neck, b, 0o644); err != nil {
		t.Fatal(err)
	}
	err = volumes(neck, filepath.Join(dir, "missing.volume.safetensors"))
	if err == nil || !strings.Contains(err.Error(), "no config set") {
		t.Fatalf("neck without a config: %v", err)
	}
	bindServed(neck, a)
	if eval.FilesOf(neck).Config != eval.FilesOf(a).Config {
		t.Fatal("served config was not recorded")
	}
	vol := filepath.Join(dir, "volume.safetensors")
	if err := volumes(neck, vol); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "vocab.float.safetensors")
	if err := vocabFloat(neck, vol, out); err != nil {
		t.Fatal(err)
	}
	st, err := OpenST(out)
	if err != nil {
		t.Fatal(err)
	}
	if got := st.LogicalShape("cochain"); !eqShape(got, []int{260, VolumeCells}) {
		t.Fatalf("cochain %v", got)
	}
	if got := st.LogicalShape("float"); !eqShape(got, []int{260, 128}) {
		t.Fatalf("float %v", got)
	}
	emb, err := OpenST(neck)
	if err != nil {
		t.Fatal(err)
	}
	E, err := emb.F32("model.embed_tokens.weight")
	if err != nil {
		t.Fatal(err)
	}
	cell, err := readI32(OpenMust(t, vol), "cell.D")
	if err != nil {
		t.Fatal(err)
	}
	want := vocabCochain(E, 128, cell, VolumeCells)
	got, err := st.F32("cochain")
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("cochain[%d]=%g, want %g", i, got[i], want[i])
		}
	}
	fl, err := st.F32("float")
	if err != nil {
		t.Fatal(err)
	}
	back := vocabCochain(fl, 128, cell, VolumeCells)
	for i := range want {
		if math.Abs(float64(back[i]-want[i])) > 1e-3*math.Max(1, math.Abs(float64(want[i]))) {
			t.Fatalf("float cell sum [%d]=%g, cochain %g", i, back[i], want[i])
		}
	}
}

func OpenMust(t *testing.T, path string) *ST {
	t.Helper()
	st, err := OpenST(path)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

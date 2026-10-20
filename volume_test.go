package main

import (
	"path/filepath"
	"testing"
)

// Every role's volume is layer x row cells x col cells on the shared axes, every cell of an axis holds
// the same number of channels, and a tensor of constant magnitude reads 1 everywhere.
func TestVolumes(t *testing.T) {
	dir := tempDir(t)
	m := tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 260}
	a := writeTiny(t, filepath.Join(dir, "a"), m, 31)
	out := filepath.Join(dir, "a.volume.safetensors")
	if err := volumes(a, out); err != nil {
		t.Fatal(err)
	}
	st, err := OpenST(out)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][]int{
		"volume.q": {2, 2 * VolumeHeadCells, VolumeCells}, "volume.k": {2, VolumeHeadCells, VolumeCells},
		"volume.o": {2, VolumeCells, 2 * VolumeHeadCells}, "volume.gate": {2, VolumeCells, VolumeCells},
		"volume.down": {2, VolumeCells, VolumeCells},
	}
	for k, sh := range want {
		if got := st.LogicalShape(k); !eqShape(got, sh) {
			t.Fatalf("%s %v, want %v", k, got, sh)
		}
	}
	for ax, n := range map[string]int{"D": 128, "F": 192} {
		c, err := st.Raw("cell." + ax)
		if err != nil {
			t.Fatal(err)
		}
		count := make([]int, VolumeCells)
		for i := 0; i < n; i++ {
			count[int(c[4*i])|int(c[4*i+1])<<8]++
		}
		for cell, k := range count {
			if k != n/VolumeCells {
				t.Fatalf("axis %s cell %d holds %d channels, want %d", ax, cell, k, n/VolumeCells)
			}
		}
	}
	v, err := st.F32("volume.gate")
	if err != nil {
		t.Fatal(err)
	}
	var mean float64
	for _, x := range v {
		mean += float64(x)
	}
	if mean /= float64(len(v)); mean < 0.9 || mean > 1.1 {
		t.Fatalf("volume.gate mean %.3f, want ~1 (cells relative to the tensor's absmean)", mean)
	}
}

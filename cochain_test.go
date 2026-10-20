package main

import (
	"math"
	"path/filepath"
	"testing"
)

// On a model laid out on its cells, the residual flow is conserved at every node, every BitLinear
// output cell is the sum of its currents, and the Hodge split leaves a divergence-free cycle part.
func TestFlowCochain(t *testing.T) {
	dir := tempDir(t)
	m := tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 260}
	a := writeTiny(t, filepath.Join(dir, "a"), m, 41)
	// packed ternary weights, as the runtime runs them: a self-mix writes the model in Champ layout
	out := filepath.Join(dir, "out", "self.bnz")
	run(tinyCfg("base"), a, []string{a}, out, "", "", true)
	model := out + ".ternary.safetensors"
	rep, err := flowRun(model, filepath.Join(dir, "self.flow.safetensors"), "hello world")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Tokens == 0 {
		t.Fatal("no tokens")
	}
	if rep.Conservation > 1e-4 {
		t.Fatalf("conservation mismatch %.2e", rep.Conservation)
	}
	if rep.Stokes > 1e-6 {
		t.Fatalf("cycle part not divergence-free: %.2e", rep.Stokes)
	}
	for c := 0; c < ecN; c++ {
		e := rep.Energy[c]
		if e[0] == 0 {
			continue
		}
		// gradient and cycle parts are orthogonal: their energies add to the flow's
		if s := (e[1] + e[2]) / e[0]; math.Abs(s-1) > 0.35 && c != ecAttn {
			t.Logf("class %s: grad+cycle %.3f of |f|^2 (per class the parts need not add; overall they must)", ecName[c], s)
		}
	}
	var f, g, cy float64
	for _, e := range rep.Energy {
		f, g, cy = f+e[0], g+e[1], cy+e[2]
	}
	if math.Abs((g+cy)/f-1) > 1e-6 {
		t.Fatalf("|grad|^2 + |cycle|^2 = %.6f of |f|^2, want 1 (orthogonal split)", (g+cy)/f)
	}
}

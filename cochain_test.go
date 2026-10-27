package main

import (
	"math"
	"path/filepath"
	"testing"

	"hodgemix/eval"
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

// Faces are digons. A gradient has no circulation. The harmonic part of a divergence-free cochain
// has none either, and it is orthogonal to the coexact part. Write-edge currents are the row sums
// of o and down.
func TestWedgeFaces(t *testing.T) {
	cells := eval.Cells{
		D: []int32{0, 1}, F: []int32{0, 1}, Q: []int32{0, 1}, KV: []int32{0, 1},
		ND: 2, NF: 2, NQ: 2, NKV: 2, Per: 1,
	}
	cx := newFlowComplex(2, cells)
	if len(cx.face) != 8 {
		t.Fatalf("faces %d, want 8", len(cx.face))
	}
	phi := make([]float64, cx.nodes)
	for i := range phi {
		phi[i] = math.Sin(float64(i + 1))
	}
	for i, g := range cx.circ(cx.grad(phi)) {
		if math.Abs(g) > 1e-12 {
			t.Fatalf("gradient circulation on face %d is %g", i, g)
		}
	}
	f := make([]float64, len(cx.tail))
	dig := cx.face[0]
	f[dig.write], f[dig.carry] = 1, -1
	if got := cx.circ(f)[0]; math.Abs(got-2) > 1e-12 {
		t.Fatalf("circulation %g, want 2", got)
	}
	_, cyc, _ := cx.hodge(f, 1e-12, 4000)
	h := cx.harmonic(cyc)
	for i, g := range cx.circ(h) {
		if math.Abs(g) > 1e-8 {
			t.Fatalf("harmonic circulation face %d %g", i, g)
		}
	}
	if math.Abs(h[dig.write]) > 1e-8 || math.Abs(h[dig.carry]) > 1e-8 {
		t.Fatalf("pure digon harmonic %g %g, want 0", h[dig.write], h[dig.carry])
	}

	rf := make([]float64, len(cx.tail))
	for i := range rf {
		rf[i] = math.Sin(float64(i*3 + 1))
	}
	p, c, _ := cx.hodge(rf, 1e-12, 8000)
	gr := cx.grad(p)
	var f2, g2, c2 float64
	for i := range rf {
		f2 += rf[i] * rf[i]
		g2 += gr[i] * gr[i]
		c2 += c[i] * c[i]
	}
	if f2 == 0 || math.Abs((g2+c2)/f2-1) > 1e-6 {
		t.Fatalf("orthogonal split %g", (g2+c2)/f2)
	}
	hh := cx.harmonic(c)
	var h2, x2, dot float64
	for i := range c {
		co := c[i] - hh[i]
		h2 += hh[i] * hh[i]
		x2 += co * co
		dot += hh[i] * co
	}
	if c2 == 0 || math.Abs((h2+x2)/c2-1) > 1e-6 {
		t.Fatalf("harmonic energy %g of |cycle|^2", (h2+x2)/c2)
	}
	if math.Abs(dot)/c2 > 1e-6 {
		t.Fatalf("harmonic not orthogonal to coexact %g", dot)
	}
	for i, g := range cx.circ(hh) {
		if math.Abs(g) > 1e-7 {
			t.Fatalf("harmonic face %d circulation %g", i, g)
		}
	}

	ls := make([]eval.FlowLayer, 2)
	for li := range ls {
		ls[li] = eval.FlowLayer{Edge: map[string][]float32{}, Node: map[string][]float32{}}
		for _, k := range []string{"q", "k", "v", "o", "gate", "up", "down"} {
			nr, nc := cells.NF, cells.ND
			switch k {
			case "q":
				nr, nc = cells.NQ, cells.ND
			case "k", "v":
				nr, nc = cells.NKV, cells.ND
			case "o":
				nr, nc = cells.ND, cells.NQ
			case "down":
				nr, nc = cells.ND, cells.NF
			}
			e := make([]float32, nr*nc)
			if k == "o" {
				e[0], e[1] = 3, 4
			}
			if k == "down" {
				e[0] = -2
			}
			ls[li].Edge[k] = e
		}
		for _, k := range []string{"D", "Dm", "A", "H"} {
			n := cells.ND
			if k == "A" {
				n = cells.NQ
			}
			if k == "H" {
				n = cells.NF
			}
			ls[li].Node[k] = make([]float32, n)
		}
		ls[li].Node["D"][0] = 5
		ls[li].Node["Dm"][1] = 7
	}
	cur, err := cx.currents(ls)
	if err != nil {
		t.Fatal(err)
	}
	if len(cur) != len(cx.tail) {
		t.Fatalf("currents %d edges %d", len(cur), len(cx.tail))
	}
	var attn0, ffn1 *flowFace
	for i := range cx.face {
		fc := &cx.face[i]
		if fc.layer == 0 && fc.kind == 0 && fc.cell == 0 {
			attn0 = fc
		}
		if fc.layer == 0 && fc.kind == 1 && fc.cell == 1 {
			ffn1 = fc
		}
	}
	if attn0 == nil || ffn1 == nil {
		t.Fatal("missing faces")
	}
	if cur[attn0.write] != 7 || cur[attn0.carry] != 5 {
		t.Fatalf("attn write %g carry %g, want 7 and 5", cur[attn0.write], cur[attn0.carry])
	}
	if cur[ffn1.carry] != 7 || cur[ffn1.write] != 0 {
		t.Fatalf("ffn carry %g write %g, want 7 and 0", cur[ffn1.carry], cur[ffn1.write])
	}
}

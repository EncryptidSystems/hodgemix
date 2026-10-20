package eval

import (
	"math"
	"math/rand"
	"testing"
)

// A BitLinear laid out on cells gives bit-identical outputs, and each row's cell currents sum to its
// output: the fabric's gathers are the edge currents.
func TestRelayoutCurrents(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	n, k, nc := 12, 240, 8 // 30 channels a cell
	w := make([]int8, n*k)
	for i := range w {
		w[i] = int8(rng.Intn(3) - 1)
	}
	d := make([]uint8, n*k)
	for i := range d {
		d[i] = uint8(rng.Intn(4))
	}
	mk := func() *lin {
		l := &lin{n: n, k: k, w: append([]int8(nil), w...), s: 0.5}
		l.pack()
		l.limbs = append(l.limbs, newLimb(append([]uint8(nil), d...), n, k, 4, 0.125))
		return l
	}
	a, b := mk(), mk()
	cell := make([]int32, k)
	perm := rng.Perm(k)
	for i, j := range perm {
		cell[j] = int32(i * nc / k) // cells of scattered channels
	}
	rowCell := make([]int32, n)
	for r := range rowCell {
		rowCell[r] = int32(r % 3)
	}
	order, size, ok := orderOf(cell, nc)
	if !ok || size != k/nc {
		t.Fatalf("orderOf: %v %d", ok, size)
	}
	b.relayout(order, size)
	x := make([]float32, k)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	ya, yb := make([]float32, n), make([]float32, n)
	a.run(&act{x: x}, ya, false)
	flow := make([]float32, 3*nc)
	b.runFlow(&flowLin{order: order, cell: size, colCell: cell, rowCell: rowCell, nr: 3, nc: nc}, &act{x: x}, yb, false, flow)
	for r := range ya {
		if ya[r] != yb[r] {
			t.Fatalf("row %d: plain %v, cell layout %v", r, ya[r], yb[r])
		}
	}
	for rc := 0; rc < 3; rc++ {
		var want, got float64
		for r := range ya {
			if int(rowCell[r]) == rc {
				want += float64(ya[r])
			}
		}
		for c := 0; c < nc; c++ {
			got += float64(flow[rc*nc+c])
		}
		if math.Abs(got-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Fatalf("row cell %d: currents sum %v, outputs %v", rc, got, want)
		}
	}
}

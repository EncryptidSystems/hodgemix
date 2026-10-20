package main

// Whitney prolongation and de Rham restriction on a periodic 1-D digit axis (paper_v2 Eq.(26)).
// Coarse cycle with n nodes/edges, fine cycle with 2n (nested: fine node 2i = coarse node i).

// Paper_v2 Eq.(26a): d0 on a cycle: (d0 a)_e = a[e+1] - a[e]
func dCycle(a []float64) []float64 {
	n := len(a)
	o := make([]float64, n)
	for e := 0; e < n; e++ {
		o[e] = a[(e+1)%n] - a[e]
	}
	return o
}

// Paper_v2 Eq.(26b): P0 = Whitney 0-form (hat) interpolation evaluated at fine nodes
func prolong0(a []float64) []float64 {
	n := len(a)
	o := make([]float64, 2*n)
	for i := 0; i < n; i++ {
		o[2*i] = a[i]
		o[2*i+1] = (a[i] + a[(i+1)%n]) / 2
	}
	return o
}

// Paper_v2 Eq.(26c): P1 = de Rham map of the Whitney 1-form of a coarse edge onto its two fine halves
func prolong1(b []float64) []float64 {
	n := len(b)
	o := make([]float64, 2*n)
	for e := 0; e < n; e++ {
		o[2*e], o[2*e+1] = b[e]/2, b[e]/2
	}
	return o
}

// Paper_v2 Eq.(26d): R0 = de Rham restriction of 0-cochains (evaluation at coarse nodes)
func restrict0(a []float64) []float64 {
	o := make([]float64, len(a)/2)
	for i := range o {
		o[i] = a[2*i]
	}
	return o
}

// Paper_v2 Eq.(26e): R1 = de Rham restriction of 1-cochains (integral over coarse edge = sum of halves)
func restrict1(b []float64) []float64 {
	o := make([]float64, len(b)/2)
	for i := range o {
		o[i] = b[2*i] + b[2*i+1]
	}
	return o
}

// Paper_v2 Eq.(26f): L1 on a cycle graph (no 2-cells): L1 b = d0 d0^T b (harmonic iff constant)
func l1Cycle(b []float64) []float64 {
	n := len(b)
	t := make([]float64, n) // d0^T b at nodes
	for e := 0; e < n; e++ {
		t[(e+1)%n] += b[e]
		t[e] -= b[e]
	}
	return dCycle(t)
}

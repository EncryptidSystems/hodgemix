package main

// Married index: the channel indexes of several models joined into one coherent index per axis.
//
// Each contributor's axis has its own mixed-radix structure (2560 = 2^9*5, 1024 = 2^10, ...) and its
// own heap. Marriage puts them in one MRH(r_M; C):
//   - r_M is the common refinement of the factorisations: the factorisation of lcm(n_1, ..., n_m),
//     finest digit first, extended by 2s until pi_k >= C+1; every model's digit hierarchy embeds in it.
//   - every channel of every model is inserted (model by model, index order), and the single
//     extract-min order is the married order. A channel's coordinate is its place in that order:
//         x = (S + (r + 1/2) * E / E_m) / N
//     with N all channels, S the channels popped before its key, E the channels with its key, E_m
//     those of its own model and r its FIFO rank among them. Equal keys share one span of the axis
//     across models; within a model they keep index order. One model married with itself gets
//     exactly its own Eq.(8) coordinates (S, E, E_m all double, the factors cancel).

import "fmt"

// IndexMode is --index: "married" (one heap per axis for all contributors of a target) or "own".
var IndexMode = "married"

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// MarriedRadices is the factorisation of lcm(ns...), finest first, covering C+1.
func MarriedRadices(ns []int, C int) []int {
	l := 1
	for _, n := range ns {
		if n > 0 {
			l = l / gcd(l, n) * n
		}
	}
	return RadicesFor(l, C)
}

// MarriedCoords places the channels of every key list (one list per model axis) on one axis.
func MarriedCoords(keys [][]int, C int) ([][]float64, []int, error) {
	lo, hi, r, err := MarriedSpans(keys, nil, C)
	if err != nil {
		return nil, nil, err
	}
	out := make([][]float64, len(keys))
	for m := range keys {
		out[m] = make([]float64, len(keys[m]))
		for i := range out[m] {
			out[m][i] = (lo[m][i] + hi[m][i]) / 2
		}
	}
	return out, r, nil
}

// MarriedSpans is the married index over units that may be wider than one channel (a kv group of
// heads): unit i of model m with width w_mi gets the span [lo, hi) of the axis, all measured in
// channels, N = sum of widths. With S the width popped before its key, E the width with its key,
// E_m its own model's width with that key and P its own model's width before it within the key,
//
//	lo = (S + P*E/E_m)/N,  hi = lo + w*E/E_m/N.
//
// widths == nil means every unit is one channel (then the midpoint is MarriedCoords' x).
func MarriedSpans(keys [][]int, widths [][]int, C int) ([][]float64, [][]float64, []int, error) {
	ns := make([]int, len(keys))
	N := 0
	wOf := func(m, i int) int {
		if widths == nil {
			return 1
		}
		return widths[m][i]
	}
	for m, k := range keys {
		for i := range k {
			ns[m] += wOf(m, i)
		}
		N += ns[m]
	}
	r := MarriedRadices(ns, C)
	h, err := NewMRH(r, C)
	if err != nil {
		return nil, nil, nil, err
	}
	type ref struct{ m, i int }
	var refs []ref
	for m, k := range keys {
		for i, kappa := range k {
			if err := h.Insert(len(refs), kappa); err != nil {
				return nil, nil, nil, fmt.Errorf("married index: model %d unit %d: %v", m, i, err)
			}
			refs = append(refs, ref{m, i})
		}
	}
	lo, hi := make([][]float64, len(keys)), make([][]float64, len(keys))
	for m := range keys {
		lo[m], hi[m] = make([]float64, len(keys[m])), make([]float64, len(keys[m]))
	}
	seen := 0
	var group []int
	flush := func() {
		if len(group) == 0 {
			return
		}
		E := 0
		own := make([]int, len(keys))
		for _, x := range group {
			w := wOf(refs[x].m, refs[x].i)
			E += w
			own[refs[x].m] += w
		}
		before := make([]int, len(keys))
		for _, x := range group {
			m, i := refs[x].m, refs[x].i
			scale := float64(E) / float64(own[m])
			lo[m][i] = (float64(seen) + float64(before[m])*scale) / float64(N)
			hi[m][i] = lo[m][i] + float64(wOf(m, i))*scale/float64(N)
			before[m] += wOf(m, i)
		}
		seen += E
		group = group[:0]
	}
	last := -1
	for h.Len() > 0 {
		x, kappa, _ := h.ExtractMin()
		if kappa != last {
			flush()
			last = kappa
		}
		group = append(group, x)
	}
	flush()
	return lo, hi, r, nil
}

package main

// Pin-mean (--pin-mean on, default): a merge must not change any BitLinear's response to its mean
// input.
//
// The inputs every reader sees share a large constant component mu. A delta dW of a reader adds
// dW mu to every token's output; attention averages over the context, which cancels the token-specific
// part and keeps that offset. With v mixed, the common-mode share of every head's values rose from
// ~0.4 to 0.72-0.92 in layers 12-29 (2B4T + Qwen3), and a 334-token chat answered garbage while 20-token
// texts scored fine. In the Hodge picture mu is the harmonic (constant) mode on the RMSNorm sphere, and
// the condition keeps the base's boundary value there:
//
//	W <- W - (dW mu^) mu^T,   dW = W - W_base,   mu^ = mu / ||mu||,
//
// a rank-one projection per tensor, before ternarization. mu comes from the base's reference forward
// (eval.FindSingular: in_attn for q/k/v, in_o for o, in_ffn for gate/up, in_down for down).
// No paper equation.

import (
	"fmt"
	"math"

	"hodgemix/eval"
)

// PinMean is --pin-mean.
var PinMean = true

func meanInput(role string) string {
	switch role {
	case "q", "k", "v":
		return "in_attn"
	case "o":
		return "in_o"
	case "gate", "up":
		return "in_ffn"
	case "down":
		return "in_down"
	}
	return ""
}

// pinMean projects the mixed tensor Wf (target shape, base units) in place; the note says what it did.
func pinMean(t target, Wf []float32, base *ST, sg *eval.Singular) (string, error) {
	in := meanInput(t.role)
	if in == "" || t.layer < 0 || len(t.shape) != 2 {
		return "", nil // not a BitLinear reading a calibrated input
	}
	if t.layer >= len(sg.Means) {
		return "", fmt.Errorf("layer %d is past the base forward's %d layers", t.layer, len(sg.Means))
	}
	mu := sg.Means[t.layer][in]
	R, C := t.shape[0], t.shape[1]
	if len(mu) != C {
		return "", fmt.Errorf("the base forward's mean %s has %d channels, the tensor %d columns", in, len(mu), C)
	}
	B, _, err := base.LoadDense(t.name)
	if err != nil {
		return "", err
	}
	if len(B) != len(Wf) {
		return "", fmt.Errorf("the base tensor has %d values, the mixed one %d", len(B), len(Wf))
	}
	var n float64
	for _, x := range mu {
		n += x * x
	}
	n = math.Sqrt(n)
	if n == 0 {
		return "", nil // a zero mean input: nothing to pin
	}
	mh := make([]float64, C)
	for i, x := range mu {
		mh[i] = x / n
	}
	resp := func(r int) (d, b float64) {
		for c := 0; c < C; c++ {
			d += (float64(Wf[r*C+c]) - float64(B[r*C+c])) * mh[c]
			b += float64(B[r*C+c]) * mh[c]
		}
		return
	}
	var d0, ref float64
	for r := 0; r < R; r++ {
		d, b := resp(r)
		d0, ref = d0+d*d, ref+b*b
	}
	parFor(R, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := Wf[r*C : (r+1)*C]
			d, _ := resp(r)
			for c := 0; c < C; c++ {
				row[c] -= float32(d * mh[c])
			}
		}
	})
	return fmt.Sprintf("%s: ||dW mu^|| was %.3g of ||W_base mu^||, now 0", in, math.Sqrt(d0/math.Max(ref, 1e-30))), nil
}

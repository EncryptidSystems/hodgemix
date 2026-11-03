package main

// Surgery (--surgery on, default): excise the singular set of the base's forward and cap it with
// the base.
//
// A ternary LM's residual stream concentrates in a few massive channels (2B4T: 17 of 2560 at
// 41-60x the median channel's rms, written by a handful of FFN spike units in layers 0-1). Every
// role mixed alone leaves them intact, but mixing every role rewrites both the rows that write them
// (o, down) and the columns that read them (q, k, v, gate, up): the spikes are squashed (layer-1
// hidden peak 0.20x the base's), the massive channels are left to whatever the mix put there, and
// the forward diverges from layer 0 (cos 0.93 -> 0.0 by layer 12, ppl 12 -> 7.6e7).
//
// The singular set is found on the base's own forward (eval.FindSingular, the reference CPU
// forward on the calibration texts): massive residual channels M (rms >= ratio x median at any
// layer) and per layer the FFN spike units S_l (peak |relu^2(gate) up| >= ratio x the layer's
// median unit peak). Surgery then puts the base's ternary codes back on exactly
//   - rows m in M of o_proj and down_proj, and columns m in M of q/k/v/gate/up (the residual side),
//   - rows u in S_l of gate_proj and up_proj, and columns u in S_l of down_proj (the dual, FFN side),
// and leaves every other code as mixed. Each tensor keeps its own weight_scale. No paper equation.

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"hodgemix/eval"
)

// SurgeryOn is --surgery; SurgeryRatio is --surgery-ratio.
var (
	SurgeryOn    = true
	SurgeryRatio = 20.0
)

func unpackTrits(b []byte, R, C int) []int8 {
	R4 := R / 4
	q := make([]int8, R*C)
	for i := 0; i < R4*C; i++ {
		for s := 0; s < 4; s++ {
			q[s*R4*C+i] = int8((b[i]>>(2*s))&3) - 1
		}
	}
	return q
}

// baseCodes are the base tensor's ternary codes: its packed codes, or a float base ternarized at
// its absmean (what the base itself would be in Champ).
func baseCodes(base *ST, name string, R, C int) ([]int8, error) {
	if base.isPacked(name) {
		b, err := base.Raw(name)
		if err != nil {
			return nil, err
		}
		return unpackTrits(b, R, C), nil
	}
	W, _, err := base.LoadDense(name)
	if err != nil {
		return nil, err
	}
	s := absmean(W)
	q := make([]int8, len(W))
	for i, v := range W {
		q[i] = int8(ternQ(float64(v), s))
	}
	return q, nil
}

type surgeryReport struct {
	Massive    []int         `json:"massive_channels"`
	Spikes     map[int][]int `json:"spike_units"`
	Codes      int           `json:"codes_restored"`
	Changed    int           `json:"codes_that_differed"`
	Tensors    int           `json:"tensors"`
	Ratio      float64       `json:"ratio"`
	SkippedWhy string        `json:"skipped,omitempty"`
}

// surgery patches outTern in place; tok is the base's tokenizer.json (for the base forward).
func surgery(basePath, outTern, tok string) (*surgeryReport, error) {
	return surgeryWith(nil, basePath, outTern, tok)
}

// surgeryWith uses a calibration already run (sg), or runs it.
func surgeryWith(sg *eval.Singular, basePath, outTern, tok string) (*surgeryReport, error) {
	rep := &surgeryReport{Ratio: SurgeryRatio}
	var err error
	if sg == nil {
		if sg, err = eval.FindSingular(basePath, tok, SurgeryRatio); err != nil {
			rep.SkippedWhy = err.Error()
			return rep, err
		}
	}
	rep.Massive, rep.Spikes = sg.Massive, sg.Spikes
	base, err := OpenST(basePath)
	if err != nil {
		return rep, err
	}
	defer base.Close()
	out, err := OpenST(outTern)
	if err != nil {
		return rep, err
	}
	defer out.Close()
	f, err := os.OpenFile(outTern, os.O_RDWR, 0)
	if err != nil {
		return rep, err
	}
	defer f.Close()
	isM := map[int]bool{}
	for _, c := range sg.Massive {
		isM[c] = true
	}
	var names []string
	for _, k := range out.Keys {
		if out.isPacked(k) && base.Has(k) && eqShape(base.LogicalShape(k), out.LogicalShape(k)) {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	for _, k := range names {
		role, l := roleOf(k), layerOf(k)
		sh := out.LogicalShape(k)
		R, C := sh[0], sh[1]
		spike := map[int]bool{}
		for _, u := range sg.Spikes[l] {
			spike[u] = true
		}
		var rows, cols map[int]bool
		switch role {
		case "o":
			rows = isM
		case "down":
			rows, cols = isM, spike
		case "gate", "up":
			rows, cols = spike, isM
		case "q", "k", "v":
			cols = isM
		default:
			continue
		}
		if len(rows) == 0 && len(cols) == 0 {
			continue
		}
		ob, err := out.Raw(k)
		if err != nil {
			return rep, err
		}
		oq := unpackTrits(ob, R, C)
		bq, err := baseCodes(base, k, R, C)
		if err != nil {
			return rep, err
		}
		n, d := 0, 0
		put := func(i int) {
			n++
			if oq[i] != bq[i] {
				d++
				oq[i] = bq[i]
			}
		}
		for r := range rows {
			if r < R {
				for c := 0; c < C; c++ {
					put(r*C + c)
				}
			}
		}
		for c := range cols {
			if c < C {
				for r := 0; r < R; r++ {
					if !rows[r] {
						put(r*C + c)
					}
				}
			}
		}
		if d > 0 {
			t := out.H[k]
			if _, err := f.WriteAt(packTrits(oq, R, C), out.base+t.Off[0]); err != nil {
				return rep, err
			}
		}
		// The cap is the base's value: dynamic MRH limbs (limbs.go) are zero on the capped rows/columns.
		for li, lb := range out.limbsOf(k) {
			c := uint8((lb.r - 1) / 2)
			for r := range rows {
				if r < R {
					for cc := 0; cc < C; cc++ {
						lb.d[r*C+cc] = c
					}
				}
			}
			for cc := range cols {
				if cc < C {
					for r := 0; r < R; r++ {
						lb.d[r*C+cc] = c
					}
				}
			}
			if err := rewriteLimb(f, out, k, li+1, lb); err != nil {
				return rep, err
			}
		}
		rep.Codes, rep.Changed, rep.Tensors = rep.Codes+n, rep.Changed+d, rep.Tensors+1
	}
	return rep, nil
}

func logSurgery(rep *surgeryReport) {
	n := 0
	var per []string
	for l := 0; l < 64; l++ {
		if u := rep.Spikes[l]; len(u) > 0 {
			n += len(u)
			per = append(per, fmt.Sprintf("L%d:%d", l, len(u)))
		}
	}
	logf("  singular set of the base forward (>= %gx median): %d massive residual channels %v; %d FFN spike units (%s)",
		rep.Ratio, len(rep.Massive), rep.Massive, n, strings.Join(per, " "))
	logf("  surgery: %d codes in %d tensors put back to the base's (%d of them had been changed by the mix)", rep.Codes, rep.Tensors, rep.Changed)
}

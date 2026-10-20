package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"time"
)

type Entry struct {
	W      float64   // depth-kernel weight
	Grid   []float64 // spline sampled on the index grid
	Bucket int
	IsBase bool
}

type SynthParams struct {
	K    KParams
	D    DParams
	Lock string // "dtft" (Algorithm D, default) | "v2" (paper_v2 omega=0 lock, for comparison)
}

type PassRec struct {
	R         float64 `json:"r"`
	Steps     int     `json:"steps"`
	Converged bool    `json:"converged"`
	Locked    int     `json:"locked"`
	NOsc      int     `json:"n_osc"`
	NModes    int     `json:"n_harmonic_modes"`
	Merged    int     `json:"merged_modes"`
	Dh        float64 `json:"dh"`
	Secs      float64 `json:"secs"`
}

type SynthRec struct {
	Shifts [][]int
	PermR  [][]int
	PermC  [][]int
	Energy [][3]float64
	RecErr float64
	Passes []PassRec
	R      []float64
	DFT    []complex128
	D      *DRec
}

// SeedAlign is --seed-align: the per-tensor seed (Eq.12-13 phase-correlation shift per radix digit,
// then Re-Basin row/column assignment) aligns each contributor's grid to the base's one tensor at a
// time. It keeps the deltas small (off: 2B4T + Qwen3 ppl 27 -> 1.3e7; the married index alone does not
// align two models), but connected weights get different permutations, so a source's v and o do not
// compose; the constant-mode damage that leaves at long context is removed by pin-mean (pinmean.go).
var SeedAlign = true

// Paper: Sec.3 and Sec.3.1 steps 1-3: Seed, Hodge split, TIES on the curl part, Kuramoto lock on h, r-weighted merge, re-projection
// Paper_v2 Alg.1 (Eq.(12)-(22)): seed, Hodge split, TIES on curl, Kuramoto lock on h, r-weighted merge, re-projection
func synth(ents []Entry, G [2]int, sp SynthParams) ([]float64, SynthRec) {
	if sp.Lock != "v2" {
		return synthD(ents, G, sp)
	}
	S := NewSpectrum(G, 0)
	dims := S.dims
	n := G[0] * G[1]
	bi := -1
	for i, e := range ents {
		if e.IsBase {
			bi = i
			break
		}
	}
	if bi < 0 {
		bi = 0
		for i, e := range ents {
			if e.W > ents[bi].W {
				bi = i
			}
		}
	}
	rec := SynthRec{Shifts: make([][]int, len(ents)), PermR: make([][]int, len(ents)), PermC: make([][]int, len(ents))}
	// Seed: phase correlation per radix digit, then assignment (starting permutation), in parallel.
	t0 := time.Now()
	ref := ents[bi].Grid
	grids := make([][]float64, len(ents))
	parts := make([]HodgeParts, len(ents))
	done := 0
	setInner("seed+hodge(DFT)", float64(len(ents)), "")
	jobs := make(chan int)
	fin := make(chan int)
	for w := 0; w < 8; w++ {
		go func() {
			for i := range jobs {
				g := ents[i].Grid
				if i != bi && SeedAlign {
					g2, sh := seedShift(g, ref, dims, 0)
					g3, pr, pc := seedAssign(g2, ref, G)
					grids[i], rec.Shifts[i], rec.PermR[i], rec.PermC[i] = g3, sh, pr, pc
				} else {
					grids[i], rec.Shifts[i] = g, make([]int, len(dims))
					rec.PermR[i], rec.PermC[i] = identity(G[0]), identity(G[1])
				}
				parts[i] = S.Hodge(grids[i])
				fin <- i
			}
		}()
	}
	go func() {
		for i := range ents {
			jobs <- i
		}
		close(jobs)
	}()
	for range ents {
		<-fin
		done++
		innerSet(float64(done), 0)
	}
	close(fin)
	endInner()
	nh := 0
	for i := range S.Harm {
		if S.Harm[i] {
			nh++
		}
	}
	for _, p := range parts {
		rec.Energy = append(rec.Energy, p.Energy)
		rec.RecErr = math.Max(rec.RecErr, p.RecErr)
	}
	logf("      seed (phase-corr per digit + %s assignment) + spectral Hodge for %d contributors: %.2fs; harmonic bins=%d of %d, max|M^-(g+c+h)|=%.1e",
		"Re-Basin weight-matching", len(ents), time.Since(t0).Seconds(), nh, n, rec.RecErr)
	ws := make([]float64, len(ents))
	var wsum float64
	for i, e := range ents {
		ws[i] = e.W
		wsum += e.W
	}
	for i := range ws {
		ws[i] /= wsum
	}
	var others []int
	for i := range ents {
		if i != bi {
			others = append(others, i)
		}
	}
	// gradient: global reorder/scale absorbed by alignment -> base frame's gradient part
	gradG := realPart(fftn(parts[bi].GradF, dims, true))
	curlB := realPart(fftn(parts[bi].CurlF, dims, true))
	curl := append([]float64(nil), curlB...)
	if len(others) > 0 {
		t1 := time.Now()
		var dl [][]float64
		var wo []float64
		for _, i := range others {
			ci := realPart(fftn(parts[i].CurlF, dims, true))
			for k := range ci {
				ci[k] -= curlB[k]
			}
			dl = append(dl, ci)
			wo = append(wo, ws[i])
		}
		tt := ternaryTIESAll(dl, wo, 1) // ws are normalised over every contributor, the base included
		for k := range curl {
			curl[k] += tt[k]
		}
		logf("      ternary TIES on curl part (%d deltas): %.2fs", len(dl), time.Since(t1).Seconds())
	}
	Href := append([]complex128(nil), parts[bi].HarmF...)
	var modes []int
	for i := range S.Harm {
		if S.Harm[i] {
			modes = append(modes, i)
		}
	}
	if len(others) == 0 || len(modes) == 0 {
		hg := realPart(fftn(Href, dims, true))
		out := make([]float64, n)
		for k := range out {
			out[k] = gradG[k] + curl[k] + hg[k]
		}
		rec.R = []float64{1}
		rec.DFT = Href
		if len(modes) == 0 {
			logf("      no low/harmonic bins on grid %v: no Kuramoto lock", G)
		}
		return out, rec
	}
	allm := append([]int{bi}, others...)
	type osc struct{ i, m int }
	var oscs []osc
	for _, i := range allm {
		for _, m := range modes {
			oscs = append(oscs, osc{i, m})
		}
	}
	no := len(oscs)
	level := make([]int, no)
	for k, o := range oscs {
		level[k] = S.Level[o.m]
	}
	// Eq.(15): c_ij = 1 within an MRH bucket or digit level, 2^-4 across
	Cm := make([]float64, no*no)
	for a := 0; a < no; a++ {
		for b := 0; b < no; b++ {
			if a == b {
				continue
			}
			c := sp.K.Weak
			if ents[oscs[a].i].Bucket == ents[oscs[b].i].Bucket || level[a] == level[b] {
				c = 1
			}
			Cm[a*no+b] = c
		}
	}
	Kb := couplingMatrix(Cm, no, sp.K.GTheta)
	bidx := map[int]int{}
	for k, o := range oscs {
		if o.i == bi {
			bidx[o.m] = k
		}
	}
	refOf := make([]int, no)
	for k, o := range oscs {
		refOf[k] = bidx[o.m]
	}
	prevH := append([]complex128(nil), Href...)
	var theta []float64
	var prevR float64
	maxOuter := S.NLevel + 1 // Eq.(20)
	for ps := 0; ps < maxOuter; ps++ {
		tp := time.Now()
		// Paper 3.1: Anneal (step 2) once from the seed phases; Refine (step 3) continues the lock from the
		// previous pass's state (same lock minimum) after each Hodge re-projection.
		kp := sp.K
		kp.Anneal = ps == 0
		kp.MaxWin = S.NLevel + 2 // Eq.(18)
		if ps == 0 {
			theta = make([]float64, no)
			for k, o := range oscs {
				theta[k] = cmplx.Phase(parts[o.i].F[o.m]) - cmplx.Phase(Href[o.m])
			}
		}
		kr := kuramotoTick(theta, Kb, level, refOf, kp, fmt.Sprintf("kuramoto pass %d", ps+1))
		// Eq.(19) merge: phases read in the base oscillator's frame; only the base's lock cluster merges;
		// per-mode order parameter r_m sets the mixing weight; unlocked modes stay with the base.
		Hn := append([]complex128(nil), Href...)
		merged := 0
		nm := len(modes)
		for mi, m := range modes {
			kb := bidx[m]
			var z complex128
			for a := range allm {
				z += cmplx.Exp(complex(0, kr.Theta[a*nm+mi]))
			}
			rm := cmplx.Abs(z) / float64(len(allm))
			if !kr.Locked[kb] {
				continue
			}
			var num complex128
			var den float64
			members := 0
			for a, i := range allm {
				k := a*nm + mi
				if !kr.Locked[k] || kr.Labels[k] != kr.Labels[kb] {
					continue
				}
				members++
				c := complex(cmplx.Abs(parts[i].F[m]), 0) * cmplx.Exp(complex(0, cmplx.Phase(parts[bi].F[m])+kr.DPhase[k]))
				num += complex(ws[i], 0) * c
				den += ws[i]
			}
			if members >= 2 && den > 0 {
				Hn[m] = complex(1-rm, 0)*Href[m] + complex(rm, 0)*num/complex(den, 0)
				merged++
			}
		}
		// re-project onto the harmonic part (Hodge decomposition each pass)
		for i := range Hn {
			if !S.Harm[i] {
				Hn[i] = 0
			}
		}
		Href = Hn
		var dn, pn float64
		for i := range Href {
			dn += sq(cmplx.Abs(Href[i] - prevH[i]))
			pn += sq(cmplx.Abs(prevH[i]))
		}
		dh := math.Sqrt(dn) / (math.Sqrt(pn) + 1e-300)
		if os.Getenv("HM_DEBUG") != "" {
			for _, m := range modes {
				if d := cmplx.Abs(Href[m] - prevH[m]); d > 1e-6 {
					fmt.Fprintf(os.Stderr, "DBG mode %d lam=%.3f lvl=%d prev=%v new=%v\n", m, S.Lam[m], S.Level[m], prevH[m], Href[m])
				}
			}
		}
		nl := 0
		for _, x := range kr.Locked {
			if x {
				nl++
			}
		}
		pr := PassRec{R: kr.R, Steps: kr.Steps, Converged: kr.Converged, Locked: nl, NOsc: no, NModes: len(modes), Merged: merged, Dh: dh, Secs: time.Since(tp).Seconds()}
		rec.Passes = append(rec.Passes, pr)
		logf("      lock pass %d: <r>=%.6f locked=%d/%d merged modes=%d/%d steps=%d converged=%v dh=%.2e (%.2fs)",
			ps+1, kr.R, nl, no, merged, len(modes), kr.Steps, kr.Converged, dh, pr.Secs)
		if ps > 0 && kr.Converged && dh < sp.K.RTol && math.Abs(kr.R-prevR) < sp.K.RTol {
			break
		}
		prevR = kr.R
		prevH = append([]complex128(nil), Href...)
	}
	hg := realPart(fftn(Href, dims, true))
	out := make([]float64, n)
	for k := range out {
		out[k] = gradG[k] + curl[k] + hg[k]
	}
	for _, p := range rec.Passes {
		rec.R = append(rec.R, p.R)
	}
	rec.DFT = fftn(toC(hg), dims, false)
	return out, rec
}

// Paper: no equation (helper: I/O, progress or utility)
func sq(x float64) float64 { return x * x }

// Paper: no equation (helper: I/O, progress or utility)
func identity(n int) []int {
	o := make([]int, n)
	for i := range o {
		o[i] = i
	}
	return o
}

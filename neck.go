package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"hodgemix/eval"
)

// packNeck writes out: tern's tensors, plus extra MRH limbs of the float checkpoint's
// packed weights. Limb 0 stays tern's ternary code. Each further limb is the neck-cut,
// Kuramoto-locked residual (residualLimbs). max is the limb count including limb 0.
func packNeck(ternPath, floatPath, outPath string, max int, tol float64) error {
	base, err := OpenST(floatPath)
	if err != nil {
		return err
	}
	src, err := OpenST(ternPath)
	if err != nil {
		return err
	}
	w, err := NewSTW(outPath, src.Meta)
	if err != nil {
		return err
	}
	was := quiet
	quiet = true
	defer func() { quiet = was }()
	t0 := time.Now()
	nLimb := 0
	for _, k := range src.Keys {
		if isLimbTensor(k) {
			continue
		}
		b, err := src.Raw(k)
		if err != nil {
			return err
		}
		ti := src.H[k]
		w.Add(k, ti.Dtype, append([]int(nil), ti.Shape...), b)
		if !src.isPacked(k) || !base.Has(k) || !eqShape(base.LogicalShape(k), src.LogicalShape(k)) {
			continue
		}
		W, _, err := base.LoadDense(k)
		if err != nil {
			return err
		}
		sh := src.LogicalShape(k)
		R, C := sh[0], sh[1]
		if R*C != len(W) {
			return fmt.Errorf("%s: float %d, shape %v", k, len(W), sh)
		}
		sc, err := src.F32(scaleKey(k))
		if err != nil || len(sc) != 1 {
			return fmt.Errorf("%s scale: %v", k, err)
		}
		ts := time.Now()
		limbs, rel := residualLimbs(W, unpackTrits(b, R, C), float64(sc[0]), max, tol, R, C)
		writeLimbs(w, k, R, C, limbs)
		nLimb += len(limbs)
		rs := make([]string, len(rel))
		for i, r := range rel {
			rs[i] = fmt.Sprintf("%.3f", r)
		}
		fmt.Fprintf(os.Stderr, "%s extra=%d rel=%v %.1fs\n", k, len(limbs), rs, time.Since(ts).Seconds())
	}
	if err := w.Close(nil); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s tensors-with-limbs sum=%d in %.0fs\n", outPath, nLimb, time.Since(t0).Seconds())
	return nil
}

// appendSing copies neckPath and adds neck limbs for the singular set the body limbs left at
// center. It does not refit or re-lock those body limbs. The float checkpoint is floatPath.
// OUT has to sit next to the Champ config.json: champCheck reads that file.
func appendSing(neckPath, floatPath, outPath string) error {
	base, err := OpenST(floatPath)
	if err != nil {
		return err
	}
	src, err := OpenST(neckPath)
	if err != nil {
		return err
	}
	w, err := NewSTW(outPath, src.Meta)
	if err != nil {
		return err
	}
	t0 := time.Now()
	for _, k := range src.Keys {
		b, err := src.Raw(k)
		if err != nil {
			return err
		}
		ti := src.H[k]
		w.Add(k, ti.Dtype, append([]int(nil), ti.Shape...), b)
	}
	nAdd := 0
	for _, k := range src.Keys {
		if !src.isPacked(k) || isLimbTensor(k) || !base.Has(k) || !eqShape(base.LogicalShape(k), src.LogicalShape(k)) {
			continue
		}
		W, _, err := base.LoadDense(k)
		if err != nil {
			return err
		}
		sh := src.LogicalShape(k)
		R, C := sh[0], sh[1]
		if R*C != len(W) {
			return fmt.Errorf("%s: float %d, shape %v", k, len(W), sh)
		}
		b, err := src.Raw(k)
		if err != nil {
			return err
		}
		sc, err := src.F32(scaleKey(k))
		if err != nil || len(sc) != 1 {
			return fmt.Errorf("%s scale: %v", k, err)
		}
		q0 := unpackTrits(b, R, C)
		gap := make([]float32, len(W))
		for i, v := range q0 {
			gap[i] = W[i] - float32(float64(sc[0])*float64(v))
		}
		sing := singularNeck(gap)
		D, _, err := src.LoadDense(k)
		if err != nil {
			return err
		}
		var before, wn float64
		for i := range W {
			d := float64(W[i] - D[i])
			before += d * d
			wn += float64(W[i]) * float64(W[i])
			if sing != nil && sing[i] {
				gap[i] = W[i] - D[i]
			} else {
				gap[i] = 0
			}
		}
		limbs := neckLimbs(gap, sing, NeckLimbs)
		for _, lb := range limbs {
			c := (lb.r - 1) / 2
			for i, dig := range lb.d {
				D[i] += float32(lb.s * float64(int(dig)-c))
			}
		}
		var after float64
		for i := range W {
			d := float64(W[i] - D[i])
			after += d * d
		}
		have := len(src.limbsOf(k))
		writeLimbsFrom(w, k, R, C, have+1, limbs)
		nAdd += len(limbs)
		den := math.Sqrt(math.Max(wn, 1e-30))
		fmt.Fprintf(os.Stderr, "%s neck-limbs=%d rel %.3f -> %.3f\n", k, len(limbs), math.Sqrt(before)/den, math.Sqrt(after)/den)
	}
	if err := w.Close(nil); err != nil {
		return err
	}
	cfg := filepath.Join(filepath.Dir(outPath), "config.json")
	if bad, _ := champCheck(outPath, cfg); len(bad) > 0 {
		return fmt.Errorf("champ:\n  %s", strings.Join(bad, "\n  "))
	}
	fmt.Fprintf(os.Stderr, "wrote %s neck-limbs=%d in %.0fs\n", outPath, nAdd, time.Since(t0).Seconds())
	return nil
}

// baseLlama3 reports whether the run file's config for base is a Llama 3 arch.
// No config set means it is not: nothing is inferred from the tensor file.
func baseLlama3(basePath string) (bool, error) {
	cp := eval.FilesOf(basePath).Config
	if cp == "" {
		return false, nil
	}
	ok, _, err := llama3Config(cp)
	return ok, err
}

// llama3Config reads a config.json. ok is rope_type llama3.
func llama3Config(path string) (ok bool, modelType string, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return false, "", err
	}
	var cfg map[string]any
	if err := json.Unmarshal(b, &cfg); err != nil {
		return false, "", fmt.Errorf("%s: %v", path, err)
	}
	mt, _ := cfg["model_type"].(string)
	return llama3Arch(cfg), mt, nil
}

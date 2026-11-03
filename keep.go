package main

// --keep: one pass over --roles. Each role is mixed alone (every other tensor stays the
// base's, copied) and scored with the reference forward. A role is kept when its
// perplexity is at most the base's. The kept roles are mixed once into OUT. The score
// is taken after the ternary file is written. It is not an input to the mix.
// No paper equation.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"hodgemix/eval"
)

type keepRow struct {
	Role   string  `json:"role"`
	PPL    float64 `json:"ppl"`
	Ratio  float64 `json:"ppl_vs_base"`
	Keep   bool    `json:"keep"`
	Greedy string  `json:"greedy,omitempty"`
	Err    string  `json:"error,omitempty"`
}

type keepReport struct {
	BasePPL float64   `json:"base_ppl"`
	OutPPL  float64   `json:"out_ppl"`
	Ratio   float64   `json:"ppl_vs_base"`
	Kept    []string  `json:"kept"`
	Roles   []keepRow `json:"roles"`
	tok     string
}

// keepRole is the accept rule. A role at the base perplexity is kept. A score
// that is not a number is not kept.
func keepRole(basePPL, rolePPL float64) bool {
	if math.IsNaN(basePPL) || math.IsNaN(rolePPL) {
		return false
	}
	return rolePPL <= basePPL
}

func pplRatio(p, base float64) float64 {
	if base == 0 || math.IsNaN(base) || math.IsNaN(p) {
		return math.NaN()
	}
	return p / base
}

func keep(c runCfg, basePath string, srcs []string, out, outTern, outMerged string, noVerify bool, tok string) {
	base, err := eval.Run(basePath, tok)
	if err != nil {
		die("keep: the base cannot be scored (%v)", err)
	}
	logScore("base", base)
	bst, err := OpenST(basePath)
	if err != nil {
		die("%v", err)
	}
	defer bst.Close()
	has := map[string]bool{}
	for _, k := range bst.Keys {
		if sh := bst.LogicalShape(k); len(sh) == 2 {
			has[roleOf(k)] = true
		}
	}
	dir := out + ".keep"
	if err := os.MkdirAll(dir, 0o755); err != nil {
		die("%v", err)
	}
	greedy := func(r *eval.Result) string {
		if len(r.Gens) == 0 {
			return ""
		}
		return r.Gens[0][1]
	}
	rec := &keepReport{BasePPL: base.PPL, Kept: []string{}, Roles: []keepRow{}, tok: tok}
	for _, role := range c.roles {
		if !has[role] {
			continue
		}
		logf("KEEP  role %s alone", role)
		rc := c
		rc.roles, rc.evalTok, rc.noBNZ, rc.keepRec = []string{role}, "", true, nil
		o := filepath.Join(dir, role+".bnz")
		run(rc, basePath, srcs, o, "", "", true)
		row := keepRow{Role: role}
		if r, err := eval.Run(o+".ternary.safetensors", tok); err != nil {
			row.Err = err.Error()
		} else {
			logScore(role, r)
			row.PPL, row.Ratio = r.PPL, pplRatio(r.PPL, base.PPL)
			row.Keep = keepRole(base.PPL, r.PPL)
			row.Greedy = greedy(r)
			if row.Keep {
				rec.Kept = append(rec.Kept, role)
			}
		}
		rec.Roles = append(rec.Roles, row)
	}
	if err := os.RemoveAll(dir); err != nil {
		die("keep: %v", err)
	}
	logf("KEEP  %s + %s, bar ppl %.1f (prompt %q):", filepath.Base(basePath), strings.Join(srcs, ", "), base.PPL, eval.Prompts[0])
	logf("  %-8s %14s %10s %6s  %s", "role", "ppl", "x base", "keep", "greedy")
	for _, r := range rec.Roles {
		if r.Err != "" {
			logf("  %-8s %s", r.Role, r.Err)
			continue
		}
		logf("  %-8s %14.1f %10.2f %6t  %q", r.Role, r.PPL, r.Ratio, r.Keep, r.Greedy)
	}
	if len(rec.Kept) == 0 {
		logf("KEEP  no role at or under the base; the output copies the base")
	} else {
		logf("KEEP  mixing %s", strings.Join(rec.Kept, ","))
	}
	rc := c
	rc.roles = rec.Kept
	rc.keepRec = rec
	run(rc, basePath, srcs, out, outTern, outMerged, noVerify)
	b, err := json.MarshalIndent(jsonSafe(reflect.ValueOf(rec).Elem()), "", " ")
	if err != nil {
		die("keep: %v", err)
	}
	if err := os.WriteFile(out+".keep.json", b, 0o644); err != nil {
		die("%v", err)
	}
	logf("KEEP  output ppl %.1f vs base %.1f (x%.2f)  table: %s", rec.OutPPL, rec.BasePPL, rec.Ratio, out+".keep.json")
}

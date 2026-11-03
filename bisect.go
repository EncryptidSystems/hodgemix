package main

// --bisect: which role breaks a mix. Each of --roles is mixed alone (every other tensor is the
// base's, copied), the ternary output is scored with the reference forward and
// the table role -> ppl is printed and kept in OUT.bisect.json. No paper equation: a diagnostic.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"hodgemix/eval"
)

type bisectRow struct {
	Role   string  `json:"role"`
	PPL    float64 `json:"ppl"`
	Ratio  float64 `json:"ppl_vs_base"`
	Greedy string  `json:"greedy"`
	Err    string  `json:"error,omitempty"`
}

func bisect(c runCfg, basePath string, srcs []string, out, tok string) {
	base, err := eval.Run(basePath, tok)
	if err != nil {
		die("bisect: the base cannot be scored (%v)", err)
	}
	logScore("base", base)
	bst, err := OpenST(basePath)
	if err != nil {
		die("%v", err)
	}
	has := map[string]bool{}
	for _, k := range bst.Keys {
		if sh := bst.LogicalShape(k); len(sh) == 2 {
			has[roleOf(k)] = true
		}
	}
	dir := out + ".bisect"
	if err := os.MkdirAll(dir, 0o755); err != nil {
		die("%v", err)
	}
	greedy := func(r *eval.Result) string {
		if len(r.Gens) == 0 {
			return ""
		}
		return r.Gens[0][1]
	}
	rows := []bisectRow{{Role: "(base)", PPL: base.PPL, Ratio: 1, Greedy: greedy(base)}}
	for _, role := range c.roles {
		if !has[role] {
			continue
		}
		logf("BISECT  role %s alone", role)
		rc := c
		rc.roles, rc.evalTok, rc.noBNZ = []string{role}, "", true
		o := filepath.Join(dir, role+".bnz")
		run(rc, basePath, srcs, o, "", "", true)
		row := bisectRow{Role: role}
		if r, err := eval.Run(o+".ternary.safetensors", tok); err != nil {
			row.Err = err.Error()
		} else {
			logScore(role, r)
			row.PPL, row.Ratio, row.Greedy = r.PPL, r.PPL/base.PPL, greedy(r)
		}
		rows = append(rows, row)
		if m, _ := filepath.Glob(o + "*"); m != nil {
			for _, f := range m {
				os.Remove(f)
			}
		}
	}
	os.Remove(filepath.Join(dir, "config.json"))
	for _, f := range []string{"generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"} {
		os.Remove(filepath.Join(dir, f))
	}
	os.Remove(dir)
	logf("BISECT  %s + %s, each role mixed alone (prompt %q):", filepath.Base(basePath), strings.Join(srcs, ", "), eval.Prompts[0])
	logf("  %-8s %14s %10s  %s", "role", "ppl", "x base", "greedy")
	for _, r := range rows {
		if r.Err != "" {
			logf("  %-8s %s", r.Role, r.Err)
			continue
		}
		logf("  %-8s %14.1f %10.2f  %q", r.Role, r.PPL, r.Ratio, r.Greedy)
	}
	b, _ := json.MarshalIndent(jsonSafe(reflect.ValueOf(rows)), "", " ")
	os.WriteFile(out+".bisect.json", b, 0o644)
	logf("  table: %s", out+".bisect.json")
}

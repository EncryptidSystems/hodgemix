package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"hodgemix/eval"
)

func TestKeepRole(t *testing.T) {
	if !keepRole(12, 12) || !keepRole(12, 11.9) {
		t.Fatal("a role at or under the bar is kept")
	}
	if keepRole(12, 14.4) || keepRole(12, math.NaN()) || keepRole(math.NaN(), 1) {
		t.Fatal("a role over the bar, or a score that is not a number, is not kept")
	}
}

func TestKeepPassTiny(t *testing.T) {
	dir := tempDir(t)
	m := tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 260}
	a := writeTiny(t, filepath.Join(dir, "a"), m, 1)
	b := writeTiny(t, filepath.Join(dir, "b"), m, 2)
	// A role pass copies every other tensor from the base. Champ loads those
	// only when the base is already packed ternary, which is what --bisect mixes.
	full := filepath.Join(dir, "full", "mix.bnz")
	run(tinyCfg("base"), a, []string{a}, full, "", "", true)
	base := full + ".ternary.safetensors"
	out := filepath.Join(dir, "out", "mix.bnz")
	c := tinyCfg("base")
	c.roles = []string{"q", "down"}
	tok := eval.FilesOf(base).Tokenizer
	if tok == "" {
		t.Fatal("ternary base has no tokenizer")
	}
	keep(c, base, []string{b}, out, "", "", true, tok)
	if _, err := os.Stat(out); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(out + ".ternary.safetensors"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(out + ".keep"); err == nil {
		t.Fatal("role directory left behind")
	}
	bjson, err := os.ReadFile(out + ".keep.json")
	if err != nil {
		t.Fatal(err)
	}
	var rec keepReport
	if err := json.Unmarshal(bjson, &rec); err != nil {
		t.Fatal(err)
	}
	if rec.BasePPL == 0 || math.IsNaN(rec.BasePPL) {
		t.Fatalf("base ppl %v", rec.BasePPL)
	}
	kept := map[string]bool{}
	for _, role := range rec.Kept {
		kept[role] = true
	}
	if len(rec.Roles) != 2 {
		t.Fatalf("roles %+v", rec.Roles)
	}
	for _, row := range rec.Roles {
		if row.Err != "" {
			t.Fatalf("%s: %s", row.Role, row.Err)
		}
		if row.Keep != keepRole(rec.BasePPL, row.PPL) || kept[row.Role] != row.Keep {
			t.Fatalf("%s keep %v ppl %v bar %v", row.Role, row.Keep, row.PPL, rec.BasePPL)
		}
	}
	rb, err := os.ReadFile(out + ".report.json")
	if err != nil {
		t.Fatal(err)
	}
	var report map[string]any
	if err := json.Unmarshal(rb, &report); err != nil {
		t.Fatal(err)
	}
	k, ok := report["keep"].(map[string]any)
	if !ok || k["base_ppl"] == nil || k["out_ppl"] == nil {
		t.Fatalf("report keep %v", report["keep"])
	}
}

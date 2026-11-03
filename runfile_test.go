package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hodgemix/eval"
)

// A run file sets every model's files (relative paths from the run file's folder), one model's shards
// as a list, the source weights, the output and the settings; an unknown key or a missing file is an
// error.
func TestRunFile(t *testing.T) {
	dir := tempDir(t)
	for _, f := range []string{"b.safetensors", "tok (1).json", "cfg (2).json", "s-1.safetensors", "s-2.safetensors", "s.json"} {
		os.WriteFile(filepath.Join(dir, f), []byte("{}"), 0o644)
	}
	rf := filepath.Join(dir, "run.yaml")
	os.WriteFile(rf, []byte(`base:
  model: b.safetensors
  tokenizer: tok (1).json
  config: cfg (2).json
sources:
  - model: [s-1.safetensors, s-2.safetensors]
    weight: 0.5
    tokenizer: s.json
    config: cfg (2).json
out: o.bnz
settings:
  limbs: 1
  repair: off
`), 0o644)
	args, err := loadRunFile(rf)
	if err != nil {
		t.Fatal(err)
	}
	b, s := filepath.Join(dir, "b.safetensors"), filepath.Join(dir, "s-1.safetensors")
	want := []string{"--base", b, "--src", s + ":0.5", "--out", filepath.Join(dir, "o.bnz"), "--limbs=1", "--repair=off"}
	if strings.Join(args, "|") != strings.Join(want, "|") {
		t.Fatalf("args %q\nwant %q", args, want)
	}
	if f := eval.FilesOf(b); f.Tokenizer != filepath.Join(dir, "tok (1).json") || f.Config != filepath.Join(dir, "cfg (2).json") || f.TokenizerConfig != "" {
		t.Fatalf("base files %+v", f)
	}
	if sh := eval.ShardsOf(s); len(sh) != 2 || sh[1] != filepath.Join(dir, "s-2.safetensors") {
		t.Fatalf("source shards %v", sh)
	}
	os.WriteFile(rf, []byte("base:\n  model: b.safetensors\n  tokenizers: x.json\n"), 0o644)
	if _, err := loadRunFile(rf); err == nil {
		t.Fatal("unknown key accepted")
	}
	os.WriteFile(rf, []byte("base:\n  model: b.safetensors\n  config: missing.json\n"), 0o644)
	if _, err := loadRunFile(rf); err == nil {
		t.Fatal("missing file accepted")
	}
	os.WriteFile(rf, []byte("base:\n  model: b.safetensors\n  tokenizer: tok (1).json\n"), 0o644)
	if _, err := loadRunFile(rf); err == nil || !strings.Contains(err.Error(), "no config set") {
		t.Fatalf("missing config: %v", err)
	}
}

func TestWriteRunFileKeepsConfig(t *testing.T) {
	dir := tempDir(t)
	model := filepath.Join(dir, "m.safetensors")
	neck := filepath.Join(dir, "m.neck.safetensors")
	cfg := filepath.Join(dir, "config.json")
	tok := filepath.Join(dir, "tokenizer.json")
	tokc := filepath.Join(dir, "tokenizer_config.json")
	for _, p := range []string{model, neck, cfg, tok, tokc} {
		if err := os.WriteFile(p, []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	out := filepath.Join(dir, "out.yaml")
	f := eval.ModelFiles{Tokenizer: tok, TokenizerConfig: tokc, Config: cfg}
	if err := writeRunFile(out, []runEntry{runEntryOf(neck, f), runEntryOf(model, f)}); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	nCfg := 0
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "config:") {
			nCfg++
		}
	}
	if nCfg != 2 {
		t.Fatalf("config not set on every model:\n%s", b)
	}
	if _, err := loadRunFile(out); err != nil {
		t.Fatal(err)
	}
	if got := eval.FilesOf(neck).Config; got != cfg {
		t.Fatalf("neck config %q", got)
	}
	if got := eval.FilesOf(model).Config; got != cfg {
		t.Fatalf("ternary config %q", got)
	}
	if err := writeRunFile(out, []runEntry{runEntryOf(model, eval.ModelFiles{Tokenizer: tok})}); err == nil {
		t.Fatal("empty config written")
	}
}

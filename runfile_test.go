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
}

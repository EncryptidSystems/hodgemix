//go:build splice

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hodgemix/bnz4"
)

// Writes Gemma 4's audio and vision tensors into the Hodge-4 safetensors.
// Not part of the default test run.
func TestSpliceHodge4Towers(t *testing.T) {
	dir := `C:\Users\grego\models\dev\hodge-4`
	dst := filepath.Join(dir, "hodge-4.bnz.ternary.safetensors")
	src := `C:\Users\grego\Downloads\model.safetensors`
	out := filepath.Join(dir, "hodge-4.with-towers.safetensors")
	base, err := OpenST(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	gemma, err := OpenST(src)
	if err != nil {
		t.Fatal(err)
	}
	defer gemma.Close()
	W, err := NewSTW(out, base.Meta)
	if err != nil {
		t.Fatal(err)
	}
	already := map[string]bool{}
	for i, k := range base.Keys {
		b, err := base.Raw(k)
		if err != nil {
			t.Fatal(err)
		}
		W.Add(k, base.H[k].Dtype, append([]int(nil), base.H[k].Shape...), b)
		already[k] = true
		if i%100 == 0 {
			t.Logf("copied %d/%d base tensors", i, len(base.Keys))
		}
	}
	n, err := copyTowers([]*ST{gemma}, W, nil, already)
	if err != nil {
		t.Fatal(err)
	}
	if err := W.Close(nil); err != nil {
		t.Fatal(err)
	}
	got, err := OpenST(out)
	if err != nil {
		t.Fatal(err)
	}
	vis, aud, emb := 0, 0, 0
	for _, k := range got.Keys {
		if strings.HasPrefix(k, "bnzmix.") {
			continue
		}
		if strings.Contains(k, "vision_tower") || strings.Contains(k, "embed_vision") {
			vis++
		}
		if strings.Contains(k, "audio_tower") || strings.Contains(k, "embed_audio") {
			aud++
		}
		if strings.Contains(k, "embed_tokens_per_layer") || strings.Contains(k, "per_layer_") ||
			strings.HasSuffix(k, "layer_scalar") || strings.Contains(k, "post_per_layer_input_norm") ||
			k == "model.language_model.embed_tokens.weight" {
			emb++
		}
	}
	got.Close()
	base.Close()
	gemma.Close()
	t.Logf("carried %d tensors; file has %d tensors, vision %d, audio %d, embedding %d", n, len(got.Keys), vis, aud, emb)
	if vis == 0 || aud == 0 || emb < 100 {
		t.Fatalf("carry %d vision %d audio %d embedding %d", n, vis, aud, emb)
	}
	if _, err := os.Stat(dst + ".pre-towers"); err == nil {
		t.Fatal("a pre-towers backup is already there; not replacing the model")
	}
	if err := os.Rename(dst, dst+".pre-towers"); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(out, dst); err != nil {
		t.Fatal(err)
	}
	model := filepath.Join(dir, "model.safetensors")
	if err := os.Rename(model, model+".pre-towers"); err != nil {
		t.Fatal(err)
	}
	in, err := os.Open(dst)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	f, err := os.Create(model)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.ReadFrom(in); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

// Rewrites hodge-4.bnz from the safetensors that now hold the towers and embeddings.
func TestCompressHodge4(t *testing.T) {
	dir := `C:\Users\grego\models\dev\hodge-4`
	src := filepath.Join(dir, "hodge-4.bnz.ternary.safetensors")
	dst := filepath.Join(dir, "hodge-4.bnz")
	tmp := dst + ".new"
	lf, err := os.Create(dst + ".new.log.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	n, err := bnz4.Compress(src, tmp, lf)
	lf.Close()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(dst, dst+".pre-towers"); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, dst); err != nil {
		t.Fatal(err)
	}
	t.Logf("bnz %d bytes", n)
}

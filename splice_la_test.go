//go:build splicela

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hodgemix/bnz4"
)

// Adds the L-A vision and audio towers and their embedding layers (writeTowers)
// to Hodge-4's ternary safetensors and packs that as BNZ. Both files are new;
// the existing model files are only read. Not part of the default test run:
//
//	go test -tags splicela -run TestSpliceHodge4LA -timeout 3h -v
func TestSpliceHodge4LA(t *testing.T) {
	dir := `C:\Users\grego\models\dev\hodge-4`
	src := filepath.Join(dir, "hodge-4.bnz.ternary.safetensors")
	out := filepath.Join(dir, "hodge-4.la.ternary.safetensors")
	bnz := filepath.Join(dir, "hodge-4.la.bnz")
	for _, p := range []string{out, bnz} {
		if _, err := os.Stat(p); err == nil {
			t.Fatalf("%s is already there; not replacing it", p)
		}
	}
	base, err := OpenST(src)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	W, err := NewSTW(out, base.Meta)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range base.Keys {
		if asideTensor(k) {
			t.Fatalf("%s already has a tower tensor (%s)", src, k)
		}
		b, err := base.Raw(k)
		if err != nil {
			t.Fatal(err)
		}
		W.Add(k, base.H[k].Dtype, append([]int(nil), base.H[k].Shape...), b)
	}
	tw, err := writeTowers(W, base, src)
	if err != nil {
		t.Fatal(err)
	}
	if err := W.Close(nil); err != nil {
		t.Fatal(err)
	}
	t.Logf("towers %v", tw)
	got, err := OpenST(out)
	if err != nil {
		t.Fatal(err)
	}
	added := 0
	for _, k := range got.Keys {
		if strings.HasPrefix(k, "bnzmix.vision_tower") || strings.HasPrefix(k, "bnzmix.audio_tower") ||
			strings.HasPrefix(k, "bnzmix.embed_vision") || strings.HasPrefix(k, "bnzmix.embed_audio") {
			added++
		}
	}
	got.Close()
	if added == 0 || len(got.Keys) != len(base.Keys)+added {
		t.Fatalf("file has %d tensors, base %d, towers %d", len(got.Keys), len(base.Keys), added)
	}
	t.Logf("added %d tower tensors", added)

	lf, err := os.Create(bnz + ".log.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	n, err := bnz4.Compress(out, bnz, lf)
	lf.Close()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: %d B", bnz, n)
	rt := bnz + ".roundtrip.safetensors"
	_, ok, err := bnz4.Decompress(bnz, rt)
	if err == nil && ok {
		ok = sameFile(rt, out)
	}
	os.Remove(rt)
	if err != nil || !ok {
		t.Fatalf("BNZ round-trip failed: %v", err)
	}
	t.Log("round-trip OK")
}

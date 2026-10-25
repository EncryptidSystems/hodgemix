//go:build spliceav

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hodgemix/bnz4"
)

// Assimilates Gemma 4's encoders into Hodge-4 with the L-A towers (hodge-4.la) and adds the
// 1536 -> 2560 bridge, then packs BNZ (assimtowers.go). Both outputs are new files; the
// inputs are only read. Not part of the default test run:
//
//	go test -tags spliceav -run TestSpliceHodge4AV -timeout 4h -v
func TestSpliceHodge4AV(t *testing.T) {
	dir := `C:\Users\grego\models\dev\hodge-4`
	src := filepath.Join(dir, "hodge-4.la.ternary.safetensors")
	gemma := `C:\Users\grego\Downloads\model.safetensors`
	baseTok := filepath.Join(dir, "tokenizer.json")
	gemmaTok := `C:\Users\grego\Downloads\tokenizer (2).json`
	out := filepath.Join(dir, "hodge-4.av.ternary.safetensors")
	bnz := filepath.Join(dir, "hodge-4.av.bnz")
	for _, p := range []string{out, bnz} {
		if _, err := os.Stat(p); err == nil {
			t.Fatalf("%s is already there; not replacing it", p)
		}
	}
	MaxLimbs = 2 // runs/hodge-4.yaml limbs: 2
	base, err := OpenST(src)
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	g, err := OpenST(gemma)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	W, err := NewSTW(out, base.Meta)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, k := range base.Keys {
		if gemmaTowerTensor(k) {
			t.Fatalf("%s already has encoder tensor %s", src, k)
		}
		b, err := base.Raw(k)
		if err != nil {
			t.Fatal(err)
		}
		W.Add(k, base.H[k].Dtype, append([]int(nil), base.H[k].Shape...), b)
		seen[k] = true
	}
	enc, err := assimilateGemmaTowers(W, g, seen)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("encoders %v", enc)
	br, err := writeBridge(W, base, baseTok, g, "model.language_model.embed_tokens.weight", gemmaTok)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("bridge %v", br)
	if err := W.Close(nil); err != nil {
		t.Fatal(err)
	}
	got, err := OpenST(out)
	if err != nil {
		t.Fatal(err)
	}
	vis, aud, bridge := 0, 0, 0
	for _, k := range got.Keys {
		switch {
		case strings.HasPrefix(k, "model.vision_tower.") || strings.HasPrefix(k, "model.embed_vision."):
			vis++
		case strings.HasPrefix(k, "model.audio_tower.") || strings.HasPrefix(k, "model.embed_audio."):
			aud++
		case strings.HasPrefix(k, bridgeEmbed+"."):
			bridge++
		}
	}
	got.Close()
	if vis == 0 || aud == 0 || bridge == 0 {
		t.Fatalf("vision %d audio %d bridge %d tensors", vis, aud, bridge)
	}
	t.Logf("tensors: vision %d, audio %d, bridge %d; file %d", vis, aud, bridge, len(got.Keys))

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

package main

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func pbVarint(v uint64) []byte {
	var out []byte
	for v >= 0x80 {
		out = append(out, byte(v)|0x80)
		v >>= 7
	}
	return append(out, byte(v))
}

func pbKey(field, wire int) []byte { return pbVarint(uint64(field<<3 | wire)) }

func pbBytes(field int, b []byte) []byte {
	return append(append(pbKey(field, 2), pbVarint(uint64(len(b)))...), b...)
}

func pbUint(field int, v uint64) []byte { return append(pbKey(field, 0), pbVarint(v)...) }

func pbPiece(text string, score float32, typ int) []byte {
	b := pbBytes(1, []byte(text))
	b = append(b, pbKey(2, 5)...)
	b = binary.LittleEndian.AppendUint32(b, math.Float32bits(score))
	if typ != typeNormal {
		b = append(b, pbUint(3, uint64(typ))...)
	}
	return b
}

// tinyModel is a Gemma-shaped model: <pad> <eos> <bos> <unk>, every byte, then pieces.
func tinyModel(extra ...piece) []byte {
	var m []byte
	add := func(p piece) { m = append(m, pbBytes(1, pbPiece(p.text, p.score, p.typ))...) }
	add(piece{"<pad>", 0, typeControl})
	add(piece{"<eos>", 0, typeControl})
	add(piece{"<bos>", 0, typeControl})
	add(piece{"<unk>", 0, typeUnknown})
	for b := 0; b < 256; b++ {
		add(piece{"<0x" + hex2(b) + ">", 0, typeByte})
	}
	for _, p := range extra {
		add(p)
	}
	trainer := append(pbUint(3, modelBPE), pbUint(35, 1)...)
	norm := append(pbBytes(1, []byte("identity")), pbUint(3, 0)...)
	norm = append(norm, pbUint(4, 0)...)
	m = append(m, pbBytes(2, trainer)...)
	m = append(m, pbBytes(3, norm)...)
	return m
}

func hex2(b int) string {
	const h = "0123456789ABCDEF"
	return string([]byte{h[b>>4], h[b&15]})
}

const tinyConfig = `{"add_bos_token": true, "add_eos_token": false, "bos_token": "<bos>", "eos_token": "<eos>",
 "added_tokens_decoder": {"0": {"content": "<pad>", "special": true}, "1": {"content": "<eos>", "special": true},
 "2": {"content": "<bos>", "special": true}, "3": {"content": "<unk>", "special": true}}}`

func TestConvert(t *testing.T) {
	dir := t.TempDir()
	mp := filepath.Join(dir, "tokenizer.model")
	cp := filepath.Join(dir, "tokenizer_config.json")
	op := filepath.Join(dir, "tokenizer.json")
	base := 4 + 256
	os.WriteFile(mp, tinyModel(
		piece{"▁", -1, typeNormal},
		piece{"t", -2, typeNormal},
		piece{"h", -3, typeNormal},
		piece{"th", -4, typeNormal},
		piece{"▁th", -5, typeNormal},
		piece{"é", -6, typeNormal},
		piece{"<start_of_turn>", 0, typeUserDefined},
	), 0o644)
	os.WriteFile(cp, []byte(tinyConfig), 0o644)
	if err := run(mp, cp, op); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(op)
	var j struct {
		Added []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
		Pre   map[string]any `json:"pre_tokenizer"`
		Post  map[string]any `json:"post_processor"`
		Model struct {
			Type   string         `json:"type"`
			Vocab  map[string]int `json:"vocab"`
			Merges [][2]string    `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &j); err != nil {
		t.Fatal(err)
	}
	alpha := byteAlphabet()
	// Every id has a key; a normal piece takes its byte, the byte piece keeps its name.
	if len(j.Model.Vocab) != base+7 {
		t.Fatalf("vocab %d keys, want %d", len(j.Model.Vocab), base+7)
	}
	if j.Model.Vocab[alpha[' ']] != base || j.Model.Vocab["<0x20>"] != 4+0x20 {
		t.Fatalf("space: %d, <0x20>: %d", j.Model.Vocab[alpha[' ']], j.Model.Vocab["<0x20>"])
	}
	if j.Model.Vocab[alpha['t']] != base+1 || j.Model.Vocab["<0x74>"] != 4+0x74 {
		t.Fatal("t did not take its byte")
	}
	if j.Model.Vocab[alpha[0x0a]] != 4+0x0a {
		t.Fatal("newline byte piece is not its byte letter")
	}
	if j.Model.Vocab[alpha[' ']+"th"] != base+4 {
		t.Fatal("▁th is not ' th' in the byte alphabet")
	}
	e := alpha[0xc3] + alpha[0xa9]
	if j.Model.Vocab[e] != base+5 {
		t.Fatal("é is not its two bytes")
	}
	// Merges by score: t h (th, -4) before Ġ th (▁th, -5) before é's bytes (-6).
	want := [][2]string{{"t", "h"}, {alpha[' '], "th"}, {alpha[0xc3], alpha[0xa9]}}
	if len(j.Model.Merges) < 3 {
		t.Fatalf("merges %v", j.Model.Merges)
	}
	for i, w := range want {
		if j.Model.Merges[i] != w {
			t.Fatalf("merge %d = %v, want %v (all %v)", i, j.Model.Merges[i], w, j.Model.Merges)
		}
	}
	// Added: the four specials, and the user-defined piece as text, not special.
	got := map[int]bool{}
	for _, a := range j.Added {
		got[a.ID] = a.Special
	}
	if len(j.Added) != 5 || !got[0] || !got[1] || !got[2] || !got[3] {
		t.Fatalf("added %v", j.Added)
	}
	if sp, ok := got[base+6]; !ok || sp {
		t.Fatal("<start_of_turn> should be an added, non-special token")
	}
	if !strings.Contains(string(raw), `"pattern": {"Regex": "(?i:'s|'t`) {
		t.Fatal("no Llama 3 Split")
	}
	single := j.Post["single"].([]any)
	first := single[0].(map[string]any)["SpecialToken"].(map[string]any)["id"]
	if first != "<bos>" {
		t.Fatalf("template starts with %v", first)
	}
}

func TestRefuses(t *testing.T) {
	var tc tokConfig
	json.Unmarshal([]byte(tinyConfig), &tc)
	m, err := parseModel(tinyModel())
	if err != nil {
		t.Fatal(err)
	}
	m.dummyPrefix = true
	if _, err := convert(m, &tc); err == nil || !strings.Contains(err.Error(), "dummy prefix") {
		t.Fatalf("dummy prefix: %v", err)
	}
	m.dummyPrefix = false
	m.modelType = 1
	if _, err := convert(m, &tc); err == nil || !strings.Contains(err.Error(), "not BPE") {
		t.Fatalf("unigram: %v", err)
	}
	m.modelType = modelBPE
	bad := tokConfig{}
	json.Unmarshal([]byte(`{"eos_token": "<eos>", "added_tokens_decoder": {"1": {"content": "</s>", "special": true}}}`), &bad)
	if _, err := convert(m, &bad); err == nil || !strings.Contains(err.Error(), "this model's piece 1") {
		t.Fatalf("other model's config: %v", err)
	}
}

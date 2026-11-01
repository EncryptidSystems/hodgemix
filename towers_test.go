package main

import (
	"bytes"
	"path/filepath"
	"testing"

	"hodgemix/eval"
	"hodgemix/tower"
)

// Audio and vision tensors from a contributor stay out of the text mix and out of
// the output. The output's towers are the L-A towers, and Champ accepts the file.
func TestSourceTowersAreNotWritten(t *testing.T) {
	dir := tempDir(t)
	base := writeTiny(t, filepath.Join(dir, "base"), tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 260}, 3)
	src := writeTiny(t, filepath.Join(dir, "src"), tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 260}, 4)
	st, err := OpenST(src)
	if err != nil {
		t.Fatal(err)
	}
	with := filepath.Join(dir, "src", "with-towers.safetensors")
	W, err := NewSTW(with, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range st.Keys {
		b, err := st.Raw(k)
		if err != nil {
			t.Fatal(err)
		}
		W.Add(k, st.H[k].Dtype, append([]int(nil), st.H[k].Shape...), b)
	}
	towers := map[string][]byte{
		"model.audio_tower.layers.0.self_attn.q_proj.linear.weight": {1, 2, 3, 4, 5, 6, 7, 8},
		"model.vision_tower.layers.0.mlp.fc1.weight":                {9, 9, 8, 8},
		"model.embed_vision.embedding_projection.weight":            {4, 0, 7, 0},
		"model.embed_audio.embedding_projection.weight":             {3, 1, 4, 1},
		"model.embed_tokens_per_layer.weight":                       {6, 0, 2, 0, 5, 0, 3, 0},
		"model.layers.0.per_layer_projection.weight":                {9, 0, 9, 0},
		"model.layers.0.layer_scalar":                               {1, 0},
		"model.layers.0.post_per_layer_input_norm.weight":           {2, 0},
	}
	shapes := map[string][]int{
		"model.audio_tower.layers.0.self_attn.q_proj.linear.weight": {2, 2},
		"model.vision_tower.layers.0.mlp.fc1.weight":                {1, 2},
		"model.embed_vision.embedding_projection.weight":            {1, 2},
		"model.embed_audio.embedding_projection.weight":             {1, 2},
		"model.embed_tokens_per_layer.weight":                       {2, 2},
		"model.layers.0.per_layer_projection.weight":                {1, 2},
		"model.layers.0.layer_scalar":                               {1},
		"model.layers.0.post_per_layer_input_norm.weight":           {1},
	}
	for name, b := range towers {
		W.Add(name, "BF16", shapes[name], b)
	}
	if err := W.Close(nil); err != nil {
		t.Fatal(err)
	}
	bf := eval.FilesOf(src)
	setFile(with, func(f *eval.ModelFiles) { *f = bf })
	out := filepath.Join(dir, "out", "mix.bnz")
	run(tinyCfg("base"), base, []string{with}, out, "", "", true)
	got, err := OpenST(out + ".ternary.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	if !got.Has("model.layers.0.self_attn.q_proj.weight") {
		t.Fatal("text q_proj missing")
	}
	for name := range towers {
		if got.Has(name) {
			t.Fatalf("%s is in the output", name)
		}
	}
	for _, name := range []string{tower.VisEmbed + ".weight", tower.AudEmbed + ".weight", tower.VisTower + ".sigma", tower.AudTower + ".sigma"} {
		if !got.Has(name) {
			t.Fatalf("missing L-A tensor %s", name)
		}
	}
	bad, notes := champCheck(out+".ternary.safetensors", champConfigPath(out))
	if len(bad) > 0 {
		t.Fatalf("champCheck: %v", bad)
	}
	for _, n := range notes {
		if bytes.Contains([]byte(n), []byte("audio, vision, embeddings")) {
			t.Fatalf("notes %v", notes)
		}
	}
}

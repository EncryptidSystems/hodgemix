package main

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hodgemix/eval"
)

// A one-layer gpt-oss checkpoint on the nibble table: conservation, the expert that fired,
// and the volume of the shared F cells.
func TestManifoldGPTOss(t *testing.T) {
	dir := tempDir(t)
	path := filepath.Join(dir, "model.safetensors")
	if err := writeOss(t, path); err != nil {
		t.Fatal(err)
	}
	cfg := filepath.Join(dir, "config.json")
	b, err := json.Marshal(map[string]any{
		"model_type": "gpt_oss", "hidden_size": 64, "intermediate_size": 64,
		"num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
		"num_hidden_layers": 1, "vocab_size": 260, "num_local_experts": 2, "num_experts_per_tok": 1,
		"rope_theta": 150000.0, "sliding_window": 128, "layer_types": []string{"sliding_attention"},
		"rope_scaling": map[string]any{"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0,
			"beta_slow": 1.0, "original_max_position_embeddings": 4096.0},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(cfg, b, 0o644); err != nil {
		t.Fatal(err)
	}
	setFile(path, func(f *eval.ModelFiles) { f.Config = cfg })
	writeTok(t, dir, 0, 0)
	out := filepath.Join(dir, "flow.safetensors")
	rep, err := flowRun(path, out, "a")
	if err != nil {
		t.Fatal(err)
	}
	if rep.Family != eval.FamGptOss {
		t.Fatalf("family %q", rep.Family)
	}
	if rep.Tokens < 1 || rep.Conservation > 1e-4 || rep.Stokes > 1e-5 {
		t.Fatalf("tokens %d conservation %g Stokes %g", rep.Tokens, rep.Conservation, rep.Stokes)
	}
	if len(rep.Experts) != 1 || rep.Experts[0] != 0 {
		t.Fatalf("experts %v", rep.Experts)
	}
	var gs float64
	for _, w := range rep.Gates {
		gs += float64(w)
	}
	if math.Abs(gs-1) > 1e-5 {
		t.Fatalf("gates %v", rep.Gates)
	}
	vol := filepath.Join(dir, "volume.safetensors")
	if err := volumes(path, vol); err != nil {
		t.Fatal(err)
	}
	st, err := OpenST(vol)
	if err != nil {
		t.Fatal(err)
	}
	if got := st.LogicalShape("volume.gate"); !eqShape(got, []int{1, VolumeCells, VolumeCells}) {
		t.Fatalf("volume.gate %v", got)
	}
	var line string
	err = eval.Flow(path, "", "a", false, evalCellsMust(t, path), func(pos, tok int, ls []eval.FlowLayer) {
		cells, cx, fam, err := manifoldSetup(path)
		if err != nil {
			t.Fatal(err)
		}
		line, err = manifoldLine(fam, pos, "a", cells, cx, ls)
		if err != nil {
			t.Fatal(err)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(line, "manifold gpt-oss") || !strings.Contains(line, "experts 0@") {
		t.Fatalf("line %s", line)
	}
}

func evalCellsMust(t *testing.T, path string) eval.Cells {
	t.Helper()
	c, _, _, err := manifoldSetup(path)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func writeOss(t *testing.T, path string) error {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	const D, F, H, KV, HD, E = 64, 64, 2, 1, 32, 2
	rng := rand.New(rand.NewSource(7))
	W, err := NewSTW(path, nil)
	if err != nil {
		return err
	}
	f32 := func(name string, shape []int, s float64) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		a := make([]float32, n)
		for i := range a {
			a[i] = float32(rng.NormFloat64() * s)
		}
		W.AddF32(name, shape, a)
	}
	ones := func(name string, n int) {
		a := make([]float32, n)
		for i := range a {
			a[i] = 1
		}
		W.AddF32(name, []int{n}, a)
	}
	f32("model.embed_tokens.weight", []int{260, D}, 0.02)
	ones("model.norm.weight", D)
	p := "model.layers.0."
	ones(p+"input_layernorm.weight", D)
	ones(p+"post_attention_layernorm.weight", D)
	f32(p+"self_attn.q_proj.weight", []int{H * HD, D}, 0.02)
	f32(p+"self_attn.k_proj.weight", []int{KV * HD, D}, 0.02)
	f32(p+"self_attn.v_proj.weight", []int{KV * HD, D}, 0.02)
	f32(p+"self_attn.o_proj.weight", []int{D, H * HD}, 0.02)
	f32(p+"self_attn.q_proj.bias", []int{H * HD}, 0.02)
	f32(p+"self_attn.k_proj.bias", []int{KV * HD}, 0.02)
	f32(p+"self_attn.v_proj.bias", []int{KV * HD}, 0.02)
	f32(p+"self_attn.o_proj.bias", []int{D}, 0.02)
	f32(p+"self_attn.sinks", []int{H}, 0.1)
	W.AddF32(p+"mlp.router.weight", []int{E, D}, make([]float32, E*D))
	W.AddF32(p+"mlp.router.bias", []int{E}, []float32{5, -5})
	f32(p+"mlp.experts.gate_up_proj_bias", []int{E, 2 * F}, 0.01)
	f32(p+"mlp.experts.down_proj_bias", []int{E, D}, 0.01)
	// B=16, groups = D/32 = 2. Gate/up rows are 2F.
	gB, gG, gRows := 16, 2, 2*F
	gb := make([]byte, E*gRows*gG*gB)
	gs := make([]byte, E*gRows*gG)
	rng.Read(gb)
	for i := range gs {
		gs[i] = 127
	}
	W.Add(p+"mlp.experts.gate_up_proj_blocks", "U8", []int{E, gRows, gG, gB}, gb)
	W.Add(p+"mlp.experts.gate_up_proj_scales", "U8", []int{E, gRows, gG}, gs)
	dRows := D
	db := make([]byte, E*dRows*gG*gB)
	ds := make([]byte, E*dRows*gG)
	rng.Read(db)
	for i := range ds {
		ds[i] = 127
	}
	W.Add(p+"mlp.experts.down_proj_blocks", "U8", []int{E, dRows, gG, gB}, db)
	W.Add(p+"mlp.experts.down_proj_scales", "U8", []int{E, dRows, gG}, ds)
	f32("lm_head.weight", []int{260, D}, 0.02)
	return W.Close(nil)
}

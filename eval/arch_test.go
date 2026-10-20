package eval

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFamily(t *testing.T) {
	cls := func(cfg map[string]any) Arch {
		t.Helper()
		a, err := Classify(cfg, 0)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	if a := cls(map[string]any{"model_type": "bitnet"}); a.Family != Fam2B4T || a.Norm != "rms" || a.MoE || a.Sandwich {
		t.Fatalf("bitnet %+v", a)
	}
	if a := cls(map[string]any{}); a.Family != FamLlama {
		t.Fatalf("empty model_type %q", a.Family)
	}
	if a := cls(map[string]any{"model_type": "mistral"}); a.Family != "mistral" {
		t.Fatalf("mistral labeled %q", a.Family)
	}
	q := cls(map[string]any{"model_type": "qwen2", "num_hidden_layers": 2.0, "sliding_window": 128.0})
	if q.Family != FamQwen || q.Window[0] != 0 || q.Window[1] != 0 {
		t.Fatalf("qwen window unused: %+v", q.Window)
	}
	q = cls(map[string]any{"model_type": "qwen2", "num_hidden_layers": 2.0, "sliding_window": 128.0, "use_sliding_window": true})
	if q.Window[0] != 128 || q.Window[1] != 128 {
		t.Fatalf("qwen window on: %+v", q.Window)
	}
	g2 := cls(map[string]any{"model_type": "gemma2", "num_hidden_layers": 2.0, "sliding_window": 16.0})
	if g2.Family != FamGemma || g2.Norm != "gemma" || !g2.EmbedScale || !g2.Sandwich || g2.Window[0] != 16 || g2.Window[1] != 0 {
		t.Fatalf("gemma2 %+v windows %v", g2, g2.Window)
	}
	g3 := cls(map[string]any{"model_type": "gemma3", "text_config": map[string]any{
		"model_type": "gemma3_text", "hidden_size": 32.0, "num_hidden_layers": 6.0, "sliding_window": 16.0,
	}})
	if g3.Family != FamGemma || !g3.Sandwich || g3.Window[0] != 16 || g3.Window[5] != 0 {
		t.Fatalf("gemma3 text_config %+v windows %v", g3, g3.Window)
	}
	vg := cls(map[string]any{"model_type": "vaultgemma", "num_hidden_layers": 2.0, "sliding_window": 512.0,
		"query_pre_attn_scalar": 256.0, "layer_types": []any{"full_attention", "full_attention"}})
	if vg.Family != FamGemma || vg.Norm != "gemma" || !vg.EmbedScale || vg.Sandwich || vg.QueryScale != 256 ||
		vg.Window[0] != 0 || vg.Window[1] != 0 {
		t.Fatalf("vaultgemma %+v windows %v", vg, vg.Window)
	}
	stamped := cls(map[string]any{"model_type": "bitnet", "architectures": []any{"VaultGemmaForCausalLM"},
		"num_hidden_layers": 2.0, "sliding_window": 512.0, "hidden_activation": "gelu_pytorch_tanh",
		"query_pre_attn_scalar": 256.0, "layer_types": []any{"full_attention", "full_attention"}})
	if stamped.Family != FamGemma || stamped.Norm != "gemma" || !stamped.EmbedScale || stamped.Sandwich ||
		stamped.QueryScale != 256 || stamped.Window[0] != 0 || stamped.Window[1] != 0 {
		t.Fatalf("stamped vaultgemma %+v windows %v", stamped, stamped.Window)
	}
	plain := cls(map[string]any{"model_type": "bitnet", "num_hidden_layers": 2.0})
	if plain.Family != Fam2B4T || plain.Norm != "rms" || plain.EmbedScale {
		t.Fatalf("bitnet %+v", plain)
	}
	bare := cls(map[string]any{"model_type": "vaultgemma", "num_hidden_layers": 2.0, "sliding_window": 512.0})
	if bare.Sandwich || bare.Window[0] != 0 || bare.Window[1] != 0 {
		t.Fatalf("vaultgemma without layer_types slid: %+v windows %v", bare, bare.Window)
	}
	oss := cls(map[string]any{"model_type": "gpt_oss", "num_hidden_layers": 2.0, "num_local_experts": 4.0,
		"num_experts_per_tok": 2.0, "sliding_window": 128.0})
	if oss.Family != FamGptOss || !oss.MoE || !oss.Sinks || oss.Experts != 4 || oss.TopK != 2 || oss.Window[0] != 128 || oss.Window[1] != 0 {
		t.Fatalf("gpt-oss %+v", oss)
	}
	if _, err := Classify(map[string]any{"model_type": "qwen2_moe"}, 0); err == nil || !strings.Contains(err.Error(), "mixture") {
		t.Fatalf("qwen2_moe: %v", err)
	}
	if _, err := Classify(map[string]any{"model_type": "mixtral"}, 0); err == nil {
		t.Fatal("mixtral accepted")
	}
}

func TestYarn(t *testing.T) {
	cfg := map[string]any{"rope_scaling": map[string]any{
		"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 8.0,
		"beta_fast": 32.0, "beta_slow": 1.0,
	}}
	f, scale, err := ropeFreqs(cfg, 10000, 4)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(f[0]-1) > 1e-12 || math.Abs(f[1]-0.005) > 1e-12 {
		t.Fatalf("yarn freqs %v", f)
	}
	want := 0.1*math.Log(2) + 1
	if math.Abs(scale-want) > 1e-12 {
		t.Fatalf("yarn scale %v want %v", scale, want)
	}
}

func TestMXFP4(t *testing.T) {
	w, sh, err := DequantMXFP4([]byte{0x21}, []byte{127}, []int{1, 1, 1, 1}, []int{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(sh) != 3 || sh[0] != 1 || sh[1] != 2 || sh[2] != 1 || math.Abs(float64(w[0]-0.5)) > 1e-6 || math.Abs(float64(w[1]-1)) > 1e-6 {
		t.Fatalf("dequant %v %v", w, sh)
	}
	m := &mx4{rows: 1, groups: 1, B: 1, stride: 1, phase: 0, blocks: []byte{0x21}, scales: []byte{127}}
	out := make([]float32, 1)
	m.gather([]float32{3, 4}, out, false, nil, nil, nil, 0)
	if math.Abs(float64(out[0]-5.5)) > 1e-5 {
		t.Fatalf("gather %v, want 5.5", out[0])
	}
}

func TestSwiGLU(t *testing.T) {
	h := swiglu([]float32{1, 9}, []float32{1, -9}, 1, 7)
	want0 := float32(2 / (1 + math.Exp(-1)))
	if math.Abs(float64(h[0]-want0)) > 1e-5 {
		t.Fatalf("swiglu(1,1) %v want %v", h[0], want0)
	}
	sig := func(g float32) float32 { return 1 / (1 + float32(math.Exp(float64(-g)))) }
	want1 := float32((-7 + 1) * 7 * float32(sig(7)))
	if math.Abs(float64(h[1]-want1)) > 1e-4 {
		t.Fatalf("clamped swiglu %v want %v", h[1], want1)
	}
}

func TestBiasFlow(t *testing.T) {
	e := make([]float32, 4)
	addBiasFlow(e, []float32{2, 4}, []int32{0, 1}, 2, 2)
	if e[0] != 1 || e[1] != 1 || e[2] != 2 || e[3] != 2 {
		t.Fatalf("bias spread %v", e)
	}
}

// VaultGemma stores the FFN pre-norm as pre_feedforward_layernorm, scales the embedding,
// and applies Gemma RMS (1+weight). sliding_window 512 with full_attention is unused.
func TestVaultGemma(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	z4 := make([]float32, 16)
	writeTensors(t, path, []ten{
		{"model.embed_tokens.weight", "F32", []int{1, 4}, []float32{1, 0, 0, 0}, nil},
		{"model.norm.weight", "F32", []int{4}, make([]float32, 4), nil},
		{"model.layers.0.input_layernorm.weight", "F32", []int{4}, make([]float32, 4), nil},
		{"model.layers.0.pre_feedforward_layernorm.weight", "F32", []int{4}, make([]float32, 4), nil},
		{"model.layers.0.self_attn.q_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.self_attn.k_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.self_attn.v_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.self_attn.o_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.mlp.gate_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.mlp.up_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.mlp.down_proj.weight", "F32", []int{4, 4}, z4, nil},
	})
	cfg := filepath.Join(dir, "config.json")
	b, _ := json.Marshal(map[string]any{
		"model_type": "vaultgemma", "hidden_size": 4, "intermediate_size": 4,
		"num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4,
		"num_hidden_layers": 1, "sliding_window": 512, "rope_theta": 10000,
		"hidden_activation": "gelu_pytorch_tanh", "query_pre_attn_scalar": 256,
		"layer_types": []any{"full_attention"}, "rms_norm_eps": 1e-6,
	})
	if err := os.WriteFile(cfg, b, 0o644); err != nil {
		t.Fatal(err)
	}
	SetFiles(path, ModelFiles{Config: cfg})
	m, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.Act != "gelu_pytorch_tanh" || m.arch.Sandwich || m.ls[0].fn == nil || m.ls[0].preFF != nil || m.arch.Window[0] != 0 {
		t.Fatalf("act %q sandwich %v fn %v preFF %v window %v", m.Act, m.arch.Sandwich, m.ls[0].fn, m.ls[0].preFF, m.arch.Window)
	}
	// Gemma RMS: weight 1 on a unit-rms vector is a gain of 2, not a multiply by 1.
	got := m.norm([]float32{2, 0, 0, 0}, []float32{1, 0, 0, 0})
	if math.Abs(float64(got[0]-4)) > 1e-4 || got[1] != 0 || got[2] != 0 || got[3] != 0 {
		t.Fatalf("gemma norm %v, want [4 0 0 0]", got)
	}
	y, err := Residual(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != 4 || math.Abs(float64(y[0]-2)) > 1e-5 || y[1] != 0 || y[2] != 0 || y[3] != 0 {
		t.Fatalf("vaultgemma residual %v, want [2 0 0 0]", y)
	}
}

func TestGemmaSandwich(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	z4 := make([]float32, 16)
	writeTensors(t, path, []ten{
		{"model.embed_tokens.weight", "F32", []int{1, 4}, []float32{1, 0, 0, 0}, nil},
		{"model.norm.weight", "F32", []int{4}, make([]float32, 4), nil},
		{"model.layers.0.input_layernorm.weight", "F32", []int{4}, make([]float32, 4), nil},
		{"model.layers.0.post_attention_layernorm.weight", "F32", []int{4}, make([]float32, 4), nil},
		{"model.layers.0.pre_feedforward_layernorm.weight", "F32", []int{4}, make([]float32, 4), nil},
		{"model.layers.0.post_feedforward_layernorm.weight", "F32", []int{4}, make([]float32, 4), nil},
		{"model.layers.0.self_attn.q_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.self_attn.k_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.self_attn.v_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.self_attn.o_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.mlp.gate_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.mlp.up_proj.weight", "F32", []int{4, 4}, z4, nil},
		{"model.layers.0.mlp.down_proj.weight", "F32", []int{4, 4}, z4, nil},
	})
	cfg := filepath.Join(dir, "config.json")
	b, _ := json.Marshal(map[string]any{
		"model_type": "gemma2", "hidden_size": 4, "intermediate_size": 4,
		"num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4,
		"num_hidden_layers": 1, "sliding_window": 16, "rope_theta": 10000,
		"hidden_activation": "gelu_pytorch_tanh",
	})
	if err := os.WriteFile(cfg, b, 0o644); err != nil {
		t.Fatal(err)
	}
	SetFiles(path, ModelFiles{Config: cfg})
	y, err := Residual(path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != 4 || math.Abs(float64(y[0]-2)) > 1e-5 || y[1] != 0 || y[2] != 0 || y[3] != 0 {
		t.Fatalf("gemma residual %v, want [2 0 0 0]", y)
	}
}

func TestMoEClosedForm(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.safetensors")
	w := float32(1 / math.Sqrt(2))
	writeTensors(t, path, moeTensors(false))
	writeMoEConfig(t, dir, path, 0)
	y := feedMoE(t, path)
	h0 := 2 / (1 + math.Exp(-1))
	if math.Abs(float64(y[0])-(1+h0)) > 1e-4 || math.Abs(float64(y[1])) > 1e-4 {
		t.Fatalf("float expert residual %v, want [1+%v 0] (norm weight %v puts [1 0] on the FFN)", y, h0, w)
	}
	mx := filepath.Join(dir, "mx.safetensors")
	writeTensors(t, mx, moeTensors(true))
	writeMoEConfig(t, dir, mx, 0)
	y2 := feedMoE(t, mx)
	if math.Abs(float64(y2[0]-y[0])) > 1e-4 || math.Abs(float64(y2[1]-y[1])) > 1e-4 {
		t.Fatalf("mxfp4 gather %v, float expert %v", y2, y)
	}
}

func feedMoE(t *testing.T, path string) []float32 {
	t.Helper()
	m, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	cells := Cells{D: []int32{0, 1}, F: []int32{0, 1}, Q: []int32{0, 1}, KV: []int32{0, 1},
		ND: 2, NF: 2, NQ: 2, NKV: 2, Per: 1}
	if err := m.instrument(cells); err != nil {
		t.Fatal(err)
	}
	m.rec = []FlowLayer{{Edge: map[string][]float32{}, Node: map[string][]float32{}}}
	y := m.feed(&seq{}, 0, 0)
	L := m.rec[0]
	if len(L.Experts) != 1 || L.Experts[0] != 0 {
		t.Fatalf("experts %v", L.Experts)
	}
	var gs float32
	for _, w := range L.Gates {
		gs += w
	}
	if math.Abs(float64(gs-1)) > 1e-5 {
		t.Fatalf("gates %v", L.Gates)
	}
	for _, role := range []string{"q", "k", "v", "gate", "up"} {
		e, n := L.Edge[role], L.Node[role]
		if e == nil || n == nil {
			t.Fatalf("no %s current", role)
		}
		nc := 2
		for r := 0; r < len(n); r++ {
			var s float32
			for c := 0; c < nc; c++ {
				s += e[r*nc+c]
			}
			if math.Abs(float64(s-n[r])) > 1e-4 {
				t.Fatalf("%s row %d current %v node %v", role, r, s, n[r])
			}
		}
	}
	return y
}

func writeMoEConfig(t *testing.T, dir, model string, eps float64) {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"model_type": "gpt_oss", "hidden_size": 2, "intermediate_size": 2,
		"num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 2,
		"num_hidden_layers": 1, "num_local_experts": 2, "num_experts_per_tok": 1,
		"rope_theta": 10000, "rms_norm_eps": eps, "swiglu_alpha": 1.0, "swiglu_limit": 7.0,
	})
	if err != nil {
		t.Fatal(err)
	}
	cfg := filepath.Join(dir, filepath.Base(model)+".config.json")
	if err := os.WriteFile(cfg, b, 0o644); err != nil {
		t.Fatal(err)
	}
	SetFiles(model, ModelFiles{Config: cfg})
}

// moeTensors is the closed-form gpt-oss block. mx packs the same identity experts as nibbles.
func moeTensors(mx bool) []ten {
	w := float32(1 / math.Sqrt(2))
	nw := []float32{w, w}
	one := []float32{1, 1}
	z4 := make([]float32, 4)
	ts := []ten{
		{"model.embed_tokens.weight", "F32", []int{1, 2}, []float32{1, 0}, nil},
		{"model.norm.weight", "F32", []int{2}, one, nil},
		{"model.layers.0.input_layernorm.weight", "F32", []int{2}, one, nil},
		{"model.layers.0.post_attention_layernorm.weight", "F32", []int{2}, nw, nil},
		{"model.layers.0.self_attn.q_proj.weight", "F32", []int{2, 2}, z4, nil},
		{"model.layers.0.self_attn.k_proj.weight", "F32", []int{2, 2}, z4, nil},
		{"model.layers.0.self_attn.v_proj.weight", "F32", []int{2, 2}, z4, nil},
		{"model.layers.0.self_attn.o_proj.weight", "F32", []int{2, 2}, z4, nil},
		{"model.layers.0.self_attn.sinks", "F32", []int{1}, []float32{0}, nil},
		{"model.layers.0.mlp.router.weight", "F32", []int{2, 2}, z4, nil},
		{"model.layers.0.mlp.router.bias", "F32", []int{2}, []float32{5, -5}, nil},
		{"model.layers.0.mlp.experts.gate_up_proj_bias", "F32", []int{2, 4}, make([]float32, 8), nil},
		{"model.layers.0.mlp.experts.down_proj_bias", "F32", []int{2, 2}, z4, nil},
	}
	if !mx {
		// [E, D, 2F], even last-index is gate. Expert 0 is identity gate and identity up.
		gu := make([]float32, 2*2*4)
		gu[0] = 1 // e0, c0, gate row 0
		gu[1] = 1 // e0, c0, up row 0
		gu[6] = 1 // e0, c1, gate row 1 (index c*4+2 = 4+2)
		gu[7] = 1 // e0, c1, up row 1
		dn := make([]float32, 2*2*2)
		dn[0] = 1 // e0, f0, d0
		dn[3] = 1 // e0, f1, d1
		ts = append(ts,
			ten{"model.layers.0.mlp.experts.gate_up_proj", "F32", []int{2, 2, 4}, gu, nil},
			ten{"model.layers.0.mlp.experts.down_proj", "F32", []int{2, 2, 2}, dn, nil},
		)
		return ts
	}
	// Packed rows of expert 0: gate0 [1,0]=0x02, up0 [1,0]=0x02, gate1 [0,1]=0x20, up1 [0,1]=0x20.
	blocks := []byte{0x02, 0x02, 0x20, 0x20, 0, 0, 0, 0}
	scales := []byte{127, 127, 127, 127, 127, 127, 127, 127}
	downB := []byte{0x02, 0x20, 0, 0}
	downS := []byte{127, 127, 127, 127}
	ts = append(ts,
		ten{"model.layers.0.mlp.experts.gate_up_proj_blocks", "U8", []int{2, 4, 1, 1}, nil, blocks},
		ten{"model.layers.0.mlp.experts.gate_up_proj_scales", "U8", []int{2, 4, 1}, nil, scales},
		ten{"model.layers.0.mlp.experts.down_proj_blocks", "U8", []int{2, 2, 1, 1}, nil, downB},
		ten{"model.layers.0.mlp.experts.down_proj_scales", "U8", []int{2, 2, 1}, nil, downS},
	)
	return ts
}

type ten struct {
	name, dtype string
	shape       []int
	f           []float32
	raw         []byte
}

func writeTensors(t *testing.T, path string, ts []ten) {
	t.Helper()
	type hdr struct {
		Dtype string  `json:"dtype"`
		Shape []int   `json:"shape"`
		Off   []int64 `json:"data_offsets"`
	}
	h := map[string]hdr{}
	var payload []byte
	var off int64
	for _, tn := range ts {
		b := tn.raw
		dt := tn.dtype
		if dt == "F32" {
			b = make([]byte, 4*len(tn.f))
			for i, v := range tn.f {
				binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
			}
		}
		h[tn.name] = hdr{dt, tn.shape, []int64{off, off + int64(len(b))}}
		payload = append(payload, b...)
		off += int64(len(b))
	}
	hb, err := json.Marshal(h)
	if err != nil {
		t.Fatal(err)
	}
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(len(hb)))
	if _, err := f.Write(n[:]); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(hb); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(payload); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

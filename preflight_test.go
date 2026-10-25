package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hodgemix/eval"
)

// splitShards writes model p's tensors into two shard files (the embedding and layers < cut, then the
// rest) and returns their paths; the shards get p's config and tokenizer files.
func splitShards(t *testing.T, p string, cut int) (string, string) {
	t.Helper()
	st, err := OpenST(p)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Dir(p)
	a, b := filepath.Join(dir, "model-00001-of-00002.safetensors"), filepath.Join(dir, "model-00002-of-00002.safetensors")
	wa, err1 := NewSTW(a, nil)
	wb, err2 := NewSTW(b, nil)
	if err1 != nil || err2 != nil {
		t.Fatal(err1, err2)
	}
	for _, k := range st.Keys {
		raw, err := st.Raw(k)
		if err != nil {
			t.Fatal(err)
		}
		w := wa
		if l := layerOf(k); l >= cut || (l < 0 && k != "model.embed_tokens.weight") {
			w = wb
		}
		w.Add(k, st.H[k].Dtype, st.H[k].Shape, raw)
	}
	if err := wa.Close(nil); err != nil {
		t.Fatal(err)
	}
	if err := wb.Close(nil); err != nil {
		t.Fatal(err)
	}
	f := eval.FilesOf(p)
	eval.SetFiles(a, f)
	eval.SetFiles(b, f)
	return a, b
}

func hasReason(bad []string, s string) bool {
	for _, b := range bad {
		if strings.Contains(b, s) {
			return true
		}
	}
	return false
}

// A Gemma 4 file is the language model plus a vision tower and an audio tower.
// The text_config describes the language model. The other towers are not a
// second, conflicting config, and they are not a reason to reject the file.
func TestMultimodalSourceAccepted(t *testing.T) {
	dir := tempDir(t)
	base := writeTiny(t, filepath.Join(dir, "base"), tinyModel{D: 16, F: 24, H: 4, KV: 1, HD: 4, L: 1, V: 300}, 1)
	srcDir := filepath.Join(dir, "gemma")
	if err := os.MkdirAll(srcDir, 0o755); err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(srcDir, "model.safetensors")
	W, err := NewSTW(p, nil)
	if err != nil {
		t.Fatal(err)
	}
	z := func(n int) []float32 { return make([]float32, n) }
	// Audio tower, layer 3, a different hidden size than the text config.
	W.AddBF16("model.audio_tower.layers.3.self_attn.q_proj.linear.weight", []int{8, 8}, z(64))
	W.AddBF16("model.audio_tower.layers.3.self_attn.k_proj.linear.weight", []int{8, 8}, z(64))
	W.AddBF16("model.audio_tower.layers.3.feed_forward1.ffw_layer_1.linear.weight", []int{8, 8}, z(64))
	W.AddBF16("model.audio_tower.output_proj.bias", []int{8}, z(8))
	W.AddBF16("model.embed_vision.embedding_projection.weight", []int{4, 4}, z(16))
	D, F, H, KV, HD := 16, 24, 4, 1, 4
	pre := "model.language_model.layers.0."
	W.AddBF16(pre+"self_attn.q_proj.weight", []int{H * HD, D}, z(H*HD*D))
	W.AddBF16(pre+"self_attn.k_proj.weight", []int{KV * HD, D}, z(KV*HD*D))
	W.AddBF16(pre+"mlp.gate_proj.weight", []int{F, D}, z(F*D))
	W.AddBF16(pre+"per_layer_projection.weight", []int{D, 4}, z(D*4))
	W.AddBF16("model.language_model.embed_tokens.weight", []int{300, D}, z(300*D))
	if err := W.Close(nil); err != nil {
		t.Fatal(err)
	}
	cfg := map[string]any{
		"model_type": "gemma4",
		"audio_config": map[string]any{"model_type": "gemma4_audio", "hidden_size": 8},
		"text_config": map[string]any{
			"model_type": "gemma4_text", "hidden_size": D, "intermediate_size": F,
			"num_attention_heads": H, "num_key_value_heads": KV, "head_dim": HD, "num_hidden_layers": 1,
		},
	}
	b, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(srcDir, "config.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	setFile(p, func(f *eval.ModelFiles) { f.Config = filepath.Join(srcDir, "config.json") })
	writeTok(t, srcDir, 0, 0)
	st, err := OpenST(p)
	if err != nil {
		t.Fatal(err)
	}
	m, err := metaOf(st)
	if err != nil || m.D != D || m.F != F || m.Heads != H || m.KV != KV || m.HD != HD || m.Layers != 1 {
		t.Fatalf("language-model topology %+v %v", m, err)
	}
	bs, err := OpenST(base)
	if err != nil {
		t.Fatal(err)
	}
	if _, bad := preflightCheck(tinyCfg("base"), base, []*ST{bs, st}, true); len(bad) > 0 {
		t.Fatalf("multimodal source refused: %v", bad)
	}
}

// Every way a run would fail or make a broken model is found before anything is written.
func TestPreflightRefuses(t *testing.T) {
	dir := tempDir(t)
	m := tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 260}
	a := writeTiny(t, filepath.Join(dir, "a"), m, 11)
	b := writeTiny(t, filepath.Join(dir, "b"), m, 12)
	open := func(ps ...string) []*ST {
		var all []*ST
		for _, p := range ps {
			st, err := OpenST(p)
			if err != nil {
				t.Fatal(err)
			}
			all = append(all, st)
		}
		return all
	}
	c := tinyCfg("base")
	if _, bad := preflightCheck(c, a, open(a, b), true); len(bad) > 0 {
		t.Fatalf("complete models refused: %v", bad)
	}
	reset := eval.FilesOf(a)
	defer eval.SetFiles(a, reset)
	cfgWith := func(extra map[string]any) {
		var cfg map[string]any
		cb, _ := os.ReadFile(filepath.Join(dir, "a", "config.json"))
		json.Unmarshal(cb, &cfg)
		for k, v := range extra {
			cfg[k] = v
		}
		nb, _ := json.Marshal(cfg)
		p := filepath.Join(dir, "a", "config-test.json")
		os.WriteFile(p, nb, 0o644)
		setFile(a, func(f *eval.ModelFiles) { f.Config = p })
	}
	for _, tc := range []struct {
		name, want string
		set        func()
	}{
		{"no config", "no config set", func() { setFile(a, func(f *eval.ModelFiles) { f.Config = "" }) }},
		{"no tokenizer", "no tokenizer set", func() { setFile(a, func(f *eval.ModelFiles) { f.Tokenizer = "" }) }},
		{"no tokenizer_config", "no tokenizer_config set", func() { setFile(a, func(f *eval.ModelFiles) { f.TokenizerConfig = "" }) }},
		{"another model's tokenizer", "another model's tokenizer", func() {
			d := filepath.Join(dir, "big")
			os.MkdirAll(d, 0o755)
			writeTok(t, d, 0, 200)
			setFile(a, func(f *eval.ModelFiles) { f.Tokenizer = filepath.Join(d, "tokenizer.json") })
		}},
		{"eos not in the tokenizer", "is not an added token", func() {
			p := filepath.Join(dir, "a", "tc-other.json")
			os.WriteFile(p, []byte(`{"eos_token": "<|im_end|>"}`), 0o644)
			setFile(a, func(f *eval.ModelFiles) { f.TokenizerConfig = p })
		}},
		{"another model's config", "another model's config", func() {
			p := filepath.Join(dir, "a", "config-other.json")
			os.WriteFile(p, []byte(`{"hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 8}`), 0o644)
			setFile(a, func(f *eval.ModelFiles) { f.Config = p })
		}},
		{"exact gelu", "hidden_act", func() { cfgWith(map[string]any{"hidden_act": "gelu"}) }},
		{"yarn", "rope_scaling", func() { cfgWith(map[string]any{"rope_scaling": map[string]any{"rope_type": "yarn", "factor": 4.0}}) }},
		{"sliding window", "sliding-window", func() { cfgWith(map[string]any{"use_sliding_window": true, "sliding_window": 4096.0}) }},
		{"no context bound", "max_position_embeddings", func() { cfgWith(map[string]any{"max_position_embeddings": nil}) }},
	} {
		eval.SetFiles(a, reset)
		tc.set()
		_, bad := preflightCheck(c, a, open(a, b), true)
		if !hasReason(bad, tc.want) {
			t.Fatalf("%s: want a reason with %q, got %v", tc.name, tc.want, bad)
		}
	}
	eval.SetFiles(a, reset)
	rc := c
	rc.rotate = true
	if _, bad := preflightCheck(rc, a, open(a, b), true); !hasReason(bad, "--rotate") {
		t.Fatalf("--rotate: %v", bad)
	}
	// A source with biases (here in a second shard of it) would lose them.
	bias := filepath.Join(dir, "b", "bias.safetensors")
	w, _ := NewSTW(bias, nil)
	w.AddBF16("model.layers.0.self_attn.q_proj.bias", []int{128}, make([]float32, 128))
	if err := w.Close(nil); err != nil {
		t.Fatal(err)
	}
	fb := eval.FilesOf(b)
	fb.Shards = []string{b, bias}
	eval.SetFiles(b, fb)
	if _, bad := preflightCheck(c, a, open(a, b), true); !hasReason(bad, "biases") {
		t.Fatalf("source biases: %v", bad)
	}
}

// Each shard of a checkpoint is its own source (equal weights), the second starting mid-stack: its
// sizes come from its first layer and the run makes a model Champ loads.
func TestShardSources(t *testing.T) {
	dir := tempDir(t)
	m := tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 3, V: 260}
	a := writeTiny(t, filepath.Join(dir, "a"), m, 21)
	b := writeTiny(t, filepath.Join(dir, "b"), m, 22)
	s1, s2 := splitShards(t, b, 2)
	st2, err := OpenST(s2)
	if err != nil {
		t.Fatal(err)
	}
	if g, err := metaOf(st2); err != nil || g.D != 128 || g.Heads != 2 || g.F != 192 {
		t.Fatalf("shard 2 topology %+v %v", g, err)
	}
	out := filepath.Join(dir, "out", "mix.bnz")
	run(tinyCfg("base"), a, []string{s1, s2}, out, "", "", true)
	if bad, _ := champCheck(out+".ternary.safetensors", champConfigPath(out)); len(bad) > 0 {
		t.Fatalf("champCheck: %v", bad)
	}
}

// The output's config keeps the base's model_type, activation, and RoPE scaling.
func TestConfigKeepsBase(t *testing.T) {
	dir := tempDir(t)
	p := filepath.Join(dir, "config.json")
	os.WriteFile(p, []byte(`{"model_type": "llama", "hidden_act": "silu", "rope_theta": 500000.0, "max_position_embeddings": 131072,
		"rope_scaling": {"rope_type": "llama3", "factor": 32.0}, "tie_word_embeddings": true}`), 0o644)
	base := filepath.Join(dir, "model.safetensors")
	eval.SetFiles(base, eval.ModelFiles{Config: p})
	b, _, err := configFor(modelMeta{D: 2048, F: 8192, Heads: 32, KV: 8, HD: 64, Layers: 16, Theta: 500000}, "", base, 128256)
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]any
	json.Unmarshal(b, &cfg)
	rs, _ := cfg["rope_scaling"].(map[string]any)
	if cfg["hidden_act"] != "silu" || rs["rope_type"] != "llama3" || cfg["model_type"] != "llama" {
		t.Fatalf("output config %v", cfg)
	}
	if _, ok := cfg["quantization_config"]; ok {
		t.Fatalf("invented quantization_config: %v", cfg["quantization_config"])
	}
	if bad := champRuntime(cfg, 2048, 8192, 32, 8, 64); len(bad) > 0 {
		t.Fatalf("champRuntime refuses a silu / llama3 config: %v", bad)
	}
	// A run whose layers are all full_attention may still carry sliding_window. One that
	// does not say which layers use the number is not assumed to slide, and is not accepted.
	cfg["sliding_window"] = 512.0
	cfg["num_hidden_layers"] = 2.0
	cfg["layer_types"] = []any{"full_attention", "full_attention"}
	if bad := champRuntime(cfg, 2048, 8192, 32, 8, 64); hasReason(bad, "sliding") {
		t.Fatalf("full_attention run refused for sliding_window 512: %v", bad)
	}
	cfg["layer_types"] = []any{"full_attention", "sliding_attention"}
	if bad := champRuntime(cfg, 2048, 8192, 32, 8, 64); !hasReason(bad, "sliding-window") {
		t.Fatalf("a sliding layer was accepted: %v", bad)
	}
	delete(cfg, "layer_types")
	if bad := champRuntime(cfg, 2048, 8192, 32, 8, 64); !hasReason(bad, "does not say which layers") {
		t.Fatalf("a bare sliding_window was assumed: %v", bad)
	}
	os.WriteFile(p, []byte(`{"model_type": "llama"}`), 0o644)
	b, _, _ = configFor(modelMeta{D: 2048, F: 8192, Heads: 32, KV: 8, HD: 64, Layers: 16}, "", base, 0)
	json.Unmarshal(b, &cfg)
	if cfg["hidden_act"] != "silu" || cfg["rope_theta"] != 10000.0 {
		t.Fatalf("unset keys did not take LlamaConfig's defaults: %v", cfg)
	}
	// VaultGemma names the activation hidden_activation. The Champ config keeps gelu tanh.
	// sliding_window 512 beside all full_attention is not a window. GEMMA norm and the
	// embed scale are what norm.wgsl and embed.wgsl run for that key.
	os.WriteFile(p, []byte(`{"model_type":"vaultgemma","hidden_activation":"gelu_pytorch_tanh","rope_theta":10000.0,
		"max_position_embeddings":1024,"sliding_window":512,"num_hidden_layers":2,
		"layer_types":["full_attention","full_attention"],"attention_bias":false,"tie_word_embeddings":true}`), 0o644)
	b, _, err = configFor(modelMeta{D: 128, F: 256, Heads: 4, KV: 4, HD: 32, Layers: 2, Theta: 10000}, "", base, 0)
	if err != nil {
		t.Fatal(err)
	}
	json.Unmarshal(b, &cfg)
	if cfg["hidden_act"] != "gelu_pytorch_tanh" || cfg["hidden_activation"] != "gelu_pytorch_tanh" || cfg["model_type"] != "vaultgemma" {
		t.Fatalf("vaultgemma output config %v", cfg)
	}
	if _, ok := cfg["quantization_config"]; ok {
		t.Fatalf("invented quantization_config: %v", cfg["quantization_config"])
	}
	if arch, ok := cfg["architectures"].([]any); ok && len(arch) > 0 {
		t.Fatalf("invented architectures: %v", arch)
	}
	bad := champRuntime(cfg, 128, 256, 4, 4, 32)
	if len(bad) > 0 {
		t.Fatalf("vaultgemma run refused: %v", bad)
	}
}

// VaultGemma's FFN pre-norm is pre_feedforward_layernorm. Champ loads that name.
// A layer that stores post_attention_layernorm still uses that name.
func TestChampAcceptsVaultGemmaPreNorm(t *testing.T) {
	dir := tempDir(t)
	D, F, H, KV, hd, V := 32, 64, 4, 4, 8, 32
	p := filepath.Join(dir, "model.safetensors")
	W, err := NewSTW(p, nil)
	if err != nil {
		t.Fatal(err)
	}
	addBF := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		W.Add(name, "BF16", shape, make([]byte, 2*n))
	}
	addU8 := func(name string, rows, cols int) {
		W.Add(name, "U8", []int{rows / 4, cols}, make([]byte, (rows/4)*cols))
		W.Add(name+"_scale", "BF16", []int{1}, make([]byte, 2))
	}
	addBF("model.embed_tokens.weight", V, D)
	addBF("model.norm.weight", D)
	pre := "model.layers.0."
	addBF(pre+"input_layernorm.weight", D)
	addBF(pre+"pre_feedforward_layernorm.weight", D)
	addBF(pre+"self_attn.attn_sub_norm.weight", H*hd)
	addBF(pre+"mlp.ffn_sub_norm.weight", F)
	addU8(pre+"self_attn.q_proj.weight", H*hd, D)
	addU8(pre+"self_attn.k_proj.weight", KV*hd, D)
	addU8(pre+"self_attn.v_proj.weight", KV*hd, D)
	addU8(pre+"self_attn.o_proj.weight", D, H*hd)
	addU8(pre+"mlp.gate_proj.weight", F, D)
	addU8(pre+"mlp.up_proj.weight", F, D)
	addU8(pre+"mlp.down_proj.weight", D, F)
	if err := W.Close(nil); err != nil {
		t.Fatal(err)
	}
	cfgp := filepath.Join(dir, "config.json")
	os.WriteFile(cfgp, []byte(`{"model_type":"vaultgemma","hidden_act":"gelu_pytorch_tanh","hidden_activation":"gelu_pytorch_tanh",
		"hidden_size":32,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":4,"head_dim":8,
		"num_hidden_layers":1,"vocab_size":32,"max_position_embeddings":1024,"rope_theta":10000,
		"sliding_window":512,"layer_types":["full_attention"],"tie_word_embeddings":true,"attention_bias":false}`), 0o644)
	if bad, _ := champCheck(p, cfgp); len(bad) > 0 {
		t.Fatalf("vaultgemma prenorm refused: %v", bad)
	}
	// The usual FFN pre-norm name still loads.
	p2 := filepath.Join(dir, "llama.safetensors")
	W2, err := NewSTW(p2, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Re-open is a new file. Copy by writing the same tensors with post_attention.
	addBF2 := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		W2.Add(name, "BF16", shape, make([]byte, 2*n))
	}
	addU82 := func(name string, rows, cols int) {
		W2.Add(name, "U8", []int{rows / 4, cols}, make([]byte, (rows/4)*cols))
		W2.Add(name+"_scale", "BF16", []int{1}, make([]byte, 2))
	}
	addBF2("model.embed_tokens.weight", V, D)
	addBF2("model.norm.weight", D)
	addBF2(pre+"input_layernorm.weight", D)
	addBF2(pre+"post_attention_layernorm.weight", D)
	addBF2(pre+"self_attn.attn_sub_norm.weight", H*hd)
	addBF2(pre+"mlp.ffn_sub_norm.weight", F)
	addU82(pre+"self_attn.q_proj.weight", H*hd, D)
	addU82(pre+"self_attn.k_proj.weight", KV*hd, D)
	addU82(pre+"self_attn.v_proj.weight", KV*hd, D)
	addU82(pre+"self_attn.o_proj.weight", D, H*hd)
	addU82(pre+"mlp.gate_proj.weight", F, D)
	addU82(pre+"mlp.up_proj.weight", F, D)
	addU82(pre+"mlp.down_proj.weight", D, F)
	if err := W2.Close(nil); err != nil {
		t.Fatal(err)
	}
	cfg2 := filepath.Join(dir, "llama-config.json")
	os.WriteFile(cfg2, []byte(`{"model_type":"llama","hidden_act":"silu",
		"hidden_size":32,"intermediate_size":64,"num_attention_heads":4,"num_key_value_heads":4,"head_dim":8,
		"num_hidden_layers":1,"vocab_size":32,"max_position_embeddings":1024,"rope_theta":10000,
		"tie_word_embeddings":true,"attention_bias":false}`), 0o644)
	if bad, _ := champCheck(p2, cfg2); len(bad) > 0 {
		t.Fatalf("post_attention prenorm refused: %v", bad)
	}
}

package main

// Compatibility with the Champ / EdgeAI BNZ4 stack (bnz-loader js/bnz-loader.js + bnz4dec.js + bitnet-rt.js,
// mirroring edgellm/bitnet/model/model.go). No paper equation: these are the consumer's interface rules.

import (
	"crypto/ed25519"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"hodgemix/eval"
)

// bnz4Prefix returns the 12-byte header + JSON index, the bytes bnz-loader verifies the Ed25519 signature over.
func bnz4Prefix(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := make([]byte, 12)
	if _, err := io.ReadFull(f, h); err != nil {
		return nil, err
	}
	if string(h[:4]) != "BNZ4" {
		return nil, fmt.Errorf("%s: not BNZ4", path)
	}
	b := make([]byte, 12+binary.LittleEndian.Uint64(h[4:]))
	copy(b, h)
	_, err = io.ReadFull(f, b[12:])
	return b, err
}

// signBNZ4 writes OUT.sig = Ed25519(seed, header+index): identical to bnz-loader's `bnz4sign sign SEED OUT`.
func signBNZ4(seedPath, out string) error {
	seed, err := os.ReadFile(seedPath)
	if err != nil {
		return err
	}
	if len(seed) != ed25519.SeedSize {
		return fmt.Errorf("%s: want a %d-byte Ed25519 seed (bnz4sign keygen format), got %d bytes", seedPath, ed25519.SeedSize, len(seed))
	}
	p, err := bnz4Prefix(out)
	if err != nil {
		return err
	}
	return os.WriteFile(out+".sig", ed25519.Sign(ed25519.NewKeyFromSeed(seed), p), 0o644)
}

// champCheck lists every way the ternary safetensors + config.json would not load or run in Champ's
// model.go; notes are what the other consumers (bitnet-rt.js, bnz-loader) do differently.
func champCheck(stPath, cfgPath string) (bad, notes []string) {
	st, err := OpenST(stPath)
	if err != nil {
		return []string{err.Error()}, nil
	}
	cfg := map[string]any{}
	b, err := os.ReadFile(cfgPath)
	if err != nil {
		return []string{"config.json: " + err.Error()}, nil
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return []string{"config.json: " + err.Error()}, nil
	}
	geti := func(k string) int {
		v, ok := cfg[k].(float64)
		if !ok {
			bad = append(bad, "config.json: missing "+k)
		}
		return int(v)
	}
	D, F, H, KV, L, V := geti("hidden_size"), geti("intermediate_size"), geti("num_attention_heads"),
		geti("num_key_value_heads"), geti("num_hidden_layers"), geti("vocab_size")
	if len(bad) > 0 || H == 0 {
		return bad, nil
	}
	if t, ok := cfg["tie_word_embeddings"].(bool); ok && !t {
		bad = append(bad, "tie_word_embeddings=false: bitnet-rt.js/model.go only support a tied LM head (Eq.24e rotation unties it; use --rotate off)")
	}
	hd := D / H
	if v, ok := cfg["head_dim"].(float64); ok {
		hd = int(v)
	}
	bad = append(bad, champRuntime(cfg, D, F, H, KV, hd)...)
	want := map[string][]int{"model.embed_tokens.weight": {V, D}, "model.norm.weight": {D}}
	type lin struct {
		n      string
		k, out int
	}
	for i := 0; i < L; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		want[p+"input_layernorm.weight"] = []int{D}
		want[ffnPreNormName(st, p)] = []int{D}
		want[p+"self_attn.attn_sub_norm.weight"] = []int{H * hd}
		want[p+"mlp.ffn_sub_norm.weight"] = []int{F}
		for _, l := range []lin{{"self_attn.q_proj", D, H * hd}, {"self_attn.k_proj", D, KV * hd}, {"self_attn.v_proj", D, KV * hd},
			{"self_attn.o_proj", H * hd, D}, {"mlp.gate_proj", D, F}, {"mlp.up_proj", D, F}, {"mlp.down_proj", F, D}} {
			if l.out%4 != 0 || l.k%4 != 0 {
				bad = append(bad, fmt.Sprintf("%s%s: outputs %d and inputs %d must be multiples of 4", p, l.n, l.out, l.k))
			}
			want[p+l.n+".weight"] = []int{l.out / 4, l.k}
			want[p+l.n+".weight_scale"] = []int{1}
		}
	}
	for n, sh := range want {
		t, ok := st.H[n]
		if !ok {
			bad = append(bad, "missing "+n)
			continue
		}
		dt := "BF16"
		if strings.HasSuffix(n, "_proj.weight") {
			dt = "U8"
		}
		if t.Dtype != dt || !eqShape(t.Shape, sh) {
			bad = append(bad, fmt.Sprintf("%s: %s%v, runtime needs %s%v", n, t.Dtype, t.Shape, dt, sh))
		}
	}
	limbed := 0
	for n, sh := range want {
		if strings.HasSuffix(n, "_proj.weight") {
			bad = append(bad, champLimbs(st, n, sh)...)
			if st.Has(limbName(n, 1)) {
				limbed++
			}
		}
	}
	if limbed > 0 {
		notes = append(notes, fmt.Sprintf("%d BitLinears have dynamic MRH limbs: model.go and bitnet-rt.js run every limb (bitlinear.wgsl CENTER, ACCUM), with no limb-count cap", limbed))
	}
	bad = append(bad, champChanMaps(st)...)
	// Champ is a legacy consumer. The check that remains is the BitNet shape
	// above. Any other model tensor in the file is counted in a note.
	extra := 0
	for n := range st.H {
		if _, ok := want[n]; ok || strings.HasPrefix(n, "bnzmix.") || isLimbTensor(n) {
			continue
		}
		if strings.HasPrefix(n, "model.") || strings.HasPrefix(n, "lm_head.") {
			extra++
		}
	}
	if extra > 0 {
		notes = append(notes, fmt.Sprintf("%d tensors besides the BitNet shape are in the file (audio, vision, embeddings)", extra))
	}
	return bad, notes
}

// ffnPreNormName is the FFN pre-norm this layer stores. Llama, Qwen, and Gemma 1
// use post_attention_layernorm. VaultGemma uses pre_feedforward_layernorm and has
// no post-attention norm. The page loads whichever of those is present.
func ffnPreNormName(st *ST, p string) string {
	post := p + "post_attention_layernorm.weight"
	pre := p + "pre_feedforward_layernorm.weight"
	if st.Has(post) || !st.Has(pre) {
		return post
	}
	return pre
}

// champConfigPath: config.json written beside OUT by copyAux.
func champConfigPath(out string) string { return filepath.Join(filepath.Dir(absP(out)), "config.json") }

// champWGStorage is WebGPU's default maxComputeWorkgroupStorageSize; model.go asks the adapter for its
// own limit, which may be larger.
const champWGStorage = 16384

// champRuntime lists what Champ's attention path and kernels (model.go LoadConfig, attention.wgsl,
// rope.wgsl, bitlinear.wgsl, lmhead.wgsl) require of the config.
func champRuntime(cfg map[string]any, D, F, H, KV, hd int) []string {
	var bad []string
	str := func(m map[string]any, k string) string { s, _ := m[k].(string); return s }
	mt := str(cfg, "model_type")
	if mt == "" {
		bad = append(bad, "config.json: no model_type")
	} else if mt == "gpt_oss" || strings.Contains(mt, "moe") || strings.Contains(mt, "mixtral") {
		bad = append(bad, fmt.Sprintf("config.json: model_type %q is not a dense block Champ runs", mt))
	}
	act := str(cfg, "hidden_act")
	if act == "" {
		act = str(cfg, "hidden_activation")
	}
	switch act {
	case "relu2", "silu", "swish", "gelu_pytorch_tanh", "gelu_new":
	default:
		bad = append(bad, fmt.Sprintf("config.json: hidden_act %q: norm.wgsl ACT runs relu2, silu and gelu tanh", act))
	}
	// hidden_activation (VaultGemma, Gemma) selects norm.wgsl GEMMA, y*(1+w), and
	// embed.wgsl SCALE = sqrt(hidden). query_pre_attn_scalar is attention.wgsl DENOM.
	if _, ok := cfg["max_position_embeddings"].(float64); !ok {
		bad = append(bad, "config.json: missing max_position_embeddings (the runtime's context bound)")
	}
	if KV <= 0 || H%KV != 0 {
		bad = append(bad, fmt.Sprintf("attention: %d query heads do not group over %d kv heads (attention.wgsl: head h reads kv head h / (H / KVH))", H, KV))
	}
	if hd%2 != 0 {
		bad = append(bad, fmt.Sprintf("attention: head_dim %d is odd (rope.wgsl pairs d with d + HD/2)", hd))
	}
	if D%4 != 0 || F%4 != 0 {
		bad = append(bad, fmt.Sprintf("hidden %d / intermediate %d must be multiples of 4 (int8 words)", D, F))
	}
	if rs := eval.RopeDict(cfg); rs != nil {
		t := eval.RopeType(rs)
		if t != "" && t != "default" && t != "linear" && t != "llama3" && t != "mrope" {
			bad = append(bad, fmt.Sprintf("config.json: rope_scaling %q: rope.wgsl runs linear and llama3", t))
		}
	}
	// Each run says which layers slide (layer_types, or use_sliding_window). A sliding_window
	// number by itself is not that: Llama carries 512 next to a layer_types of full_attention.
	if slide, detail, errS := runSlides(cfg); errS != "" {
		bad = append(bad, "config.json: "+errS)
	} else if slide {
		bad = append(bad, fmt.Sprintf("config.json: sliding-window attention (%s): attention.wgsl is full causal attention", detail))
	}
	if b, _ := cfg["attention_bias"].(bool); b {
		bad = append(bad, "config.json: attention_bias: attention.wgsl has no bias")
	}
	// Workgroup memory: bitlinear holds its K/4 input words + 256 vec4<i32>, lmhead D floats + 256 floats.
	for _, k := range []struct {
		what string
		n    int
	}{{"BitLinear input " + fmt.Sprint(D), D + 4096}, {"BitLinear input " + fmt.Sprint(H*hd), H*hd + 4096}, {"BitLinear input " + fmt.Sprint(F), F + 4096}, {"LM head width " + fmt.Sprint(D), 4*D + 1024}} {
		if k.n > champWGStorage {
			bad = append(bad, fmt.Sprintf("%s needs %d B of workgroup memory, over WebGPU's default %d (works only where the adapter allows more)", k.what, k.n, champWGStorage))
		}
	}
	return bad
}

// runSlides is this run's own attention configuration. layer_types wins, then use_sliding_window.
// A sliding_window number with neither is not assumed: the run has to say which layers use it.
// The same rule is in bitnet model.go LoadConfig.
func runSlides(cfg map[string]any) (slide bool, detail, errS string) {
	w, hasW := cfg["sliding_window"].(float64)
	use, hasUse := cfg["use_sliding_window"].(bool)
	raw, hasTypes := cfg["layer_types"].([]any)
	if hasTypes && len(raw) > 0 {
		if L, ok := cfg["num_hidden_layers"].(float64); ok && int(L) > 0 && int(L) != len(raw) {
			return false, "", fmt.Sprintf("layer_types has %d entries, the model has %d layers", len(raw), int(L))
		}
		n := 0
		for i, v := range raw {
			s, _ := v.(string)
			switch s {
			case "full_attention":
			case "sliding_attention":
				n++
				if !hasW || w <= 0 {
					return false, "", fmt.Sprintf("layer %d is sliding_attention and sliding_window is %v", i, cfg["sliding_window"])
				}
			default:
				return false, "", fmt.Sprintf("layer_types %q is not full_attention or sliding_attention", s)
			}
		}
		if n == 0 {
			return false, "", ""
		}
		return true, fmt.Sprintf("%d of %d layers sliding_attention, window %g", n, len(raw), w), ""
	}
	if hasUse {
		if !use {
			return false, "", ""
		}
		if !hasW || w <= 0 {
			return false, "", "use_sliding_window is true and sliding_window is unset"
		}
		return true, fmt.Sprintf("use_sliding_window, window %g", w), ""
	}
	if hasW && w > 0 {
		return false, "", fmt.Sprintf("sliding_window is %g but this run does not say which layers use it (layer_types or use_sliding_window)", w)
	}
	return false, "", ""
}

// champLimbs checks a BitLinear's dynamic MRH limbs (limbs.go) as model.go loads them: numbered from 1
// without gaps, each U8 packed like limb 0, a BF16[1] scale and an I32[1] radix in {2, 3, 4}.
func champLimbs(st *ST, k string, packed []int) []string {
	var bad []string
	for i := 1; ; i++ {
		n := limbName(k, i)
		t, ok := st.H[n]
		if !ok {
			if _, next := st.H[limbName(k, i+1)]; next {
				bad = append(bad, fmt.Sprintf("%s: limb %d missing before limb %d", k, i, i+1))
			}
			return bad
		}
		if t.Dtype != "U8" || !eqShape(t.Shape, packed) {
			bad = append(bad, fmt.Sprintf("%s: %s%v, model.go needs U8%v", n, t.Dtype, t.Shape, packed))
		}
		if s, ok := st.H[n+"_scale"]; !ok || s.Dtype != "BF16" || numel(s.Shape) != 1 {
			bad = append(bad, n+"_scale: missing or not BF16[1]")
		}
		if r, ok := st.H[n+"_radix"]; !ok || r.Dtype != "I32" || numel(r.Shape) != 1 {
			bad = append(bad, n+"_radix: missing or not I32[1]")
		} else if b, err := st.Raw(n + "_radix"); err == nil && len(b) == 4 {
			if rv := int(binary.LittleEndian.Uint32(b)); rv < 2 || rv > 4 {
				bad = append(bad, fmt.Sprintf("%s: radix %d (bitlinear.wgsl reads 2-bit digits: radix 2, 3 or 4)", n, rv))
			}
		}
	}
}

// champChanMaps checks each BNZ4 channel-map chunk: its radix tuples multiply to at least its grid (the
// MRH(r; C) levels the run used cover the grid).
func champChanMaps(st *ST) []string {
	var bad []string
	for _, k := range st.Keys {
		if !strings.HasPrefix(k, "bnzmix.bnz4.") {
			continue
		}
		b, err := st.Raw(k)
		if err != nil {
			continue
		}
		cm, err := DecodeChanMap(b)
		if err != nil {
			bad = append(bad, k+": "+err.Error())
			continue
		}
		prod := func(r []uint8) uint64 {
			p := uint64(1)
			for _, x := range r {
				p *= uint64(x)
			}
			return p
		}
		if prod(cm.RadR) < uint64(cm.Gr) || prod(cm.RadC) < uint64(cm.Gc) {
			bad = append(bad, fmt.Sprintf("%s: radix tuples %v x %v do not cover grid %dx%d", k, cm.RadR, cm.RadC, cm.Gr, cm.Gc))
		}
	}
	return bad
}

// promoteRope copies rope_parameters into rope_scaling when that is the only place the
// checkpoint wrote it, so Champ's rope.wgsl (which reads rope_scaling) sees llama3 or linear.
// mrope stays the default frequencies; the section list is kept on the copied block.
func promoteRope(cfg map[string]any) string {
	if rs, ok := cfg["rope_scaling"].(map[string]any); ok && len(rs) > 0 {
		return ""
	}
	src := eval.RopeDict(cfg)
	if src == nil {
		return ""
	}
	out := map[string]any{}
	for k, v := range src {
		out[k] = v
	}
	t := eval.RopeType(src)
	switch t {
	case "", "mrope":
		out["rope_type"] = "default"
	default:
		out["rope_type"] = t
	}
	cfg["rope_scaling"] = out
	if t == "" {
		t = "default"
	}
	return "rope_scaling from rope_parameters (" + t + ")"
}

package eval

// Architectures the forward instruments. Each one is the checkpoint's own block, not a conversion
// into BitNet. The initial text runs are 2b4t (packed ternary, relu²), llama3, vaultgemma
// (Gemma RMS and embed scale, no sandwich, full attention), and gpt-oss (top-k experts, attention
// sinks, YaRN). Qwen stays the dense GQA block it already is. The manifold is the same cell complex
// for all of them; a mixture writes its selected experts into the shared F cells.

import (
	"fmt"
	"math"
	"strings"
)

// Family names the forward prints.
const (
	Fam2B4T   = "2b4t"
	FamLlama  = "llama"
	FamQwen   = "qwen"
	FamGemma  = "gemma"
	FamGptOss = "gpt-oss"
)

// Arch is how one config departs from a dense pre-norm GQA block.
type Arch struct {
	Family       string
	ModelType    string
	Norm         string // "rms" (x * w) or "gemma" (x * (1+w))
	EmbedScale   bool   // gemma multiplies the embedding by sqrt(hidden)
	Sandwich     bool   // gemma2/3: norm the sublayer output before the residual add
	Sinks        bool   // gpt-oss learned per-head attention sink
	MoE          bool
	Experts      int
	TopK         int
	SwigluAlpha  float64
	SwigluLimit  float64
	QueryScale   float64 // attention denominator; 0 means head_dim
	AttnSoftcap  float64
	FinalSoftcap float64
	LocalTheta   float64 // gemma3 RoPE base on sliding layers; 0 means the global RoPE
	Window       []int   // per layer, 0 = full causal
}

// Classify reads the architecture out of a config. layers is the tensor stack; 0 uses
// num_hidden_layers. A Gemma 3 multimodal config contributes its text_config.
func Classify(cfg map[string]any, layers int) (Arch, error) {
	cfg = flattenText(cfg)
	mt, _ := cfg["model_type"].(string)
	a := Arch{ModelType: mt, Norm: "rms"}
	switch mt {
	case "bitnet", "hodge":
		a.Family = Fam2B4T
	case "", "llama":
		a.Family = FamLlama
	case "qwen", "qwen2", "qwen3":
		a.Family = FamQwen
	case "gemma", "gemma2", "gemma3", "gemma3_text", "vaultgemma":
		a.Family = FamGemma
		a.Norm = "gemma"
		a.EmbedScale = true
		// VaultGemma is Gemma 2 with the post-attention and post-FFN norms removed, and
		// full attention on every layer. Sandwich stays off for it and for Gemma 1.
		a.Sandwich = mt == "gemma2" || strings.HasPrefix(mt, "gemma3")
	case "gpt_oss":
		a.Family = FamGptOss
		a.Sinks = true
		a.MoE = true
		a.SwigluAlpha = 1.702
		a.SwigluLimit = 7
	default:
		if strings.Contains(mt, "moe") || strings.Contains(mt, "mixtral") {
			return a, fmt.Errorf("model_type %q is a mixture; the mixture forward is gpt-oss", mt)
		}
		// A dense checkpoint under another name still runs the llama block. The log uses its own name.
		a.Family = mt
	}
	// A packed mix used to stamp model_type bitnet. Gemma, including VaultGemma, still names
	// the activation hidden_activation. That key is the block: RMS is (1+w), embed scale is
	// sqrt(hidden). Sandwich stays off unless the source model_type is gemma2 or gemma3.
	if a.Norm != "gemma" && gemmaBlock(cfg) {
		a.Family = FamGemma
		a.Norm = "gemma"
		a.EmbedScale = true
	}
	if layers == 0 {
		layers = num(cfg, "num_hidden_layers")
	}
	if v, ok := cfg["query_pre_attn_scalar"].(float64); ok && v > 0 {
		a.QueryScale = v
	}
	if v, ok := cfg["attn_logit_softcapping"].(float64); ok && v > 0 {
		a.AttnSoftcap = v
	}
	if v, ok := cfg["final_logit_softcapping"].(float64); ok && v > 0 {
		a.FinalSoftcap = v
	}
	if v, ok := cfg["rope_local_base_freq"].(float64); ok && v > 0 && a.Family == FamGemma {
		a.LocalTheta = v
	}
	if a.MoE {
		a.Experts = num(cfg, "num_local_experts")
		if a.Experts == 0 {
			a.Experts = num(cfg, "num_experts")
		}
		a.TopK = num(cfg, "num_experts_per_tok")
		if a.TopK == 0 {
			a.TopK = num(cfg, "experts_per_token")
		}
		if a.Experts < 1 || a.TopK < 1 {
			return a, fmt.Errorf("gpt-oss needs num_local_experts and num_experts_per_tok")
		}
		if a.TopK > a.Experts {
			return a, fmt.Errorf("gpt-oss top-%d of %d experts", a.TopK, a.Experts)
		}
		if v, ok := cfg["swiglu_alpha"].(float64); ok && v > 0 {
			a.SwigluAlpha = v
		}
		if v, ok := cfg["swiglu_limit"].(float64); ok && v > 0 {
			a.SwigluLimit = v
		}
	}
	w, err := windows(cfg, layers, a.Family)
	if err != nil {
		return a, err
	}
	a.Window = w
	return a, nil
}

// gemmaBlock is true when the config is a Gemma model under any model_type stamp.
func gemmaBlock(cfg map[string]any) bool {
	if s, _ := cfg["hidden_activation"].(string); s != "" {
		return true
	}
	if s, _ := cfg["rms_norm"].(string); s == "gemma" {
		return true
	}
	arch, _ := cfg["architectures"].([]any)
	for _, v := range arch {
		s, _ := v.(string)
		if strings.Contains(strings.ToLower(s), "gemma") {
			return true
		}
	}
	return false
}

func num(cfg map[string]any, k string) int {
	v, _ := cfg[k].(float64)
	return int(v)
}

// flattenText lifts Gemma 3's text_config when the outer config is the multimodal wrapper.
func flattenText(cfg map[string]any) map[string]any {
	if _, ok := cfg["hidden_size"]; ok {
		return cfg
	}
	tc, ok := cfg["text_config"].(map[string]any)
	if !ok {
		return cfg
	}
	out := map[string]any{}
	for k, v := range tc {
		out[k] = v
	}
	for k, v := range cfg {
		if k == "text_config" || k == "vision_config" {
			continue
		}
		out[k] = v
	}
	return out
}

// windows is the causal window of each layer. 0 is full attention. layer_types wins. Gemma 2
// alternates, even layers sliding. Gemma 3's published pattern keeps every 6th layer full.
// VaultGemma publishes sliding_window and then marks every layer full_attention; with no
// layer_types it still does not alternate. Qwen carries a window it does not use unless
// use_sliding_window is true.
func windows(cfg map[string]any, layers int, family string) ([]int, error) {
	sw := num(cfg, "sliding_window")
	if raw, ok := cfg["layer_types"].([]any); ok && len(raw) > 0 {
		if layers > 0 && len(raw) != layers {
			return nil, fmt.Errorf("layer_types has %d entries, the model has %d layers", len(raw), layers)
		}
		out := make([]int, len(raw))
		for i, v := range raw {
			s, _ := v.(string)
			switch s {
			case "full_attention":
			case "sliding_attention":
				if sw <= 0 {
					return nil, fmt.Errorf("layer %d is sliding_attention and sliding_window is %d", i, sw)
				}
				out[i] = sw
			default:
				return nil, fmt.Errorf("layer_types %q is not full_attention or sliding_attention", s)
			}
		}
		return out, nil
	}
	out := make([]int, layers)
	switch family {
	case FamGemma:
		mt, _ := cfg["model_type"].(string)
		if sw <= 0 {
			return out, nil
		}
		switch {
		case mt == "gemma2":
			for i := range out {
				if i%2 == 0 {
					out[i] = sw
				}
			}
		case strings.HasPrefix(mt, "gemma3"):
			pat := num(cfg, "_sliding_window_pattern")
			if pat <= 0 {
				pat = 6
			}
			for i := range out {
				if (i+1)%pat != 0 {
					out[i] = sw
				}
			}
		}
	case FamQwen:
		use, has := cfg["use_sliding_window"].(bool)
		if has && use && sw > 0 {
			for i := range out {
				out[i] = sw
			}
		}
	case FamGptOss:
		if sw > 0 {
			for i := range out {
				if i%2 == 0 {
					out[i] = sw
				}
			}
		}
	}
	return out, nil
}

// ropeFreqs is the rotary inverse frequencies and the extra magnitude YaRN applies to the
// rotated queries and keys (1 for every other scaling).
func ropeFreqs(cfg map[string]any, theta float64, hd int) ([]float64, float64, error) {
	if p, ok := cfg["partial_rotary_factor"].(float64); ok && p != 1 {
		return nil, 0, fmt.Errorf("partial_rotary_factor %g is not modelled", p)
	}
	f := make([]float64, hd/2)
	for i := range f {
		f[i] = math.Pow(theta, -2*float64(i)/float64(hd))
	}
	rs, _ := cfg["rope_scaling"].(map[string]any)
	if rs == nil {
		return f, 1, nil
	}
	typ, _ := rs["rope_type"].(string)
	if typ == "" {
		typ, _ = rs["type"].(string)
	}
	numf := func(k string, d float64) float64 {
		if v, ok := rs[k].(float64); ok {
			return v
		}
		return d
	}
	switch typ {
	case "", "default":
	case "dynamic":
	case "linear":
		fac := numf("factor", 1)
		for i := range f {
			f[i] /= fac
		}
	case "llama3":
		fac, lo, hi := numf("factor", 8), numf("low_freq_factor", 1), numf("high_freq_factor", 4)
		orig := numf("original_max_position_embeddings", 8192)
		loW, hiW := orig/lo, orig/hi
		for i, v := range f {
			wl := 2 * math.Pi / v
			switch {
			case wl < hiW:
			case wl > loW:
				f[i] = v / fac
			default:
				sm := (orig/wl - lo) / (hi - lo)
				f[i] = (1-sm)*v/fac + sm*v
			}
		}
	case "yarn":
		fac, ok := rs["factor"].(float64)
		if !ok || fac <= 0 {
			return nil, 0, fmt.Errorf("rope_scaling yarn needs factor")
		}
		orig, ok := rs["original_max_position_embeddings"].(float64)
		if !ok || orig <= 0 {
			return nil, 0, fmt.Errorf("rope_scaling yarn needs original_max_position_embeddings")
		}
		if tr, ok := rs["truncate"].(bool); ok && tr {
			return nil, 0, fmt.Errorf("rope_scaling yarn truncate is not modelled")
		}
		betaFast, betaSlow := numf("beta_fast", 32), numf("beta_slow", 1)
		dim := float64(hd)
		corr := func(rot float64) float64 {
			return (dim * math.Log(orig/(rot*2*math.Pi))) / (2 * math.Log(theta))
		}
		low := int(math.Max(0, math.Floor(corr(betaFast))))
		high := int(math.Min(dim-1, math.Ceil(corr(betaSlow))))
		span := float64(high - low)
		if span == 0 {
			span = 0.001
		}
		for i, v := range f {
			ramp := (float64(i) - float64(low)) / span
			if ramp < 0 {
				ramp = 0
			} else if ramp > 1 {
				ramp = 1
			}
			mask := 1 - ramp
			f[i] = (v/fac)*(1-mask) + v*mask
		}
		scale := 0.1*math.Log(fac) + 1
		if v, ok := rs["attention_factor"].(float64); ok && v > 0 {
			scale = v
		}
		return f, scale, nil
	default:
		return nil, 0, fmt.Errorf("rope_scaling %q is not modelled", typ)
	}
	return f, 1, nil
}

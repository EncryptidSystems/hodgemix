package main

// Repair (--repair on, default): put back the base's per-channel statistics with the free BF16 gains.
//
// A merge leaves every channel's spread off the base's (variance collapse in merged networks; REPAIR,
// Jordan et al. 2022), and singularity_surgery.pdf step 7 repairs it for the singular units through the
// gains, which are BF16 and not ternary. Here every channel of every RMSNorm a BitLinear (or the head)
// reads is repaired the same way: with sigma_c the rms over the calibration tokens of channel c after
// the gain (eval.NormStats),
//     g_c <- g_c * clamp(sigma_c^base / sigma_c^out, 1/2, 2)        (the T5 band, one BF16 binade)
// for input_layernorm, attn_sub_norm, the FFN pre-norm, ffn_sub_norm and model.norm, when the
// base checkpoint stores that gain. The FFN pre-norm is post_attention_layernorm, or
// pre_feedforward_layernorm when that is the name the base stores (VaultGemma).
// Champ inserts attn_sub_norm and ffn_sub_norm at unit gain for a
// base that has neither. The base's tap there is the raw sublayer, so the ratio is not a correction
// of a shared gain: three clamps walk the inserted gain 1 → 1/2 → 1/4 → 1/8 and every later layer
// scales both residual branches by 1/8. Those gains stay at 1. The output scales linearly with g_c,
// so one pass fixes each shared norm given its input; fixing early layers moves later inputs, so
// passes continue until the measured spread is below repairStop. That stop is in the 0.1s, short of
// a ratio of 1, and a pass already under it writes nothing. Ternary codes and weight_scales are untouched;
// gains are written in place. Forward passes only, no training.
// No paper equation.

import (
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"hodgemix/eval"
)

// RepairOn is --repair.
var RepairOn = true

// repairStop: a pass whose mean |log2| channel spread is below this ends the repair.
// 0.2 is still in the tenths, short of a ratio of 1, so the last small correction is not written.
const repairStop = 0.2

func normTensor(base *ST, key string) string {
	parts := strings.SplitN(key, "/", 2)
	l, _ := strconv.Atoi(parts[0])
	p := fmt.Sprintf("model.layers.%d.", l)
	switch parts[1] {
	case "in_attn":
		return p + "input_layernorm.weight"
	case "in_o":
		return p + "self_attn.attn_sub_norm.weight"
	case "in_ffn":
		post := p + "post_attention_layernorm.weight"
		pre := p + "pre_feedforward_layernorm.weight"
		if base != nil && base.Has(pre) && !base.Has(post) {
			return pre
		}
		return post
	case "in_down":
		return p + "mlp.ffn_sub_norm.weight"
	case "final":
		return "model.norm.weight"
	}
	return ""
}

type repairPass struct {
	MaxLog2  float64 `json:"max_abs_log2_ratio"`  // largest channel correction this pass (before the clamp)
	MeanLog2 float64 `json:"mean_abs_log2_ratio"` // mean over channels
	Clamped  int     `json:"clamped_channels"`
}

type repairReport struct {
	Tokens int          `json:"calibration_tokens"`
	Passes []repairPass `json:"passes"`
}

// repairRatio is clamp(base/out, 1/2, 2). A missing denominator leaves the gain unchanged.
func repairRatio(base, out float64) float64 {
	r := 1.0
	if out > 0 && base > 0 {
		r = base / out
	}
	if r > 2 || r < 0.5 {
		r = math.Max(0.5, math.Min(2, r))
	}
	return r
}

// baseOwnsGain is false for a norm Champ inserted. The base has no gain to match.
func baseOwnsGain(base *ST, name string) bool { return base.Has(name) }

func repair(basePath, outTern, tok string) (*repairReport, error) {
	baseST, err := OpenST(basePath)
	if err != nil {
		return nil, err
	}
	ref, n, err := eval.NormStats(basePath, tok)
	if err != nil {
		return nil, err
	}
	rep := &repairReport{Tokens: n}
	keys := make([]string, 0, len(ref))
	for k := range ref {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var skipped []string
	for pass := 0; ; pass++ {
		cur, _, err := eval.NormStats(outTern, tok)
		if err != nil {
			return rep, err
		}
		out, err := OpenST(outTern)
		if err != nil {
			return rep, err
		}
		f, err := os.OpenFile(outTern, os.O_RDWR, 0)
		if err != nil {
			return rep, err
		}
		var pr repairPass
		var sum float64
		var cnt int
		bufs := map[string][]byte{}
		for _, k := range keys {
			name := normTensor(baseST, k)
			sb, so := ref[k], cur[k]
			if name == "" {
				f.Close()
				return rep, fmt.Errorf("norm stage %s has no gain tensor", k)
			}
			if !baseOwnsGain(baseST, name) {
				if pass == 0 {
					skipped = append(skipped, name)
				}
				continue
			}
			if !out.Has(name) || out.H[name].Dtype != "BF16" {
				f.Close()
				return rep, fmt.Errorf("%s: missing from the output or not BF16", name)
			}
			g, err := out.F32(name)
			if err != nil {
				f.Close()
				return rep, err
			}
			if len(sb) != len(so) || len(g) != len(sb) {
				f.Close()
				return rep, fmt.Errorf("%s: %d gains, base forward %d channels, output forward %d", name, len(g), len(sb), len(so))
			}
			buf := make([]byte, 2*len(g))
			for c := range g {
				raw := 1.0
				if so[c] > 0 && sb[c] > 0 {
					raw = sb[c] / so[c]
				}
				r := repairRatio(sb[c], so[c])
				l2 := math.Abs(math.Log2(raw))
				pr.MaxLog2 = math.Max(pr.MaxLog2, l2)
				sum += l2
				cnt++
				if raw > 2 || raw < 0.5 {
					pr.Clamped++
				}
				u := bf16(float32(float64(g[c]) * r))
				buf[2*c], buf[2*c+1] = byte(u), byte(u>>8)
			}
			bufs[name] = buf
		}
		if cnt > 0 {
			pr.MeanLog2 = sum / float64(cnt)
		}
		rep.Passes = append(rep.Passes, pr)
		if pr.MeanLog2 < repairStop {
			f.Close()
			logf("  repair pass %d: channel spread vs base |log2| mean %.3f max %.2f < stop %.3g: gains left as they are", pass+1, pr.MeanLog2, pr.MaxLog2, repairStop)
			break
		}
		for name, buf := range bufs {
			if _, err := f.WriteAt(buf, out.base+out.H[name].Off[0]); err != nil {
				f.Close()
				return rep, err
			}
		}
		f.Close()
		logf("  repair pass %d: channel spread vs base |log2| mean %.3f max %.2f (%d channels clamped to [1/2, 2])", pass+1, pr.MeanLog2, pr.MaxLog2, pr.Clamped)
	}
	if len(skipped) > 0 {
		logf("  repair left %d gains the base does not have at unit gain", len(skipped))
	}
	return rep, nil
}

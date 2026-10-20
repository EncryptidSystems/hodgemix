package main

// Residual-stream rotation plan (paper_v2 Eq.(24)): computational invariance of RMSNorm transformers
// under x -> Q^T x after folding the RMSNorm gains of residual readers into their weights.

import (
	"regexp"
	"strings"
)

type rotPlan struct {
	Q      *Rot
	H      int
	gain   map[string][]float32 // reader tensor -> folded norm gain
	folded map[string]bool      // norm tensors replaced by ones
	finalG []float32
	embed  string
	lmHead string
	untied bool
}

var preNormRe = regexp.MustCompile(`(input_layernorm|attention_norm|ln_1)\.weight$`)
var postNormRe = regexp.MustCompile(`(post_attention_layernorm|ffn_norm|ln_2)\.weight$`)

// Paper_v2 Eq.(24): readers of the residual stream (W -> W diag(g) Q) and writers (W -> Q^T W)
func rotKind(role string) string {
	switch role {
	case "q", "k", "v", "qkv", "gate", "up", "lm_head":
		return "reader"
	case "o", "down":
		return "writer"
	case "embed":
		return "embed"
	}
	return ""
}

// Paper_v2 Eq.(24): build the plan, or return nil (logged) when invariance cannot be guaranteed
func planRotation(T []target, base *ST, seed int64, Tp *[]target) *rotPlan {
	p := &rotPlan{gain: map[string][]float32{}, folded: map[string]bool{}}
	pre, post := map[int]target{}, map[int]target{}
	var final *target
	for i, t := range T {
		switch {
		case t.role == "embed":
			p.embed, p.H = t.name, t.shape[1]
		case t.role == "lm_head" && len(t.shape) == 2:
			p.lmHead = t.name
		case len(t.shape) == 1 && preNormRe.MatchString(t.name) && t.layer >= 0:
			pre[t.layer] = t
		case len(t.shape) == 1 && postNormRe.MatchString(t.name) && t.layer >= 0:
			post[t.layer] = t
		case len(t.shape) == 1 && t.layer < 0 && strings.Contains(t.name, "norm"):
			final = &T[i]
		}
	}
	fail := func(why string) *rotPlan { logf("  rotation (Eq.24) NOT applied: %s", why); return nil }
	if p.embed == "" || final == nil {
		return fail("embedding or final norm missing from the output tensor list")
	}
	gainOf := func(t target) []float32 {
		if base.Has(t.name) && eqShape(base.LogicalShape(t.name), t.shape) {
			a, _ := base.F32(t.name)
			return a
		}
		o := make([]float32, t.shape[0])
		for i := range o {
			o[i] = 1
		}
		return o
	}
	for _, t := range T {
		if len(t.shape) != 2 || t.role == "embed" {
			continue
		}
		k := rotKind(t.role)
		switch {
		case k == "":
			return fail("2-D tensor " + t.name + " has no residual role")
		case k == "writer" && t.shape[0] != p.H, k == "reader" && t.shape[1] != p.H:
			return fail("shape of " + t.name + " does not touch hidden size")
		case t.role == "lm_head":
			continue
		}
		if k == "reader" {
			nt, ok := pre[t.layer]
			if t.role == "gate" || t.role == "up" {
				nt, ok = post[t.layer]
			}
			if !ok {
				return fail("no RMSNorm before " + t.name)
			}
			p.gain[t.name] = gainOf(nt)
			p.folded[nt.name] = true
		}
	}
	p.finalG = gainOf(*final)
	p.folded[final.name] = true
	if p.lmHead == "" {
		var V int
		for _, t := range T {
			if t.name == p.embed {
				V = t.shape[0]
			}
		}
		p.lmHead, p.untied = "lm_head.weight", true
		*Tp = append(*Tp, target{"lm_head.weight", "lm_head", -1, []int{V, p.H}})
		logf("  rotation: tied embeddings -> writing untied lm_head.weight (tie_word_embeddings=false)")
	}
	p.Q = NewRot(p.H, seed)
	logf("  rotation (Eq.23-24) ON: H=%d Hadamard block=%d, %d readers folded, %d norms -> 1", p.H, p.Q.B, len(p.gain), len(p.folded))
	return p
}

// Paper: helper
func (p *rotPlan) handles(t target) bool {
	return t.name == p.embed || t.name == p.lmHead || p.folded[t.name]
}

// Paper_v2 Eq.(24c)-(24e): embedding E -> E Q, lm_head -> W diag(g_final) Q, folded norms -> 1
func (p *rotPlan) nonIndex(t target, base *ST) []float32 {
	n := numel(t.shape)
	if p.folded[t.name] {
		a := make([]float32, n)
		for i := range a {
			a[i] = 1
		}
		return a
	}
	src := t.name
	if t.name == p.lmHead && (p.untied || !base.Has(src)) {
		src = p.embed
	}
	var a []float32
	if base.Has(src) {
		b, _ := base.F32(src)
		sh := base.LogicalShape(src)
		a = resize2(b, sh[0], sh[1], t.shape[0], t.shape[1])
	} else {
		a = make([]float32, n)
	}
	if t.name == p.lmHead {
		p.Q.Reader(a, t.shape[0], p.finalG)
	} else {
		p.Q.Reader(a, t.shape[0], nil)
	}
	return a
}

// Paper_v2 Eq.(24a)-(24b)
func (p *rotPlan) apply(t target, W []float32) {
	switch rotKind(t.role) {
	case "reader":
		p.Q.Reader(W, t.shape[0], p.gain[t.name])
	case "writer":
		p.Q.Writer(W, t.shape[1])
	}
}

package main

// Attention after marriage (--attn on, default).
//
// The married index keeps heads whole and kv groups intact, but three things about attention are not
// carried by the weights' layout alone:
//
//  1. QK-norm. The Champ runtime has none, so every model with per-head q/k RMSNorm (q_norm /
//     k_norm, Qwen3), the base included, has it folded: each head's rows are scaled to the unit rms
//     QK-norm gives on a normalized input, c_h = sqrt(hd) / ||W_h diag(g_in)||_F (g_in the layer's
//     input_layernorm gain), and the q_norm/k_norm gain is folded per dim. The output drops
//     q_norm / k_norm (champLayout).
//  2. RoPE frequency. The head_dim axis is married by rotation frequency: target pair d takes the
//     contributor's rows at s = d ln(theta_t) / ln(theta_c) (rotate-half layout, both halves), so a
//     model trained at another rope_theta rotates at the target's frequencies. q and k get the same map.
//  3. Logit temperature. Once a layer's q and k are mixed, q's weight_scale is set so the mean
//     per-head logit spread sigma_h = sqrt(tr(Q_h Q_h^T K_g K_g^T)) (query head h, its kv head g;
//     the std of q.k for independent unit inputs) equals the base layer's. Skipped when the base
//     has QK-norm: its folded heads already sit at the scale QK-norm gives. A self-mix gives x1.

import (
	"bytes"
	"encoding/binary"
	"math"
	"strconv"
	"strings"
)

// AttnOn is --attn; SrcRopeTheta is --src-rope-theta (models whose config gives none, base included).
var (
	AttnOn       = true
	SrcRopeTheta = 0.0
)

func layerPrefix(name string) string {
	if i := strings.Index(name, "self_attn."); i >= 0 {
		return name[:i]
	}
	return ""
}

func onesOr(st *ST, name string, n int) []float32 {
	if st.Has(name) {
		if v, err := st.F32(name); err == nil && len(v) == n {
			return v
		}
	}
	v := make([]float32, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

// prepQK applies 1 and 2 to a contributor's raw q or k rows ([heads*hd, D]); notes say what it did.
func (tp *topology) prepQK(m int, st *ST, name, role string, W []float32, sh []int) ([]float32, []string) {
	if tp == nil || !AttnOn || (role != "q" && role != "k") || len(sh) != 2 {
		return W, nil
	}
	mm := tp.metas[m]
	hd, D := mm.HD, sh[1]
	if hd <= 0 || sh[0]%hd != 0 {
		return W, nil
	}
	heads := sh[0] / hd
	var notes []string
	pre := layerPrefix(name)
	if mm.QKNorm && pre != "" { // the Champ runtime has no QK-norm: fold it, base included
		gin := onesOr(st, pre+"input_layernorm.weight", D)
		gain := onesOr(st, pre+"self_attn."+role+"_norm.weight", hd)
		out := make([]float32, len(W))
		parFor(heads, func(lo, hi int) {
			for h := lo; h < hi; h++ {
				var fro float64
				for r := h * hd; r < (h+1)*hd; r++ {
					for c := 0; c < D; c++ {
						v := float64(W[r*D+c]) * float64(gin[c])
						fro += v * v
					}
				}
				ch := 0.0
				if fro > 0 {
					ch = math.Sqrt(float64(hd) / fro)
				}
				for d := 0; d < hd; d++ {
					f := float32(ch * float64(gain[d]))
					r := h*hd + d
					for c := 0; c < D; c++ {
						out[r*D+c] = W[r*D+c] * f
					}
				}
			}
		})
		W = out
		notes = append(notes, "QK-norm folded")
	}
	tc, tt := mm.Theta, tp.tgt.Theta
	if tc == 0 && m > 0 {
		tc = SrcRopeTheta
	}
	if tc > 1 && tt > 1 && math.Abs(tc-tt) > 1e-9*tt {
		W = ropeRemap(W, heads, hd, D, tc, tt)
		notes = append(notes, "RoPE theta "+fmtG(tc)+" -> "+fmtG(tt))
	} else if tc == 0 && m > 0 && tt > 1 {
		notes = append(notes, "RoPE theta unknown (pass --src-rope-theta): frequencies left as trained")
	}
	return W, notes
}

func fmtG(x float64) string { return strconv.FormatFloat(x, 'g', -1, 64) }

// ropeRemap marries the head_dim axis by frequency: target pair d <- contributor rows at
// s = d ln(tt)/ln(tc), linearly between pairs, the same for both rotate-half halves.
func ropeRemap(W []float32, heads, hd, D int, tc, tt float64) []float32 {
	half := hd / 2
	ratio := math.Log(tt) / math.Log(tc)
	out := make([]float32, len(W))
	for h := 0; h < heads; h++ {
		for d := 0; d < half; d++ {
			s := math.Min(float64(d)*ratio, float64(half-1))
			i0 := int(s)
			i1 := min(i0+1, half-1)
			f := float32(s - float64(i0))
			for _, off := range []int{0, half} {
				dst := (h*hd + off + d) * D
				a, b := (h*hd+off+i0)*D, (h*hd+off+i1)*D
				for c := 0; c < D; c++ {
					out[dst+c] = W[a+c] + f*(W[b+c]-W[a+c])
				}
			}
		}
	}
	return out
}

// ---- 3: logit temperature ----

// headSigma is the mean over (sampled) query heads of sqrt(tr(Q_h Q_h^T K_g K_g^T)) for q [H*hd, D],
// k [KV*hd, D], query head h reading kv head h/G. Heads are sampled evenly when all of them would
// cost more than ~4e9 multiply-adds.
func headSigma(q, k func(r, c int) float64, H, KV, hd, D int) float64 {
	G := max(1, H/max(1, KV))
	step := 1
	if cost := float64(H) * float64(hd*hd) * float64(D) * 2; cost > 4e9 {
		step = int(math.Ceil(cost / 4e9))
	}
	var hs []int
	for h := 0; h < H; h += step {
		hs = append(hs, h)
	}
	gram := func(m func(r, c int) float64, base int) []float64 {
		g := make([]float64, hd*hd)
		for i := 0; i < hd; i++ {
			for j := i; j < hd; j++ {
				var s float64
				for c := 0; c < D; c++ {
					s += m(base+i, c) * m(base+j, c)
				}
				g[i*hd+j], g[j*hd+i] = s, s
			}
		}
		return g
	}
	kg := map[int][]float64{}
	for _, h := range hs {
		kg[h/G] = nil
	}
	gs := make([]int, 0, len(kg))
	for g := range kg {
		gs = append(gs, g)
	}
	kgr := make([][]float64, len(gs))
	parFor(len(gs), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			kgr[i] = gram(k, gs[i]*hd)
		}
	})
	for i, g := range gs {
		kg[g] = kgr[i]
	}
	sig := make([]float64, len(hs))
	parFor(len(hs), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			gq, gk := gram(q, hs[i]*hd), kg[hs[i]/G]
			var tr float64
			for a := range gq {
				tr += gq[a] * gk[a] // both symmetric: tr(A B) = sum A_ij B_ij
			}
			sig[i] = math.Sqrt(math.Max(tr, 0))
		}
	})
	var s float64
	for _, v := range sig {
		s += v
	}
	return s / float64(len(sig))
}

type calTensor struct {
	name  string
	codes []int8
	ws    float64
	R, C  int
}

// attnCal pairs each target layer's mixed q and k, then sets q's weight_scale (Patch) so the
// layer's logit spread equals the base layer's.
type attnCal struct {
	tp    *topology
	base  *ST
	q, k  map[int]*calTensor
	notes []string
}

func newAttnCal(tp *topology, base *ST) *attnCal {
	if tp == nil || !AttnOn || tp.metas[0].QKNorm {
		return nil
	}
	return &attnCal{tp: tp, base: base, q: map[int]*calTensor{}, k: map[int]*calTensor{}}
}

// add records a written q or k tensor; when its layer has both, q's weight_scale is patched in W.
func (a *attnCal) add(t target, codes []int8, ws float64, W *STW) {
	if a == nil || t.layer < 0 || (t.role != "q" && t.role != "k") {
		return
	}
	ct := &calTensor{t.name, codes, ws, t.shape[0], t.shape[1]}
	if t.role == "q" {
		a.q[t.layer] = ct
	} else {
		a.k[t.layer] = ct
	}
	q, k := a.q[t.layer], a.k[t.layer]
	if q == nil || k == nil {
		return
	}
	delete(a.q, t.layer)
	delete(a.k, t.layer)
	g := a.tp.tgt
	mix := headSigma(func(r, c int) float64 { return float64(q.codes[r*q.C+c]) },
		func(r, c int) float64 { return float64(k.codes[r*k.C+c]) }, g.Heads, g.KV, g.HD, q.C) * q.ws * k.ws
	bm := a.tp.metas[0]
	bl := a.tp.baseLayerFor(t.layer)
	bq := strings.Replace(q.name, "."+itoa(t.layer)+".", "."+itoa(bl)+".", 1)
	bk := strings.Replace(k.name, "."+itoa(t.layer)+".", "."+itoa(bl)+".", 1)
	Q, qsh, err1 := a.base.LoadDense(bq)
	K, _, err2 := a.base.LoadDense(bk)
	if err1 != nil || err2 != nil {
		die("attention temperature layer %d: %v %v", t.layer, err1, err2)
	}
	if mix <= 0 {
		die("attention temperature layer %d: the mixed q/k have no logit spread", t.layer)
	}
	ref := headSigma(func(r, c int) float64 { return float64(Q[r*qsh[1]+c]) },
		func(r, c int) float64 { return float64(K[r*qsh[1]+c]) }, bm.Heads, bm.KV, bm.HD, qsh[1])
	if ref <= 0 {
		die("attention temperature layer %d: the base's q/k have no logit spread", t.layer)
	}
	s := ref / mix
	b := make([]byte, 2)
	binary.LittleEndian.PutUint16(b, bf16(float32(q.ws*s)))
	if err := W.Patch(scaleKey(q.name), b); err != nil {
		die("attention temperature %s: %v", q.name, err)
	}
	logf("      attention temperature layer %d: logit spread %.4g vs base %.4g -> q weight_scale x%.4f", t.layer, mix, ref, s)
}

// champLayout makes the target tensor list one the Champ runtime loads whatever the base's family:
// q_norm / k_norm are dropped when --attn folds them into q / k, and a layer without BitNet's
// attn_sub_norm / ffn_sub_norm gets them with unit gain (sized by the layer's q rows and gate rows),
// and an lm_head.weight byte-identical to the embeddings (a tied model that stores both, Qwen3) is
// dropped: the runtime's head is the tied embedding.
func champLayout(T []target, tp *topology, base *ST) ([]target, []string) {
	have := map[string]bool{}
	for _, t := range T {
		have[t.name] = true
	}
	var out []target
	var notes []string
	dropped := 0
	tied := false
	if base != nil && base.Has("lm_head.weight") && base.Has("model.embed_tokens.weight") {
		a, e1 := base.Raw("lm_head.weight")
		b, e2 := base.Raw("model.embed_tokens.weight")
		tied = e1 == nil && e2 == nil && bytes.Equal(a, b)
	}
	for _, t := range T {
		if tied && t.name == "lm_head.weight" {
			notes = append(notes, "lm_head.weight (identical to the tied embeddings)")
			continue
		}
		if tp != nil && AttnOn && (strings.HasSuffix(t.name, "self_attn.q_norm.weight") || strings.HasSuffix(t.name, "self_attn.k_norm.weight")) {
			dropped++
			continue
		}
		out = append(out, t)
		if t.layer < 0 || len(t.shape) != 2 {
			continue
		}
		p := t.name[:strings.Index(t.name, "."+itoa(t.layer)+".")+len(itoa(t.layer))+2]
		add := func(n string, size int) {
			if !have[p+n] {
				have[p+n] = true
				out = append(out, target{p + n, "norm", t.layer, []int{size}})
				if t.layer == 0 {
					notes = append(notes, n+" (unit gain)")
				}
			}
		}
		switch t.role {
		case "q":
			add("self_attn.attn_sub_norm.weight", t.shape[0])
		case "gate":
			add("mlp.ffn_sub_norm.weight", t.shape[0])
		}
	}
	if dropped > 0 {
		notes = append(notes, itoa(dropped)+" q_norm/k_norm (folded into q/k)")
	}
	return out, notes
}

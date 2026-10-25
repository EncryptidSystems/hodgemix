package main

// Topology: the axis graph of each model, the married index per axis node, and the target topology
// both models coexist in.
//
// A transformer's tensors are edges between axis nodes. Every tensor on a node must see that node in
// one order, or the network stops computing one function:
//
//	D      the residual stream: q,k,v,gate,up columns; o,down rows; embeddings, lm_head, D norms
//	F@l    layer l's FFN: gate,up rows; down columns; ffn_sub_norm
//	KV@l   layer l's key/value heads: k,v rows
//	Q@l    layer l's query heads: q rows; o columns; attn_sub_norm
//
// Heads stay whole (RoPE rotates pairs inside a head) and grouped-query structure stays intact
// (query heads h*G..h*G+G-1 read kv head h), so the attention unit is a kv group: one kv head and
// its G query heads, ordered once on KV and carried to Q. D and F units are single channels.
//
// Each axis node of each model gets one MRH key per unit, the mean of the Eq.(7) keys of every
// tensor that touches it. For a target axis node, the (model, layer) instances that contribute to
// it (Eq.(11) depth hat) are married into one MRH (married.go); every tensor on that node then uses
// the same married coordinates. --topology picks the target sizes: base (the base model's), max
// (the larger per axis) or married (lcm per axis: every model's digit hierarchy embeds exactly).

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"hodgemix/eval"
)

type axisKind int

const (
	axD axisKind = iota
	axF
	axKV
	axQ
)

func (k axisKind) String() string { return [...]string{"D", "F", "KV", "Q"}[k] }

// roleAxes is (row axis, column axis) of a BitLinear role.
func roleAxes(role string) (axisKind, axisKind, bool) {
	switch role {
	case "q":
		return axQ, axD, true
	case "k", "v":
		return axKV, axD, true
	case "o":
		return axD, axQ, true
	case "gate", "up":
		return axF, axD, true
	case "down":
		return axD, axF, true
	}
	return 0, 0, false
}

// vecAxis is the axis a 1-D tensor lies on (norm gains, biases); ok=false keeps its length.
func vecAxis(name string) (axisKind, bool) {
	switch {
	case strings.Contains(name, "attn_sub_norm"):
		return axQ, true
	case strings.Contains(name, "ffn_sub_norm"):
		return axF, true
	case strings.Contains(name, "q_norm"), strings.Contains(name, "k_norm"):
		return 0, false // per head_dim: unchanged
	case strings.HasSuffix(name, ".bias"):
		if r, _, ok := roleAxes(roleOf(name)); ok {
			return r, true
		}
		return 0, false
	case strings.HasSuffix(name, "norm.weight"):
		return axD, true
	}
	return 0, false
}

type modelMeta struct {
	D, F, Heads, KV, HD, Layers int
	FBy                         []int   `json:",omitempty"` // per-layer intermediate when it is not F (Gemma 4 double-wide MLP)
	KVBy                        []int   `json:",omitempty"` // per-layer KV*head_dim when a layer's head is wider (Gemma 4 full attention)
	QBy                         []int   `json:",omitempty"` // per-layer Heads*head_dim, same layers
	Theta                       float64 // rope_theta; 0 = unknown
	QKNorm                      bool    // per-head q_norm / k_norm (Qwen3)
	Config                      string  `json:",omitempty"` // the config set for the model, when it agreed with the tensors
}

// noteF records one layer's intermediate size. gate and up win over down.
func (m *modelMeta) noteF(layer, f int, prefer bool) {
	if layer < 0 || f <= 0 {
		return
	}
	if layer >= len(m.FBy) {
		n := make([]int, layer+1)
		copy(n, m.FBy)
		m.FBy = n
	}
	if m.FBy[layer] == 0 || prefer {
		m.FBy[layer] = f
	}
}

// fAt is the intermediate size of one layer. A model whose every layer matches F has no FBy.
func (m modelMeta) fAt(layer int) int {
	if layer >= 0 && layer < len(m.FBy) && m.FBy[layer] > 0 {
		return m.FBy[layer]
	}
	return m.F
}

// finishF drops FBy when every recorded layer has the model's F.
func (m *modelMeta) finishF() {
	for _, f := range m.FBy {
		if f > 0 && f != m.F {
			return
		}
	}
	m.FBy = nil
}

func (m *modelMeta) noteAxis(dst *[]int, layer, n int) {
	if layer < 0 || n <= 0 {
		return
	}
	if layer >= len(*dst) {
		grown := make([]int, layer+1)
		copy(grown, *dst)
		*dst = grown
	}
	if (*dst)[layer] == 0 {
		(*dst)[layer] = n
	}
}

// kvLen is one layer's k/v rows. qLen is its q rows and its o columns.
func (m modelMeta) kvLen(layer int) int {
	if layer >= 0 && layer < len(m.KVBy) && m.KVBy[layer] > 0 {
		return m.KVBy[layer]
	}
	return m.KV * m.HD
}

func (m modelMeta) qLen(layer int) int {
	if layer >= 0 && layer < len(m.QBy) && m.QBy[layer] > 0 {
		return m.QBy[layer]
	}
	return m.Heads * m.HD
}

// hdAt is the head dim of one layer, from its k/v rows when those differ from HD.
func (m modelMeta) hdAt(layer int) int {
	n := m.kvLen(layer)
	if m.KV > 0 && n > 0 && n%m.KV == 0 {
		return n / m.KV
	}
	if m.HD > 0 {
		return m.HD
	}
	return 1
}

// finishAxes drops the per-layer tables when every layer matches the model.
func (m *modelMeta) finishAxes() {
	m.finishF()
	same := true
	for _, n := range m.KVBy {
		if n > 0 && n != m.KV*m.HD {
			same = false
			break
		}
	}
	for _, n := range m.QBy {
		if n > 0 && n != m.Heads*m.HD {
			same = false
			break
		}
	}
	if same {
		m.KVBy, m.QBy = nil, nil
	}
}

func (m modelMeta) G() int { return m.Heads / max(1, m.KV) }

// expertIntermediate is F from a gpt-oss expert pack when the layer has no dense gate_proj.
// Float gate_up is [experts, hidden, 2F]. Blocks are [experts, 2F, groups, bytes].
func expertIntermediate(st *ST) (int, error) {
	found := 0
	for _, k := range st.Keys {
		sh := st.H[k].Shape
		var f int
		switch {
		case strings.HasSuffix(k, "mlp.experts.gate_up_proj"):
			if len(sh) == 3 && sh[2]%2 == 0 {
				f = sh[2] / 2
			}
		case strings.HasSuffix(k, "mlp.experts.gate_up_proj_blocks"):
			if len(sh) == 4 && sh[1]%2 == 0 {
				f = sh[1] / 2
			}
		default:
			continue
		}
		if f == 0 {
			return 0, fmt.Errorf("%s: %s shape %v is not an expert gate_up", st.Path, k, sh)
		}
		if found != 0 && found != f {
			return 0, fmt.Errorf("%s: expert intermediate %d and %d", st.Path, found, f)
		}
		found = f
	}
	return found, nil
}

func (m modelMeta) size(k axisKind) int {
	switch k {
	case axD:
		return m.D
	case axF:
		return m.F
	case axKV:
		return m.KV * m.HD
	default:
		return m.Heads * m.HD
	}
}

// metaOf reads a model's sizes from its tensors, with heads from the config set for it when that
// config agrees with the tensors, else from q_norm's length.
func metaOf(st *ST) (modelMeta, error) {
	var m modelMeta
	var qr, kr int
	// Each size from the first layer that has the tensor: a shard of a checkpoint may start mid-stack,
	// even mid-layer.
	first := map[string]int{}
	lm := languageModel(st)
	for _, k := range st.Keys {
		// The text stack is the language model when the file has one. An audio
		// or vision q_proj is a different hidden size, not a second config.
		if asideTensor(k) || (lm && !strings.Contains(k, "model.language_model.")) {
			continue
		}
		sh := st.LogicalShape(k)
		l := layerOf(k)
		if l+1 > m.Layers {
			m.Layers = l + 1
		}
		if l < 0 {
			continue
		}
		r := roleOf(k)
		// Limbs are stored at the packed row count, and bnzmix blobs are not the
		// weight. Only the weight itself says how wide that layer's FFN is.
		if len(sh) == 2 && strings.HasSuffix(k, ".weight") && !strings.HasPrefix(k, "bnzmix.") && !isLimbTensor(k) {
			switch r {
			case "gate", "up":
				m.noteF(l, sh[0], true)
			case "down":
				m.noteF(l, sh[1], false)
			case "k", "v":
				m.noteAxis(&m.KVBy, l, sh[0])
			case "q":
				m.noteAxis(&m.QBy, l, sh[0])
			case "o":
				m.noteAxis(&m.QBy, l, sh[1])
			}
		}
		if len(sh) == 1 && (strings.Contains(k, "q_norm") || strings.Contains(k, "k_norm")) {
			r = "qknorm"
		} else if len(sh) != 2 || (r != "q" && r != "k" && r != "gate") {
			continue
		}
		if f, ok := first[r]; ok && f <= l {
			continue
		}
		first[r] = l
		switch r {
		case "qknorm":
			m.HD = sh[0]
			m.QKNorm = true
		case "q":
			qr, m.D = sh[0], sh[1]
		case "k":
			kr = sh[0]
		case "gate":
			m.F = sh[0]
		}
	}
	if m.F == 0 {
		f, err := expertIntermediate(st)
		if err != nil {
			return m, err
		}
		m.F = f
	}
	if qr == 0 || kr == 0 || m.D == 0 || m.F == 0 {
		return m, fmt.Errorf("%s: no q/k/gate projections to read the topology from", st.Path)
	}
	// The config set for the model: it must describe these tensors. Without one only q_norm's length
	// gives head_dim; the q/k row counts admit several (Llama-3.2-1B's 2048 / 512 rows are 32 / 8 heads
	// of 64, or 16 / 4 of 128).
	if cp := eval.FilesOf(st.Path).Config; cp != "" {
		b, err := os.ReadFile(cp)
		if err != nil {
			return m, err
		}
		var cfg map[string]any
		if err := json.Unmarshal(b, &cfg); err != nil {
			return m, fmt.Errorf("%s: %v", cp, err)
		}
		if _, ok := cfg["hidden_size"]; !ok {
			if tc, ok := cfg["text_config"].(map[string]any); ok {
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
				cfg = out
			}
		}
		num := func(k string) int { v, _ := cfg[k].(float64); return int(v) }
		h, kv, hd := num("num_attention_heads"), num("num_key_value_heads"), num("head_dim")
		if kv == 0 {
			kv = h
		}
		if hd == 0 && h > 0 {
			hd = num("hidden_size") / h
		}
		if !(h > 0 && hd > 0 && h*hd == qr && kv*hd == kr && num("hidden_size") == m.D) {
			return m, fmt.Errorf("%s does not describe %s (hidden %d, q %d / k %d rows): it is another model's config", cp, st.Path, m.D, qr, kr)
		}
		if fi := num("intermediate_size"); fi > 0 && fi != m.F {
			return m, fmt.Errorf("%s: intermediate_size %d, the tensors say %d", cp, fi, m.F)
		}
		m.Heads, m.KV, m.HD = h, kv, hd
		m.Theta, _ = cfg["rope_theta"].(float64)
		m.Config = cp
	} else {
		if m.HD == 0 || qr%m.HD != 0 || kr%m.HD != 0 {
			return m, fmt.Errorf("%s: no config set (hidden %d, q %d / k %d rows), so its heads are unknown: set its config in the run file", st.Path, m.D, qr, kr)
		}
		m.Heads, m.KV = qr/m.HD, kr/m.HD
	}
	if m.Heads%m.KV != 0 {
		return m, fmt.Errorf("%s: %d query heads do not group over %d kv heads", st.Path, m.Heads, m.KV)
	}
	m.finishAxes()
	return m, nil
}

func lcm(a, b int) int { return a / gcd(a, b) * b }

// planTarget is the target topology for --topology base|max|married.
func planTarget(policy string, metas []modelMeta) (modelMeta, error) {
	t := metas[0]
	if policy == "base" {
		return t, nil
	}
	for _, m := range metas[1:] {
		if m.HD != t.HD {
			fmt.Fprintf(os.Stderr, "hodgemix: head_dim %d vs base %d: heads are resampled into the base head_dim\n", m.HD, t.HD)
		}
		switch policy {
		case "max":
			g := max(t.G(), m.G())
			t.D, t.F, t.KV = max(t.D, m.D), max(t.F, m.F), max(t.KV, m.KV)
			t.Heads = t.KV * g
		case "married":
			g := lcm(t.G(), m.G())
			t.D, t.F, t.KV = lcm(t.D, m.D), lcm(t.F, m.F), lcm(t.KV, m.KV)
			t.Heads = t.KV * g
		default:
			return t, fmt.Errorf("--topology must be base, max or married")
		}
		t.Layers = max(t.Layers, m.Layers)
	}
	return t, nil
}

func (m modelMeta) params(V int) float64 {
	per := float64(m.Heads*m.HD*m.D)*2 + float64(m.KV*m.HD*m.D)*2 + float64(m.F*m.D)*3
	return per*float64(m.Layers) + float64(V*m.D)
}

// ---- axis keys ----

type axisInst struct {
	m     int
	kind  axisKind
	layer int // -1 for D
}

type topology struct {
	policy    string
	metas     []modelMeta
	tgt       modelMeta
	keys      map[axisInst][]int // one key per unit (D, F: channel; KV: kv group)
	w         []float64          // model mixing weights (base first)
	cache     map[[2]int]*marriage
	spanCache map[[2]int]*unitSpans
}

func newTopology(policy string, sts []*ST, E []*entry, w []float64) (*topology, error) {
	tp := &topology{policy: policy, w: w, keys: map[axisInst][]int{}, cache: map[[2]int]*marriage{}, spanCache: map[[2]int]*unitSpans{}}
	for _, st := range sts {
		m, err := metaOf(st)
		if err != nil {
			return nil, err
		}
		if m.Theta == 0 {
			m.Theta = SrcRopeTheta
		}
		tp.metas = append(tp.metas, m)
	}
	t, err := planTarget(policy, tp.metas)
	if err != nil {
		return nil, err
	}
	tp.tgt = t
	type acc struct{ sum, n []float64 }
	accs := map[axisInst]*acc{}
	add := func(in axisInst, units int, perChan []int, unitOf func(c int) int) {
		a := accs[in]
		if a == nil {
			a = &acc{make([]float64, units), make([]float64, units)}
			accs[in] = a
		}
		for c, k := range perChan {
			u := unitOf(c)
			if u >= 0 && u < units {
				a.sum[u] += float64(k)
				a.n[u]++
			}
		}
	}
	for _, e := range E {
		rk, ck, ok := roleAxes(e.role)
		if !ok {
			continue
		}
		mm := tp.metas[e.src]
		l := layerOf(e.name)
		for side, kind := range []axisKind{rk, ck} {
			keys := e.kr
			if side == 1 {
				keys = e.kc
			}
			in := axisInst{e.src, kind, l}
			units := mm.size(kind)
			full := units
			hd := mm.hdAt(l)
			unitOf := func(c int) int { return c }
			switch kind {
			case axD:
				in.layer = -1
			case axF:
				full = mm.fAt(l)
				units = full
			case axKV:
				units = mm.KV
				full = mm.kvLen(l)
				unitOf = func(c int) int { return c / hd }
			case axQ: // query head h belongs to kv group h / G
				in.kind = axKV
				units = mm.KV
				full = mm.qLen(l)
				ghd := mm.G() * hd
				unitOf = func(c int) int { return c / ghd }
			}
			if len(keys) != full {
				continue // not the shape the topology reads (fused or unusual tensor)
			}
			add(in, units, keys, unitOf)
		}
	}
	for in, a := range accs {
		k := make([]int, len(a.sum))
		for u := range k {
			if a.n[u] > 0 {
				k[u] = min(keyUniverse, int(math.Round(a.sum[u]/a.n[u])))
			}
		}
		tp.keys[in] = k
	}
	return tp, nil
}

// ---- marriage per target axis node ----

// marriage is one target axis node: every contributing (model, layer) instance's channel
// coordinates on the married axis, and the target channels' coordinates.
type marriage struct {
	coords map[axisInst][]float64 // per channel of that instance's axis (kind D/F/KV/Q)
	target []float64
	radix  []int
}

type contrib struct {
	m, layer int
	w        float64
}

// contributors of target layer t to a per-layer axis: Eq.(11) depth hat, as for tensors.
func (tp *topology) contributors(kind axisKind, t int) []contrib {
	if kind == axD {
		var cs []contrib
		for m := range tp.metas {
			if tp.w[m] > 0 {
				cs = append(cs, contrib{m, -1, tp.w[m]})
			}
		}
		return cs
	}
	ut := 0.5
	if tp.tgt.Layers > 1 {
		ut = float64(t) / float64(tp.tgt.Layers-1)
	}
	var cs []contrib
	for m, mm := range tp.metas {
		for l := 0; l < mm.Layers; l++ {
			us := 0.5
			if mm.Layers > 1 {
				us = float64(l) / float64(mm.Layers-1)
			}
			if w := tp.w[m] * depthHat(us, ut, mm.Layers); w > 0 {
				cs = append(cs, contrib{m, l, w})
			}
		}
	}
	return cs
}

// unitSpans is the married index of the units of a target axis node: D and F channels, or the kv
// groups shared by KV and Q (so a group sits at one place on both axes and GQA pairing holds).
type unitSpans struct {
	ins    []axisInst
	lo, hi [][]float64
	radix  []int
}

func (tp *topology) spansFor(unitKind axisKind, t int) (*unitSpans, error) {
	key := [2]int{100 + int(unitKind), t}
	if unitKind == axD {
		key[1] = -1
	}
	if s, ok := tp.spanCache[key]; ok {
		return s, nil
	}
	var keys, widths [][]int
	var ins []axisInst
	for _, c := range tp.contributors(unitKind, t) {
		in := axisInst{c.m, unitKind, c.layer}
		k, ok := tp.keys[in]
		if !ok {
			continue
		}
		w := make([]int, len(k))
		for i := range w {
			w[i] = 1
			if unitKind == axKV {
				w[i] = tp.metas[c.m].hdAt(c.layer)
			}
		}
		keys, widths, ins = append(keys, k), append(widths, w), append(ins, in)
	}
	if len(keys) == 0 {
		return nil, fmt.Errorf("axis %s@%d: no contributor has keys", unitKind, t)
	}
	lo, hi, r, err := MarriedSpans(keys, widths, keyUniverse)
	if err != nil {
		return nil, err
	}
	s := &unitSpans{ins, lo, hi, r}
	tp.spanCache[key] = s
	return s, nil
}

// marriageFor is the married index of target axis node (kind, t): every contributing instance's
// channel coordinates, subdividing its unit spans (a query channel inside its kv group's span).
func (tp *topology) marriageFor(kind axisKind, t int) (*marriage, error) {
	unitKind := kind
	if kind == axQ {
		unitKind = axKV
	}
	key := [2]int{int(kind), t}
	if kind == axD {
		key[1] = -1
	}
	if mg := tp.cache[key]; mg != nil {
		return mg, nil
	}
	sp, err := tp.spansFor(unitKind, t)
	if err != nil {
		return nil, err
	}
	ins, lo, hi := sp.ins, sp.lo, sp.hi
	mg := &marriage{coords: map[axisInst][]float64{}, radix: sp.radix}
	for j, in := range ins {
		mm := tp.metas[in.m]
		hd := mm.hdAt(in.layer)
		n := mm.size(kind)
		switch kind {
		case axF:
			n = mm.fAt(in.layer)
		case axKV:
			n = mm.kvLen(in.layer)
		case axQ:
			n = mm.qLen(in.layer)
		}
		x := make([]float64, n)
		for c := range x {
			u, off, per := c, 0, 1
			switch kind {
			case axKV:
				u, off, per = c/hd, c%hd, hd
			case axQ:
				per = mm.G() * hd
				u, off = c/per, c%per
			}
			x[c] = lo[j][u] + (float64(off)+.5)/float64(per)*(hi[j][u]-lo[j][u])
		}
		key := in
		key.kind = kind
		mg.coords[key] = x
	}
	if tp.policy == "base" {
		bin := axisInst{0, kind, -1}
		if kind != axD {
			bin.layer = t
		}
		if x, ok := mg.coords[bin]; ok {
			mg.target = x
		}
	}
	if mg.target == nil {
		n := tp.tgt.size(kind)
		mg.target = make([]float64, n)
		for i := range mg.target {
			mg.target[i] = (float64(i) + .5) / float64(n)
		}
	}
	tp.cache[key] = mg
	return mg, nil
}

// instFor is the axis instance a contributor tensor's rows or columns lie on.
func instFor(m int, kind axisKind, layer int) axisInst {
	if kind == axD {
		layer = -1
	}
	return axisInst{m, kind, layer}
}

// ---- resampling along an axis ----

// axisInterp moves values given at coordinates xs (any order) to coordinates ys, linearly in the
// coordinate (the Whitney P0 hat interpolant on the married axis), clamped at the ends.
type axisInterp struct {
	i0, i1 []int
	f      []float64
}

func newAxisInterp(xs, ys []float64) axisInterp {
	o := make([]int, len(xs))
	for i := range o {
		o[i] = i
	}
	sort.Slice(o, func(a, b int) bool { return xs[o[a]] < xs[o[b]] })
	sx := make([]float64, len(o))
	for i, k := range o {
		sx[i] = xs[k]
	}
	a := axisInterp{make([]int, len(ys)), make([]int, len(ys)), make([]float64, len(ys))}
	for i, y := range ys {
		p := sort.SearchFloat64s(sx, y)
		switch {
		case p <= 0:
			a.i0[i], a.i1[i] = o[0], o[0]
		case p >= len(sx):
			a.i0[i], a.i1[i] = o[len(o)-1], o[len(o)-1]
		default:
			lo, hi := p-1, p
			a.i0[i], a.i1[i] = o[lo], o[hi]
			if d := sx[hi] - sx[lo]; d > 0 {
				a.f[i] = (y - sx[lo]) / d
			}
		}
	}
	return a
}

func (a axisInterp) at(get func(i int) float64, i int) float64 {
	v0 := get(a.i0[i])
	if a.f[i] == 0 {
		return v0
	}
	return v0 + a.f[i]*(get(a.i1[i])-v0)
}

func (a axisInterp) vec(v []float32) []float32 {
	out := make([]float32, len(a.f))
	for i := range out {
		out[i] = float32(a.at(func(k int) float64 { return float64(v[k]) }, i))
	}
	return out
}

// rows resamples the rows of an R x C matrix: (len(ys)) x C.
func (a axisInterp) rows(M []float64, C int) []float64 {
	out := make([]float64, len(a.f)*C)
	parFor(len(a.f), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			r0, r1, f := a.i0[i]*C, a.i1[i]*C, a.f[i]
			for c := 0; c < C; c++ {
				out[i*C+c] = M[r0+c] + f*(M[r1+c]-M[r0+c])
			}
		}
	})
	return out
}

// cols resamples the columns of an R x C matrix: R x (len(ys)).
func (a axisInterp) cols(M []float64, R, C int) []float64 {
	n := len(a.f)
	out := make([]float64, R*n)
	parFor(R, func(lo, hi int) {
		for r := lo; r < hi; r++ {
			row := M[r*C : (r+1)*C]
			for i := 0; i < n; i++ {
				out[r*n+i] = row[a.i0[i]] + a.f[i]*(row[a.i1[i]]-row[a.i0[i]])
			}
		}
	})
	return out
}

// ---- target tensor list ----

// targets is the target tensor list in the base model's own layout, resized to tgt and extended to
// tgt.Layers by repeating the base's layer pattern.
func (tp *topology) targets(base *ST) ([]target, int) {
	var T []target
	var pattern []string
	seen := map[string]bool{}
	for _, k := range base.Keys {
		if (strings.HasSuffix(k, "weight_scale") && base.Has(strings.TrimSuffix(k, "_scale"))) || isLimbTensor(k) { // limbs belong to their weight
			continue
		}
		l := layerOf(k)
		if l < 0 {
			T = append(T, tp.resized(base, k, k, -1))
			continue
		}
		if l == 0 {
			rest := strings.SplitN(k, ".0.", 2)
			if len(rest) == 2 && !seen[rest[1]] {
				seen[rest[1]] = true
				pattern = append(pattern, k)
			}
		}
	}
	for t := 0; t < tp.tgt.Layers; t++ {
		for _, k := range pattern {
			name := strings.Replace(k, ".0.", "."+itoa(t)+".", 1)
			T = append(T, tp.resized(base, k, name, t))
		}
	}
	return T, tp.tgt.Layers
}

func itoa(n int) string { return fmt.Sprint(n) }

func (tp *topology) resized(base *ST, src, name string, layer int) target {
	sh := append([]int(nil), base.LogicalShape(src)...)
	role := "norm"
	if len(sh) == 2 {
		role = roleOf(src)
		if rk, ck, ok := roleAxes(role); ok {
			sh = []int{tp.tgt.size(rk), tp.tgt.size(ck)}
		} else if role == "embed" || role == "lm_head" {
			sh[1] = tp.tgt.D
		}
	} else if len(sh) == 1 {
		if k, ok := vecAxis(src); ok {
			sh[0] = tp.tgt.size(k)
		}
	}
	return target{name, role, layer, sh}
}

// configDefaults is what the config class means by a key the base's config leaves out (BitNetConfig
// for bitnet, LlamaConfig otherwise). The output keeps the source model_type. These fill holes only.
func configDefaults(modelType string) map[string]any {
	if ternaryFamily(modelType) {
		return map[string]any{"hidden_act": "relu2", "rope_theta": 500000.0, "rms_norm_eps": 1e-5}
	}
	return map[string]any{"hidden_act": "silu", "rope_theta": 10000.0, "rms_norm_eps": 1e-6}
}

// configFor is the config.json for the output: cfgPath's (else the base's config) with the sizes the
// output's tensors have. The source model_type, architectures, and activation stay. A missing
// hidden_act is copied from hidden_activation (VaultGemma, Gemma). Keys the class leaves out take
// that class's defaults. tie_word_embeddings is set when the key is absent, because the output head
// is the embedding. changed lists what moved.
func configFor(t modelMeta, cfgPath, basePath string, vocab int) (b []byte, changed []string, err error) {
	cfg := map[string]any{}
	src := cfgPath
	b, err = os.ReadFile(cfgPath)
	if err != nil {
		src = eval.FilesOf(basePath).Config
		if src == "" {
			return nil, nil, fmt.Errorf("no config set for the base")
		}
		b, err = os.ReadFile(src)
	}
	if err != nil {
		return nil, nil, err
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, nil, fmt.Errorf("%s: %v", src, err)
	}
	mt, _ := cfg["model_type"].(string)
	// Gemma, including VaultGemma, names the activation hidden_activation. Champ reads
	// hidden_act. Copy before the class default, or a VaultGemma run is stamped silu.
	if _, ok := cfg["hidden_act"]; !ok {
		if act, ok := cfg["hidden_activation"].(string); ok && act != "" {
			cfg["hidden_act"] = act
			changed = append(changed, fmt.Sprintf("hidden_act = %s (from hidden_activation)", act))
		}
	}
	for k, v := range configDefaults(mt) {
		if _, ok := cfg[k]; !ok {
			cfg[k] = v
			changed = append(changed, fmt.Sprintf("%s = %v (the %s default)", k, v, mt))
		}
	}
	set := func(k string, v float64) {
		if old, ok := cfg[k].(float64); !ok || old != v {
			if ok {
				changed = append(changed, k+" "+fmtG(old)+" -> "+fmtG(v))
			} else {
				changed = append(changed, k+" = "+fmtG(v))
			}
			cfg[k] = v
		}
	}
	set("hidden_size", float64(t.D))
	set("intermediate_size", float64(t.F))
	set("num_attention_heads", float64(t.Heads))
	set("num_key_value_heads", float64(t.KV))
	if _, ok := cfg["head_dim"]; ok || t.HD*t.Heads != t.D {
		set("head_dim", float64(t.HD))
	}
	set("num_hidden_layers", float64(t.Layers))
	if vocab > 0 {
		set("vocab_size", float64(vocab))
	}
	if t.Theta > 0 {
		set("rope_theta", t.Theta)
	}
	if _, ok := cfg["tie_word_embeddings"]; !ok {
		cfg["tie_word_embeddings"] = true // the output's head is the embedding (champLayout, preflight)
		changed = append(changed, "tie_word_embeddings = true")
	}
	b, err = json.MarshalIndent(cfg, "", "  ")
	return b, changed, err
}

// tokenizerVocab is the token count of a tokenizer.json (base vocab + added tokens), 0 if unreadable.
func tokenizerVocab(path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	var tk struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
		Added []struct {
			ID int `json:"id"`
		} `json:"added_tokens"`
	}
	if json.Unmarshal(b, &tk) != nil {
		return 0
	}
	n := 0
	for _, id := range tk.Model.Vocab {
		n = max(n, id+1)
	}
	for _, a := range tk.Added {
		n = max(n, a.ID+1)
	}
	return n
}

// baseLayerFor is the base layer whose per-layer vectors a target layer resamples (nearest depth).
func (tp *topology) baseLayerFor(t int) int {
	bl := tp.metas[0].Layers
	if bl <= 1 || tp.tgt.Layers <= 1 {
		return 0
	}
	return int(math.Round(float64(t) / float64(tp.tgt.Layers-1) * float64(bl-1)))
}

// baseScaleFor is the absmean of the base's tensor with the target's role at the base layer
// nearest the target layer, for a target tensor the base does not have at this shape.
func baseScaleFor(E []*entry, t target, tp *topology) float64 {
	if tp == nil || t.layer < 0 {
		return 0
	}
	bl := tp.baseLayerFor(t.layer)
	for _, e := range E {
		if e.src == 0 && e.role == t.role && layerOf(e.name) == bl {
			return e.scale
		}
	}
	return 0
}

// copyResampled writes a tensor that is not mixed (embeddings, lm_head, norm gains, biases) in a new
// topology: the base's tensor (at the nearest base layer) carried along its married axis onto the
// target's channels, so it reads the residual stream, FFN or heads in the same order as every mixed
// tensor. A tensor on no resized axis (q_norm, k_norm) is copied.
func (tp *topology) copyResampled(t target, base *ST, W, MW *STW, FD string, kept *[][2]string) {
	src := t.name
	if t.layer >= 0 {
		src = strings.Replace(t.name, "."+itoa(t.layer)+".", "."+itoa(tp.baseLayerFor(t.layer))+".", 1)
	}
	if !base.Has(src) {
		a := make([]float32, numel(t.shape))
		if t.role == "norm" {
			for i := range a {
				a[i] = 1
			}
		}
		W.AddFloat(t.name, t.shape, a, FD)
		*kept = append(*kept, [2]string{t.name, "init (base has no " + src + ")"})
		return
	}
	dt := base.H[src].Dtype
	if dt != "F32" && dt != "F16" && dt != "BF16" {
		b, err := base.Raw(src)
		if err != nil {
			die("%s: %v", src, err)
		}
		W.Add(t.name, dt, base.H[src].Shape, b)
		*kept = append(*kept, [2]string{t.name, "copied verbatim from " + src})
		return
	}
	a, err := base.F32(src)
	if err != nil {
		die("%s: %v", src, err)
	}
	sh := base.H[src].Shape
	out := a
	how := "copied from " + src
	bl := tp.baseLayerFor(t.layer)
	switch {
	case len(sh) == 1:
		if k, ok := vecAxis(src); ok {
			mg, err := tp.marriageFor(k, t.layer)
			if err != nil {
				die("%s: %v", t.name, err)
			}
			if x, ok := mg.coords[instFor(0, k, bl)]; ok && len(x) == sh[0] {
				out = newAxisInterp(x, mg.target).vec(a)
				how = "resampled from " + src + " along married " + k.String()
			}
		}
	case len(sh) == 2 && (t.role == "embed" || t.role == "lm_head"):
		mg, err := tp.marriageFor(axD, -1)
		if err != nil {
			die("%s: %v", t.name, err)
		}
		if x, ok := mg.coords[instFor(0, axD, -1)]; ok && len(x) == sh[1] {
			M := make([]float64, len(a))
			for i, v := range a {
				M[i] = float64(v)
			}
			M = newAxisInterp(x, mg.target).cols(M, sh[0], sh[1])
			out = f32s(M)
			how = "resampled from " + src + " along married D"
		}
	}
	if numel(t.shape) != len(out) {
		die("%s: resampled to %d values, target %v", t.name, len(out), t.shape)
	}
	W.AddFloat(t.name, t.shape, out, FD)
	if MW != nil {
		MW.AddFloat(t.name, t.shape, out, FD)
	}
	*kept = append(*kept, [2]string{t.name, how})
}

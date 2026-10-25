package eval

// The model eval runs: exactly the safetensors files set for it (files.go), its architecture read from
// its tensors and the config set for it, checked against them. Linear
// weights are either packed ternary (U8, BitLinear: int8 activations, LUT fabric, dynamic MRH limbs)
// or float (BF16 / F16 / F32, a plain matmul on the float activations), per matrix; the BitNet
// sub-norms and the Qwen3 per-head q/k norms are used where the tensors exist.

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"sync"
)

// Shape is a model's architecture.
type Shape struct {
	D, F, L, H, KVH, HD int
	Theta, Eps          float64
	Act                 string // hidden_act
	Config              string // the config*.json it was read from
}

type tinfo struct {
	Dtype string
	Shape []int
	Off   [2]int64 `json:"data_offsets"`
	f     int
}

// shards is a model's tensors over one or more safetensors files, read tensor by tensor.
type shards struct {
	fs   []*os.File
	base []int64
	hdr  map[string]tinfo
}

// openShards opens exactly the files set for the model (ShardsOf): nothing is looked up beside it.
func openShards(path string) (*shards, error) {
	fl := ShardsOf(path)
	s := &shards{hdr: map[string]tinfo{}}
	for i, p := range fl {
		f, err := os.Open(p)
		if err != nil {
			s.close()
			return nil, err
		}
		s.fs = append(s.fs, f)
		var nb [8]byte
		if _, err := f.ReadAt(nb[:], 0); err != nil {
			s.close()
			return nil, fmt.Errorf("%s: %v", p, err)
		}
		n := int64(binary.LittleEndian.Uint64(nb[:]))
		hb := make([]byte, n)
		if _, err := f.ReadAt(hb, 8); err != nil {
			s.close()
			return nil, fmt.Errorf("%s: %v", p, err)
		}
		var h map[string]json.RawMessage
		if err := json.Unmarshal(hb, &h); err != nil {
			s.close()
			return nil, fmt.Errorf("%s: %v", p, err)
		}
		s.base = append(s.base, 8+n)
		for k, v := range h {
			if k == "__metadata__" {
				continue
			}
			var t tinfo
			json.Unmarshal(v, &t)
			t.f = i
			s.hdr[k] = t
		}
	}
	return s, nil
}

func (s *shards) close() {
	for _, f := range s.fs {
		f.Close()
	}
}

func (s *shards) raw(n string) ([]byte, error) {
	t, ok := s.hdr[n]
	if !ok {
		return nil, errors.New("no tensor " + n)
	}
	b := make([]byte, t.Off[1]-t.Off[0])
	_, err := s.fs[t.f].ReadAt(b, s.base[t.f]+t.Off[0])
	return b, err
}

func isFloat(dt string) bool { return dt == "BF16" || dt == "F16" || dt == "F32" }

func elemSize(dt string) int {
	if dt == "F32" {
		return 4
	}
	return 2
}

func bf(b []byte, i int) float32 {
	return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16)
}

func f16(h uint16) float32 {
	sign := uint32(h>>15) << 31
	e := int(h>>10) & 31
	m := uint32(h & 1023)
	switch {
	case e == 0:
		return math.Float32frombits(sign) + float32(m)*float32(math.Ldexp(1, -24))*sgn(sign)
	case e == 31:
		return math.Float32frombits(sign | 0x7f800000 | m<<13)
	}
	return math.Float32frombits(sign | uint32(e+112)<<23 | m<<13)
}

func sgn(sign uint32) float32 {
	if sign != 0 {
		return -1
	}
	return 1
}

// at is element i of a float tensor's bytes.
func at(b []byte, dt string, i int) float32 {
	switch dt {
	case "BF16":
		return bf(b, i)
	case "F16":
		return f16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
}

// vec is float tensor n as float32; want > 0 checks its length.
func (s *shards) vec(n string, want int) ([]float32, error) {
	t, ok := s.hdr[n]
	if !ok {
		return nil, errors.New("no tensor " + n)
	}
	if !isFloat(t.Dtype) {
		return nil, fmt.Errorf("%s is %s, want a float tensor", n, t.Dtype)
	}
	b, err := s.raw(n)
	if err != nil {
		return nil, err
	}
	c := len(b) / elemSize(t.Dtype)
	if want > 0 && c != want {
		return nil, fmt.Errorf("%s has %d values, want %d", n, c, want)
	}
	o := make([]float32, c)
	for i := range o {
		o[i] = at(b, t.Dtype, i)
	}
	return o, nil
}

// rows is the output size of linear weight n (a packed U8 row holds four).
func (s *shards) rows(n string) int {
	t, ok := s.hdr[n]
	if !ok || len(t.Shape) != 2 {
		return 0
	}
	if t.Dtype == "U8" {
		return 4 * t.Shape[0]
	}
	return t.Shape[0]
}

// lin loads linear n (rows x k): packed ternary with its limbs, or float; with its bias if it has one.
func (s *shards) lin(n string, rows, k int) (*lin, error) {
	t, ok := s.hdr[n+".weight"]
	if !ok {
		return nil, fmt.Errorf("no tensor %s.weight", n)
	}
	l := &lin{n: rows, k: k}
	switch {
	case t.Dtype == "U8":
		if len(t.Shape) != 2 || t.Shape[0]*4 != rows || t.Shape[1] != k {
			return nil, fmt.Errorf("%s.weight is U8 %v, want U8 [%d %d]", n, t.Shape, rows/4, k)
		}
		b, err := s.raw(n + ".weight")
		if err != nil {
			return nil, err
		}
		sc, err := s.vec(n+".weight_scale", 1)
		if err != nil {
			return nil, err
		}
		np := t.Shape[0]
		l.w, l.s = make([]int8, rows*k), sc[0]
		for r := 0; r < np; r++ {
			for j := 0; j < k; j++ {
				c := b[r*k+j]
				for i := 0; i < 4; i++ {
					l.w[(i*np+r)*k+j] = int8((c>>(2*i))&3) - 1
				}
			}
		}
		l.pack()
		// Dynamic MRH limbs beyond the ternary one (hodgemix limbs.go): X.weight_limb<i> packed like
		// limb 0 with raw digits 0..r-1, X.weight_limb<i>_scale, X.weight_limb<i>_radix.
		for li := 1; ; li++ {
			ln := fmt.Sprintf("%s.weight_limb%d", n, li)
			lt, ok := s.hdr[ln]
			if !ok || lt.Dtype != "U8" || len(lt.Shape) != 2 || lt.Shape[0] != np || lt.Shape[1] != k {
				break
			}
			lb, err1 := s.raw(ln)
			ls, err2 := s.vec(ln+"_scale", 1)
			rb, err3 := s.raw(ln + "_radix")
			if err1 != nil || err2 != nil || err3 != nil || len(rb) != 4 {
				return nil, fmt.Errorf("%s: incomplete limb", ln)
			}
			radix := int(binary.LittleEndian.Uint32(rb))
			d := make([]uint8, rows*k)
			for r := 0; r < np; r++ {
				for j := 0; j < k; j++ {
					c := lb[r*k+j]
					for i := 0; i < 4; i++ {
						d[(i*np+r)*k+j] = (c >> (2 * i)) & 3
					}
				}
			}
			l.limbs = append(l.limbs, newLimb(d, rows, k, radix, ls[0]))
		}
	case isFloat(t.Dtype):
		if len(t.Shape) != 2 || t.Shape[0] != rows || t.Shape[1] != k {
			return nil, fmt.Errorf("%s.weight is %s %v, want [%d %d]", n, t.Dtype, t.Shape, rows, k)
		}
		var err error
		if l.f, err = s.vec(n+".weight", rows*k); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("%s.weight is %s", n, t.Dtype)
	}
	if _, ok := s.hdr[n+".bias"]; ok {
		var err error
		if l.b, err = s.vec(n+".bias", rows); err != nil {
			return nil, err
		}
	}
	return l, nil
}

var layerRE = regexp.MustCompile(`^model\.layers\.(\d+)\.`)

// config is the model's config file as set for it, checked against its tensors.
func config(cp string, sh Shape, qr, kr int) (map[string]any, error) {
	if cp == "" {
		return nil, errors.New("no config set for this model (set its config in the run file)")
	}
	b, err := os.ReadFile(cp)
	if err != nil {
		return nil, err
	}
	var cfg map[string]any
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("%s: %v", cp, err)
	}
	cfg = flattenText(cfg)
	num := func(k string) int { v, _ := cfg[k].(float64); return int(v) }
	h, kv, hd := num("num_attention_heads"), num("num_key_value_heads"), num("head_dim")
	if kv == 0 {
		kv = h
	}
	if hd == 0 && h > 0 {
		hd = num("hidden_size") / h
	}
	if !(h > 0 && hd > 0 && h*hd == qr && kv*hd == kr && num("hidden_size") == sh.D &&
		(num("num_hidden_layers") == 0 || num("num_hidden_layers") == sh.L) &&
		(num("intermediate_size") == 0 || num("intermediate_size") == sh.F)) {
		return nil, fmt.Errorf("%s does not describe these tensors (hidden %d, %d layers, ffn %d, q %d / k %d rows): it is another model's config",
			cp, sh.D, sh.L, sh.F, qr, kr)
	}
	return cfg, nil
}

func load(path string) (*model, error) {
	s, err := openShards(path)
	if err != nil {
		return nil, err
	}
	defer s.close()
	et, ok := s.hdr["model.embed_tokens.weight"]
	if !ok || len(et.Shape) != 2 || !isFloat(et.Dtype) {
		return nil, fmt.Errorf("model.embed_tokens.weight is %s %v, want a float [vocab, hidden]", et.Dtype, et.Shape)
	}
	m := &model{}
	m.D, m.nEmb, m.embDT = et.Shape[1], et.Shape[0], et.Dtype
	if m.emb, err = s.raw("model.embed_tokens.weight"); err != nil {
		return nil, err
	}
	m.head, m.headDT, m.nv = m.emb, m.embDT, m.nEmb
	if ht, ok := s.hdr["lm_head.weight"]; ok {
		if len(ht.Shape) != 2 || ht.Shape[1] != m.D || !isFloat(ht.Dtype) {
			return nil, fmt.Errorf("lm_head.weight is %s %v, want a float [vocab, %d]", ht.Dtype, ht.Shape, m.D)
		}
		if m.head, err = s.raw("lm_head.weight"); err != nil {
			return nil, err
		}
		m.headDT, m.nv = ht.Dtype, ht.Shape[0]
	}
	for k := range s.hdr {
		if g := layerRE.FindStringSubmatch(k); g != nil {
			i, _ := strconv.Atoi(g[1])
			m.L = max(m.L, i+1)
		}
	}
	p0 := "model.layers.0."
	qr, kr := s.rows(p0+"self_attn.q_proj.weight"), s.rows(p0+"self_attn.k_proj.weight")
	var ferr error
	m.F, ferr = s.ffnWidth(m.D)
	if ferr != nil {
		return nil, ferr
	}
	if m.L == 0 || qr == 0 || kr == 0 || m.F == 0 {
		return nil, errors.New("no layer-0 q_proj / k_proj / gate_proj (or gpt-oss experts)")
	}
	cp := FilesOf(path).Config
	cfg, err := config(cp, m.Shape, qr, kr)
	if err != nil {
		return nil, err
	}
	m.Config = cp
	num := func(k string) int { v, _ := cfg[k].(float64); return int(v) }
	m.H, m.KVH, m.HD = num("num_attention_heads"), num("num_key_value_heads"), num("head_dim")
	if m.KVH == 0 {
		m.KVH = m.H
	}
	if m.HD == 0 {
		m.HD = m.D / m.H
	}
	if m.H%m.KVH != 0 || m.HD%2 != 0 {
		return nil, fmt.Errorf("%s: %d heads over %d kv heads, head_dim %d", cp, m.H, m.KVH, m.HD)
	}
	// Unset keys take the config class's defaults (BitNetConfig for bitnet, LlamaConfig otherwise).
	m.Act, m.Theta, m.Eps = "silu", 10000, 1e-6
	if cfg["model_type"] == "bitnet" || cfg["model_type"] == "hodge" {
		m.Act, m.Theta, m.Eps = "relu2", 500000, 1e-5
	}
	if v, ok := cfg["hidden_act"].(string); ok {
		m.Act = v
	}
	if v, ok := cfg["hidden_activation"].(string); ok && v != "" {
		m.Act = v // gemma3 names it hidden_activation
	}
	if v, ok := cfg["rope_theta"].(float64); ok {
		m.Theta = v
	}
	if v, ok := cfg["rms_norm_eps"].(float64); ok {
		m.Eps = v
	}
	if m.arch, err = Classify(cfg, m.L); err != nil {
		return nil, fmt.Errorf("%s: %v", cp, err)
	}
	m.attnScale = 1
	if actFn(m.Act) == nil {
		return nil, fmt.Errorf("%s: hidden_act %q is not modelled", cp, m.Act)
	}
	if m.invFreq, m.attnScale, err = ropeFreqs(cfg, m.Theta, m.HD); err != nil {
		return nil, fmt.Errorf("%s: %v", cp, err)
	}
	if m.arch.LocalTheta > 0 {
		if m.localFreq, _, err = ropeFreqs(map[string]any{}, m.arch.LocalTheta, m.HD); err != nil {
			return nil, err
		}
	}
	if m.fnorm, err = s.vec("model.norm.weight", m.D); err != nil {
		return nil, err
	}
	D, F, A, KV := m.D, m.F, m.H*m.HD, m.KVH*m.HD
	m.ls = make([]layer, m.L)
	var mu sync.Mutex
	var first error
	var wg sync.WaitGroup
	sem := make(chan struct{}, 4)
	for i := 0; i < m.L; i++ {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int) {
			defer func() { <-sem; wg.Done() }()
			err := m.loadLayer(s, i, D, F, A, KV)
			if err != nil {
				mu.Lock()
				if first == nil {
					first = err
				}
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	if first != nil {
		return nil, first
	}
	if err := m.loadTowers(s); err != nil {
		return nil, err
	}
	return m, nil
}

// ffnWidth is the intermediate size: a dense gate_proj, or a gpt-oss expert pack.
func (s *shards) ffnWidth(hidden int) (int, error) {
	if r := s.rows("model.layers.0.mlp.gate_proj.weight"); r > 0 {
		return r, nil
	}
	if t, ok := s.hdr["model.layers.0.mlp.experts.gate_up_proj"]; ok {
		if len(t.Shape) == 3 && t.Shape[2]%2 == 0 && (hidden == 0 || t.Shape[1] == hidden) {
			return t.Shape[2] / 2, nil
		}
		return 0, fmt.Errorf("experts.gate_up_proj is %s %v, want [experts, hidden, 2*intermediate]", t.Dtype, t.Shape)
	}
	if t, ok := s.hdr["model.layers.0.mlp.experts.gate_up_proj_blocks"]; ok {
		if len(t.Shape) == 4 && t.Shape[3] > 0 && t.Shape[2] > 0 && t.Shape[1]%2 == 0 {
			return t.Shape[1] / 2, nil
		}
		return 0, fmt.Errorf("experts.gate_up_proj_blocks is %v, want [experts, 2*intermediate, groups, bytes]", t.Shape)
	}
	return 0, nil
}

func (m *model) loadLayer(s *shards, i, D, F, A, KV int) error {
	p := fmt.Sprintf("model.layers.%d.", i)
	Lr := &m.ls[i]
	var err error
	// The FFN pre-norm. Llama, Qwen, and Gemma 1 store it as post_attention_layernorm.
	// VaultGemma stores that same pre-norm as pre_feedforward_layernorm and has no
	// post-attention norm. On a sandwich layer, post_attention is the attention output
	// norm and pre_feedforward stays the FFN pre-norm.
	ffnPre := "post_attention_layernorm.weight"
	if !m.arch.Sandwich {
		if _, hasPost := s.hdr[p+"post_attention_layernorm.weight"]; !hasPost {
			if _, hasPre := s.hdr[p+"pre_feedforward_layernorm.weight"]; hasPre {
				ffnPre = "pre_feedforward_layernorm.weight"
			}
		}
	}
	for _, x := range []struct {
		dst      *[]float32
		n        string
		k        int
		optional bool
	}{{&Lr.an, "input_layernorm.weight", D, false}, {&Lr.fn, ffnPre, D, false},
		{&Lr.sn, "self_attn.attn_sub_norm.weight", A, true}, {&Lr.fsn, "mlp.ffn_sub_norm.weight", F, true},
		{&Lr.qn, "self_attn.q_norm.weight", m.HD, true}, {&Lr.kn, "self_attn.k_norm.weight", m.HD, true},
		{&Lr.preFF, "pre_feedforward_layernorm.weight", D, true}, {&Lr.postFF, "post_feedforward_layernorm.weight", D, true}} {
		// pre_feedforward is already loaded into fn. Skip only the duplicate slot.
		if x.dst == &Lr.preFF && ffnPre == x.n {
			continue
		}
		if _, ok := s.hdr[p+x.n]; !ok && x.optional {
			continue
		}
		if *x.dst, err = s.vec(p+x.n, x.k); err != nil {
			return err
		}
	}
	// An all-ones sub-norm is the one Champ's layout inserts. Leave the sublayer as the
	// checkpoint computed it; the BitLinear still quantizes its own input.
	if unitGain(Lr.sn) {
		Lr.sn = nil
	}
	if unitGain(Lr.fsn) {
		Lr.fsn = nil
	}
	if m.arch.Sandwich && (Lr.preFF == nil || Lr.postFF == nil) {
		return fmt.Errorf("layer %d: gemma2/3 need pre_feedforward_layernorm and post_feedforward_layernorm", i)
	}
	if m.arch.Sinks {
		if Lr.sink, err = s.vec(p+"self_attn.sinks", m.H); err != nil {
			return err
		}
	}
	for _, x := range []struct {
		dst     **lin
		n       string
		rows, k int
	}{{&Lr.q, "self_attn.q_proj", A, D}, {&Lr.k, "self_attn.k_proj", KV, D}, {&Lr.v, "self_attn.v_proj", KV, D},
		{&Lr.o, "self_attn.o_proj", D, A}} {
		if *x.dst, err = s.lin(p+x.n, x.rows, x.k); err != nil {
			return err
		}
	}
	if _, moe := s.hdr[p+"mlp.experts.gate_up_proj"]; moe {
		return m.loadExperts(s, i, D, F)
	}
	if _, moe := s.hdr[p+"mlp.experts.gate_up_proj_blocks"]; moe {
		return m.loadExperts(s, i, D, F)
	}
	if m.arch.MoE {
		return fmt.Errorf("layer %d: gpt-oss has no experts", i)
	}
	for _, x := range []struct {
		dst     **lin
		n       string
		rows, k int
	}{{&Lr.g, "mlp.gate_proj", F, D}, {&Lr.u, "mlp.up_proj", F, D}, {&Lr.d, "mlp.down_proj", D, F}} {
		if *x.dst, err = s.lin(p+x.n, x.rows, x.k); err != nil {
			return err
		}
	}
	if Lr.fsn != nil && m.arch.MoE {
		return fmt.Errorf("layer %d: ffn_sub_norm on a mixture is not modelled", i)
	}
	return nil
}

// loadExperts reads one layer's gpt-oss mixture: float gate_up/down, or MXFP4 blocks and scales.
func (m *model) loadExperts(s *shards, i, D, F int) error {
	if !m.arch.MoE {
		return fmt.Errorf("layer %d: expert tensors on a %s checkpoint", i, m.arch.Family)
	}
	p := fmt.Sprintf("model.layers.%d.mlp.", i)
	Lr := &m.ls[i]
	if Lr.fsn != nil {
		return fmt.Errorf("layer %d: ffn_sub_norm on a mixture is not modelled", i)
	}
	if _, ok := s.hdr[p+"experts.gate_up_proj_blocks"]; ok {
		return m.loadMXExperts(s, i, D, F)
	}
	var gu, dn []float32
	var guS, dnS []int
	var err error
	if gu, guS, err = s.dense3(p + "experts.gate_up_proj"); err != nil {
		return err
	}
	if dn, dnS, err = s.dense3(p + "experts.down_proj"); err != nil {
		return err
	}
	if len(guS) != 3 || guS[0] != m.arch.Experts || guS[1] != D || guS[2] != 2*F {
		return fmt.Errorf("layer %d: gate_up is %v, want [%d %d %d]", i, guS, m.arch.Experts, D, 2*F)
	}
	if len(dnS) != 3 || dnS[0] != m.arch.Experts || dnS[1] != F || dnS[2] != D {
		return fmt.Errorf("layer %d: down is %v, want [%d %d %d]", i, dnS, m.arch.Experts, F, D)
	}
	gates, ups, err := SplitGateUp(gu, guS)
	if err != nil {
		return err
	}
	downs, err := SplitDown(dn, dnS)
	if err != nil {
		return err
	}
	gb, ub, err := s.expertBias(p+"experts.gate_up_proj_bias", m.arch.Experts, 2*F)
	if err != nil {
		return err
	}
	db, err := s.vec(p+"experts.down_proj_bias", m.arch.Experts*D)
	if err != nil {
		return err
	}
	Lr.experts = make([]expert, m.arch.Experts)
	for e := 0; e < m.arch.Experts; e++ {
		Lr.experts[e] = expert{
			gate: &lin{n: F, k: D, f: gates[e], b: gb[e]},
			up:   &lin{n: F, k: D, f: ups[e], b: ub[e]},
			down: &lin{n: D, k: F, f: downs[e], b: db[e*D : (e+1)*D]},
		}
	}
	rw, err := s.vec(p+"router.weight", m.arch.Experts*D)
	if err != nil {
		return err
	}
	rb, err := s.vec(p+"router.bias", m.arch.Experts)
	if err != nil {
		return err
	}
	Lr.router = &lin{n: m.arch.Experts, k: D, f: rw, b: rb}
	return nil
}

// loadMXExperts keeps each expert as its nibble table. Gate and up are even and odd rows of one
// gate_up pack; the bytes are not expanded into a float matrix.
func (m *model) loadMXExperts(s *shards, i, D, F int) error {
	p := fmt.Sprintf("model.layers.%d.mlp.", i)
	Lr := &m.ls[i]
	E := m.arch.Experts
	gu, err := s.mxRaw(p + "experts.gate_up_proj")
	if err != nil {
		return err
	}
	dn, err := s.mxRaw(p + "experts.down_proj")
	if err != nil {
		return err
	}
	gsh, dsh := gu.bShape, dn.bShape
	if len(gsh) != 4 || gsh[0] != E || gsh[1] != 2*F || gsh[2]*gsh[3]*2 != D {
		return fmt.Errorf("layer %d: gate_up blocks %v, want [%d %d groups bytes] covering hidden %d", i, gsh, E, 2*F, D)
	}
	if len(dsh) != 4 || dsh[0] != E || dsh[1] != D || dsh[2]*dsh[3]*2 != F {
		return fmt.Errorf("layer %d: down blocks %v, want [%d %d groups bytes] covering intermediate %d", i, dsh, E, D, F)
	}
	gb, ub, err := s.expertBias(p+"experts.gate_up_proj_bias", E, 2*F)
	if err != nil {
		return err
	}
	db, err := s.vec(p+"experts.down_proj_bias", E*D)
	if err != nil {
		return err
	}
	gRows, gG, gB := gsh[1], gsh[2], gsh[3]
	dRows, dG, dB := dsh[1], dsh[2], dsh[3]
	gPer, gsPer := gRows*gG*gB, gRows*gG
	dPer, dsPer := dRows*dG*dB, dRows*dG
	Lr.experts = make([]expert, E)
	for e := 0; e < E; e++ {
		eb, es := gu.blocks[e*gPer:(e+1)*gPer], gu.scales[e*gsPer:(e+1)*gsPer]
		dbb, dss := dn.blocks[e*dPer:(e+1)*dPer], dn.scales[e*dsPer:(e+1)*dsPer]
		Lr.experts[e] = expert{
			gate: &lin{n: F, k: D, b: gb[e], mx: &mx4{rows: F, groups: gG, B: gB, stride: 2, phase: 0, blocks: eb, scales: es}},
			up:   &lin{n: F, k: D, b: ub[e], mx: &mx4{rows: F, groups: gG, B: gB, stride: 2, phase: 1, blocks: eb, scales: es}},
			down: &lin{n: D, k: F, b: db[e*D : (e+1)*D], mx: &mx4{rows: D, groups: dG, B: dB, stride: 1, phase: 0, blocks: dbb, scales: dss}},
		}
	}
	rw, err := s.vec(p+"router.weight", E*D)
	if err != nil {
		return err
	}
	rb, err := s.vec(p+"router.bias", E)
	if err != nil {
		return err
	}
	Lr.router = &lin{n: E, k: D, f: rw, b: rb}
	return nil
}

type mxPack struct {
	blocks, scales []byte
	bShape         []int
}

func (s *shards) mxRaw(proj string) (mxPack, error) {
	var z mxPack
	bt, okB := s.hdr[proj+"_blocks"]
	st, okS := s.hdr[proj+"_scales"]
	if !okB || !okS || bt.Dtype != "U8" || st.Dtype != "U8" {
		return z, fmt.Errorf("%s mxfp4 blocks/scales missing", proj)
	}
	if len(st.Shape) != len(bt.Shape)-1 {
		return z, fmt.Errorf("%s scales %v, blocks %v", proj, st.Shape, bt.Shape)
	}
	bb, err := s.raw(proj + "_blocks")
	if err != nil {
		return z, err
	}
	sb, err := s.raw(proj + "_scales")
	if err != nil {
		return z, err
	}
	return mxPack{blocks: bb, scales: sb, bShape: bt.Shape}, nil
}

func (s *shards) expertBias(name string, experts, two int) (gate, up [][]float32, err error) {
	b, err := s.vec(name, experts*two)
	if err != nil {
		return nil, nil, err
	}
	return SplitBias(b, experts, two)
}

func (s *shards) dense3(name string) ([]float32, []int, error) {
	t, ok := s.hdr[name]
	if !ok || len(t.Shape) != 3 || !isFloat(t.Dtype) {
		return nil, nil, fmt.Errorf("%s is %s %v, want a float rank-3 tensor", name, t.Dtype, t.Shape)
	}
	b, err := s.raw(name)
	if err != nil {
		return nil, nil, err
	}
	o := make([]float32, len(b)/elemSize(t.Dtype))
	for i := range o {
		o[i] = at(b, t.Dtype, i)
	}
	return o, t.Shape, nil
}

// actFn is the gated FFN's activation on the gate.
func actFn(a string) func(float32) float32 {
	switch a {
	case "relu2":
		return func(g float32) float32 { r := max(g, 0); return r * r }
	case "silu", "swish":
		return func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }
	case "gelu":
		return func(g float32) float32 { x := float64(g); return float32(0.5 * x * (1 + math.Erf(x/math.Sqrt2))) }
	case "gelu_pytorch_tanh", "gelu_new":
		return func(g float32) float32 {
			x := float64(g)
			return float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x))))
		}
	}
	return nil
}

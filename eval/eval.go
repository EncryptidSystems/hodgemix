// Package eval scores a safetensors model on the CPU: greedy continuations and teacher-forced NLL
// on fixed text, and the calibration passes hodgemix reads. The model is whatever it is given
// (model.go): its sizes, activation, norms and rotary frequencies come from its own tensors and
// config. On a BitNet b1.58 file the forward is champ-learn's reference probe
// (edgellm/sparse-research/probe), whose logits match bitnet-rt.js / model.go exactly; it is used
// here only to compare a hodgemix output with its base, never inside the mixing math.
package eval

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
)

// Prompts for greedy continuations and texts for teacher-forced NLL. Fixed so runs compare.
var Prompts = []string{"The capital of France is", "Q: What is 17 times 23?\nA:"}
var Texts = []string{
	"The capital of France is Paris, and the capital of Germany is Berlin.",
	"Water boils at 100 degrees Celsius at sea level.",
	"The quick brown fox jumps over the lazy dog.",
	"In 1969, Neil Armstrong became the first person to walk on the Moon.",
	"func main() { fmt.Println(\"hello, world\") }",
	"Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
}

type Result struct {
	Model   string      `json:"model"`
	Gens    [][2]string `json:"greedy"`
	NLL     []float64   `json:"nll_per_text"`
	MeanNLL float64     `json:"mean_nll"`
	PPL     float64     `json:"ppl"`
	Tokens  int         `json:"tokens"`
}

type lin struct {
	n, k  int
	w     []int8 // [n][k], until pack (lut.go) turns it into codes
	s     float32
	limbs []*limb   // dynamic MRH digit limbs (lut.go)
	f     []float32 // a float weight [n][k] instead (no activation quantization)
	b     []float32 // bias, if the model has one
	mx    *mx4      // an MXFP4 expert: the nibble table, not a dense expansion of it
}

// expert is one gpt-oss expert, weights in the dense layout (output rows x input columns).
type expert struct {
	gate, up, down *lin
}

type layer struct {
	an, fn, sn, fsn, qn, kn []float32 // sn, fsn (BitNet sub-norms), qn, kn (per-head q/k norms) may be nil
	preFF, postFF           []float32 // gemma2/3 pre-ffn norm and the norm on the ffn output
	sink                    []float32 // gpt-oss per-head attention sink
	q, k, v, o, g, u, d     *lin
	router                  *lin
	experts                 []expert
	upd                     *expert // an improvement on the same cells; admitted per token
}

type model struct {
	Shape
	arch      Arch
	attnScale float64   // YaRN magnitude on rotated q and k; 1 otherwise
	localFreq []float64 // gemma3 RoPE on sliding layers
	ls        []layer
	emb, head []byte // embedding and LM head rows (the same bytes when tied)
	embDT     string
	headDT    string
	nEmb, nv  int // embedding rows, LM head rows
	fnorm     []float32
	invFreq   []float64
	flows     []map[string]*flowLin // per layer and role, when instrumented (flow.go)
	rec       []FlowLayer           // the token being fed's currents, when recording
	cells     Cells
}

// act is one linear layer input: its float activations and, for the BitLinear readers, their
// per-token absmax int8 quantization and LUT tables, made once and shared.
type act struct {
	x       []float32
	once    sync.Once
	in      *input
	xs      float32
	omu     sync.Mutex
	byOrder map[*int]*input // the input in a cell-major column order (flow.go)
}

func (a *act) ternary() (*input, float32) {
	a.once.Do(func() {
		q, s := quant(a.x)
		a.in, a.xs = newInput(q), s
	})
	return a.in, a.xs
}

// run is out (+)= W x + b. An MXFP4 expert gathers its nibble table. A packed BitLinear gathers
// the digit LUT. A float matrix is the probe path for a checkpoint that is already float.
func (l *lin) run(a *act, out []float32, accum bool) {
	switch {
	case l.mx != nil:
		l.mx.gather(a.x, out, accum, nil, nil, nil, 0)
	case l.f != nil:
		l.matmul(a.x, out, accum)
	default:
		in, xs := a.ternary()
		l.applyIn(in, xs, out, accum)
	}
	for i, v := range l.b {
		out[i] += v
	}
}

func (l *lin) matmul(x, out []float32, accum bool) {
	var wg sync.WaitGroup
	const P = 8
	for p := 0; p < P; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for r := p; r < l.n; r += P {
				row := l.f[r*l.k : (r+1)*l.k]
				var a float32
				for j, v := range x {
					a += v * row[j]
				}
				if accum {
					out[r] += a
				} else {
					out[r] = a
				}
			}
		}(p)
	}
	wg.Wait()
}

// quant: per-token absmax int8, as AutoBitLinear's ActQuant and norm.wgsl.
func quant(y []float32) ([]int8, float32) {
	mx := float32(1e-5)
	for _, v := range y {
		if a := float32(math.Abs(float64(v))); a > mx {
			mx = a
		}
	}
	s := 127 / mx
	q := make([]int8, len(y))
	for i, v := range y {
		q[i] = int8(math.Max(-128, math.Min(127, math.Round(float64(v*s)))))
	}
	return q, 1 / s
}

func (l *lin) apply(q []int8, xs float32, out []float32, accum bool) {
	var wg sync.WaitGroup
	const P = 8
	for p := 0; p < P; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for r := p; r < l.n; r += P {
				row := l.w[r*l.k : (r+1)*l.k]
				var acc int32
				for j, a := range q {
					acc += int32(a) * int32(row[j])
				}
				v := float32(acc) * xs * l.s
				if accum {
					out[r] += v
				} else {
					out[r] = v
				}
			}
		}(p)
	}
	wg.Wait()
}

// unitGain is a norm Champ inserted at ones. RMSNorm of that gain is not the identity, and it is
// not a gain the checkpoint stored.
func unitGain(w []float32) bool {
	if len(w) == 0 {
		return false
	}
	for _, v := range w {
		if v != 1 {
			return false
		}
	}
	return true
}

func rms(x, w []float32, eps float64) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v * v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+eps))
	o := make([]float32, len(x))
	for i := range x {
		o[i] = x[i] * inv * w[i]
	}
	return o
}

// headNorm is the per-head RMSNorm (q_norm / k_norm) of x, in place.
// norm is the model's RMSNorm: Llama/Qwen/BitNet/gpt-oss multiply by the weight, Gemma by 1+weight.
func (m *model) norm(x, w []float32) []float32 {
	if m.arch.Norm != "gemma" {
		return rms(x, w, m.Eps)
	}
	var ss float64
	for _, v := range x {
		ss += float64(v * v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+m.Eps))
	o := make([]float32, len(x))
	for i := range x {
		o[i] = x[i] * inv * (1 + w[i])
	}
	return o
}

func (m *model) headNorm(x, w []float32) {
	for h := 0; h < len(x)/m.HD; h++ {
		copy(x[h*m.HD:(h+1)*m.HD], m.norm(x[h*m.HD:(h+1)*m.HD], w))
	}
}

// rope applies rotary frequencies. scale is the YaRN magnitude (1 except yarn).
func (m *model) rope(x []float32, pos int, freq []float64, scale float64) {
	hd := m.HD
	sc := float32(scale)
	for h := 0; h < len(x)/hd; h++ {
		for d := 0; d < hd/2; d++ {
			f := float64(pos) * freq[d]
			c, s := float32(math.Cos(f))*sc, float32(math.Sin(f))*sc
			a, b := x[h*hd+d], x[h*hd+d+hd/2]
			x[h*hd+d], x[h*hd+d+hd/2] = a*c-b*s, a*s+b*c
		}
	}
}

func (m *model) logits(x []float32) []float32 {
	y := m.norm(x, m.fnorm)
	out := make([]float32, m.nv)
	D := m.D
	var wg sync.WaitGroup
	for p := 0; p < 8; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for r := p; r < m.nv; r += 8 {
				var a float32
				o := r * D
				if m.headDT == "BF16" {
					for j := 0; j < D; j++ {
						a += y[j] * bf(m.head, o+j)
					}
				} else {
					for j := 0; j < D; j++ {
						a += y[j] * at(m.head, m.headDT, o+j)
					}
				}
				out[r] = a
			}
		}(p)
	}
	wg.Wait()
	if cap := m.arch.FinalSoftcap; cap > 0 {
		for i, v := range out {
			out[i] = float32(math.Tanh(float64(v)/cap) * cap)
		}
	}
	return out
}

type seq struct {
	kc, vc [][][]float32
	tap    func(li int, stage string, v []float32) // Trace: the residual after each sublayer, the FFN hidden
}

// step is one token through every layer and the LM head: the next-token logits.
func (m *model) step(s *seq, tok, pos int) []float32 { return m.logits(m.feed(s, tok, pos)) }

// feed is one token through every layer, without the LM head (vocab x hidden, more than all the
// ternary layers): for prompt tokens whose logits are not read, and for calibration.
func (m *model) feed(s *seq, tok, pos int) []float32 {
	D, H, KVH, HD := m.D, m.H, m.KVH, m.HD
	if s.kc == nil {
		s.kc, s.vc = make([][][]float32, m.L), make([][][]float32, m.L)
	}
	x := make([]float32, D)
	for j := 0; j < D; j++ {
		x[j] = at(m.emb, m.embDT, tok*D+j)
	}
	if m.arch.EmbedScale {
		sc := float32(math.Sqrt(float64(D)))
		for j := range x {
			x[j] *= sc
		}
	}
	for li := 0; li < m.L; li++ {
		Lr := &m.ls[li]
		m.node(li, "D", x, m.cells.D, m.cells.ND)
		xa := m.norm(x, Lr.an)
		if s.tap != nil {
			s.tap(li, "in_attn", xa)
		}
		ta := &act{x: xa}
		qv, kv, vv := make([]float32, H*HD), make([]float32, KVH*HD), make([]float32, KVH*HD)
		m.lin(li, "q", Lr.q, ta, qv, false)
		m.lin(li, "k", Lr.k, ta, kv, false)
		m.lin(li, "v", Lr.v, ta, vv, false)
		m.node(li, "q", qv, m.cells.Q, m.cells.NQ)
		m.node(li, "k", kv, m.cells.KV, m.cells.NKV)
		m.node(li, "v", vv, m.cells.KV, m.cells.NKV)
		if s.tap != nil {
			s.tap(li, "v", vv)
		}
		if Lr.qn != nil {
			m.headNorm(qv, Lr.qn)
		}
		if Lr.kn != nil {
			m.headNorm(kv, Lr.kn)
		}
		freq, scale := m.invFreq, m.attnScale
		win := 0
		if li < len(m.arch.Window) {
			win = m.arch.Window[li]
		}
		if win > 0 && m.localFreq != nil {
			freq, scale = m.localFreq, 1
		}
		m.rope(qv, pos, freq, scale)
		m.rope(kv, pos, freq, scale)
		s.kc[li] = append(s.kc[li], kv)
		s.vc[li] = append(s.vc[li], vv)
		ao := m.attend(li, qv, s, pos, win)
		m.node(li, "A", ao, m.cells.Q, m.cells.NQ)
		xo := ao
		if Lr.sn != nil {
			xo = m.norm(ao, Lr.sn)
		}
		if s.tap != nil {
			s.tap(li, "in_o", xo)
		}
		if m.arch.Sandwich {
			y := make([]float32, D)
			m.lin(li, "o", Lr.o, &act{x: xo}, y, false)
			yn := m.norm(y, Lr.fn)
			m.scaleEdge(li, "o", y, yn, m.cells.D, m.cells.ND)
			for i := range x {
				x[i] += yn[i]
			}
		} else {
			m.lin(li, "o", Lr.o, &act{x: xo}, x, true)
		}
		m.node(li, "Dm", x, m.cells.D, m.cells.ND)
		if s.tap != nil {
			s.tap(li, "attn", x)
		}
		var xf []float32
		if m.arch.Sandwich {
			xf = m.norm(x, Lr.preFF)
		} else {
			xf = m.norm(x, Lr.fn)
		}
		if s.tap != nil {
			s.tap(li, "in_ffn", xf)
		}
		z := m.ffn(li, xf, s)
		if m.arch.Sandwich {
			zn := m.norm(z, Lr.postFF)
			m.scaleEdge(li, "down", z, zn, m.cells.D, m.cells.ND)
			z = zn
		}
		for i := range x {
			x[i] += z[i]
		}
		m.node(li, "Dout", x, m.cells.D, m.cells.ND)
		if s.tap != nil {
			s.tap(li, "mlp", x)
		}
	}
	if s.tap != nil {
		s.tap(m.L, "final", m.norm(x, m.fnorm))
	}
	return x
}

// attend is one layer's causal attention. A sliding window drops keys at or before pos-window.
// A sink is an extra logit that takes softmax mass and reads no value. Softcap is Gemma's tanh cap.
func (m *model) attend(li int, qv []float32, s *seq, pos, win int) []float32 {
	H, KVH, HD := m.H, m.KVH, m.HD
	Lr := &m.ls[li]
	ao := make([]float32, H*HD)
	T := len(s.kc[li])
	den := math.Sqrt(float64(HD))
	if m.arch.QueryScale > 0 {
		den = math.Sqrt(m.arch.QueryScale)
	}
	cap := m.arch.AttnSoftcap
	for h := 0; h < H; h++ {
		g := h / (H / KVH)
		sc := make([]float64, T)
		keep := make([]bool, T)
		mx := -1e30
		for t := 0; t < T; t++ {
			if win > 0 && pos-t >= win {
				continue
			}
			keep[t] = true
			var a float64
			for d := 0; d < HD; d++ {
				a += float64(qv[h*HD+d] * s.kc[li][t][g*HD+d])
			}
			sc[t] = a / den
			if cap > 0 {
				sc[t] = math.Tanh(sc[t]/cap) * cap
			}
			mx = math.Max(mx, sc[t])
		}
		sink := -1e30
		if Lr.sink != nil {
			sink = float64(Lr.sink[h])
			mx = math.Max(mx, sink)
		}
		var z float64
		if Lr.sink != nil {
			z = math.Exp(sink - mx)
		}
		for t := range sc {
			if !keep[t] {
				continue
			}
			sc[t] = math.Exp(sc[t] - mx)
			z += sc[t]
		}
		for t := range sc {
			if !keep[t] {
				continue
			}
			w := float32(sc[t] / z)
			for d := 0; d < HD; d++ {
				ao[h*HD+d] += w * s.vc[li][t][g*HD+d]
			}
		}
	}
	return ao
}

// ffn is the residual contribution of the feed-forward. A mixture routes top-k and writes the
// weighted experts into the same F cells the dense gate/up/down use.
func (m *model) ffn(li int, xf []float32, s *seq) []float32 {
	Lr := &m.ls[li]
	if len(Lr.experts) > 0 {
		return m.consider(li, xf, s, m.mixture(li, xf, s))
	}
	F := m.F
	gate := actFn(m.Act)
	gv, uv := make([]float32, F), make([]float32, F)
	tf := &act{x: xf}
	m.lin(li, "gate", Lr.g, tf, gv, false)
	m.lin(li, "up", Lr.u, tf, uv, false)
	m.node(li, "gate", gv, m.cells.F, m.cells.NF)
	m.node(li, "up", uv, m.cells.F, m.cells.NF)
	hv := make([]float32, F)
	for i := range hv {
		hv[i] = gate(gv[i]) * uv[i]
	}
	if s.tap != nil {
		s.tap(li, "hidden", hv)
	}
	xd := hv
	if Lr.fsn != nil {
		xd = m.norm(hv, Lr.fsn)
	}
	if s.tap != nil {
		s.tap(li, "in_down", xd)
	}
	m.node(li, "H", hv, m.cells.F, m.cells.NF)
	z := make([]float32, m.D)
	m.lin(li, "down", Lr.d, &act{x: xd}, z, false)
	return m.consider(li, xf, s, z)
}

// consider admits an attached improvement when the combined feed-forward keeps the base's
// residual cell and its sign. A refused update is left out of the residual and out of the cochain.
func (m *model) consider(li int, xf []float32, s *seq, z []float32) []float32 {
	Lr := &m.ls[li]
	if Lr.upd == nil {
		return z
	}
	saved := m.rec
	var scratch []FlowLayer
	if saved != nil {
		scratch = make([]FlowLayer, m.L)
		for i := range scratch {
			scratch[i] = FlowLayer{Edge: map[string][]float32{}, Node: map[string][]float32{}}
		}
		m.rec = scratch
	}
	z1 := m.denseExpert(li, xf, s, Lr.upd)
	m.rec = saved
	ok := m.admit(z, z1)
	if saved != nil {
		if ok {
			addEdges(saved[li].Edge, scratch[li].Edge)
			saved[li].Experts = []int{0, 1}
			saved[li].Gates = []float32{1, 1}
		} else {
			saved[li].Experts = []int{0}
			saved[li].Gates = []float32{1}
		}
	}
	if ok {
		for i := range z {
			z[i] += z1[i]
		}
	}
	return z
}

// denseExpert is one feed-forward on the base's activation, recorded into the current cochain.
func (m *model) denseExpert(li int, xf []float32, s *seq, ex *expert) []float32 {
	Lr := &m.ls[li]
	gate := actFn(m.Act)
	gv, uv := make([]float32, m.F), make([]float32, m.F)
	tf := &act{x: xf}
	m.lin(li, "gate", ex.gate, tf, gv, false)
	m.lin(li, "up", ex.up, tf, uv, false)
	hv := make([]float32, m.F)
	for i := range hv {
		hv[i] = gate(gv[i]) * uv[i]
	}
	xd := hv
	if Lr.fsn != nil {
		xd = m.norm(hv, Lr.fsn)
	}
	if s != nil && s.tap != nil {
		s.tap(li, "update", xd)
	}
	z := make([]float32, m.D)
	m.lin(li, "down", ex.down, &act{x: xd}, z, false)
	return z
}

// admit is true when base+update moves the same residual cell, the same way.
// With no cell layout the update is refused: there is nothing to check it against.
func (m *model) admit(base, upd []float32) bool {
	if m.cells.ND == 0 || len(m.cells.D) != len(base) {
		return false
	}
	c0, d0 := hotCell(base, m.cells.D, m.cells.ND)
	sum := make([]float32, len(base))
	for i := range sum {
		sum[i] = base[i] + upd[i]
	}
	c1, d1 := hotCell(sum, m.cells.D, m.cells.ND)
	const eps = 1e-8
	if math.Abs(d0) < eps {
		return math.Abs(d1) < eps
	}
	return c1 == c0 && d0*d1 > 0
}

func hotCell(v []float32, cell []int32, n int) (int, float64) {
	s := cellSums(v, cell, n)
	c, d := 0, 0.0
	for i, x := range s {
		if math.Abs(float64(x)) > math.Abs(d) {
			c, d = i, float64(x)
		}
	}
	return c, d
}

func addEdges(dst, src map[string][]float32) {
	for k, v := range src {
		e := dst[k]
		if e == nil {
			e = make([]float32, len(v))
			dst[k] = e
		}
		for i := range v {
			e[i] += v[i]
		}
	}
}

// attach copies update's feed-forward onto each layer as the improvement expert.
func (m *model) attach(path string) error {
	o, err := load(path)
	if err != nil {
		return err
	}
	if o.L != m.L || o.D != m.D || o.F != m.F {
		return fmt.Errorf("update %s is %d layers, D=%d F=%d; base is %d layers, D=%d F=%d", path, o.L, o.D, o.F, m.L, m.D, m.F)
	}
	for i := range m.ls {
		var ex expert
		if len(o.ls[i].experts) > 0 {
			ex = o.ls[i].experts[0]
		} else {
			ex = expert{gate: o.ls[i].g, up: o.ls[i].u, down: o.ls[i].d}
		}
		if ex.gate == nil || ex.up == nil || ex.down == nil {
			return fmt.Errorf("update %s layer %d has no feed-forward", path, i)
		}
		e := ex
		m.ls[i].upd = &e
	}
	return nil
}

// mixture is gpt-oss: top-k on the router, softmax over the selected logits only, then the clamped
// SwiGLU (gate and up interleaved, up shifted by one) scaled by that weight.
func (m *model) mixture(li int, xf []float32, s *seq) []float32 {
	Lr := &m.ls[li]
	logits := make([]float32, Lr.router.n)
	Lr.router.run(&act{x: xf}, logits, false)
	idx, wt := topkSoftmax(logits, m.arch.TopK)
	if m.rec != nil {
		m.rec[li].Experts = append([]int(nil), idx...)
		m.rec[li].Gates = append([]float32(nil), wt...)
	}
	alpha, limit := float32(m.arch.SwigluAlpha), float32(m.arch.SwigluLimit)
	F := m.F
	gAcc, uAcc, hAcc := make([]float32, F), make([]float32, F), make([]float32, F)
	tf := &act{x: xf}
	hidden := make([][]float32, len(idx))
	for t, e := range idx {
		ex := &Lr.experts[e]
		w := wt[t]
		g := m.mixLin(li, "gate", ex.gate, tf, w, gAcc)
		u := m.mixLin(li, "up", ex.up, tf, w, uAcc)
		h := swiglu(g, u, alpha, limit)
		hidden[t] = h
		for i := range h {
			hAcc[i] += w * h[i]
		}
	}
	m.node(li, "gate", gAcc, m.cells.F, m.cells.NF)
	m.node(li, "up", uAcc, m.cells.F, m.cells.NF)
	if s.tap != nil {
		s.tap(li, "hidden", hAcc)
		s.tap(li, "in_down", hAcc)
	}
	m.node(li, "H", hAcc, m.cells.F, m.cells.NF)
	z := make([]float32, m.D)
	for t, e := range idx {
		m.mixLin(li, "down", Lr.experts[e].down, &act{x: hidden[t]}, wt[t], z)
	}
	return z
}

// mixLin adds weight * (W x + b) into acc and, when recording, the same weight times the edge
// currents. acc may be nil.
func (m *model) mixLin(li int, role string, l *lin, a *act, weight float32, acc []float32) []float32 {
	y := make([]float32, l.n)
	if m.rec == nil {
		l.run(a, y, false)
	} else {
		fl := m.flows[li][role]
		tmp := make([]float32, fl.nr*fl.nc)
		l.runFlow(fl, a, y, false, tmp)
		addBiasFlow(tmp, l.b, fl.rowCell, fl.nr, fl.nc)
		e := m.rec[li].Edge[role]
		if e == nil {
			e = make([]float32, len(tmp))
			m.rec[li].Edge[role] = e
		}
		for i, v := range tmp {
			e[i] += weight * v
		}
	}
	if acc != nil {
		for i, v := range y {
			acc[i] += weight * v
		}
	}
	return y
}

func swiglu(gate, up []float32, alpha, limit float32) []float32 {
	h := make([]float32, len(gate))
	for i := range h {
		g, u := gate[i], up[i]
		if g > limit {
			g = limit
		}
		if u > limit {
			u = limit
		} else if u < -limit {
			u = -limit
		}
		sig := float32(1 / (1 + math.Exp(float64(-alpha*g))))
		h[i] = (u + 1) * (g * sig)
	}
	return h
}

// topkSoftmax keeps the k largest logits (lower index wins a tie) and softmaxes over those only.
func topkSoftmax(logits []float32, k int) ([]int, []float32) {
	if k > len(logits) {
		k = len(logits)
	}
	idx := make([]int, k)
	used := make([]bool, len(logits))
	for t := 0; t < k; t++ {
		best, bi := float32(-1e30), 0
		for i, v := range logits {
			if used[i] {
				continue
			}
			if v > best || (v == best && i < bi) {
				best, bi = v, i
			}
		}
		used[bi] = true
		idx[t] = bi
	}
	mx := logits[idx[0]]
	for _, i := range idx[1:] {
		if logits[i] > mx {
			mx = logits[i]
		}
	}
	w := make([]float32, k)
	var z float32
	for t, i := range idx {
		w[t] = float32(math.Exp(float64(logits[i] - mx)))
		z += w[t]
	}
	for t := range w {
		w[t] /= z
	}
	return idx, w
}

// open loads a model and its tokenizer and checks that the tokenizer is the model's: every id it
// can produce has an embedding row.
func open(modelPath, tokPath string) (*model, *tok, error) {
	if tokPath == "" {
		tokPath = FilesOf(modelPath).Tokenizer
	}
	if tokPath == "" {
		return nil, nil, fmt.Errorf("%s: no tokenizer set (set its tokenizer in the run file)", modelPath)
	}
	tk, err := loadTok(tokPath, FilesOf(modelPath).TokenizerConfig)
	if err != nil {
		return nil, nil, err
	}
	m, err := load(modelPath)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %v", modelPath, err)
	}
	if u := UpdateOf(modelPath); u != "" {
		if err := m.attach(u); err != nil {
			return nil, nil, err
		}
	}
	if tk.n > m.nEmb {
		return nil, nil, fmt.Errorf("%s has ids up to %d but %s embeds %d rows: it is not this model's tokenizer",
			tokPath, tk.n-1, modelPath, m.nEmb)
	}
	return m, tk, nil
}

// ---- tokenizer (byte-level BPE) ----

type tok struct {
	b2u     [256]rune
	u2b     map[rune]byte
	vocab   map[string]int
	inv     map[int]string
	ranks   map[string]int
	special map[string]int // added tokens: encodeChat keeps them whole
	bos     []int          // what the post-processor puts before a sequence (none for most chat models)
	eot     string         // the turn end: the tokenizer config's eos_token (none without one)
	n       int            // 1 + the largest id
}

func loadTok(p, cfgPath string) (*tok, error) {
	t := &tok{u2b: map[rune]byte{}, ranks: map[string]int{}}
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE) {
			t.b2u[b] = rune(b)
		} else {
			t.b2u[b] = rune(256 + n)
			n++
		}
		t.u2b[t.b2u[b]] = byte(b)
	}
	f, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var j struct {
		Model struct {
			Vocab  map[string]int  `json:"vocab"`
			Merges json.RawMessage `json:"merges"`
		} `json:"model"`
		Added []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
		Post json.RawMessage `json:"post_processor"`
	}
	if err := json.Unmarshal(f, &j); err != nil {
		return nil, err
	}
	if len(j.Model.Vocab) == 0 {
		return nil, errors.New(p + ": no model.vocab")
	}
	t.vocab = j.Model.Vocab
	// Byte-level BPE: every byte has a token, so every piece of any text encodes (a missing one would
	// otherwise read as id 0).
	for b := 0; b < 256; b++ {
		if _, ok := t.vocab[string(t.b2u[b])]; !ok {
			return nil, fmt.Errorf("%s: byte %#x has no token: not a byte-level BPE tokenizer", p, b)
		}
	}
	t.inv = map[int]string{}
	for k, v := range t.vocab {
		t.inv[v] = k
	}
	t.special = map[string]int{}
	for _, a := range j.Added {
		t.special[a.Content] = a.ID
		t.inv[a.ID] = a.Content
	}
	var ms []string
	if json.Unmarshal(j.Model.Merges, &ms) != nil {
		var mp [][2]string
		json.Unmarshal(j.Model.Merges, &mp)
		for _, m := range mp {
			ms = append(ms, m[0]+" "+m[1])
		}
	}
	for i, m := range ms {
		t.ranks[m] = i
	}
	for id := range t.inv {
		t.n = max(t.n, id+1)
	}
	t.bos = templateBOS(j.Post)
	if cfgPath != "" {
		b, err := os.ReadFile(cfgPath)
		if err != nil {
			return nil, err
		}
		var tc struct {
			EOS json.RawMessage `json:"eos_token"`
		}
		if err := json.Unmarshal(b, &tc); err != nil {
			return nil, fmt.Errorf("%s: %v", cfgPath, err)
		}
		var e string
		if json.Unmarshal(tc.EOS, &e) != nil {
			var o struct {
				Content string `json:"content"`
			}
			json.Unmarshal(tc.EOS, &o)
			e = o.Content
		}
		if _, ok := t.special[e]; !ok {
			return nil, fmt.Errorf("%s: eos_token %q is not an added token of %s", cfgPath, e, p)
		}
		t.eot = e
	}
	return t, nil
}

// templateBOS is the special tokens a TemplateProcessing post-processor (alone or inside a Sequence)
// puts before a single sequence.
func templateBOS(raw json.RawMessage) []int {
	var pp struct {
		Processors []json.RawMessage `json:"processors"`
		Single     []struct {
			Special *struct {
				ID string `json:"id"`
			} `json:"SpecialToken"`
		} `json:"single"`
		Specials map[string]struct {
			IDs []int `json:"ids"`
		} `json:"special_tokens"`
	}
	if json.Unmarshal(raw, &pp) != nil {
		return nil
	}
	for _, p := range pp.Processors {
		if b := templateBOS(p); b != nil {
			return b
		}
	}
	var out []int
	for _, e := range pp.Single {
		if e.Special == nil {
			break
		}
		out = append(out, pp.Specials[e.Special.ID].IDs...)
	}
	return out
}

var pre = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]{1,3}| ?[^\s\p{L}\p{N}]+|\s+`)

func (t *tok) encode(s string) []int { return append(append([]int(nil), t.bos...), t.pieces(s)...) }

// pieces is s's BPE ids, no BOS.
func (t *tok) pieces(s string) []int {
	var out []int
	for _, w := range pre.FindAllString(s, -1) {
		var sym []string
		for _, b := range []byte(w) {
			sym = append(sym, string(t.b2u[b]))
		}
		for {
			best, bi := 1<<30, -1
			for i := 0; i+1 < len(sym); i++ {
				if r, ok := t.ranks[sym[i]+" "+sym[i+1]]; ok && r < best {
					best, bi = r, i
				}
			}
			if bi < 0 {
				break
			}
			sym = append(sym[:bi], append([]string{sym[bi] + sym[bi+1]}, sym[bi+2:]...)...)
		}
		for _, x := range sym {
			out = append(out, t.vocab[x])
		}
	}
	return out
}

func (t *tok) decode(id int) string {
	var b []byte
	for _, r := range t.inv[id] {
		b = append(b, t.u2b[r])
	}
	return string(b)
}

func argmax(x []float32) int {
	b := 0
	for i := range x {
		if x[i] > x[b] {
			b = i
		}
	}
	return b
}

func logsoft(x []float32, i int) float64 {
	m := float64(x[argmax(x)])
	var z float64
	for _, v := range x {
		z += math.Exp(float64(v) - m)
	}
	return float64(x[i]) - m - math.Log(z)
}

// Run loads modelPath and scores it with tokenizer tokPath.
func Run(modelPath, tokPath string) (*Result, error) {
	m, tk, err := open(modelPath, tokPath)
	if err != nil {
		return nil, err
	}
	r := &Result{Model: modelPath}
	for _, p := range Prompts {
		ids := tk.encode(p)
		s := &seq{}
		var lg []float32
		for i, id := range ids {
			if i < len(ids)-1 {
				m.feed(s, id, i)
			} else {
				lg = m.step(s, id, i)
			}
		}
		var sb strings.Builder
		for j := 0; j < 10; j++ {
			nx := argmax(lg)
			sb.WriteString(tk.decode(nx))
			lg = m.step(s, nx, len(ids)+j)
		}
		r.Gens = append(r.Gens, [2]string{p, sb.String()})
	}
	var tot float64
	for _, tx := range Texts {
		ids := tk.encode(tx)
		s := &seq{}
		var nll float64
		for i := 0; i < len(ids)-1; i++ {
			nll -= logsoft(m.step(s, ids[i], i), ids[i+1])
		}
		r.NLL = append(r.NLL, nll/float64(len(ids)-1))
		tot += nll
		r.Tokens += len(ids) - 1
	}
	r.MeanNLL = tot / float64(r.Tokens)
	r.PPL = math.Exp(r.MeanNLL)
	return r, nil
}

// SublayerTrace compares two models (same shape) on one text, sublayer by sublayer: Cos is the mean
// cosine between their residual streams after the sublayer, Norm the mean ratio ||x_b|| / ||x_a||,
// and for the FFN hidden (relu^2(gate) * up) Peak is max|h_b| / max|h_a|. Top lists the residual
// channels carrying most of the difference (the singular set to cut, if any).
type SublayerTrace struct {
	Layer int     `json:"layer"`
	Stage string  `json:"stage"`
	Cos   float64 `json:"cos"`
	Norm  float64 `json:"norm_ratio"`
	Peak  float64 `json:"peak_ratio,omitempty"`
	Top   []int   `json:"top_channels,omitempty"`
	TopFr float64 `json:"top_fraction,omitempty"`   // share of ||x_b - x_a||^2 in Top
	Big   []int   `json:"base_massive,omitempty"`   // the first model's largest residual channels (rms)
	BigX  float64 `json:"base_massive_x,omitempty"` // their rms over the median channel's
}

// Trace's posCos is, per position, the cosine of the two models' last-layer residuals (chat: text is
// sent in the runtime's chat format with the added tokens whole).
func Trace(aPath, bPath, tokPath, text string, chat bool) ([]SublayerTrace, []float64, error) {
	ma, tk, err := open(aPath, tokPath)
	if err != nil {
		return nil, nil, err
	}
	mb, _, err := open(bPath, tokPath)
	if err != nil {
		return nil, nil, err
	}
	if ma.D != mb.D || ma.L != mb.L || ma.F != mb.F {
		return nil, nil, fmt.Errorf("%s (%d x %d, ffn %d) and %s (%d x %d, ffn %d) are not the same shape", aPath, ma.L, ma.D, ma.F, bPath, mb.L, mb.D, mb.F)
	}
	type key struct {
		li    int
		stage string
	}
	type acc struct {
		cos, norm, peakA, peakB float64
		n                       int
		diff, ea                []float64
	}
	accs := map[key]*acc{}
	var order []key
	var cur map[key][]float32
	sa, sb := &seq{}, &seq{}
	var lastB []float32
	sa.tap = func(li int, st string, v []float32) { cur[key{li, st}] = append([]float32(nil), v...) }
	sb.tap = func(li int, st string, v []float32) {
		if li == mb.L-1 && st == "mlp" {
			lastB = append(lastB[:0], v...)
		}
		k := key{li, st}
		a := cur[k]
		ac := accs[k]
		if ac == nil {
			ac = &acc{}
			accs[k] = ac
			order = append(order, k)
		}
		if st == "hidden" {
			for i := range v {
				ac.peakA = math.Max(ac.peakA, math.Abs(float64(a[i])))
				ac.peakB = math.Max(ac.peakB, math.Abs(float64(v[i])))
			}
			return
		}
		var dot, na, nb float64
		if ac.diff == nil {
			ac.diff, ac.ea = make([]float64, len(v)), make([]float64, len(v))
		}
		for i := range v {
			x, y := float64(a[i]), float64(v[i])
			dot, na, nb = dot+x*y, na+x*x, nb+y*y
			ac.diff[i] += (y - x) * (y - x)
			ac.ea[i] += x * x
		}
		ac.cos += dot / math.Sqrt(na*nb+1e-30)
		ac.norm += math.Sqrt(nb / (na + 1e-30))
		ac.n++
	}
	if chat && text == "" {
		text = chatPrompt(tk)
	}
	ids := tk.encode(text)
	if chat {
		ids = tk.encodeChat(text)
	}
	var posCos []float64
	for i, id := range ids {
		cur = map[key][]float32{}
		ma.feed(sa, id, i)
		mb.feed(sb, id, i)
		if a, b := cur[key{ma.L - 1, "mlp"}], lastB; a != nil && b != nil {
			var dot, na, nb float64
			for j := range a {
				x, y := float64(a[j]), float64(b[j])
				dot, na, nb = dot+x*y, na+x*x, nb+y*y
			}
			posCos = append(posCos, dot/math.Sqrt(na*nb+1e-30))
		}
	}
	var out []SublayerTrace
	for _, k := range order {
		ac := accs[k]
		t := SublayerTrace{Layer: k.li, Stage: k.stage}
		if k.stage == "hidden" {
			t.Peak = ac.peakB / math.Max(ac.peakA, 1e-30)
			t.Cos, t.Norm = math.NaN(), math.NaN()
		} else {
			t.Cos, t.Norm = ac.cos/float64(ac.n), ac.norm/float64(ac.n)
			idx := make([]int, len(ac.diff))
			var tot float64
			for i := range idx {
				idx[i] = i
				tot += ac.diff[i]
			}
			sort.Slice(idx, func(i, j int) bool { return ac.diff[idx[i]] > ac.diff[idx[j]] })
			var top float64
			for _, i := range idx[:8] {
				top += ac.diff[i]
			}
			t.Top, t.TopFr = idx[:8], top/math.Max(tot, 1e-30)
			big := append([]int(nil), idx...)
			sort.Slice(big, func(i, j int) bool { return ac.ea[big[i]] > ac.ea[big[j]] })
			t.Big, t.BigX = big[:8], math.Sqrt(ac.ea[big[7]]/math.Max(ac.ea[big[len(big)/2]], 1e-30))
		}
		out = append(out, t)
	}
	return out, posCos, nil
}

// Singular is where a model's forward concentrates: Massive are the residual channels whose rms
// (over the calibration texts, at any layer) is >= Ratio x the median channel's (massive
// activations / attention sinks), and Spikes[l] the FFN hidden units of layer l whose peak
// |relu^2(gate) * up| is >= Ratio x the layer's median unit peak (the units that write them).
// MassiveX and SpikeX are each set's smallest ratio, for the log.
type Singular struct {
	Massive  []int             `json:"massive_channels"`
	MassiveX []float64         `json:"massive_ratio"`
	Spikes   map[int][]int     `json:"spike_units"`
	SpikeX   map[int][]float64 `json:"spike_ratio"`
	// Means[l][input] is the mean BitLinear input over the calibration tokens (input: in_attn read by
	// q/k/v, in_o by o, in_ffn by gate/up, in_down by down): the harmonic (constant) mode a merge must
	// leave each reader's response to.
	Means  []map[string][]float64 `json:"-"`
	Tokens int                    `json:"calibration_tokens"`
}

func FindSingular(path, tokPath string, ratio float64) (*Singular, error) {
	m, tk, err := open(path, tokPath)
	if err != nil {
		return nil, err
	}
	ea, peak := make([][]float64, m.L), make([][]float64, m.L)
	means := make([]map[string][]float64, m.L)
	ntok := 0
	s := &seq{}
	s.tap = func(li int, st string, v []float32) {
		if strings.HasPrefix(st, "in_") {
			if means[li] == nil {
				means[li] = map[string][]float64{}
			}
			mv := means[li][st]
			if mv == nil {
				mv = make([]float64, len(v))
				means[li][st] = mv
			}
			for i, x := range v {
				mv[i] += float64(x)
			}
			return
		}
		switch st {
		case "mlp", "attn":
			if ea[li] == nil {
				ea[li] = make([]float64, len(v))
			}
			for i, x := range v {
				ea[li][i] += float64(x) * float64(x)
			}
		case "hidden":
			if peak[li] == nil {
				peak[li] = make([]float64, len(v))
			}
			for i, x := range v {
				peak[li][i] = math.Max(peak[li][i], math.Abs(float64(x)))
			}
		}
	}
	var calib [][]int
	for _, tx := range append(append([]string{}, Texts...), Prompts...) {
		calib = append(calib, tk.encode(tx))
	}
	calib = append(calib, tk.encodeChat(chatPrompt(tk)))
	for _, ids := range calib {
		*s = seq{tap: s.tap}
		for i, id := range ids {
			m.feed(s, id, i)
		}
		ntok += len(ids)
	}
	median := func(a []float64) float64 {
		b := append([]float64(nil), a...)
		sort.Float64s(b)
		return b[len(b)/2]
	}
	out := &Singular{Spikes: map[int][]int{}, SpikeX: map[int][]float64{}, Tokens: ntok}
	for li := range means {
		for _, mv := range means[li] {
			for i := range mv {
				mv[i] /= float64(ntok)
			}
		}
	}
	out.Means = means
	best := map[int]float64{}
	for li := 0; li < m.L; li++ {
		if ea[li] == nil {
			continue
		}
		md := math.Sqrt(median(ea[li]))
		for c, e := range ea[li] {
			if r := math.Sqrt(e) / math.Max(md, 1e-30); r >= ratio && r > best[c] {
				best[c] = r
			}
		}
		pm := median(peak[li])
		for u, p := range peak[li] {
			if r := p / math.Max(pm, 1e-30); r >= ratio {
				out.Spikes[li] = append(out.Spikes[li], u)
				out.SpikeX[li] = append(out.SpikeX[li], r)
			}
		}
	}
	for c := range best {
		out.Massive = append(out.Massive, c)
	}
	sort.Ints(out.Massive)
	for _, c := range out.Massive {
		out.MassiveX = append(out.MassiveX, best[c])
	}
	return out, nil
}

// encodeChat is encode with the added tokens taken whole, as the runtime's
// tokenizer does for a chat turn.
func (t *tok) encodeChat(s string) []int {
	return append(append([]int(nil), t.bos...), t.chatPieces(s)...)
}

// chatPieces is encodeChat without the BOS (a later turn of a conversation).
func (t *tok) chatPieces(s string) []int {
	var out []int
	for len(s) > 0 {
		at, id, n := len(s), -1, 0
		for sp, i := range t.special {
			if j := strings.Index(s, sp); j >= 0 && (j < at || (j == at && len(sp) > n)) {
				at, id, n = j, i, len(sp)
			}
		}
		out = append(out, t.pieces(s[:at])...)
		if id < 0 {
			break
		}
		out = append(out, id)
		s = s[at+n:]
	}
	return out
}

// ChatSystem and ChatUser are the page's system message and a first question, in the runtime's chat
// format (examples/bitnet main.go: "Role: content" and the turn end, then "Assistant: "). The greedy reply
// to them is the page's first answer: a ~300-token context, where the short Texts are ~20.
const ChatSystem = `You are Champ, named for the lake monster, and you run on this device. Answer the person directly in plain words. Do not begin with your name. Do not introduce yourself. Do not say that you are an AI who cannot eat or feel, unless they asked what you are. Use a tool only when the person asks you to search, look something up, or read a page. To search, reply with only {"type":"tool_call","tool":"search","args":{"q":"query"}}. To read one page, reply with only {"type":"tool_call","tool":"gather","args":{"url":"https://example.com"}}. When the person asks you to save, open or list their notes, reply with only {"type":"tool_call","tool":"notes","args":{"op":"write","path":"plans/today.md","text":"..."}} (op is write, read or list). For their task list: {"type":"tool_call","tool":"tasks","args":{"op":"add","title":"..."}} (op is add, list, done or start, with id for done and start). To draw something on their canvas: {"type":"tool_call","tool":"canvas","args":{"op":"write","title":"...","text":"markdown"}}. To run a shell command when they ask you to run code: {"type":"tool_call","tool":"exec","args":{"cmd":"ls"}}. To write a note, plan, or answer for them to read here, do not use a tool.`

const ChatUser = "What is the capital of France?"

// Chat is the greedy reply (up to n tokens, stopping at the turn end) to ChatSystem + ChatUser.
func Chat(modelPath, tokPath string, n int) (string, int, error) {
	m, tk, err := open(modelPath, tokPath)
	if err != nil {
		return "", 0, err
	}
	sys := "System: " + ChatSystem + tk.eot
	if os.Getenv("HODGEMIX_CHAT_NOSYS") != "" {
		sys = ""
	}
	ids := tk.encodeChat(sys + "User: " + ChatUser + tk.eot + "Assistant: ")
	eot, hasEot := tk.special[tk.eot]
	s := &seq{}
	var lg []float32
	for i, id := range ids {
		if i < len(ids)-1 {
			m.feed(s, id, i)
		} else {
			lg = m.step(s, id, i)
		}
	}
	var sb strings.Builder
	for j := 0; j < n; j++ {
		nx := argmax(lg)
		if hasEot && nx == eot {
			break
		}
		sb.WriteString(tk.decode(nx))
		lg = m.step(s, nx, len(ids)+j)
	}
	return sb.String(), len(ids), nil
}

// Generate feeds prompt and then up to n new tokens. chat uses the page's system prompt and
// ChatUser when user is empty. When cells are set, every fed token is instrumented and each is
// called with that token's currents; next is the id the forward chose (−1 while the prompt is
// teacher-forced), and piece is that id decoded.
func Generate(modelPath, tokPath, user string, n int, chat bool, cells Cells, each func(pos, fed, next int, piece string, layers []FlowLayer)) (string, int, error) {
	m, tk, err := open(modelPath, tokPath)
	if err != nil {
		return "", 0, err
	}
	if cells.ND > 0 {
		if err := m.instrument(cells); err != nil {
			return "", 0, err
		}
	}
	var ids []int
	if chat {
		if user == "" {
			user = ChatUser
		}
		sys := "System: " + ChatSystem + tk.eot
		if os.Getenv("HODGEMIX_CHAT_NOSYS") != "" {
			sys = ""
		}
		ids = tk.encodeChat(sys + "User: " + user + tk.eot + "Assistant: ")
	} else {
		ids = tk.encode(user)
	}
	if len(ids) == 0 {
		return "", 0, errors.New("empty prompt")
	}
	s := &seq{}
	var lg []float32
	feed := func(id, pos int, wantLogits bool) {
		if cells.ND > 0 {
			m.rec = make([]FlowLayer, m.L)
			for li := range m.rec {
				m.rec[li] = FlowLayer{Edge: map[string][]float32{}, Node: map[string][]float32{}}
			}
		}
		if wantLogits {
			lg = m.step(s, id, pos)
		} else {
			m.feed(s, id, pos)
		}
	}
	report := func(pos, fed, next int) {
		if each == nil || m.rec == nil {
			return
		}
		piece := ""
		if next >= 0 {
			piece = tk.decode(next)
		}
		each(pos, fed, next, piece, m.rec)
		m.rec = nil
	}
	for i, id := range ids {
		last := i == len(ids)-1
		feed(id, i, last)
		next := -1
		if last {
			next = argmax(lg)
		}
		report(i, id, next)
	}
	eot, hasEot := tk.special[tk.eot]
	var sb strings.Builder
	for j := 0; j < n; j++ {
		nx := argmax(lg)
		if hasEot && nx == eot {
			break
		}
		sb.WriteString(tk.decode(nx))
		pos := len(ids) + j
		feed(nx, pos, true)
		report(pos, nx, argmax(lg))
	}
	return sb.String(), len(ids), nil
}

// Residual is the residual stream after feeding one token at position 0. Architecture proofs use it.
func Residual(path string, tok int) ([]float32, error) {
	m, err := load(path)
	if err != nil {
		return nil, err
	}
	return m.feed(&seq{}, tok, 0), nil
}

// NormStats is, for every RMSNorm (layer l < L: in_attn, in_o, in_ffn, in_down; layer L: final), the rms
// over the calibration tokens of each channel after its gain: what each BitLinear (and the head) reads.
func NormStats(path, tokPath string) (map[string][]float64, int, error) {
	m, tk, err := open(path, tokPath)
	if err != nil {
		return nil, 0, err
	}
	ss := map[string][]float64{}
	ntok := 0
	s := &seq{}
	s.tap = func(li int, st string, v []float32) {
		if !strings.HasPrefix(st, "in_") && st != "final" {
			return
		}
		k := fmt.Sprintf("%d/%s", li, st)
		a := ss[k]
		if a == nil {
			a = make([]float64, len(v))
			ss[k] = a
		}
		for i, x := range v {
			a[i] += float64(x) * float64(x)
		}
	}
	for _, ids := range calibration(tk) {
		*s = seq{tap: s.tap}
		for i, id := range ids {
			m.feed(s, id, i)
		}
		ntok += len(ids)
	}
	for _, a := range ss {
		for i := range a {
			a[i] = math.Sqrt(a[i] / float64(ntok))
		}
	}
	return ss, ntok, nil
}

// calibration is the token sequences every calibration pass runs: the eval texts and prompts and the
// page's chat prompt.
func calibration(tk *tok) [][]int {
	var c [][]int
	for _, tx := range append(append([]string{}, Texts...), Prompts...) {
		c = append(c, tk.encode(tx))
	}
	return append(c, tk.encodeChat(chatPrompt(tk)))
}

// chatPrompt is the page's system message and first question in its chat format, with the
// tokenizer's own turn end.
func chatPrompt(tk *tok) string {
	return "System: " + ChatSystem + tk.eot + "User: " + ChatUser + tk.eot + "Assistant: "
}

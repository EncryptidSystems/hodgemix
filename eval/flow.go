package eval

// Flow: the forward as a cochain on the model's cell complex (hodgemix volume.go lays the cells out).
//
// Every axis node (residual D, FFN F, attention Q and KV) is cut into cells; a BitLinear's weight is a
// 1-cochain on the edges input cell -> output cell, and a token puts a current on each edge: the sum
// of W_ij x_j over the edge's channels. The LUT fabric computes these currents as it runs: each
// weight's digit columns are laid out in cell order with a block size dividing the cell size, so every
// gathered table entry T_b[c] is the current from one block of one input cell into one row, and a row's
// output is the sum of its cells' currents (bit-identical to the plain layout: the integer sums are the
// same). The inputs are read through the norms (the metric); the residual stream is a conserved flow:
// at every residual node, the carried value plus the writes into it is the node's value.
// No paper equation.

import (
	"fmt"
	"sync"
)

// Cells is the cell layout of a model's axis nodes: D[c] is residual channel c's cell, and so on; Per
// is the cells per attention head.
type Cells struct {
	D, F, Q, KV          []int32
	ND, NF, NQ, NKV, Per int
}

// FlowLayer is one token's currents in one layer: Edge[role] is [row cells x col cells] (q, k, v, o,
// gate, up, down), Node[name] the cell sums of the vectors at each stage: D (residual entering the
// layer), Dm (after the attention write), Dout (after the FFN write), A (attention output, o's input),
// H (FFN hidden, down's input), and the outputs q, k, v, gate, up. Experts and Gates are the mixture
// selection on a gpt-oss layer (empty on a dense layer).
type FlowLayer struct {
	Edge    map[string][]float32
	Node    map[string][]float32
	Experts []int
	Gates   []float32
}

type flowLin struct {
	order   []int   // input column order (cell-major); nil for a float matrix
	cell    int     // channels per input cell
	colCell []int32 // input channel -> cell (float matrices)
	rowCell []int32 // output channel -> cell
	nr, nc  int
}

// relayout reorders a packed BitLinear's digit columns into cell order and repacks every limb with a
// block size dividing the cell size.
func (l *lin) relayout(order []int, cell int) {
	for li, lb := range l.limbs {
		d := make([]uint8, l.n*l.k)
		pw := make([]int, lb.K)
		pw[0] = 1
		for t := 1; t < lb.K; t++ {
			pw[t] = pw[t-1] * lb.r
		}
		for r := 0; r < l.n; r++ {
			codes := lb.code[r*lb.nb : (r+1)*lb.nb]
			for i, j := range order {
				d[r*l.k+i] = uint8(int(codes[j/lb.K]) / pw[j%lb.K] % lb.r)
			}
		}
		K := blockFor(lb.r, cell)
		nl := &limb{r: lb.r, K: K, nb: l.k / K, code: make([]uint8, l.n*(l.k/K)), s: lb.s}
		for r := 0; r < l.n; r++ {
			row := d[r*l.k : (r+1)*l.k]
			for b := 0; b < nl.nb; b++ {
				c, p := 0, 1
				for t := 0; t < K; t++ {
					c += int(row[b*K+t]) * p
					p *= lb.r
				}
				nl.code[r*nl.nb+b] = uint8(c)
			}
		}
		l.limbs[li] = nl
	}
}

// ordered is the act's int8 input in the given column order, with its tables.
func (a *act) ordered(order []int) (*input, float32) {
	in, xs := a.ternary()
	a.omu.Lock()
	defer a.omu.Unlock()
	if a.byOrder == nil {
		a.byOrder = map[*int]*input{}
	}
	if p := a.byOrder[&order[0]]; p != nil {
		return p, xs
	}
	q := make([]int8, len(order))
	for i, j := range order {
		q[i] = in.q[j]
	}
	p := newInput(q)
	a.byOrder[&order[0]] = p
	return p, xs
}

// runFlow is run that also adds each edge's current into flow [nr x nc].
func (l *lin) runFlow(fl *flowLin, a *act, out []float32, accum bool, flow []float32) {
	if l.mx != nil {
		l.mx.gather(a.x, out, accum, flow, fl.colCell, fl.rowCell, fl.nc)
		for i, v := range l.b {
			out[i] += v
		}
		return
	}
	const P = 8
	parts := make([][]float32, P)
	var wg sync.WaitGroup
	if l.f != nil {
		for p := 0; p < P; p++ {
			wg.Add(1)
			go func(p int) {
				defer wg.Done()
				fb := make([]float32, fl.nr*fl.nc)
				cs := make([]float32, fl.nc)
				for r := p; r < l.n; r += P {
					for i := range cs {
						cs[i] = 0
					}
					row := l.f[r*l.k : (r+1)*l.k]
					var y float32
					for j, x := range a.x {
						v := x * row[j]
						y += v
						cs[fl.colCell[j]] += v
					}
					if accum {
						out[r] += y
					} else {
						out[r] = y
					}
					rc := int(fl.rowCell[r]) * fl.nc
					for c, v := range cs {
						fb[rc+c] += v
					}
				}
				parts[p] = fb
			}(p)
		}
	} else {
		in, xs := a.ordered(fl.order)
		tbs := make([]*luts, len(l.limbs))
		for i, lb := range l.limbs {
			tbs[i] = in.tables(lb.r, lb.K)
		}
		for p := 0; p < P; p++ {
			wg.Add(1)
			go func(p int) {
				defer wg.Done()
				fb := make([]float32, fl.nr*fl.nc)
				cs := make([]int32, fl.nc)
				for r := p; r < l.n; r += P {
					var v float32
					rc := int(fl.rowCell[r]) * fl.nc
					for i, lb := range l.limbs {
						for c := range cs {
							cs[c] = 0
						}
						t, size, nb := tbs[i].t, tbs[i].size, lb.nb
						codes := lb.code[r*nb : (r+1)*nb]
						per := fl.cell / lb.K // blocks per cell
						var acc int32
						for b, c := range codes {
							e := t[b*size+int(c)]
							acc += e
							cs[b/per] += e
						}
						v += float32(acc) * xs * lb.s
						for c, e := range cs {
							fb[rc+c] += float32(e) * xs * lb.s
						}
					}
					if accum {
						out[r] += v
					} else {
						out[r] = v
					}
				}
				parts[p] = fb
			}(p)
		}
	}
	wg.Wait()
	for _, fb := range parts {
		for i, v := range fb {
			flow[i] += v
		}
	}
	for i, v := range l.b {
		out[i] += v
	}
}

// cellSums is v summed over each cell of an axis.
func cellSums(v []float32, cell []int32, n int) []float32 {
	o := make([]float32, n)
	for i, x := range v {
		o[cell[i]] += x
	}
	return o
}

// orderOf is an axis' channels in cell-major order and the cell size; every cell must hold the same
// number of channels.
func orderOf(cell []int32, n int) ([]int, int, bool) {
	count := make([]int, n)
	for _, c := range cell {
		count[c]++
	}
	for _, k := range count {
		if k != count[0] || k == 0 {
			return nil, 0, false
		}
	}
	order := make([]int, 0, len(cell))
	for c := 0; c < n; c++ {
		for j, x := range cell {
			if int(x) == c {
				order = append(order, j)
			}
		}
	}
	return order, count[0], true
}

// addBiasFlow spreads a per-channel bias uniformly across that channel's input cells, binned into
// the output cell. A bias has no input channel; the row sum stays the biased output, which is what
// the residual actually receives.
func addBiasFlow(e, b []float32, rowCell []int32, nr, nc int) {
	if len(b) == 0 || nc == 0 || len(rowCell) == 0 {
		return
	}
	share := float32(1) / float32(nc)
	for i, v := range b {
		r := int(rowCell[i])
		if r < 0 || r >= nr {
			continue
		}
		s := v * share
		base := r * nc
		for c := 0; c < nc; c++ {
			e[base+c] += s
		}
	}
}

// scaleEdge makes an output cell's currents sum to the value that was actually added (a norm
// applied after the linear, as in Gemma 2/3). The split across input cells is kept.
func (m *model) scaleEdge(li int, role string, before, after []float32, cell []int32, n int) {
	if m.rec == nil || n == 0 {
		return
	}
	e := m.rec[li].Edge[role]
	if e == nil {
		return
	}
	sb, sa := cellSums(before, cell, n), cellSums(after, cell, n)
	nc := len(e) / n
	for r := 0; r < n; r++ {
		base := r * nc
		if sb[r] == 0 {
			e[base] += sa[r]
			continue
		}
		sc := sa[r] / sb[r]
		for c := 0; c < nc; c++ {
			e[base+c] *= sc
		}
	}
}

// lin runs l, recording its currents when the model is recording.
func (m *model) lin(li int, role string, l *lin, a *act, out []float32, accum bool) {
	if m.rec == nil {
		l.run(a, out, accum)
		return
	}
	fl := m.flows[li][role]
	e := m.rec[li].Edge[role]
	if e == nil {
		e = make([]float32, fl.nr*fl.nc)
		m.rec[li].Edge[role] = e
	}
	l.runFlow(fl, a, out, accum, e)
	addBiasFlow(e, l.b, fl.rowCell, fl.nr, fl.nc)
}

// node records v's cell sums as layer li's node name, when the model is recording.
func (m *model) node(li int, name string, v []float32, cell []int32, n int) {
	if m.rec != nil {
		m.rec[li].Node[name] = cellSums(v, cell, n)
	}
}

// instrument lays every layer's linears out on the cells.
func (m *model) instrument(c Cells) error {
	if len(c.D) != m.D || len(c.F) != m.F || len(c.Q) != m.H*m.HD || len(c.KV) != m.KVH*m.HD {
		return errorf("cells cover D %d F %d Q %d KV %d channels, the model has %d %d %d %d",
			len(c.D), len(c.F), len(c.Q), len(c.KV), m.D, m.F, m.H*m.HD, m.KVH*m.HD)
	}
	type ax struct {
		cell  []int32
		n     int
		order []int
		size  int
	}
	axes := map[string]*ax{"D": {cell: c.D, n: c.ND}, "F": {cell: c.F, n: c.NF}, "Q": {cell: c.Q, n: c.NQ}, "KV": {cell: c.KV, n: c.NKV}}
	for name, a := range axes {
		o, size, ok := orderOf(a.cell, a.n)
		if !ok {
			return errorf("axis %s: every cell must hold the same number of channels", name)
		}
		a.order, a.size = o, size
	}
	io := map[string][2]string{"q": {"Q", "D"}, "k": {"KV", "D"}, "v": {"KV", "D"}, "o": {"D", "Q"},
		"gate": {"F", "D"}, "up": {"F", "D"}, "down": {"D", "F"}}
	m.cells = c
	m.flows = make([]map[string]*flowLin, m.L)
	for li := range m.ls {
		Lr := &m.ls[li]
		m.flows[li] = map[string]*flowLin{}
		roles := map[string][]*lin{
			"q": {Lr.q}, "k": {Lr.k}, "v": {Lr.v}, "o": {Lr.o},
			"gate": {Lr.g}, "up": {Lr.u}, "down": {Lr.d},
		}
		if len(Lr.experts) > 0 {
			g, u, d := make([]*lin, len(Lr.experts)), make([]*lin, len(Lr.experts)), make([]*lin, len(Lr.experts))
			for i := range Lr.experts {
				g[i], u[i], d[i] = Lr.experts[i].gate, Lr.experts[i].up, Lr.experts[i].down
			}
			roles["gate"], roles["up"], roles["down"] = g, u, d
		}
		if Lr.upd != nil {
			roles["gate"] = append(roles["gate"], Lr.upd.gate)
			roles["up"] = append(roles["up"], Lr.upd.up)
			roles["down"] = append(roles["down"], Lr.upd.down)
		}
		for role, ls := range roles {
			r, in := axes[io[role][0]], axes[io[role][1]]
			fl := &flowLin{cell: in.size, colCell: in.cell, rowCell: r.cell, nr: r.n, nc: in.n}
			for _, l := range ls {
				if l == nil {
					return errorf("layer %d role %s has no weight", li, role)
				}
				if l.mx != nil {
					continue // the nibble table is already addressed by input channel
				}
				if l.f == nil {
					fl.order = in.order
					l.relayout(in.order, in.size)
				}
			}
			m.flows[li][role] = fl
		}
	}
	return nil
}

// Flow feeds text (in the page's chat format when chat) through the model laid out on the cells and
// calls each with every token's currents.
func Flow(modelPath, tokPath, text string, chat bool, c Cells, each func(pos, tok int, layers []FlowLayer)) error {
	m, tk, err := open(modelPath, tokPath)
	if err != nil {
		return err
	}
	if err := m.instrument(c); err != nil {
		return err
	}
	ids := tk.encode(text)
	if chat {
		ids = tk.encodeChat(text)
	}
	s := &seq{}
	for pos, id := range ids {
		m.rec = make([]FlowLayer, m.L)
		for li := range m.rec {
			m.rec[li] = FlowLayer{Edge: map[string][]float32{}, Node: map[string][]float32{}}
		}
		m.feed(s, id, pos)
		each(pos, id, m.rec)
	}
	m.rec = nil
	return nil
}

func errorf(f string, a ...any) error { return fmt.Errorf(f, a...) }

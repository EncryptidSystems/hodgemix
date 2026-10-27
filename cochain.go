package main

// The forward's cochain complex and its Hodge split (hodgemix --config RUN.yaml --flow MODEL OUT "text").
//
// 0-cells, per layer l: the residual cells D_l (entering the layer), Dm_l (after the attention write)
// and D_{l+1} (after the FFN write, shared with layer l+1), and the cells of q, k, v, A (attention
// output), gate, up and H (FFN hidden). 1-cells carry one token's currents (eval/flow.go):
//
//	q, k, v      D_l  -> q, k, v     every voxel, W x through the input norm (the metric)
//	attention    v    -> A           A's cell sums: attention carries v, mixed over the context
//	o            A    -> Dm_l        every voxel
//	carry        D_l  -> Dm_l        the residual carried past the attention block
//	gate, up     Dm_l -> gate, up    every voxel
//	gating       up   -> H           H's cell sums: the gate scales up's transport
//	down         H    -> D_{l+1}     every voxel
//	carry        Dm_l -> D_{l+1}     the residual carried past the FFN block
//
// The metric is the model's own: each (layer, class) block of currents is measured in units of its
// rms for the token (every sublayer input passes an RMSNorm, so the model is scale-free per block).
// On those currents, d0 phi(e) = phi(head) - phi(tail); the gradient part of f is d0 phi with L0 phi = d0^T f
// (L0 = d0^T d0, the graph Laplacian, conjugate gradient), the rest f - d0 phi is the cycle part:
// divergence-free, so d0^T of it is 0 (the Stokes check). Each residual cell also carries two 2-cells,
// digons of the net sublayer write against the residual carry (attention, then feed-forward). The
// write edge's current is that net write. Circulation around a face is write minus carry, and the
// harmonic part is the divergence-free part with that circulation removed.
// No paper equation.

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"

	"hodgemix/eval"
)

// Edge classes.
const (
	ecQ = iota
	ecK
	ecV
	ecAttn
	ecO
	ecCarryA
	ecGate
	ecUp
	ecGating
	ecDown
	ecCarryF
	ecWriteA
	ecWriteF
	ecN
)

var ecName = [ecN]string{"q", "k", "v", "attention", "o", "carry attn", "gate", "up", "gating", "down", "carry ffn", "write attn", "write ffn"}

// complex is the 1-skeleton: every edge's tail and head node, its class and layer.
type flowComplex struct {
	nodes      int
	tail, head []int32
	class      []uint8
	layer      []int16
	off        []int // layer l's node block starts at off[l] (D_l first); off[L] is D_L
	face       []flowFace
	c          eval.Cells
	L          int
}

// flowFace is one residual digon: the net write edge against the carry edge of the same cell.
// kind 0 is attention, kind 1 is the feed-forward. Circulation is f(write) - f(carry).
type flowFace struct {
	write, carry int
	layer        int16
	cell         int32
	kind         uint8
}

// node block of one layer: D, Dm, q, k, v, A, gate, up, H (D_{l+1} is the next block's D).
func (c *flowComplex) blk(l int) (D, Dm, q, k, v, A, g, u, H int) {
	o := c.off[l]
	D = o
	Dm = D + c.c.ND
	q = Dm + c.c.ND
	k = q + c.c.NQ
	v = k + c.c.NKV
	A = v + c.c.NKV
	g = A + c.c.NQ
	u = g + c.c.NF
	H = u + c.c.NF
	return
}

func newFlowComplex(L int, cells eval.Cells) *flowComplex {
	c := &flowComplex{c: cells, L: L}
	per := 2*cells.ND + 2*cells.NQ + 2*cells.NKV + 3*cells.NF
	c.off = make([]int, L+1)
	for l := 0; l <= L; l++ {
		c.off[l] = l * per
	}
	c.nodes = L*per + cells.ND
	add := func(t, h int, cl, l int) {
		c.tail, c.head = append(c.tail, int32(t)), append(c.head, int32(h))
		c.class, c.layer = append(c.class, uint8(cl)), append(c.layer, int16(l))
	}
	vox := func(cl, l, tail, head, nr, nc int) {
		for r := 0; r < nr; r++ {
			for col := 0; col < nc; col++ {
				add(tail+col, head+r, cl, l)
			}
		}
	}
	G := cells.NQ / cells.NKV // q cells per kv cell group
	for l := 0; l < L; l++ {
		D, Dm, q, k, v, A, g, u, H := c.blk(l)
		Dn := c.off[l+1]
		vox(ecQ, l, D, q, cells.NQ, cells.ND)
		vox(ecK, l, D, k, cells.NKV, cells.ND)
		vox(ecV, l, D, v, cells.NKV, cells.ND)
		for i := 0; i < cells.NQ; i++ { // q cell i = head h, sub s reads kv head h/(H/KVH), sub s
			h, s := i/cells.Per, i%cells.Per
			add(v+(h/(G))*cells.Per+s, A+i, ecAttn, l)
		}
		vox(ecO, l, A, Dm, cells.ND, cells.NQ)
		for i := 0; i < cells.ND; i++ {
			add(D+i, Dm+i, ecCarryA, l)
		}
		vox(ecGate, l, Dm, g, cells.NF, cells.ND)
		vox(ecUp, l, Dm, u, cells.NF, cells.ND)
		for i := 0; i < cells.NF; i++ {
			add(u+i, H+i, ecGating, l)
		}
		vox(ecDown, l, H, Dn, cells.ND, cells.NF)
		for i := 0; i < cells.ND; i++ {
			add(Dm+i, Dn+i, ecCarryF, l)
		}
		// Net writes, parallel to the carries. Their currents are the row sums of o and down.
		for i := 0; i < cells.ND; i++ {
			add(D+i, Dm+i, ecWriteA, l)
		}
		for i := 0; i < cells.ND; i++ {
			add(Dm+i, Dn+i, ecWriteF, l)
		}
	}
	c.pairFaces()
	return c
}

// pairFaces matches each write edge with the carry it runs against, in cell order.
func (c *flowComplex) pairFaces() {
	type key struct {
		layer int16
		class uint8
	}
	idx := map[key][]int{}
	for e := range c.tail {
		k := key{c.layer[e], c.class[e]}
		idx[k] = append(idx[k], e)
	}
	for l := 0; l < c.L; l++ {
		pair := func(carry, write uint8, kind uint8) {
			ce, we := idx[key{int16(l), carry}], idx[key{int16(l), write}]
			n := len(ce)
			if len(we) < n {
				n = len(we)
			}
			for i := 0; i < n; i++ {
				c.face = append(c.face, flowFace{write: we[i], carry: ce[i], layer: int16(l), cell: int32(i), kind: kind})
			}
		}
		pair(ecCarryA, ecWriteA, 0)
		pair(ecCarryF, ecWriteF, 1)
	}
}

// currents is the flow 1-cochain of one token, in the complex's edge order.
func (c *flowComplex) currents(ls []eval.FlowLayer) ([]float64, error) {
	f := make([]float64, 0, len(c.tail))
	put := func(v []float32) {
		for _, x := range v {
			f = append(f, float64(x))
		}
	}
	// putSums is the net current into each output cell, the write edge of that cell's face.
	putSums := func(e []float32, nr, nc int) {
		for r := 0; r < nr; r++ {
			var s float64
			if e != nil && nc > 0 {
				base := r * nc
				for col := 0; col < nc && base+col < len(e); col++ {
					s += float64(e[base+col])
				}
			}
			f = append(f, s)
		}
	}
	for l := 0; l < c.L; l++ {
		e, n := ls[l].Edge, ls[l].Node
		for _, k := range []string{"q", "k", "v", "o", "gate", "up", "down"} {
			if e[k] == nil {
				return nil, fmt.Errorf("layer %d: no %s currents", l, k)
			}
		}
		for _, k := range []string{"D", "Dm", "A", "H"} {
			if n[k] == nil {
				return nil, fmt.Errorf("layer %d: no %s node", l, k)
			}
		}
		put(e["q"])
		put(e["k"])
		put(e["v"])
		put(n["A"])
		put(e["o"])
		put(n["D"])
		put(e["gate"])
		put(e["up"])
		put(n["H"])
		put(e["down"])
		put(n["Dm"])
		putSums(e["o"], c.c.ND, c.c.NQ)
		putSums(e["down"], c.c.ND, c.c.NF)
	}
	if len(f) != len(c.tail) {
		return nil, fmt.Errorf("%d currents for %d edges", len(f), len(c.tail))
	}
	return f, nil
}

// conservation is the largest mismatch, over every residual node and BitLinear output cell, between
// what flows in and the node's value, relative to the sum of the magnitudes flowing in (the scale of
// float32 rounding: currents that cancel leave a small value but not a small rounding error).
func conservation(cells eval.Cells, ls []eval.FlowLayer) float64 {
	var worst float64
	chk := func(in, abs, val float64) {
		worst = math.Max(worst, math.Abs(in-val)/math.Max(abs, 1e-30))
	}
	rows := func(e []float32, nc int, r int) (s, a float64) {
		for c := 0; c < nc; c++ {
			x := float64(e[r*nc+c])
			s, a = s+x, a+math.Abs(x)
		}
		return
	}
	for _, L := range ls {
		e, n := L.Edge, L.Node
		for i := 0; i < cells.ND; i++ {
			s, a := rows(e["o"], cells.NQ, i)
			chk(float64(n["D"][i])+s, math.Abs(float64(n["D"][i]))+a, float64(n["Dm"][i]))
			s, a = rows(e["down"], cells.NF, i)
			chk(float64(n["Dm"][i])+s, math.Abs(float64(n["Dm"][i]))+a, float64(n["Dout"][i]))
		}
		for _, k := range []struct {
			e      string
			nr, nc int
		}{{"q", cells.NQ, cells.ND}, {"k", cells.NKV, cells.ND}, {"v", cells.NKV, cells.ND},
			{"gate", cells.NF, cells.ND}, {"up", cells.NF, cells.ND}} {
			for r := 0; r < k.nr; r++ {
				s, a := rows(e[k.e], k.nc, r)
				chk(s, a, float64(n[k.e][r]))
			}
		}
	}
	return worst
}

// div is d0^T f: what flows into each node minus what flows out.
func (c *flowComplex) div(f []float64) []float64 {
	d := make([]float64, c.nodes)
	for e, x := range f {
		d[c.head[e]] += x
		d[c.tail[e]] -= x
	}
	return d
}

// grad is d0 phi.
func (c *flowComplex) grad(phi []float64) []float64 {
	g := make([]float64, len(c.tail))
	for e := range g {
		g[e] = phi[c.head[e]] - phi[c.tail[e]]
	}
	return g
}

// hodge splits f into its gradient part d0 phi and its cycle part, by conjugate gradient on
// L0 phi = d0^T f (Jacobi preconditioned, mean-zero gauge); it returns phi and the cycle part.
func (c *flowComplex) hodge(f []float64, tol float64, maxIt int) (phi, cyc []float64, it int) {
	b := c.div(f)
	deg := make([]float64, c.nodes)
	for e := range c.tail {
		deg[c.head[e]]++
		deg[c.tail[e]]++
	}
	center := func(v []float64) {
		var s float64
		for _, x := range v {
			s += x
		}
		s /= float64(len(v))
		for i := range v {
			v[i] -= s
		}
	}
	lap := func(x []float64) []float64 { return c.div(c.grad(x)) }
	dot := func(a, b []float64) float64 {
		var s float64
		for i := range a {
			s += a[i] * b[i]
		}
		return s
	}
	// L0 phi = d0^T f: div is d0^T, lap = div . grad = d0^T d0.
	phi = make([]float64, c.nodes)
	r := append([]float64(nil), b...)
	center(r)
	z := make([]float64, c.nodes)
	for i := range z {
		z[i] = r[i] / math.Max(deg[i], 1)
	}
	p := append([]float64(nil), z...)
	rz := dot(r, z)
	bn := math.Sqrt(dot(b, b))
	for it = 0; it < maxIt && math.Sqrt(dot(r, r)) > tol*bn; it++ {
		Ap := lap(p)
		a := rz / dot(p, Ap)
		for i := range phi {
			phi[i] += a * p[i]
			r[i] -= a * Ap[i]
		}
		for i := range z {
			z[i] = r[i] / math.Max(deg[i], 1)
		}
		rz2 := dot(r, z)
		for i := range p {
			p[i] = z[i] + rz2/rz*p[i]
		}
		rz = rz2
	}
	center(phi)
	g := c.grad(phi)
	cyc = make([]float64, len(f))
	for e := range f {
		cyc[e] = f[e] - g[e]
	}
	return phi, cyc, it
}

// circ is d1 f: write minus carry on each face.
func (c *flowComplex) circ(f []float64) []float64 {
	o := make([]float64, len(c.face))
	for i, fc := range c.face {
		o[i] = f[fc.write] - f[fc.carry]
	}
	return o
}

// harmonic removes face circulation from a divergence-free cochain. On a digon the coexact part is
// (γ/2, -γ/2) with γ = write - carry, and the harmonic part is the mean of the two edges, which has
// circulation 0. Edges off a face are already harmonic.
func (c *flowComplex) harmonic(cyc []float64) []float64 {
	h := append([]float64(nil), cyc...)
	for _, fc := range c.face {
		if fc.write >= len(h) || fc.carry >= len(h) {
			continue
		}
		h[fc.write] = 0.5 * (cyc[fc.write] + cyc[fc.carry])
		h[fc.carry] = h[fc.write]
	}
	return h
}

// flowReport is what --flow measured over a text.
type flowReport struct {
	Tokens       int
	Family       string
	Conservation float64         // largest relative in-vs-value mismatch at any node
	Stokes       float64         // largest ||d0^T cycle|| / ||d0^T f||
	CGIters      int             // most conjugate-gradient iterations a token took
	Energy       [ecN][3]float64 // per class: |f|^2, |grad|^2, |cycle|^2, summed over tokens
	Layer        [][3]float64    // per layer: the same
	Nodes, Edges int
	Faces        int       // residual digons
	Harmonic     float64   // |h|^2 summed over tokens
	Cycle        float64   // |cycle|^2 summed over tokens
	HarmCirc     float64   // largest |d1 h|
	Experts      []int     // last token, the layer that fired experts
	Gates        []float32 // those experts' router weights
	Signs        []int8    // last token, layer-major: attention sign, feed-forward sign
	Keeps        []int8    // last token, layer-major: 1 when that write was added
}

// flowRun feeds text through the model laid out on its cells, writes every token's currents and
// potential to outPath (with the complex), and returns the measurements.
func flowRun(modelPath, outPath, text string) (*flowReport, error) {
	st, err := OpenST(modelPath)
	if err != nil {
		return nil, err
	}
	g, err := metaOf(st)
	if err != nil {
		return nil, err
	}
	axes, err := cellLayout(st, g)
	if err != nil {
		return nil, err
	}
	cells := evalCells(axes)
	cx := newFlowComplex(g.Layers, cells)
	fam, err := familyOf(modelPath)
	if err != nil {
		return nil, err
	}
	logf("FLOW  %s (%s): %d nodes, %d edges, %d faces per token", modelPath, fam, cx.nodes, len(cx.tail), len(cx.face))
	W, err := NewSTW(outPath, [][2]string{{"format", "pt"}, {"flow_of", modelPath}, {"text", text}})
	if err != nil {
		return nil, err
	}
	ti, hi, cl, ly := make([]int32, len(cx.tail)), make([]int32, len(cx.tail)), make([]byte, len(cx.tail)), make([]int32, len(cx.tail))
	for e := range cx.tail {
		ti[e], hi[e], cl[e], ly[e] = cx.tail[e], cx.head[e], cx.class[e], int32(cx.layer[e])
	}
	W.AddI32("edge.tail", []int{len(ti)}, ti)
	W.AddI32("edge.head", []int{len(hi)}, hi)
	W.Add("edge.class", "U8", []int{len(cl)}, cl)
	W.AddI32("edge.layer", []int{len(ly)}, ly)
	fw, fc, fk, fl := make([]int32, len(cx.face)), make([]int32, len(cx.face)), make([]byte, len(cx.face)), make([]int32, len(cx.face))
	for i, f := range cx.face {
		fw[i], fc[i], fk[i], fl[i] = int32(f.write), int32(f.carry), f.kind, int32(f.layer)
	}
	W.AddI32("face.write", []int{len(fw)}, fw)
	W.AddI32("face.carry", []int{len(fc)}, fc)
	W.Add("face.kind", "U8", []int{len(fk)}, fk)
	W.AddI32("face.layer", []int{len(fl)}, fl)
	for k, a := range axes {
		W.AddI32("cell."+map[axisKind]string{axD: "D", axF: "F", axQ: "Q", axKV: "KV"}[k], []int{a.n}, a.cell)
	}
	rep := &flowReport{Family: fam, Layer: make([][3]float64, g.Layers), Nodes: cx.nodes, Edges: len(cx.tail), Faces: len(cx.face)}
	var ferr error
	err = eval.Flow(modelPath, "", text, false, cells, func(pos, tok int, ls []eval.FlowLayer) {
		if ferr != nil {
			return
		}
		f, err := cx.currents(ls)
		if err != nil {
			ferr = err
			return
		}
		rep.Conservation = math.Max(rep.Conservation, conservation(cells, ls))
		// The metric: each (class, layer) block of edges in units of its own rms this token, as the
		// model's norms measure every sublayer input; the split runs on the unit-free currents.
		scale := cx.blockScales(f)
		fn := make([]float64, len(f))
		for e, x := range f {
			if s := scale[cx.block(e)]; s > 0 {
				fn[e] = x / s
			}
		}
		phi, cyc, it := cx.hodge(fn, 1e-9, 4000)
		rep.CGIters = max(rep.CGIters, it)
		harm := cx.harmonic(cyc)
		var h2, c2 float64
		for i := range cyc {
			h2 += harm[i] * harm[i]
			c2 += cyc[i] * cyc[i]
		}
		rep.Harmonic += h2
		rep.Cycle += c2
		for _, g := range cx.circ(harm) {
			rep.HarmCirc = math.Max(rep.HarmCirc, math.Abs(g))
		}
		var dn, cn float64
		for _, x := range cx.div(fn) {
			dn += x * x
		}
		for _, x := range cx.div(cyc) {
			cn += x * x
		}
		if dn > 0 {
			rep.Stokes = math.Max(rep.Stokes, math.Sqrt(cn/dn))
		}
		gr := cx.grad(phi)
		for e := range fn {
			c, l := cx.class[e], cx.layer[e]
			rep.Energy[c][0] += fn[e] * fn[e]
			rep.Energy[c][1] += gr[e] * gr[e]
			rep.Energy[c][2] += cyc[e] * cyc[e]
			rep.Layer[l][0] += fn[e] * fn[e]
			rep.Layer[l][1] += gr[e] * gr[e]
			rep.Layer[l][2] += cyc[e] * cyc[e]
		}
		f32, p32, s32 := make([]float32, len(f)), make([]float32, len(phi)), make([]float32, len(scale))
		for i, x := range f {
			f32[i] = float32(x)
		}
		for i, x := range phi {
			p32[i] = float32(x)
		}
		for i, x := range scale {
			s32[i] = float32(x)
		}
		W.AddF32(fmt.Sprintf("t%d.f", pos), []int{len(f32)}, f32)
		W.AddF32(fmt.Sprintf("t%d.scale", pos), []int{g.Layers, ecN}, s32)
		W.AddF32(fmt.Sprintf("t%d.phi", pos), []int{len(p32)}, p32)
		for i := len(ls) - 1; i >= 0; i-- {
			if len(ls[i].Experts) > 0 {
				rep.Experts = append([]int(nil), ls[i].Experts...)
				rep.Gates = append([]float32(nil), ls[i].Gates...)
				break
			}
		}
		rep.Signs, rep.Keeps = rep.Signs[:0], rep.Keeps[:0]
		sg, kp, wg := make([]int32, 0, len(ls)*2), make([]int32, 0, len(ls)*2), make([]float32, 0, len(ls)*2)
		for _, L := range ls {
			rep.Signs = append(rep.Signs, L.Sign[0], L.Sign[1])
			rep.Keeps = append(rep.Keeps, L.Keep[0], L.Keep[1])
			sg = append(sg, int32(L.Sign[0]), int32(L.Sign[1]))
			kp = append(kp, int32(L.Keep[0]), int32(L.Keep[1]))
			wg = append(wg, L.Wedge[0], L.Wedge[1])
		}
		W.AddI32(fmt.Sprintf("t%d.sign", pos), []int{len(ls), 2}, sg)
		W.AddI32(fmt.Sprintf("t%d.keep", pos), []int{len(ls), 2}, kp)
		W.AddF32(fmt.Sprintf("t%d.wedge", pos), []int{len(ls), 2}, wg)
		rep.Tokens++
		logf("  token %d (id %d): conservation %.1e, Stokes %.1e, CG %d iterations, %s", pos, tok, conservation(cells, ls), math.Sqrt(cn/math.Max(dn, 1e-300)), it, wedgeLeaf(ls))
	})
	if err == nil {
		err = ferr
	}
	if err != nil {
		return nil, err
	}
	if err := W.Close(nil); err != nil {
		return nil, err
	}
	return rep, nil
}

func familyOf(modelPath string) (string, error) {
	cp := eval.FilesOf(modelPath).Config
	if cp == "" {
		return "", fmt.Errorf("%s: no config set", modelPath)
	}
	b, err := os.ReadFile(cp)
	if err != nil {
		return "", err
	}
	var cfg map[string]any
	if err := json.Unmarshal(b, &cfg); err != nil {
		return "", fmt.Errorf("%s: %v", cp, err)
	}
	a, err := eval.Classify(cfg, 0)
	if err != nil {
		return "", err
	}
	return a.Family, nil
}

func manifoldSetup(modelPath string) (eval.Cells, *flowComplex, string, error) {
	st, err := OpenST(modelPath)
	if err != nil {
		return eval.Cells{}, nil, "", err
	}
	g, err := metaOf(st)
	if err != nil {
		return eval.Cells{}, nil, "", err
	}
	axes, err := cellLayout(st, g)
	if err != nil {
		return eval.Cells{}, nil, "", err
	}
	fam, err := familyOf(modelPath)
	if err != nil {
		return eval.Cells{}, nil, "", err
	}
	cells := evalCells(axes)
	return cells, newFlowComplex(g.Layers, cells), fam, nil
}

// manifoldLine is one token of the cochain: conservation, the Hodge split, the hottest layer,
// the residual cell that moved most, and the experts that fired.
func manifoldLine(family string, pos int, piece string, cells eval.Cells, cx *flowComplex, ls []eval.FlowLayer) (string, error) {
	f, err := cx.currents(ls)
	if err != nil {
		return "", err
	}
	cons := conservation(cells, ls)
	scale := cx.blockScales(f)
	fn := make([]float64, len(f))
	for e, x := range f {
		if s := scale[cx.block(e)]; s > 0 {
			fn[e] = x / s
		}
	}
	phi, cyc, _ := cx.hodge(fn, 1e-9, 4000)
	var dn, cn, f2, g2 float64
	for _, x := range cx.div(fn) {
		dn += x * x
	}
	for _, x := range cx.div(cyc) {
		cn += x * x
	}
	gr := cx.grad(phi)
	layerE := make([]float64, cx.L)
	for e := range fn {
		f2 += fn[e] * fn[e]
		g2 += gr[e] * gr[e]
		layerE[cx.layer[e]] += fn[e] * fn[e]
	}
	stokes := 0.0
	if dn > 0 {
		stokes = math.Sqrt(cn / dn)
	}
	grad := 0.0
	if f2 > 0 {
		grad = 100 * g2 / f2
	}
	hot := 0
	for l, e := range layerE {
		if e > layerE[hot] {
			hot = l
		}
	}
	share := 0.0
	if f2 > 0 {
		share = 100 * layerE[hot] / f2
	}
	last := ls[len(ls)-1]
	cell, delta := 0, 0.0
	if d, ok := last.Node["D"]; ok {
		dout := last.Node["Dout"]
		for i := range d {
			if i >= len(dout) {
				break
			}
			del := float64(dout[i] - d[i])
			if math.Abs(del) > math.Abs(delta) {
				cell, delta = i, del
			}
		}
	}
	ex, gt := last.Experts, last.Gates
	if len(ls[hot].Experts) > 0 {
		ex, gt = ls[hot].Experts, ls[hot].Gates
	}
	b := fmt.Sprintf("manifold %s pos %d next %q conservation %.1e Stokes %.1e gradient %.1f%% layer %d holds %.0f%% of |f|^2 residual cell %d Δ %+.2f",
		family, pos, piece, cons, stokes, grad, hot, share, cell, delta)
	if len(ex) > 0 {
		parts := make([]string, len(ex))
		for i, e := range ex {
			w := float32(0)
			if i < len(gt) {
				w = gt[i]
			}
			parts[i] = fmt.Sprintf("%d@%.2f", e, w)
		}
		b += " experts " + strings.Join(parts, " ")
	}
	if len(ls) > 0 {
		b += " wedge " + wedgeLeaf(ls)
	}
	return b, nil
}

func branchWord(sign, keep int8) string {
	if keep == 1 {
		return "kept"
	}
	if sign < 0 {
		return "refused-"
	}
	return "refused0"
}

func wedgeLeaf(ls []eval.FlowLayer) string {
	parts := make([]string, len(ls))
	for i, L := range ls {
		parts[i] = fmt.Sprintf("L%d attn %s, ffn %s", i, branchWord(L.Sign[0], L.Keep[0]), branchWord(L.Sign[1], L.Keep[1]))
	}
	return strings.Join(parts, "; ")
}

func logFlow(r *flowReport) {
	if r.Family != "" {
		logf("  family %s", r.Family)
	}
	logf("  %d tokens; conservation %.1e, Stokes %.1e, CG <= %d iterations", r.Tokens, r.Conservation, r.Stokes, r.CGIters)
	share := 0.0
	if r.Cycle > 0 {
		share = 100 * r.Harmonic / r.Cycle
	}
	logf("  %d faces; harmonic %.1f%% of |cycle|^2, largest face circulation of h %.1e", r.Faces, share, r.HarmCirc)
	if len(r.Signs) > 0 {
		leaf := make([]string, 0, len(r.Signs)/2)
		for i := 0; i+1 < len(r.Signs) && i+1 < len(r.Keeps); i += 2 {
			leaf = append(leaf, fmt.Sprintf("L%d attn %s, ffn %s", i/2, branchWord(r.Signs[i], r.Keeps[i]), branchWord(r.Signs[i+1], r.Keeps[i+1])))
		}
		logf("  last token %s", strings.Join(leaf, "; "))
	}
	// The gradient space is d0's range, rank nodes - 1 (connected): isotropic random currents put
	// that fraction of their energy in it, the reference for how potential-driven a flow is.
	logf("  random currents on this complex would be %.2f%% gradient (rank d0 / edges = %d / %d)", 100*float64(r.Nodes-1)/float64(r.Edges), r.Nodes-1, r.Edges)
	logf("  %-12s %14s %8s %8s", "class", "|f|^2", "grad", "cycle")
	for c := 0; c < ecN; c++ {
		e := r.Energy[c]
		if e[0] > 0 {
			logf("  %-12s %14.4g %7.1f%% %7.1f%%", ecName[c], e[0], 100*e[1]/e[0], 100*e[2]/e[0])
		}
	}
	logf("  %-6s %14s %8s %8s", "layer", "|f|^2", "grad", "cycle")
	for l, e := range r.Layer {
		if e[0] > 0 {
			logf("  %-6d %14.4g %7.1f%% %7.1f%%", l, e[0], 100*e[1]/e[0], 100*e[2]/e[0])
		}
	}
}

// block is edge e's (layer, class) block index, layer-major.
func (c *flowComplex) block(e int) int { return int(c.layer[e])*ecN + int(c.class[e]) }

// blockScales is the rms current of every (layer, class) block.
func (c *flowComplex) blockScales(f []float64) []float64 {
	ss, n := make([]float64, c.L*ecN), make([]float64, c.L*ecN)
	for e, x := range f {
		b := c.block(e)
		ss[b] += x * x
		n[b]++
	}
	for b := range ss {
		if n[b] > 0 {
			ss[b] = math.Sqrt(ss[b] / n[b])
		}
	}
	return ss
}

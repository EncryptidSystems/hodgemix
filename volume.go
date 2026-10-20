package main

// Weight volumes: a model's weights as one 3-D matrix per BitLinear role, layer x output cell x input
// cell, on the model's own MRH index (every run writes it beside its output as volume.safetensors).
//
// Channels are grouped into cells along each axis node, and the node is shared: the residual axis (D)
// has one order for every tensor that reads or writes it (q, k, v and gate, up columns; o and down
// rows), the FFN axis (F) one for gate and up rows and down columns, so a residual channel sits at the
// same coordinate in every volume. That order is the MRH pop order (Eq.8) of the axis' married key:
// each channel's Eq.(7) key averaged over every tensor on the node. The attention axes (Q, KV) keep
// heads whole, head-major, a fixed number of cells per head.
//
// A voxel is the cell's mean |W| over its tensor's absmean (comparable across roles and layers), and
// its balance mean W / mean |W| in [-1, 1]. cell.<axis> maps every channel to its cell, so activations
// on a channel land on its voxels. No paper equation.

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"hodgemix/eval"
)

// VolumeCells is the cell count of the D and F axes; the attention axes get VolumeHeadCells per head.
const (
	VolumeCells     = 64
	VolumeHeadCells = 4
)

type volAxis struct {
	n, cells int
	cell     []int32 // channel -> cell
}

// marriedAxis orders an axis node's n channels by the MRH pop order of their averaged Eq.(7) keys and
// cuts it into equal cells.
func marriedAxis(keys [][]int, n, cells int) volAxis {
	agg := make([]int, n)
	for i := range agg {
		var s float64
		for _, k := range keys {
			s += float64(k[i])
		}
		agg[i] = int(math.Round(s / float64(len(keys))))
	}
	x := chanCoords(agg)
	a := volAxis{n: n, cells: cells, cell: make([]int32, n)}
	for i, c := range x {
		a.cell[i] = int32(min(cells-1, int(c*float64(cells))))
	}
	return a
}

// headAxis keeps heads whole: head h's dims fill cells [h*per, (h+1)*per) in order.
func headAxis(heads, hd, per int) volAxis {
	a := volAxis{n: heads * hd, cells: heads * per, cell: make([]int32, heads*hd)}
	for i := range a.cell {
		h, d := i/hd, i%hd
		a.cell[i] = int32(h*per + d*per/hd)
	}
	return a
}

var volRoles = []string{"q", "k", "v", "o", "gate", "up", "down"}

func volName(l int, r string) string {
	if r == "q" || r == "k" || r == "v" || r == "o" {
		return fmt.Sprintf("model.layers.%d.self_attn.%s_proj.weight", l, r)
	}
	return fmt.Sprintf("model.layers.%d.mlp.%s_proj.weight", l, r)
}

// cellLayout is the cells of a model's axis nodes (shared D and F orders, heads whole).
func cellLayout(st *ST, g modelMeta) (map[axisKind]volAxis, error) {
	if moeLayout(st) {
		return moeAxes(st, g)
	}
	return denseLayout(st, g)
}

func denseLayout(st *ST, g modelMeta) (map[axisKind]volAxis, error) {
	roles, name := volRoles, volName
	for l := 0; l < g.Layers; l++ {
		for _, r := range roles {
			if !st.Has(name(l, r)) {
				return nil, fmt.Errorf("%s: no %s", st.Path, name(l, r))
			}
		}
	}
	// Married keys of the shared axis nodes, one pass over the weights.
	var dKeys, fKeys [][]int
	for l := 0; l < g.Layers; l++ {
		for _, r := range roles {
			W, sh, err := st.LoadDense(name(l, r))
			if err != nil {
				return nil, err
			}
			rk, ck, _ := roleAxes(r)
			if rk == axD {
				dKeys = append(dKeys, heapKey(W, sh[0], sh[1], 0))
			}
			if ck == axD {
				dKeys = append(dKeys, heapKey(W, sh[0], sh[1], 1))
			}
			if rk == axF {
				fKeys = append(fKeys, heapKey(W, sh[0], sh[1], 0))
			}
			if ck == axF {
				fKeys = append(fKeys, heapKey(W, sh[0], sh[1], 1))
			}
		}
	}
	return map[axisKind]volAxis{
		axD:  marriedAxis(dKeys, g.D, VolumeCells),
		axF:  marriedAxis(fKeys, g.F, VolumeCells),
		axQ:  headAxis(g.Heads, g.HD, VolumeHeadCells),
		axKV: headAxis(g.KV, g.HD, VolumeHeadCells),
	}, nil
}

// evalCells is the layout as eval takes it.
func evalCells(axes map[axisKind]volAxis) eval.Cells {
	return eval.Cells{D: axes[axD].cell, F: axes[axF].cell, Q: axes[axQ].cell, KV: axes[axKV].cell,
		ND: axes[axD].cells, NF: axes[axF].cells, NQ: axes[axQ].cells, NKV: axes[axKV].cells, Per: VolumeHeadCells}
}

func volumes(modelPath, outPath string) error {
	st, err := OpenST(modelPath)
	if err != nil {
		return err
	}
	g, err := metaOf(st)
	if err != nil {
		return err
	}
	roles, name := volRoles, volName
	logf("VOLUME  %s: %d layers, D=%d F=%d heads %d/%d x %d", modelPath, g.Layers, g.D, g.F, g.Heads, g.KV, g.HD)
	axes, err := cellLayout(st, g)
	if err != nil {
		return err
	}
	if moeLayout(st) {
		roles = []string{"q", "k", "v", "o"}
	}
	W, err := NewSTW(outPath, [][2]string{{"format", "pt"}, {"volume_of", modelPath}})
	if err != nil {
		return err
	}
	for _, r := range roles {
		rk, ck, _ := roleAxes(r)
		ar, ac := axes[rk], axes[ck]
		en := make([]float32, g.Layers*ar.cells*ac.cells)
		bal := make([]float32, len(en))
		for l := 0; l < g.Layers; l++ {
			M, sh, err := st.LoadDense(name(l, r))
			if err != nil {
				return err
			}
			R, C := sh[0], sh[1]
			if R != ar.n || C != ac.n {
				return fmt.Errorf("%s is %dx%d, the axes %dx%d", name(l, r), R, C, ar.n, ac.n)
			}
			s := absmean(M)
			sum, abs, cnt := make([]float64, ar.cells*ac.cells), make([]float64, ar.cells*ac.cells), make([]float64, ar.cells*ac.cells)
			for i := 0; i < R; i++ {
				ci := int(ar.cell[i]) * ac.cells
				row := M[i*C : (i+1)*C]
				for j, w := range row {
					v := ci + int(ac.cell[j])
					sum[v] += float64(w)
					abs[v] += math.Abs(float64(w))
					cnt[v]++
				}
			}
			off := l * ar.cells * ac.cells
			for v := range sum {
				if cnt[v] > 0 && s > 0 {
					en[off+v] = float32(abs[v] / cnt[v] / s)
				}
				if abs[v] > 0 {
					bal[off+v] = float32(sum[v] / abs[v])
				}
			}
		}
		W.AddF32("volume."+r, []int{g.Layers, ar.cells, ac.cells}, en)
		W.AddF32("balance."+r, []int{g.Layers, ar.cells, ac.cells}, bal)
		logf("  volume.%s [%d %d %d]", r, g.Layers, ar.cells, ac.cells)
	}
	if moeLayout(st) {
		if err := writeExpertVolumes(st, g, axes, W); err != nil {
			return err
		}
	}
	axisName := map[axisKind]string{axD: "D", axF: "F", axQ: "Q", axKV: "KV"}
	keys := make([]axisKind, 0, len(axes))
	for k := range axes {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	for _, k := range keys {
		W.AddI32("cell."+axisName[k], []int{axes[k].n}, axes[k].cell)
	}
	if err := W.Close(nil); err != nil {
		return err
	}
	logf("  wrote %s", outPath)
	return nil
}

func moeLayout(st *ST) bool {
	for _, k := range st.Keys {
		if strings.Contains(k, "mlp.experts.gate_up_proj") && !strings.Contains(k, "bias") {
			return true
		}
	}
	return false
}

// moeAxes averages Eq.(7) keys one expert at a time. The slice passed to marriedAxis is that average.
func moeAxes(st *ST, g modelMeta) (map[axisKind]volAxis, error) {
	dSum, fSum := make([]float64, g.D), make([]float64, g.F)
	nd, nf := 0, 0
	add := func(sum []float64, k []int, n *int) error {
		if len(k) != len(sum) {
			return fmt.Errorf("%s: key length %d, axis %d", st.Path, len(k), len(sum))
		}
		for i, v := range k {
			sum[i] += float64(v)
		}
		*n++
		return nil
	}
	for l := 0; l < g.Layers; l++ {
		for _, r := range []string{"q", "k", "v", "o"} {
			if !st.Has(volName(l, r)) {
				return nil, fmt.Errorf("%s: no %s", st.Path, volName(l, r))
			}
			W, sh, err := st.LoadDense(volName(l, r))
			if err != nil {
				return nil, err
			}
			rk, ck, _ := roleAxes(r)
			if rk == axD {
				if err := add(dSum, heapKey(W, sh[0], sh[1], 0), &nd); err != nil {
					return nil, err
				}
			}
			if ck == axD {
				if err := add(dSum, heapKey(W, sh[0], sh[1], 1), &nd); err != nil {
					return nil, err
				}
			}
		}
		E, err := expertCount(st, l)
		if err != nil {
			return nil, err
		}
		for e := 0; e < E; e++ {
			gate, up, down, err := loadOneExpert(st, l, e)
			if err != nil {
				return nil, err
			}
			if err := add(dSum, heapKey(gate, g.F, g.D, 1), &nd); err != nil {
				return nil, err
			}
			if err := add(fSum, heapKey(gate, g.F, g.D, 0), &nf); err != nil {
				return nil, err
			}
			if err := add(dSum, heapKey(up, g.F, g.D, 1), &nd); err != nil {
				return nil, err
			}
			if err := add(fSum, heapKey(up, g.F, g.D, 0), &nf); err != nil {
				return nil, err
			}
			if err := add(dSum, heapKey(down, g.D, g.F, 0), &nd); err != nil {
				return nil, err
			}
			if err := add(fSum, heapKey(down, g.D, g.F, 1), &nf); err != nil {
				return nil, err
			}
		}
	}
	if nd == 0 || nf == 0 {
		return nil, fmt.Errorf("%s: no expert weights to lay out", st.Path)
	}
	avg := func(sum []float64, n int) []int {
		o := make([]int, len(sum))
		for i, s := range sum {
			o[i] = int(math.Round(s / float64(n)))
		}
		return o
	}
	return map[axisKind]volAxis{
		axD:  marriedAxis([][]int{avg(dSum, nd)}, g.D, VolumeCells),
		axF:  marriedAxis([][]int{avg(fSum, nf)}, g.F, VolumeCells),
		axQ:  headAxis(g.Heads, g.HD, VolumeHeadCells),
		axKV: headAxis(g.KV, g.HD, VolumeHeadCells),
	}, nil
}

func expertCount(st *ST, layer int) (int, error) {
	p := fmt.Sprintf("model.layers.%d.mlp.experts.gate_up_proj", layer)
	if st.Has(p + "_blocks") {
		sh := st.H[p+"_blocks"].Shape
		if len(sh) != 4 || sh[0] < 1 {
			return 0, fmt.Errorf("%s: %s_blocks shape %v", st.Path, p, sh)
		}
		return sh[0], nil
	}
	if st.Has(p) {
		sh := st.H[p].Shape
		if len(sh) != 3 || sh[0] < 1 {
			return 0, fmt.Errorf("%s: %s shape %v", st.Path, p, sh)
		}
		return sh[0], nil
	}
	return 0, fmt.Errorf("%s: layer %d has no experts", st.Path, layer)
}

func loadOneExpert(st *ST, layer, expert int) (gate, up, down []float32, err error) {
	p := fmt.Sprintf("model.layers.%d.mlp.experts.", layer)
	if st.Has(p + "gate_up_proj_blocks") {
		return loadOneExpertMX(st, p, expert)
	}
	gw, err := st.F32(p + "gate_up_proj")
	if err != nil {
		return nil, nil, nil, err
	}
	gsh := st.H[p+"gate_up_proj"].Shape
	if len(gsh) != 3 || expert < 0 || expert >= gsh[0] || gsh[2]%2 != 0 {
		return nil, nil, nil, fmt.Errorf("%s: gate_up %v expert %d", st.Path, gsh, expert)
	}
	D, two := gsh[1], gsh[2]
	F := two / 2
	gs, us, err := eval.SplitGateUp(gw[expert*D*two:(expert+1)*D*two], []int{1, D, two})
	if err != nil {
		return nil, nil, nil, err
	}
	dw, err := st.F32(p + "down_proj")
	if err != nil {
		return nil, nil, nil, err
	}
	dsh := st.H[p+"down_proj"].Shape
	if len(dsh) != 3 || dsh[0] != gsh[0] || dsh[1] != F || dsh[2] != D {
		return nil, nil, nil, fmt.Errorf("%s: down %v, want [%d %d %d]", st.Path, dsh, gsh[0], F, D)
	}
	ds, err := eval.SplitDown(dw[expert*F*D:(expert+1)*F*D], []int{1, F, D})
	if err != nil {
		return nil, nil, nil, err
	}
	return gs[0], us[0], ds[0], nil
}

func loadOneExpertMX(st *ST, p string, expert int) (gate, up, down []float32, err error) {
	gb, err := st.Raw(p + "gate_up_proj_blocks")
	if err != nil {
		return nil, nil, nil, err
	}
	gs, err := st.Raw(p + "gate_up_proj_scales")
	if err != nil {
		return nil, nil, nil, err
	}
	db, err := st.Raw(p + "down_proj_blocks")
	if err != nil {
		return nil, nil, nil, err
	}
	ds, err := st.Raw(p + "down_proj_scales")
	if err != nil {
		return nil, nil, nil, err
	}
	gsh, ssh := st.H[p+"gate_up_proj_blocks"].Shape, st.H[p+"gate_up_proj_scales"].Shape
	dsh, dss := st.H[p+"down_proj_blocks"].Shape, st.H[p+"down_proj_scales"].Shape
	eb, esh, err := sliceExpert(gb, gsh, expert)
	if err != nil {
		return nil, nil, nil, err
	}
	es, ess, err := sliceExpert(gs, ssh, expert)
	if err != nil {
		return nil, nil, nil, err
	}
	w, osh, err := eval.DequantMXFP4(eb, es, esh, ess)
	if err != nil {
		return nil, nil, nil, err
	}
	gu, uu, err := eval.SplitGateUp(w, osh)
	if err != nil {
		return nil, nil, nil, err
	}
	xb, xsh, err := sliceExpert(db, dsh, expert)
	if err != nil {
		return nil, nil, nil, err
	}
	xs, xss, err := sliceExpert(ds, dss, expert)
	if err != nil {
		return nil, nil, nil, err
	}
	dw, dosh, err := eval.DequantMXFP4(xb, xs, xsh, xss)
	if err != nil {
		return nil, nil, nil, err
	}
	downs, err := eval.SplitDown(dw, dosh)
	if err != nil {
		return nil, nil, nil, err
	}
	return gu[0], uu[0], downs[0], nil
}

func sliceExpert(b []byte, shape []int, e int) ([]byte, []int, error) {
	if len(shape) < 2 || e < 0 || e >= shape[0] {
		return nil, nil, fmt.Errorf("expert %d of %v", e, shape)
	}
	rest := 1
	for _, d := range shape[1:] {
		rest *= d
	}
	if len(b) < shape[0]*rest {
		return nil, nil, fmt.Errorf("expert tensor is %d bytes, shape %v", len(b), shape)
	}
	sh := append([]int{1}, shape[1:]...)
	return append([]byte(nil), b[e*rest:(e+1)*rest]...), sh, nil
}

// writeExpertVolumes writes gate, up and down as the mean, over experts, of each expert's cell ratio.
// One expert is in memory at a time.
func writeExpertVolumes(st *ST, g modelMeta, axes map[axisKind]volAxis, W *STW) error {
	type role struct {
		name       string
		rows, cols int
		ar, ac     volAxis
	}
	roles := []role{
		{"gate", g.F, g.D, axes[axF], axes[axD]},
		{"up", g.F, g.D, axes[axF], axes[axD]},
		{"down", g.D, g.F, axes[axD], axes[axF]},
	}
	accE := make([][]float64, len(roles))
	accB := make([][]float64, len(roles))
	for i, r := range roles {
		accE[i] = make([]float64, g.Layers*r.ar.cells*r.ac.cells)
		accB[i] = make([]float64, len(accE[i]))
	}
	for l := 0; l < g.Layers; l++ {
		E, err := expertCount(st, l)
		if err != nil {
			return err
		}
		sumE := make([][]float64, len(roles))
		sumB := make([][]float64, len(roles))
		for i, r := range roles {
			sumE[i] = make([]float64, r.ar.cells*r.ac.cells)
			sumB[i] = make([]float64, len(sumE[i]))
		}
		for e := 0; e < E; e++ {
			gate, up, down, err := loadOneExpert(st, l, e)
			if err != nil {
				return err
			}
			for i, M := range [][]float32{gate, up, down} {
				r := roles[i]
				if len(M) != r.rows*r.cols {
					return fmt.Errorf("layer %d %s expert %d has %d values, want %d", l, r.name, e, len(M), r.rows*r.cols)
				}
				s := absmean(M)
				cellSum := make([]float64, r.ar.cells*r.ac.cells)
				cellAbs := make([]float64, len(cellSum))
				cnt := make([]float64, len(cellSum))
				for row := 0; row < r.rows; row++ {
					ci := int(r.ar.cell[row]) * r.ac.cells
					line := M[row*r.cols : (row+1)*r.cols]
					for j, w := range line {
						v := ci + int(r.ac.cell[j])
						cellSum[v] += float64(w)
						cellAbs[v] += math.Abs(float64(w))
						cnt[v]++
					}
				}
				for v := range cellSum {
					if cnt[v] > 0 && s > 0 {
						sumE[i][v] += cellAbs[v] / cnt[v] / s
					}
					if cellAbs[v] > 0 {
						sumB[i][v] += cellSum[v] / cellAbs[v]
					}
				}
			}
		}
		for i := range roles {
			off := l * len(sumE[i])
			for v := range sumE[i] {
				accE[i][off+v] = sumE[i][v] / float64(E)
				accB[i][off+v] = sumB[i][v] / float64(E)
			}
		}
	}
	for i, r := range roles {
		en, bal := make([]float32, len(accE[i])), make([]float32, len(accB[i]))
		for v := range en {
			en[v], bal[v] = float32(accE[i][v]), float32(accB[i][v])
		}
		W.AddF32("volume."+r.name, []int{g.Layers, r.ar.cells, r.ac.cells}, en)
		W.AddF32("balance."+r.name, []int{g.Layers, r.ar.cells, r.ac.cells}, bal)
		logf("  volume.%s [%d %d %d]", r.name, g.Layers, r.ar.cells, r.ac.cells)
	}
	return nil
}

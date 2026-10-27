package eval

// Wedge decision on one residual loop.
//
// A layer has two faces. The attention face pairs the residual carried into the
// attention block with the attention write. The feed-forward face pairs the
// residual carried into the feed-forward with that write. On each residual cell
// the two currents are the cell sums, and the wedge is their product summed
// over cells. With no cell layout every channel is its own cell.
//
// The sign is the branch. A positive wedge keeps the write and adds it to the
// residual. Zero means the two currents do not span the face, and a negative
// wedge means they oppose, so the write is refused and the carry stays.
// No paper equation.

// wedgeKeep is the branch at one face: sign -1, 0 or +1, the oriented area, and
// whether the write is added to the residual.
func wedgeKeep(carry, write []float32, cell []int32, n int) (sign int, area float64, keep bool) {
	area = wedgeArea(carry, write, cell, n)
	switch {
	case area > 0:
		sign = 1
	case area < 0:
		sign = -1
	}
	keep = sign > 0
	return
}

func wedgeArea(carry, write []float32, cell []int32, n int) float64 {
	if n > 0 && len(cell) == len(carry) {
		cs, ws := cellSums(carry, cell, n), cellSums(write, cell, n)
		var s float64
		for i := range cs {
			wi := float32(0)
			if i < len(ws) {
				wi = ws[i]
			}
			s += float64(cs[i]) * float64(wi)
		}
		return s
	}
	var s float64
	for i, c := range carry {
		w := float32(0)
		if i < len(write) {
			w = write[i]
		}
		s += float64(c) * float64(w)
	}
	return s
}

// admitWrite keeps write on the residual when its wedge with carry is positive, and records the
// branch. kind 0 is the attention write (edge "o"), kind 1 the feed-forward write (edge "down").
// A refused write is removed from the cochain so the recorded currents match the residual.
func (m *model) admitWrite(li, kind int, carry, write []float32) bool {
	sign, area, keep := wedgeKeep(carry, write, m.cells.D, m.cells.ND)
	if m.rec != nil && li >= 0 && li < len(m.rec) {
		m.rec[li].Wedge[kind] = float32(area)
		m.rec[li].Sign[kind] = int8(sign)
		if keep {
			m.rec[li].Keep[kind] = 1
		}
	}
	if !keep {
		role := "o"
		if kind == 1 {
			role = "down"
		}
		m.dropEdge(li, role)
	}
	return keep
}

func (m *model) dropEdge(li int, role string) {
	if m.rec == nil || li < 0 || li >= len(m.rec) {
		return
	}
	e := m.rec[li].Edge[role]
	for i := range e {
		e[i] = 0
	}
}

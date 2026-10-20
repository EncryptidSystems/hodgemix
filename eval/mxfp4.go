package eval

// MXFP4 experts as gpt-oss stores them (transformers integrations/mxfp4.py, the OpenAI pack).
// A block is 32 values: 16 bytes, low nibble then high nibble, times an E8M0 scale 2^(s-127).
// The forward gathers that 16-entry table. It does not expand the checkpoint into a dense matrix.
// Dequant exists so a volume can read one expert's |W| and so the table can be checked.

import (
	"fmt"
	"math"
)

// fp4 is the E2M1 table, nibble order. Index 8 is −0.
var fp4 = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}

// mx4 is one linear as a view of the packed table. Row r of this linear is packed row
// phase + r*stride (gate takes the even gate_up rows, up the odd ones, down every row).
type mx4 struct {
	rows, groups, B int
	stride, phase   int
	blocks          []byte // packed rows of this expert: [packedRows, groups, B]
	scales          []byte // [packedRows, groups]
}

// gather is y = W x through the nibble table. When flow is set, each product is binned
// into the edge (output cell, input cell). Bias is not included.
func (m *mx4) gather(x, out []float32, accum bool, flow []float32, colCell, rowCell []int32, nc int) {
	if m == nil {
		return
	}
	span := m.groups * m.B
	for r := 0; r < m.rows; r++ {
		src := m.phase + r*m.stride
		var y float32
		var cs []float32
		if flow != nil {
			cs = make([]float32, nc)
		}
		rb := m.blocks[src*span : (src+1)*span]
		rs := m.scales[src*m.groups : (src+1)*m.groups]
		for g := 0; g < m.groups; g++ {
			scale := float32(math.Ldexp(1, int(rs[g])-127))
			blk := rb[g*m.B : (g+1)*m.B]
			base := g * m.B * 2
			for j, b := range blk {
				c0 := base + 2*j
				v0 := fp4[b&0x0f] * scale * x[c0]
				v1 := fp4[b>>4] * scale * x[c0+1]
				y += v0 + v1
				if cs != nil {
					cs[colCell[c0]] += v0
					cs[colCell[c0+1]] += v1
				}
			}
		}
		if accum {
			out[r] += y
		} else {
			out[r] = y
		}
		if flow != nil {
			rc := int(rowCell[r]) * nc
			for c, v := range cs {
				flow[rc+c] += v
			}
		}
	}
}

// DequantMXFP4 unpacks blocks [..., rows, groups, bytes] and scales [..., rows, groups] into
// [..., cols, rows] with cols = groups * bytes * 2. The last two axes are transposed, which is
// the float checkpoint's layout.
func DequantMXFP4(blocks, scales []byte, bShape, sShape []int) ([]float32, []int, error) {
	if len(bShape) < 3 || len(sShape) != len(bShape)-1 {
		return nil, nil, fmt.Errorf("mxfp4 shapes blocks %v scales %v", bShape, sShape)
	}
	for i := range sShape {
		if bShape[i] != sShape[i] {
			return nil, nil, fmt.Errorf("mxfp4 blocks %v do not match scales %v", bShape, sShape)
		}
	}
	B := bShape[len(bShape)-1]
	groups := bShape[len(bShape)-2]
	rows := 1
	for _, v := range bShape[:len(bShape)-2] {
		rows *= v
	}
	if B < 1 || groups < 1 || rows < 1 || len(blocks) != rows*groups*B || len(scales) != rows*groups {
		return nil, nil, fmt.Errorf("mxfp4 bytes blocks %d scales %d, shapes %v %v", len(blocks), len(scales), bShape, sShape)
	}
	cols := groups * B * 2
	raw := make([]float32, rows*cols) // [rows, cols] before the last-two transpose
	for r := 0; r < rows; r++ {
		for g := 0; g < groups; g++ {
			exp := int(scales[r*groups+g]) - 127
			blk := blocks[(r*groups+g)*B : (r*groups+g+1)*B]
			dst := raw[r*cols+g*B*2:]
			for j, b := range blk {
				dst[2*j] = float32(math.Ldexp(float64(fp4[b&0x0f]), exp))
				dst[2*j+1] = float32(math.Ldexp(float64(fp4[b>>4]), exp))
			}
		}
	}
	// bShape is [prefix..., rowAxis, groups, bytes]. `rows` above folds prefix into rowAxis.
	rowAxis := bShape[len(bShape)-3]
	if rowAxis < 1 || rows%rowAxis != 0 {
		return nil, nil, fmt.Errorf("mxfp4 row axis %d does not divide %d", rowAxis, rows)
	}
	pre := rows / rowAxis
	out := make([]float32, len(raw))
	for p := 0; p < pre; p++ {
		for r := 0; r < rowAxis; r++ {
			for c := 0; c < cols; c++ {
				out[(p*cols+c)*rowAxis+r] = raw[(p*rowAxis+r)*cols+c]
			}
		}
	}
	osh := make([]int, 0, len(bShape)-1)
	osh = append(osh, bShape[:len(bShape)-3]...)
	osh = append(osh, cols, rowAxis)
	return out, osh, nil
}

// SplitGateUp splits gate_up [experts, hidden, 2*intermediate], interleaved even/odd along the
// last axis, into gate and up of shape [intermediate, hidden] (row = output).
func SplitGateUp(w []float32, shape []int) (gate, up [][]float32, err error) {
	if len(shape) != 3 || shape[2]%2 != 0 {
		return nil, nil, fmt.Errorf("gate_up %v, want [experts, hidden, 2*intermediate]", shape)
	}
	e, d, two := shape[0], shape[1], shape[2]
	f := two / 2
	gate, up = make([][]float32, e), make([][]float32, e)
	for i := 0; i < e; i++ {
		g, u := make([]float32, f*d), make([]float32, f*d)
		base := i * d * two
		for r := 0; r < f; r++ {
			for c := 0; c < d; c++ {
				g[r*d+c] = w[base+c*two+2*r]
				u[r*d+c] = w[base+c*two+2*r+1]
			}
		}
		gate[i], up[i] = g, u
	}
	return gate, up, nil
}

// SplitDown turns down [experts, intermediate, hidden] into [hidden, intermediate] per expert.
func SplitDown(w []float32, shape []int) ([][]float32, error) {
	if len(shape) != 3 {
		return nil, fmt.Errorf("down %v, want [experts, intermediate, hidden]", shape)
	}
	e, f, d := shape[0], shape[1], shape[2]
	out := make([][]float32, e)
	for i := 0; i < e; i++ {
		m := make([]float32, d*f)
		base := i * f * d
		for r := 0; r < d; r++ {
			for c := 0; c < f; c++ {
				m[r*f+c] = w[base+c*d+r]
			}
		}
		out[i] = m
	}
	return out, nil
}

// SplitBias splits an interleaved gate/up bias [experts, 2*intermediate] into two [intermediate] vectors.
func SplitBias(b []float32, experts, two int) (gate, up [][]float32, err error) {
	if two%2 != 0 || len(b) != experts*two {
		return nil, nil, fmt.Errorf("gate_up bias length %d, want %d experts x %d", len(b), experts, two)
	}
	f := two / 2
	gate, up = make([][]float32, experts), make([][]float32, experts)
	for i := 0; i < experts; i++ {
		g, u := make([]float32, f), make([]float32, f)
		row := b[i*two : (i+1)*two]
		for r := 0; r < f; r++ {
			g[r], u[r] = row[2*r], row[2*r+1]
		}
		gate[i], up[i] = g, u
	}
	return gate, up, nil
}

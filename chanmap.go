package main

// BNZ4 channel-map chunk (paper_v2 Sec.6, Table 2). Little-endian. Stored as a U8 tensor
// "bnzmix.bnz4.<tensor>" beside the ternary weight and its weight_scale (carried losslessly by the BNZ3 container).
//
//  off  size  field
//   0    4    magic "BNZ4"
//   4    2    version = 1
//   6    2    name length L, then L bytes UTF-8 tensor name
//   .    4+4  R_out, C_out (u32)
//   .    4+4  G_r, G_c index grid (u32)
//   .    1+1  n_r, n_c digit counts; then n_r + n_c radices (u8 each, coarsest first)
//   .    2    S contributors; per contributor:
//               u16 src model id, u8 MRH bucket, f32 depth weight w_s,
//               n_r+n_c u8 phase-correlation digit shifts,
//               G_r i32 + G_c i32 SIGNED permutation (value = +(p+1) or -(p+1); sign = channel sign flip)
//   .    R_out*(4+4+4)  row prolongation: u32 left grid node, f32 left weight, f32 right weight (Whitney hat, Eq.10)
//   .    C_out*(4+4+4)  col prolongation, same layout
//   .    4    B low-frequency bins; per bin u32 flat bin index, f32 re, f32 im (harmonic/low spectrum, Eq.5)
//   end  4    CRC-32 (IEEE) of all preceding bytes
// Invertible: every signed permutation has inverse p^-1 with signs carried; the decoder checks bijectivity.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"math"
)

type Contrib struct {
	Src    uint16
	Bucket uint8
	W      float32
	Shift  []uint8
	PR, PC []int32 // signed 1-based
}

type Prolong struct {
	Left   uint32
	WL, WR float32
}

type ChanMap struct {
	Name       string
	Rout, Cout uint32
	Gr, Gc     uint32
	RadR, RadC []uint8
	C          []Contrib
	ProR, ProC []Prolong
	Bins       []uint32
	Re, Im     []float32
	DT         *DtftChunk // dtft_lock Sec.6 proposal: version 2 only
}

// dtft_lock "Proposed changes" Sec.6 BNZ4 (version 2 trailer, before the CRC):
//
//	u8 mode axis (0 rows,1 cols, 255 none), f32 lambda_gap (H2), f32 c_P/c_S, u16 M band modes, M x u32 A-axis bin,
//	u8 L loops (long sequence-axis digits), then for each contributor (chunk order) and mode:
//	BF16 nu_hat (E1), u16 phi 12-bit (turns*4096 mod 4096), u8 flags (1 PSR gate, 2 L2, 4 winding, 8 lost-clear, 16 merged),
//	L x (int8 winding q (H4), BF16 period p (H3))
type DtftChunk struct {
	Axis   uint8
	LGap   float32
	CPS    float32
	Modes  []uint32
	NLoops uint8
	Nu     [][]uint16
	Phi    [][]uint16
	Flags  [][]uint8
	Q      [][][]int8
	P      [][][]uint16
}

// Paper_v2 Sec.6 Table 2: encode
func (m *ChanMap) Encode() []byte {
	var b bytes.Buffer
	w := func(v any) { binary.Write(&b, binary.LittleEndian, v) }
	b.WriteString("BNZ4")
	ver := uint16(1)
	if m.DT != nil {
		ver = 2
	}
	w(ver)
	w(uint16(len(m.Name)))
	b.WriteString(m.Name)
	w(m.Rout)
	w(m.Cout)
	w(m.Gr)
	w(m.Gc)
	w(uint8(len(m.RadR)))
	w(uint8(len(m.RadC)))
	b.Write(m.RadR)
	b.Write(m.RadC)
	w(uint16(len(m.C)))
	for _, c := range m.C {
		w(c.Src)
		w(c.Bucket)
		w(c.W)
		b.Write(c.Shift)
		w(c.PR)
		w(c.PC)
	}
	for _, p := range append(append([]Prolong{}, m.ProR...), m.ProC...) {
		w(p.Left)
		w(p.WL)
		w(p.WR)
	}
	w(uint32(len(m.Bins)))
	for i := range m.Bins {
		w(m.Bins[i])
		w(m.Re[i])
		w(m.Im[i])
	}
	if d := m.DT; d != nil {
		w(d.Axis)
		w(d.LGap)
		w(d.CPS)
		w(uint16(len(d.Modes)))
		w(d.Modes)
		w(d.NLoops)
		for c := range m.C {
			for k := range d.Modes {
				w(d.Nu[c][k])
				w(d.Phi[c][k])
				w(d.Flags[c][k])
				for l := 0; l < int(d.NLoops); l++ {
					w(d.Q[c][k][l])
					w(d.P[c][k][l])
				}
			}
		}
	}
	w(crc32.ChecksumIEEE(b.Bytes()))
	return b.Bytes()
}

// Paper_v2 Sec.6 Table 2: decode + CRC + signed-permutation bijectivity check
func DecodeChanMap(d []byte) (*ChanMap, error) {
	if len(d) < 8 || string(d[:4]) != "BNZ4" {
		return nil, errors.New("bnz4: bad magic")
	}
	if crc32.ChecksumIEEE(d[:len(d)-4]) != binary.LittleEndian.Uint32(d[len(d)-4:]) {
		return nil, errors.New("bnz4: crc mismatch")
	}
	r := bytes.NewReader(d[4 : len(d)-4])
	rd := func(v any) { binary.Read(r, binary.LittleEndian, v) }
	var ver, nl uint16
	rd(&ver)
	rd(&nl)
	nm := make([]byte, nl)
	r.Read(nm)
	m := &ChanMap{Name: string(nm)}
	rd(&m.Rout)
	rd(&m.Cout)
	rd(&m.Gr)
	rd(&m.Gc)
	var nr, nc uint8
	rd(&nr)
	rd(&nc)
	m.RadR, m.RadC = make([]uint8, nr), make([]uint8, nc)
	r.Read(m.RadR)
	r.Read(m.RadC)
	var S uint16
	rd(&S)
	for i := 0; i < int(S); i++ {
		c := Contrib{Shift: make([]uint8, int(nr)+int(nc)), PR: make([]int32, m.Gr), PC: make([]int32, m.Gc)}
		rd(&c.Src)
		rd(&c.Bucket)
		rd(&c.W)
		r.Read(c.Shift)
		rd(c.PR)
		rd(c.PC)
		for _, p := range [][]int32{c.PR, c.PC} {
			seen := make([]bool, len(p))
			for _, v := range p {
				a := int(v)
				if a < 0 {
					a = -a
				}
				if a < 1 || a > len(p) || seen[a-1] {
					return nil, errors.New("bnz4: permutation not invertible")
				}
				seen[a-1] = true
			}
		}
		m.C = append(m.C, c)
	}
	m.ProR, m.ProC = make([]Prolong, m.Rout), make([]Prolong, m.Cout)
	for i := range m.ProR {
		rd(&m.ProR[i].Left)
		rd(&m.ProR[i].WL)
		rd(&m.ProR[i].WR)
	}
	for i := range m.ProC {
		rd(&m.ProC[i].Left)
		rd(&m.ProC[i].WL)
		rd(&m.ProC[i].WR)
	}
	var B uint32
	rd(&B)
	m.Bins, m.Re, m.Im = make([]uint32, B), make([]float32, B), make([]float32, B)
	for i := range m.Bins {
		rd(&m.Bins[i])
		rd(&m.Re[i])
		rd(&m.Im[i])
	}
	if ver >= 2 {
		d := &DtftChunk{}
		rd(&d.Axis)
		rd(&d.LGap)
		rd(&d.CPS)
		var M uint16
		rd(&M)
		d.Modes = make([]uint32, M)
		rd(d.Modes)
		rd(&d.NLoops)
		for range m.C {
			nu, ph, fl := make([]uint16, M), make([]uint16, M), make([]uint8, M)
			qq, pp := make([][]int8, M), make([][]uint16, M)
			for k := 0; k < int(M); k++ {
				rd(&nu[k])
				rd(&ph[k])
				rd(&fl[k])
				qq[k], pp[k] = make([]int8, d.NLoops), make([]uint16, d.NLoops)
				for l := range qq[k] {
					rd(&qq[k][l])
					rd(&pp[k][l])
				}
			}
			d.Nu, d.Phi, d.Flags, d.Q, d.P = append(d.Nu, nu), append(d.Phi, ph), append(d.Flags, fl), append(d.Q, qq), append(d.P, pp)
		}
		m.DT = d
	}
	if r.Len() != 0 {
		return nil, errors.New("bnz4: trailing bytes")
	}
	return m, nil
}

// Paper_v2 Eq.(10): Whitney hat weights of target coordinate x on the cell-centred index grid of size G
// (matches sampleGrid/interp exactly, including clamping at the ends)
func hatWeights(x []float64, G int) []Prolong {
	o := make([]Prolong, len(x))
	for i, v := range x {
		p := v*float64(G) - .5
		if p <= 0 {
			o[i] = Prolong{0, 1, 0}
			continue
		}
		if p >= float64(G-1) {
			o[i] = Prolong{uint32(G - 1), 1, 0}
			continue
		}
		l := math.Floor(p)
		t := p - l
		o[i] = Prolong{uint32(l), float32(1 - t), float32(t)}
	}
	return o
}

// Paper_v2 Sec.6: signed 1-based encoding of a permutation (all signs + : the seed never flips sign)
func signedPerm(p []int) []int32 {
	o := make([]int32, len(p))
	for i, v := range p {
		o[i] = int32(v + 1)
	}
	return o
}

package bnz4

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"sort"
	"sync"

	"github.com/klauspost/compress/zstd"
)

var (
	zencOnce sync.Once
	zenc     *zstd.Encoder
	zdec     *zstd.Decoder
)

func zinit() {
	zencOnce.Do(func() {
		zenc, _ = zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedBestCompression), zstd.WithEncoderConcurrency(1), zstd.WithWindowSize(1<<27))
		zdec, _ = zstd.NewReader(nil, zstd.WithDecoderConcurrency(0), zstd.WithDecoderMaxMemory(1<<32), zstd.WithDecoderMaxWindow(1<<28))
	})
}
func Z(b []byte) []byte           { zinit(); return zenc.EncodeAll(b, nil) }
func UZ(b []byte) ([]byte, error) { zinit(); return zdec.DecodeAll(b, nil) }

func pk(parts ...[]byte) []byte {
	var o bytes.Buffer
	for _, p := range parts {
		binary.Write(&o, binary.LittleEndian, uint64(len(p)))
		o.Write(p)
	}
	return o.Bytes()
}
func upk(b []byte) ([][]byte, error) {
	var out [][]byte
	for i := 0; i < len(b); {
		if i+8 > len(b) {
			return nil, errors.New("bad pack")
		}
		n := int(binary.LittleEndian.Uint64(b[i:]))
		i += 8
		if n < 0 || i+n > len(b) {
			return nil, errors.New("bad pack")
		}
		out = append(out, b[i:i+n])
		i += n
	}
	return out, nil
}
func u32bytes(v []uint32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}
func bytesU32(b []byte) []uint32 {
	v := make([]uint32, len(b)/4)
	for i := range v {
		v[i] = binary.LittleEndian.Uint32(b[4*i:])
	}
	return v
}

// ---------------- ternary ----------------
// Trit plane layout: a is R x C bytes; trit row n = s*R + r holds (a[r][c] >> 2s) & 3.

type ternModel struct {
	N, C int
	mode int
	rc   []uint32 // N*3
	cc   []uint32 // C*3
	rowF []uint32 // mode0: N*2 (f0,f1)
	colF []uint32 // mode1: C*2
	ra   []float64
	ca   []float64
	g    [3]float64
}

func quant3(p0, p1, p2 float64) (uint32, uint32) {
	s := p0 + p1 + p2
	f0 := 1 + uint32(math.Floor(float64(p0/s)*float64(ProbT-3)))
	f1 := 1 + uint32(math.Floor(float64(p1/s)*float64(ProbT-3)))
	return f0, f1 // f2 = T - f0 - f1 >= 1
}

func newTernModel(rc, cc []uint32, N, C, mode int) *ternModel {
	m := &ternModel{N: N, C: C, mode: mode, rc: rc, cc: cc}
	switch mode {
	case 0:
		m.rowF = make([]uint32, 2*N)
		for n := 0; n < N; n++ {
			m.rowF[2*n], m.rowF[2*n+1] = quant3(float64(rc[3*n])+.5, float64(rc[3*n+1])+.5, float64(rc[3*n+2])+.5)
		}
	case 1:
		m.colF = make([]uint32, 2*C)
		for c := 0; c < C; c++ {
			m.colF[2*c], m.colF[2*c+1] = quant3(float64(cc[3*c])+.5, float64(cc[3*c+1])+.5, float64(cc[3*c+2])+.5)
		}
	case 2:
		var gc [3]uint64
		for n := 0; n < N; n++ {
			for k := 0; k < 3; k++ {
				gc[k] += uint64(rc[3*n+k])
			}
		}
		for k := 0; k < 3; k++ {
			m.g[k] = float64(gc[k]) + .5
		}
		m.ra = make([]float64, 3*N)
		for i := range m.ra {
			m.ra[i] = float64(rc[i]) + .5
		}
		m.ca = make([]float64, 3*C)
		for i := range m.ca {
			m.ca[i] = float64(cc[i]) + .5
		}
	}
	return m
}

// freqs returns (f0,f1) for row n, col c. Explicit float64() conversions forbid FMA fusion.
func (m *ternModel) freqs(n, c int) (uint32, uint32) {
	switch m.mode {
	case 0:
		return m.rowF[2*n], m.rowF[2*n+1]
	case 1:
		return m.colF[2*c], m.colF[2*c+1]
	}
	r, q := m.ra[3*n:3*n+3], m.ca[3*c:3*c+3]
	return quant3(float64(float64(r[0]*q[0])/m.g[0]), float64(float64(r[1]*q[1])/m.g[1]), float64(float64(r[2]*q[2])/m.g[2]))
}

var log2T [ProbT + 1]float32

func init() {
	for i := 1; i <= ProbT; i++ {
		log2T[i] = float32(ProbBits - math.Log2(float64(i)))
	}
}

func symFS(t byte, f0, f1 uint32) (uint32, uint32) {
	switch t {
	case 0:
		return 0, f0
	case 1:
		return f0, f1
	}
	return f0 + f1, ProbT - f0 - f1
}

func EncTern(raw []byte, R, C int) []byte {
	N := 4 * R
	trit := func(n, c int) byte { s, r := n/R, n%R; return (raw[r*C+c] >> (2 * s)) & 3 }
	rc := make([]uint32, 3*N)
	cc := make([]uint32, 3*C)
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			t := trit(n, c)
			if t == 3 {
				return nil
			}
			rc[3*n+int(t)]++
			cc[3*c+int(t)]++
		}
	}
	// pick the mode with the smallest exact quantized code length
	best, bestBits := 0, math.Inf(1)
	var wg sync.WaitGroup
	bits := make([]float64, 3)
	for mode := 0; mode < 3; mode++ {
		wg.Add(1)
		go func(mode int) {
			defer wg.Done()
			m := newTernModel(rc, cc, N, C, mode)
			var tot float64
			for n := 0; n < N; n++ {
				var row float32
				for c := 0; c < C; c++ {
					f0, f1 := m.freqs(n, c)
					_, f := symFS(trit(n, c), f0, f1)
					row += log2T[f]
				}
				tot += float64(row)
			}
			bits[mode] = tot
		}(mode)
	}
	wg.Wait()
	for i, b := range bits {
		if b < bestBits {
			best, bestBits = i, b
		}
	}
	m := newTernModel(rc, cc, N, C, best)
	e := NewEncoder(int(bestBits/8) + 1024)
	for n := N - 1; n >= 0; n-- {
		for c := C - 1; c >= 0; c-- {
			f0, f1 := m.freqs(n, c)
			s, f := symFS(trit(n, c), f0, f1)
			e.Put(s, f)
		}
	}
	tab := Z(append(u32bytes(rc), u32bytes(cc)...))
	return pk([]byte{byte(best)}, tab, e.Bytes())
}

func DecTern(b []byte, R, C int) ([]byte, error) {
	N := 4 * R
	p, err := upk(b)
	if err != nil || len(p) != 3 || len(p[0]) != 1 {
		return nil, errors.New("tern: bad payload")
	}
	tb, err := UZ(p[1])
	if err != nil || len(tb) != 4*3*(N+C) {
		return nil, errors.New("tern: bad tables")
	}
	tv := bytesU32(tb)
	m := newTernModel(tv[:3*N], tv[3*N:], N, C, int(p[0][0]))
	d := NewDecoder(p[2])
	out := make([]byte, R*C)
	for n := 0; n < N; n++ {
		s, r := n/R, n%R
		row := out[r*C : r*C+C]
		sh := uint(2 * s)
		for c := 0; c < C; c++ {
			f0, f1 := m.freqs(n, c)
			x := d.Slot()
			var t byte
			if x < f0 {
				d.Advance(0, f0)
			} else if x < f0+f1 {
				t = 1
				d.Advance(f0, f1)
			} else {
				t = 2
				d.Advance(f0+f1, ProbT-f0-f1)
			}
			row[c] |= t << sh
		}
	}
	if d.Overrun() {
		return nil, errors.New("tern: stream overrun")
	}
	return out, nil
}

// ---------------- bf16 ----------------
// value = sign<<15 | (exp)<<7 | man. Exponent coded with a per-column table, mantissa with a
// per-exponent table, both in row-major order interleaved (exp then man). Signs packed + zstd.

type table struct {
	only int
	f    []uint32 // nil => deterministic symbol `only`
	cum  []uint32
	lut  []uint8 // slot -> symbol (mantissa tables only)
}

func mkTable(counts []uint32, lut bool) *table {
	t := &table{only: Only(counts), f: Quantize(counts)}
	if t.f != nil {
		t.cum = Cum(t.f)
		if lut {
			t.lut = make([]uint8, ProbT)
			for s := range t.f {
				for i := t.cum[s]; i < t.cum[s+1]; i++ {
					t.lut[i] = uint8(s)
				}
			}
		}
	}
	return t
}

func (t *table) find(x uint32) int {
	if t.lut != nil {
		return int(t.lut[x])
	}
	return sort.Search(len(t.f), func(i int) bool { return t.cum[i+1] > x })
}

func EncBF16(raw []byte, shape []int) []byte {
	n := len(raw) / 2
	C := 1
	if len(shape) == 2 {
		C = shape[1]
	}
	v := make([]uint16, n)
	emin, emax := 255, 0
	for i := range v {
		v[i] = binary.LittleEndian.Uint16(raw[2*i:])
		e := int(v[i]>>7) & 255
		emin, emax = min(emin, e), max(emax, e)
	}
	K := emax - emin + 1
	ecnt := make([]uint32, C*K)
	mcnt := make([]uint32, K*128)
	sign := make([]byte, (n+7)/8)
	for i, x := range v {
		e := int(x>>7)&255 - emin
		ecnt[(i%C)*K+e]++
		mcnt[e*128+int(x&127)]++
		if x>>15 != 0 {
			sign[i/8] |= 0x80 >> (i % 8)
		}
	}
	et := make([]*table, C)
	for c := range et {
		et[c] = mkTable(ecnt[c*K:(c+1)*K], false)
	}
	mt := make([]*table, K)
	for k := range mt {
		mt[k] = mkTable(mcnt[k*128:(k+1)*128], false)
	}
	enc := NewEncoder(n)
	for i := n - 1; i >= 0; i-- {
		x := v[i]
		e := int(x>>7)&255 - emin
		if t := mt[e]; t.f != nil {
			s := int(x & 127)
			enc.Put(t.cum[s], t.f[s])
		}
		if t := et[i%C]; t.f != nil {
			enc.Put(t.cum[e], t.f[e])
		}
	}
	hdr := make([]byte, 12)
	binary.LittleEndian.PutUint32(hdr, uint32(int32(emin)))
	binary.LittleEndian.PutUint32(hdr[4:], uint32(K))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(C))
	return pk(hdr, Z(append(u32bytes(ecnt), u32bytes(mcnt)...)), Z(sign), enc.Bytes())
}

func DecBF16(b []byte, shape []int) ([]byte, error) {
	p, err := upk(b)
	if err != nil || len(p) != 4 || len(p[0]) != 12 {
		return nil, errors.New("bf16: bad payload")
	}
	emin := int(int32(binary.LittleEndian.Uint32(p[0])))
	K := int(binary.LittleEndian.Uint32(p[0][4:]))
	C := int(binary.LittleEndian.Uint32(p[0][8:]))
	n := 1
	for _, s := range shape {
		n *= s
	}
	tb, err := UZ(p[1])
	if err != nil || len(tb) != 4*(C*K+K*128) || C <= 0 {
		return nil, errors.New("bf16: bad tables")
	}
	tv := bytesU32(tb)
	et := make([]*table, C)
	for c := range et {
		et[c] = mkTable(tv[c*K:(c+1)*K], false)
	}
	mt := make([]*table, K)
	for k := range mt {
		mt[k] = mkTable(tv[C*K+k*128:C*K+(k+1)*128], true)
	}
	sign, err := UZ(p[2])
	if err != nil || len(sign) != (n+7)/8 {
		return nil, errors.New("bf16: bad signs")
	}
	d := NewDecoder(p[3])
	out := make([]byte, 2*n)
	for i := 0; i < n; i++ {
		t := et[i%C]
		e := t.only
		if t.f != nil {
			e = t.find(d.Slot())
			d.Advance(t.cum[e], t.f[e])
		}
		m := mt[e]
		s := m.only
		if m.f != nil {
			s = m.find(d.Slot())
			d.Advance(m.cum[s], m.f[s])
		}
		x := uint16(e+emin)<<7 | uint16(s)
		if sign[i/8]&(0x80>>(i%8)) != 0 {
			x |= 0x8000
		}
		binary.LittleEndian.PutUint16(out[2*i:], x)
	}
	if d.Overrun() {
		return nil, errors.New("bf16: stream overrun")
	}
	return out, nil
}

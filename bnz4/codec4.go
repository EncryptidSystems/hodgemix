package bnz4

// BNZ4: chunked, GPU-parallel variants of the ternary and bf16 methods.
// Every chunk is an independent rANS stream (4-byte BE state + bytes), so one GPU
// thread decodes one chunk and owns every output word of it. All model arithmetic is
// 32-bit integer so Go and WGSL produce identical frequencies.

import (
	"encoding/binary"
	"errors"
	"math"
	"math/bits"
	"runtime"
	"sync"
)

const (
	MTern4     = 4
	MBF164     = 5
	TernSyms   = 16384 // target trits per chunk / 4 -> rows per chunk = max(1, TernSyms/C)
	BF16Elems  = 32768 // elements per bf16 chunk (even, so each chunk owns whole u32 words)
	maxTernDim = 32768
)

// Cost table: 1/256-bit units of -log2(f/2^16); shared by Go and the GPU cost kernel.
var CostL [ProbT + 1]uint32

func init() {
	for f := 1; f <= ProbT; f++ {
		CostL[f] = uint32(math.Round(256 * (ProbBits - math.Log2(float64(f)))))
	}
}

func Mulhi(a, b uint32) uint32 { return uint32((uint64(a) * uint64(b)) >> 32) }

// Quant3i: a_k >= 1, a_k < 2^16. f_k = 1 + a_k*(T-3)/S for k=0,1.
func Quant3i(a0, a1, a2 uint32) (uint32, uint32) {
	s := a0 + a1 + a2
	return 1 + a0*(ProbT-3)/s, 1 + a1*(ProbT-3)/s
}

// TernW: mode-2 weights W_k = floor((2^32-1)*G2min/G2_k), G2_k = 2*gc_k+1.
func TernW(rc []uint32, N int) [3]uint32 {
	var g [3]uint64
	for n := 0; n < N; n++ {
		for k := 0; k < 3; k++ {
			g[k] += uint64(rc[3*n+k])
		}
	}
	var g2 [3]uint64
	mn := uint64(math.MaxUint64)
	for k := range g {
		g2[k] = 2*g[k] + 1
		mn = min(mn, g2[k])
	}
	var w [3]uint32
	for k := range g {
		w[k] = uint32((uint64(math.MaxUint32) * mn) / g2[k])
	}
	return w
}

// Mode2 frequencies for row n (rc counts r0..r2) and column (cc counts c0..c2).
func Mode2F(r0, r1, r2, c0, c1, c2 uint32, w [3]uint32) (uint32, uint32) {
	p0 := Mulhi((2*r0+1)*(2*c0+1), w[0])
	p1 := Mulhi((2*r1+1)*(2*c1+1), w[1])
	p2 := Mulhi((2*r2+1)*(2*c2+1), w[2])
	m := max(p0, p1, p2)
	sh := 0
	if bl := bits.Len32(m); bl > 15 {
		sh = bl - 15
	}
	return Quant3i(p0>>sh+1, p1>>sh+1, p2>>sh+1)
}

type tern4 struct {
	R, C, N, mode int
	rc, cc        []uint32
	rowF, colF    []uint32 // f0 | f1<<16
	w             [3]uint32
}

func newTern4(rc, cc []uint32, R, C int) *tern4 {
	N := 4 * R
	m := &tern4{R: R, C: C, N: N, rc: rc, cc: cc, rowF: make([]uint32, N), colF: make([]uint32, C)}
	for n := 0; n < N; n++ {
		f0, f1 := Quant3i(2*rc[3*n]+1, 2*rc[3*n+1]+1, 2*rc[3*n+2]+1)
		m.rowF[n] = f0 | f1<<16
	}
	for c := 0; c < C; c++ {
		f0, f1 := Quant3i(2*cc[3*c]+1, 2*cc[3*c+1]+1, 2*cc[3*c+2]+1)
		m.colF[c] = f0 | f1<<16
	}
	m.w = TernW(rc, N)
	return m
}

func (m *tern4) freqs(mode, n, c int) (uint32, uint32) {
	switch mode {
	case 0:
		return m.rowF[n] & 0xFFFF, m.rowF[n] >> 16
	case 1:
		return m.colF[c] & 0xFFFF, m.colF[c] >> 16
	}
	r, q := m.rc[3*n:3*n+3], m.cc[3*c:3*c+3]
	return Mode2F(r[0], r[1], r[2], q[0], q[1], q[2], m.w)
}

func TernRPC(C int) int { return max(1, TernSyms/C) }

func parFor(n int, f func(i int)) {
	var wg sync.WaitGroup
	ch := make(chan int, n)
	for i := 0; i < n; i++ {
		ch <- i
	}
	close(ch)
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				f(i)
			}
		}()
	}
	wg.Wait()
}

// chunk directory: u32 offsets[n+1] relative to the streams blob.
func joinChunks(cs [][]byte) (dir, streams []byte) {
	dir = make([]byte, 4*(len(cs)+1))
	tot := 0
	for i, c := range cs {
		binary.LittleEndian.PutUint32(dir[4*i:], uint32(tot))
		tot += len(c)
	}
	binary.LittleEndian.PutUint32(dir[4*len(cs):], uint32(tot))
	streams = make([]byte, 0, tot)
	for _, c := range cs {
		streams = append(streams, c...)
	}
	return
}

func TernOK(R, C int) bool { return C%4 == 0 && C < maxTernDim && 4*R < maxTernDim && R > 0 }

// TernCounts returns rc[N*3], cc[C*3] or nil if a trit is 3.
func TernCounts(raw []byte, R, C int) ([]uint32, []uint32) {
	N := 4 * R
	rc := make([]uint32, 3*N)
	cc := make([]uint32, 3*C)
	for r := 0; r < R; r++ {
		for c := 0; c < C; c++ {
			b := raw[r*C+c]
			for s := 0; s < 4; s++ {
				t := (b >> (2 * s)) & 3
				if t == 3 {
					return nil, nil
				}
				rc[3*(s*R+r)+int(t)]++
				cc[3*c+int(t)]++
			}
		}
	}
	return rc, cc
}

func EncTern4(raw []byte, R, C int) []byte {
	if !TernOK(R, C) {
		return nil
	}
	rc, cc := TernCounts(raw, R, C)
	if rc == nil {
		return nil
	}
	m := newTern4(rc, cc, R, C)
	rpc := TernRPC(C)
	nch := (R + rpc - 1) / rpc
	cost := make([][3]uint64, nch)
	parFor(nch, func(k int) {
		for mode := 0; mode < 3; mode++ {
			var s uint32
			for r := k * rpc; r < min(R, (k+1)*rpc); r++ {
				for c := 0; c < C; c++ {
					b := raw[r*C+c]
					for sp := 0; sp < 4; sp++ {
						f0, f1 := m.freqs(mode, sp*R+r, c)
						_, f := symFS((b>>(2*sp))&3, f0, f1)
						s += CostL[f]
					}
				}
			}
			cost[k][mode] = uint64(s)
		}
	})
	best := 0
	var tot [3]uint64
	for k := range cost {
		for i := 0; i < 3; i++ {
			tot[i] += cost[k][i]
		}
	}
	for i := 1; i < 3; i++ {
		if tot[i] < tot[best] {
			best = i
		}
	}
	return PackTern4(best, rpc, rc, cc, encTernChunks(raw, m, best, rpc))
}

func PackTern4(mode, rpc int, rc, cc []uint32, chunks [][]byte) []byte {
	hdr := make([]byte, 12)
	binary.LittleEndian.PutUint32(hdr, uint32(mode))
	binary.LittleEndian.PutUint32(hdr[4:], uint32(rpc))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(len(chunks)))
	dir, st := joinChunks(chunks)
	return pk(hdr, Z(append(u32bytes(rc), u32bytes(cc)...)), dir, st)
}

func encTernChunks(raw []byte, m *tern4, mode, rpc int) [][]byte {
	R, C := m.R, m.C
	nch := (R + rpc - 1) / rpc
	out := make([][]byte, nch)
	parFor(nch, func(k int) {
		r0, r1 := k*rpc, min(R, (k+1)*rpc)
		e := NewEncoder((r1 - r0) * C)
		for r := r1 - 1; r >= r0; r-- {
			for c := C - 1; c >= 0; c-- {
				b := raw[r*C+c]
				for sp := 3; sp >= 0; sp-- {
					f0, f1 := m.freqs(mode, sp*R+r, c)
					s, f := symFS((b>>(2*sp))&3, f0, f1)
					e.Put(s, f)
				}
			}
		}
		out[k] = e.Bytes()
	})
	return out
}

type Tern4Payload struct {
	Mode, RPC, NCh int
	RC, CC         []uint32
	Dir            []uint32
	Streams        []byte
}

func ParseTern4(b []byte, R, C int) (*Tern4Payload, error) {
	p, err := upk(b)
	if err != nil || len(p) != 4 || len(p[0]) != 12 {
		return nil, errors.New("tern4: bad payload")
	}
	N := 4 * R
	t := &Tern4Payload{Mode: int(binary.LittleEndian.Uint32(p[0])), RPC: int(binary.LittleEndian.Uint32(p[0][4:])), NCh: int(binary.LittleEndian.Uint32(p[0][8:]))}
	tb, err := UZ(p[1])
	if err != nil || len(tb) != 4*3*(N+C) || t.Mode > 2 || t.RPC < 1 || t.NCh != (R+t.RPC-1)/t.RPC || len(p[2]) != 4*(t.NCh+1) {
		return nil, errors.New("tern4: bad tables")
	}
	tv := bytesU32(tb)
	t.RC, t.CC, t.Dir, t.Streams = tv[:3*N], tv[3*N:], bytesU32(p[2]), p[3]
	if int(t.Dir[t.NCh]) != len(t.Streams) {
		return nil, errors.New("tern4: bad dir")
	}
	return t, nil
}

func DecTern4(b []byte, R, C int) ([]byte, error) {
	t, err := ParseTern4(b, R, C)
	if err != nil {
		return nil, err
	}
	m := newTern4(t.RC, t.CC, R, C)
	out := make([]byte, R*C)
	var bad sync.Once
	var e error
	parFor(t.NCh, func(k int) {
		r0, r1 := k*t.RPC, min(R, (k+1)*t.RPC)
		d := NewDecoder(t.Streams[t.Dir[k]:t.Dir[k+1]])
		for r := r0; r < r1; r++ {
			for c := 0; c < C; c++ {
				var by byte
				for sp := 0; sp < 4; sp++ {
					f0, f1 := m.freqs(t.Mode, sp*R+r, c)
					x := d.Slot()
					var tt byte
					if x < f0 {
						d.Advance(0, f0)
					} else if x < f0+f1 {
						tt = 1
						d.Advance(f0, f1)
					} else {
						tt = 2
						d.Advance(f0+f1, ProbT-f0-f1)
					}
					by |= tt << (2 * sp)
				}
				out[r*C+c] = by
			}
		}
		if d.Overrun() {
			bad.Do(func() { e = errors.New("tern4: overrun") })
		}
	})
	return out, e
}

// ---------------- bf16 chunked ----------------

func BF16OK(n int) bool { return n >= 2 && n%2 == 0 && n < 1<<31 }

func EncBF164(raw []byte, shape []int) []byte {
	n := len(raw) / 2
	if !BF16OK(n) {
		return nil
	}
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
	nch := (n + BF16Elems - 1) / BF16Elems
	chunks := make([][]byte, nch)
	parFor(nch, func(k int) {
		i0, i1 := k*BF16Elems, min(n, (k+1)*BF16Elems)
		enc := NewEncoder(2 * (i1 - i0))
		for i := i1 - 1; i >= i0; i-- {
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
		chunks[k] = enc.Bytes()
	})
	return PackBF164(emin, K, C, ecnt, mcnt, sign, chunks)
}

func PackBF164(emin, K, C int, ecnt, mcnt []uint32, sign []byte, chunks [][]byte) []byte {
	hdr := make([]byte, 20)
	binary.LittleEndian.PutUint32(hdr, uint32(int32(emin)))
	binary.LittleEndian.PutUint32(hdr[4:], uint32(K))
	binary.LittleEndian.PutUint32(hdr[8:], uint32(C))
	binary.LittleEndian.PutUint32(hdr[12:], BF16Elems)
	binary.LittleEndian.PutUint32(hdr[16:], uint32(len(chunks)))
	dir, st := joinChunks(chunks)
	return pk(hdr, Z(append(u32bytes(ecnt), u32bytes(mcnt)...)), Z(sign), dir, st)
}

type BF164Payload struct {
	Emin, K, C, EPC, NCh int
	Cnt                  []uint32 // ecnt[C*K] ++ mcnt[K*128]
	Sign                 []byte
	Dir                  []uint32
	Streams              []byte
}

func ParseBF164(b []byte, n int) (*BF164Payload, error) {
	p, err := upk(b)
	if err != nil || len(p) != 5 || len(p[0]) != 20 {
		return nil, errors.New("bf16x: bad payload")
	}
	h := p[0]
	t := &BF164Payload{Emin: int(int32(binary.LittleEndian.Uint32(h))), K: int(binary.LittleEndian.Uint32(h[4:])),
		C: int(binary.LittleEndian.Uint32(h[8:])), EPC: int(binary.LittleEndian.Uint32(h[12:])), NCh: int(binary.LittleEndian.Uint32(h[16:]))}
	if t.C <= 0 || t.K <= 0 || t.K > 256 || t.EPC <= 0 || t.EPC%2 != 0 || t.NCh != (n+t.EPC-1)/t.EPC || len(p[3]) != 4*(t.NCh+1) {
		return nil, errors.New("bf16x: bad header")
	}
	tb, err := UZ(p[1])
	if err != nil || len(tb) != 4*(t.C*t.K+t.K*128) {
		return nil, errors.New("bf16x: bad tables")
	}
	t.Cnt = bytesU32(tb)
	if t.Sign, err = UZ(p[2]); err != nil || len(t.Sign) != (n+7)/8 {
		return nil, errors.New("bf16x: bad signs")
	}
	t.Dir, t.Streams = bytesU32(p[3]), p[4]
	if int(t.Dir[t.NCh]) != len(t.Streams) {
		return nil, errors.New("bf16x: bad dir")
	}
	return t, nil
}

func DecBF164(b []byte, shape []int) ([]byte, error) {
	n := 1
	for _, s := range shape {
		n *= s
	}
	t, err := ParseBF164(b, n)
	if err != nil {
		return nil, err
	}
	C, K := t.C, t.K
	et := make([]*table, C)
	for c := range et {
		et[c] = mkTable(t.Cnt[c*K:(c+1)*K], false)
	}
	mt := make([]*table, K)
	for k := range mt {
		mt[k] = mkTable(t.Cnt[C*K+k*128:C*K+(k+1)*128], true)
	}
	out := make([]byte, 2*n)
	var bad sync.Once
	var e error
	parFor(t.NCh, func(k int) {
		i0, i1 := k*t.EPC, min(n, (k+1)*t.EPC)
		d := NewDecoder(t.Streams[t.Dir[k]:t.Dir[k+1]])
		for i := i0; i < i1; i++ {
			tb := et[i%C]
			ex := tb.only
			if tb.f != nil {
				ex = tb.find(d.Slot())
				d.Advance(tb.cum[ex], tb.f[ex])
			}
			m := mt[ex]
			s := m.only
			if m.f != nil {
				s = m.find(d.Slot())
				d.Advance(m.cum[s], m.f[s])
			}
			x := uint16(ex+t.Emin)<<7 | uint16(s)
			if t.Sign[i/8]&(0x80>>(i%8)) != 0 {
				x |= 0x8000
			}
			binary.LittleEndian.PutUint16(out[2*i:], x)
		}
		if d.Overrun() {
			bad.Do(func() { e = errors.New("bf16x: overrun") })
		}
	})
	return out, e
}

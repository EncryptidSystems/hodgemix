package bnz4

// 32-bit rANS, byte-wise renormalization, 16-bit quantized probabilities.
const (
	ProbBits = 16
	ProbT    = 1 << ProbBits
	ransL    = 1 << 23
)

// Encoder encodes symbols in reverse order; Bytes() returns the stream for forward decoding.
type Encoder struct {
	x   uint32
	buf []byte // written backwards (appended, reversed at the end)
}

func NewEncoder(sizeHint int) *Encoder { return &Encoder{x: ransL, buf: make([]byte, 0, sizeHint)} }

func (e *Encoder) Put(start, freq uint32) {
	x := e.x
	xmax := ((ransL >> ProbBits) << 8) * freq
	for x >= xmax {
		e.buf = append(e.buf, byte(x))
		x >>= 8
	}
	e.x = (x/freq)<<ProbBits + x%freq + start
}

func (e *Encoder) Bytes() []byte {
	x := e.x
	for i := 0; i < 4; i++ {
		e.buf = append(e.buf, byte(x))
		x >>= 8
	}
	b := e.buf
	for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}
	return b
}

type Decoder struct {
	x   uint32
	b   []byte
	pos int
}

func NewDecoder(b []byte) *Decoder {
	d := &Decoder{b: b}
	for i := 0; i < 4; i++ {
		d.x = d.x<<8 | uint32(d.byte())
	}
	return d
}

func (d *Decoder) byte() byte {
	if d.pos < len(d.b) {
		c := d.b[d.pos]
		d.pos++
		return c
	}
	d.pos++
	return 0
}

func (d *Decoder) Slot() uint32 { return d.x & (ProbT - 1) }

func (d *Decoder) Advance(start, freq uint32) {
	x := freq*(d.x>>ProbBits) + d.x&(ProbT-1) - start
	for x < ransL {
		x = x<<8 | uint32(d.byte())
	}
	d.x = x
}

func (d *Decoder) Overrun() bool { return d.pos > len(d.b) }

// Quantize maps counts to frequencies summing to ProbT, every nonzero count gets >=1.
// Integer only, so encoder and decoder agree on every platform.
// Returns nil when fewer than two symbols are present (deterministic, no bits coded).
func Quantize(counts []uint32) []uint32 {
	var total uint64
	nz := 0
	for _, c := range counts {
		total += uint64(c)
		if c > 0 {
			nz++
		}
	}
	if nz < 2 {
		return nil
	}
	f := make([]uint32, len(counts))
	var sum int64
	for i, c := range counts {
		if c > 0 {
			v := uint64(c) * ProbT / total
			if v < 1 {
				v = 1
			}
			f[i] = uint32(v)
			sum += int64(v)
		}
	}
	for sum != ProbT {
		// adjust the largest frequency (lowest index on ties)
		bi := 0
		for i := range f {
			if f[i] > f[bi] {
				bi = i
			}
		}
		if sum < ProbT {
			f[bi] += uint32(ProbT - sum)
			sum = ProbT
		} else {
			d := sum - ProbT
			if int64(f[bi])-1 < d {
				d = int64(f[bi]) - 1
			}
			f[bi] -= uint32(d)
			sum -= d
		}
	}
	return f
}

func Only(counts []uint32) int {
	for i, c := range counts {
		if c > 0 {
			return i
		}
	}
	return 0
}

func Cum(f []uint32) []uint32 {
	c := make([]uint32, len(f)+1)
	for i, v := range f {
		c[i+1] = c[i] + v
	}
	return c
}

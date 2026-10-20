package eval

// BitLinear as a LUT fabric over dynamic MRH digit limbs (digit-ring exterior calculus,
// unified_main.pdf; WASMApollo DECLUT).
//
// A weight matrix is a sum of digit limbs, W = sum_l s_l D_l, each D_l a matrix of digits of radix
// r_l in {2, 3, 4} (the TLUT-native alphabet, 2 bits a digit), digit d meaning the integer d - c_r with
// c_r = (r-1)/2 (r = 3: {-1, 0, +1}, the ternary weight; r = 4: {-1..2}; r = 2: {0, 1}). Limb 0 of a
// BitNet checkpoint is its ternary weight; further limbs carry what the earlier ones did not.
//
// Each limb is a signed incidence operator: D x is a coboundary of the activation cochain, additions
// only. The input axis is cut into blocks of K channels on its own radix levels: K is the largest
// divisor of the axis length with r^K <= 256 (2560 = 2^9 5 -> K = 5 at r = 3; 6912 = 2^8 3^3 -> K = 4),
// so blocks tile the axis exactly and a row's block is one byte, the mixed-radix numeral
// c = sum_t d_t r^t. That numeral is the address of a TLUT over the block: per token
// T_b[c] = sum_t (digit_t(c) - c_r) x_{Kb+t}, filled digit by digit (the levelized compile), and every
// row is gather + add, y_r = sum_b T_b[c_rb]. Exact integer arithmetic; no multiplies, no SIMD.

import "sync"

// limb is one digit matrix of a BitLinear.
type limb struct {
	r, K, nb int
	code     []uint8 // [n][nb] block numerals
	s        float32 // the limb's scale
}

func (lb *limb) center() int { return (lb.r - 1) / 2 }

// blockFor is the block size for radix r on an axis of length k: the largest divisor of k with
// r^K <= 256 (one-byte codes).
func blockFor(r, k int) int {
	best := 1
	p := r
	for K := 1; p <= 256; K++ {
		if k%K == 0 {
			best = K
		}
		p *= r
	}
	return best
}

// newLimb packs digits d[n][k] (each 0..r-1) of radix r.
func newLimb(d []uint8, n, k, r int, s float32) *limb {
	K := blockFor(r, k)
	nb := k / K
	lb := &limb{r: r, K: K, nb: nb, code: make([]uint8, n*nb), s: s}
	for i := 0; i < n; i++ {
		row := d[i*k : (i+1)*k]
		for b := 0; b < nb; b++ {
			c, p := 0, 1
			for t := 0; t < K; t++ {
				c += int(row[b*K+t]) * p
				p *= r
			}
			lb.code[i*nb+b] = uint8(c)
		}
	}
	return lb
}

// pack turns the expanded ternary weights into limb 0 and drops them.
func (l *lin) pack() {
	d := make([]uint8, len(l.w))
	for i, v := range l.w {
		d[i] = uint8(v + 1)
	}
	l.limbs = []*limb{newLimb(d, l.n, l.k, 3, l.s)}
	l.w = nil
}

// luts is one input's tables for one (radix, K): block b's r^K signed sums at t[b*size:].
type luts struct {
	size int
	t    []int32
}

// input is one BitLinear input (int8 activations) and its tables, built once per (radix, K) and
// shared by every limb and every matrix that reads it.
type input struct {
	q     []int8
	mu    sync.Mutex
	cache map[[2]int]*luts
}

func newInput(q []int8) *input { return &input{q: q, cache: map[[2]int]*luts{}} }

func (in *input) tables(r, K int) *luts {
	in.mu.Lock()
	defer in.mu.Unlock()
	key := [2]int{r, K}
	if tb := in.cache[key]; tb != nil {
		return tb
	}
	q := in.q
	size := 1
	for t := 0; t < K; t++ {
		size *= r
	}
	nb := len(q) / K
	c0 := int32((r - 1) / 2)
	tb := &luts{size: size, t: make([]int32, nb*size)}
	fill := func(b int) {
		dst := tb.t[b*size : (b+1)*size]
		dst[0] = 0
		n := 1
		for t := 0; t < K; t++ {
			a := int32(q[b*K+t])
			// Entries [0, n) hold the sums of the lower digits; digit t = d adds (d - c_r) a and moves
			// the entry to c + d n. Walking c downwards reads each entry before it is overwritten.
			for c := n - 1; c >= 0; c-- {
				v := dst[c]
				for d := int32(r - 1); d >= 0; d-- {
					dst[c+int(d)*n] = v + (d-c0)*a
				}
			}
			n *= r
		}
	}
	if nb < 64 {
		for b := 0; b < nb; b++ {
			fill(b)
		}
	} else {
		var wg sync.WaitGroup
		const P = 8
		for p := 0; p < P; p++ {
			wg.Add(1)
			go func(p int) {
				defer wg.Done()
				for b := p; b < nb; b += P {
					fill(b)
				}
			}(p)
		}
		wg.Wait()
	}
	in.cache[key] = tb
	return tb
}

// applyIn is y = (sum_l s_l D_l q) xs by gather + add over the input's tables.
func (l *lin) applyIn(in *input, xs float32, out []float32, accum bool) {
	tbs := make([]*luts, len(l.limbs))
	for i, lb := range l.limbs {
		tbs[i] = in.tables(lb.r, lb.K)
	}
	var wg sync.WaitGroup
	const P = 8
	for p := 0; p < P; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for r := p; r < l.n; r += P {
				var v float32
				for i, lb := range l.limbs {
					t, size, nb := tbs[i].t, tbs[i].size, lb.nb
					codes := lb.code[r*nb : (r+1)*nb]
					var acc int32
					for b, c := range codes {
						acc += t[b*size+int(c)]
					}
					v += float32(acc) * xs * lb.s // the original order: one ternary limb is bit-identical
				}
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

package eval

import (
	"math"
	"math/rand"
	"testing"
)

// One ternary limb (gather + add over block numerals) is exactly the multiply-accumulate.
func TestLUTMatchesMAC(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, sz := range [][2]int{{37, 25}, {8, 5}, {64, 2560}, {16, 6912}, {3, 7}} {
		n, k := sz[0], sz[1]
		w := make([]int8, n*k)
		for i := range w {
			w[i] = int8(rng.Intn(3) - 1)
		}
		q := make([]int8, k)
		for i := range q {
			q[i] = int8(rng.Intn(256) - 128)
		}
		mac := &lin{n: n, k: k, w: append([]int8(nil), w...), s: 0.75}
		lut := &lin{n: n, k: k, w: append([]int8(nil), w...), s: 0.75}
		lut.pack()
		a, b := make([]float32, n), make([]float32, n)
		mac.apply(q, 0.5, a, false)
		lut.applyIn(newInput(q), 0.5, b, false)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%dx%d row %d: MAC %v, LUT %v", n, k, i, a[i], b[i])
			}
		}
	}
}

// Limbs of radix 2, 3 and 4 sum to the dense sum_l s_l (d_l - c_r) q, and the blocks tile each axis
// on its radix levels.
func TestMultiLimb(t *testing.T) {
	if K := blockFor(3, 2560); K != 5 {
		t.Fatalf("radix 3 on 2560: K=%d, want 5", K)
	}
	if K := blockFor(3, 6912); K != 4 {
		t.Fatalf("radix 3 on 6912: K=%d, want 4", K)
	}
	if K := blockFor(4, 2560); K != 4 {
		t.Fatalf("radix 4 on 2560: K=%d, want 4", K)
	}
	rng := rand.New(rand.NewSource(2))
	n, k := 19, 240
	q := make([]int8, k)
	for i := range q {
		q[i] = int8(rng.Intn(256) - 128)
	}
	l := &lin{n: n, k: k}
	ref := make([]float64, n)
	for _, rs := range []struct {
		r int
		s float32
	}{{3, 0.5}, {4, 0.125}, {2, 0.03125}} {
		d := make([]uint8, n*k)
		for i := range d {
			d[i] = uint8(rng.Intn(rs.r))
		}
		l.limbs = append(l.limbs, newLimb(d, n, k, rs.r, rs.s))
		c := (rs.r - 1) / 2
		for i := 0; i < n; i++ {
			var acc float64
			for j := 0; j < k; j++ {
				acc += float64(int(d[i*k+j])-c) * float64(q[j])
			}
			ref[i] += acc * float64(rs.s) * 0.25
		}
	}
	out := make([]float32, n)
	l.applyIn(newInput(q), 0.25, out, false)
	for i := range out {
		if math.Abs(float64(out[i])-ref[i]) > 1e-3*math.Max(1, math.Abs(ref[i])) {
			t.Fatalf("row %d: LUT %v, dense %v", i, out[i], ref[i])
		}
	}
}

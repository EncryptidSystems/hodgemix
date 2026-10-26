package eval

import (
	"math/rand"
	"testing"
)

func nafValue(d [8]int8) int {
	v := 0
	for t := 0; t < 8; t++ {
		v += int(d[t]) << uint(t)
	}
	return v
}

// TestNAFIdentity checks every int8 code: one non-adjacent form, weight at most 4,
// and that form is the one NAFDigits returns.
func TestNAFIdentity(t *testing.T) {
	var cur [8]int8
	count := make([]int, 256)
	which := make([][8]int8, 256)
	var walk func(tPos int, prev bool)
	walk = func(tPos int, prev bool) {
		if tPos == 8 {
			v := nafValue(cur)
			if v < -128 || v > 127 {
				return
			}
			count[v+128]++
			which[v+128] = cur
			return
		}
		cur[tPos] = 0
		walk(tPos+1, false)
		if prev {
			return
		}
		cur[tPos] = 1
		walk(tPos+1, true)
		cur[tPos] = -1
		walk(tPos+1, true)
	}
	walk(0, false)
	for q := -128; q <= 127; q++ {
		if count[q+128] != 1 {
			t.Fatalf("code %d has %d non-adjacent forms in 8 digits", q, count[q+128])
		}
		d := NAFDigits(q)
		if d != which[q+128] {
			t.Fatalf("code %d: NAFDigits %v, the only form is %v", q, d, which[q+128])
		}
		if nafValue(d) != q {
			t.Fatalf("code %d reconstructs to %d", q, nafValue(d))
		}
		for tPos := 0; tPos < 7; tPos++ {
			if d[tPos] != 0 && d[tPos+1] != 0 {
				t.Fatalf("code %d has adjacent digits %v", q, d)
			}
		}
		if w := NAFWeight(q); w > 4 {
			t.Fatalf("code %d weight %d", q, w)
		}
	}
}

func TestEventLinearExact(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, r := range []int{2, 3, 4} {
		const n, k = 12, 16
		digits := make([]uint8, n*k)
		for i := range digits {
			digits[i] = uint8(rng.Intn(r))
		}
		q := make([]int8, k)
		for i := range q {
			q[i] = int8(rng.Intn(256) - 128)
		}
		q[0], q[3], q[k-1] = 0, 0, 0
		center := (r - 1) / 2
		got := EventAcc(digits, n, k, center, q)
		naf := nafAcc(digits, n, k, center, q)
		for i := range got {
			if got[i] != naf[i] {
				t.Fatalf("radix %d row %d: graded %d, NAF %d", r, i, got[i], naf[i])
			}
		}
		const xs, scale float32 = 0.25, 0.5
		l := &lin{n: n, k: k, limbs: []*limb{newLimb(digits, n, k, r, scale)}}
		out := make([]float32, n)
		l.applyIn(newInput(q), xs, out, false)
		for i := range got {
			want := float32(got[i]) * xs * scale
			if out[i] != want {
				t.Fatalf("radix %d row %d: applyIn %v, event %v", r, i, out[i], want)
			}
		}
	}

	// Two limbs: the sum of the two scaled accumulators, in applyIn's order.
	const n, k = 8, 16
	rng = rand.New(rand.NewSource(2))
	d0 := make([]uint8, n*k)
	d1 := make([]uint8, n*k)
	for i := range d0 {
		d0[i] = uint8(rng.Intn(3))
		d1[i] = uint8(rng.Intn(4))
	}
	q := make([]int8, k)
	for i := range q {
		q[i] = int8(rng.Intn(256) - 128)
	}
	const xs float32 = 1.5
	const s0, s1 float32 = 0.125, 2
	a0 := EventAcc(d0, n, k, 1, q)
	a1 := EventAcc(d1, n, k, 1, q)
	l := &lin{n: n, k: k, limbs: []*limb{newLimb(d0, n, k, 3, s0), newLimb(d1, n, k, 4, s1)}}
	out := make([]float32, n)
	l.applyIn(newInput(q), xs, out, false)
	for i := range out {
		want := float32(a0[i])*xs*s0 + float32(a1[i])*xs*s1
		if out[i] != want {
			t.Fatalf("two limbs row %d: applyIn %v, events %v", i, out[i], want)
		}
	}
}

// Every int8 code, one column: the NAF expansion equals the graded spike.
func TestNAFMatchesGraded(t *testing.T) {
	digits := []uint8{0, 1, 2, 3}
	q := []int8{0}
	for code := -128; code <= 127; code++ {
		q[0] = int8(code)
		g := EventAcc(digits, 4, 1, 1, q)
		n := nafAcc(digits, 4, 1, 1, q)
		for i := range g {
			if g[i] != n[i] {
				t.Fatalf("code %d row %d: graded %d, NAF %d", code, i, g[i], n[i])
			}
		}
	}
}

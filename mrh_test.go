package main

import (
	"math/rand"
	"sort"
	"testing"
)

// Random monotone (Dijkstra-shaped) sequences: inserts in [mu, mu+C], decrease-keys, extract-mins.
// Every extract must equal the reference minimum (FIFO among equal keys) and I1-I4 must hold after
// every operation, for uniform, binary and unequal radix tuples.
func TestMRHAgainstReference(t *testing.T) {
	cases := [][]int{{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, {16, 16, 16}, {5, 2, 2, 2, 2, 2, 2, 2, 2, 2}, {3, 7, 5, 2, 11}, {7}}
	for ci, r := range cases {
		C := 1
		for _, x := range r {
			C *= x
		}
		C--
		rng := rand.New(rand.NewSource(int64(ci + 1)))
		h, err := NewMRH(r, C)
		if err != nil {
			t.Fatal(err)
		}
		type rec struct{ key, seq int }
		ref := map[int]rec{}
		seq, next := 0, 0
		for step := 0; step < 4000; step++ {
			switch op := rng.Intn(10); {
			case op < 5 || len(ref) == 0:
				k := h.Mu + rng.Intn(C+1)
				if err := h.Insert(next, k); err != nil {
					t.Fatalf("%v insert: %v", r, err)
				}
				ref[next] = rec{k, seq}
				seq++
				next++
			case op < 7:
				for x, v := range ref {
					if v.key > h.Mu {
						k := h.Mu + rng.Intn(v.key-h.Mu)
						if err := h.DecreaseKey(x, k); err != nil {
							t.Fatalf("%v decrease: %v", r, err)
						}
						ref[x] = rec{k, seq}
						seq++
					}
					break
				}
			default:
				best, bx := rec{1 << 62, 0}, -1
				for x, v := range ref {
					if v.key < best.key || v.key == best.key && v.seq < best.seq {
						best, bx = v, x
					}
				}
				x, k, ok := h.ExtractMin()
				if !ok || k != best.key || x != bx {
					t.Fatalf("%v step %d: extract (%d,%d,%v), want (%d,%d)", r, step, x, k, ok, bx, best.key)
				}
				delete(ref, x)
			}
			if err := h.Validate(); err != nil {
				t.Fatalf("%v step %d: %v", r, step, err)
			}
			if h.Len() != len(ref) {
				t.Fatalf("len %d want %d", h.Len(), len(ref))
			}
		}
	}
}

// The MRH pop order of channel keys is the (key, index) order, for the dynamic radices of real
// channel counts; this is what makes Eq.(8) channel coordinates heap-defined.
func TestPopOrderIsKeyIndexOrder(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, n := range []int{1024, 2560, 2880, 3072, 6912, 151936 / 64} {
		keys := make([]int, n)
		for i := range keys {
			keys[i] = rng.Intn(1 << KeyBits)
		}
		r := RadicesFor(n, 1<<KeyBits-1)
		got, err := PopOrder(keys, r, 1<<KeyBits-1)
		if err != nil {
			t.Fatal(err)
		}
		want := make([]int, n)
		for i := range want {
			want[i] = i
		}
		sort.SliceStable(want, func(a, b int) bool { return keys[want[a]] < keys[want[b]] })
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("n=%d r=%v: pop %d is %d, want %d", n, r, i, got[i], want[i])
			}
		}
	}
}

func TestRadicesFor(t *testing.T) {
	for _, c := range []struct {
		n    int
		want []int
	}{
		{2560, []int{2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2}},  // 2^9*5 = 2560 < 4096: one more 2
		{6912, []int{2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3}},  // 2^8*3^3 = 6912 >= 4096
		{2880, []int{2, 2, 2, 2, 2, 2, 3, 3, 5, 2}},     // 2^6*3^2*5 = 2880: one more 2
		{1187, []int{1187, 2, 2}},                       // a prime channel count is one radix
	} {
		got := RadicesFor(c.n, 4095)
		if len(got) != len(c.want) {
			t.Fatalf("RadicesFor(%d) = %v, want %v", c.n, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("RadicesFor(%d) = %v, want %v", c.n, got, c.want)
			}
		}
	}
}

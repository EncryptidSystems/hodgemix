package main

// Multi-radix heap MRH(r; C), as defined in multi_radix_heap.pdf (Disney, technical note):
// a monotone integer min-priority queue whose buckets form a k-level hierarchy addressed by the
// mixed-radix place values pi_0 = 1, pi_{j+1} = pi_j * r_j (Def. 2). Level j has r_j buckets of
// width pi_j covering [L_j, L_j + pi_{j+1}) (Def. 3); the block of level j-1 is the active bucket
// B_j[alpha_j] of level j (I3); an item sits in the least level whose block holds its key (I4).
// Buckets are FIFO lists, so equal keys leave in insertion order (paper_v2 Eq.8: ties FIFO by index).

import (
	"errors"
	"fmt"
)

type MRH struct {
	R     []int // radices r_0..r_{k-1}
	Pi    []int // place values pi_0..pi_k
	C     int   // universe bound: keys lie in [mu, mu+C] when inserted (I5)
	Mu    int   // frontier
	L     []int // level lower bounds L_j
	Alpha []int // active indices alpha_j

	key, lvl, bkt, next, prev []int // item records and handles; -1 = none
	head, tail                [][]int
	n                         int
}

// NewMRH builds MRH(r; C). The radices must satisfy r_i >= 2 and prod r_i >= C+1 (Def. 2).
func NewMRH(r []int, C int) (*MRH, error) {
	if len(r) == 0 {
		return nil, errors.New("mrh: no radices")
	}
	h := &MRH{R: append([]int(nil), r...), C: C, Pi: make([]int, len(r)+1)}
	h.Pi[0] = 1
	for j, rj := range r {
		if rj < 2 {
			return nil, fmt.Errorf("mrh: radix r_%d = %d < 2", j, rj)
		}
		h.Pi[j+1] = h.Pi[j] * rj
	}
	if h.Pi[len(r)] < C+1 {
		return nil, fmt.Errorf("mrh: prod r = %d < C+1 = %d", h.Pi[len(r)], C+1)
	}
	k := len(r)
	h.L, h.Alpha = make([]int, k), make([]int, k)
	h.head, h.tail = make([][]int, k), make([][]int, k)
	for j := range r {
		h.head[j], h.tail[j] = fillNeg(r[j]), fillNeg(r[j])
	}
	return h, nil
}

func fillNeg(n int) []int {
	s := make([]int, n)
	for i := range s {
		s[i] = -1
	}
	return s
}

// RadicesFor is the dynamic radix tuple for n channels: the factorisation of n, finest digit first
// (level 0 is the finest), extended by 2s until the place values cover the key universe C+1.
func RadicesFor(n, C int) []int {
	f := radices(n) // coarsest first
	r := make([]int, 0, len(f)+4)
	for i := len(f) - 1; i >= 0; i-- {
		if f[i] >= 2 {
			r = append(r, f[i])
		}
	}
	p := 1
	for _, x := range r {
		p *= x
	}
	for p < C+1 {
		r = append(r, 2)
		p *= 2
	}
	return r
}

// Len is the number of items in the heap.
func (h *MRH) Len() int { return h.n }

// Key is item x's current key.
func (h *MRH) Key(x int) int { return h.key[x] }

// Handle is item x's (level, bucket), or (-1, -1) if it is not in the heap.
func (h *MRH) Handle(x int) (int, int) {
	if x >= len(h.lvl) {
		return -1, -1
	}
	return h.lvl[x], h.bkt[x]
}

// find is Sec. 4.1: the least level j whose block [L_j, L_j + pi_{j+1}) holds kappa, and its bucket.
func (h *MRH) find(kappa int) (int, int, bool) {
	for j := range h.R {
		if kappa >= h.L[j] && kappa < h.L[j]+h.Pi[j+1] {
			return j, (kappa - h.L[j]) / h.Pi[j], true
		}
	}
	return -1, -1, false
}

func (h *MRH) grow(x int) {
	for len(h.key) <= x {
		h.key = append(h.key, 0)
		h.lvl = append(h.lvl, -1)
		h.bkt = append(h.bkt, -1)
		h.next = append(h.next, -1)
		h.prev = append(h.prev, -1)
	}
}

func (h *MRH) link(x, j, t int) {
	h.lvl[x], h.bkt[x], h.next[x], h.prev[x] = j, t, -1, h.tail[j][t]
	if h.tail[j][t] >= 0 {
		h.next[h.tail[j][t]] = x
	} else {
		h.head[j][t] = x
	}
	h.tail[j][t] = x
}

func (h *MRH) unlink(x int) {
	j, t := h.lvl[x], h.bkt[x]
	if h.prev[x] >= 0 {
		h.next[h.prev[x]] = h.next[x]
	} else {
		h.head[j][t] = h.next[x]
	}
	if h.next[x] >= 0 {
		h.prev[h.next[x]] = h.prev[x]
	} else {
		h.tail[j][t] = h.prev[x]
	}
	h.lvl[x], h.bkt[x], h.next[x], h.prev[x] = -1, -1, -1, -1
}

// rebase restarts the hierarchy at the frontier when a key in [mu, mu+C] lies past the top block
// (the modular wrap of Def. 3, done by reinsertion). Keys keep their FIFO order within a key.
func (h *MRH) rebase() {
	var items []int
	for j := range h.R {
		for t := range h.R[j] {
			for x := h.head[j][t]; x >= 0; x = h.next[x] {
				items = append(items, x)
			}
		}
	}
	for _, x := range items {
		h.unlink(x)
	}
	for j := range h.R {
		h.L[j], h.Alpha[j] = h.Mu, 0
	}
	for _, x := range items {
		j, t, _ := h.find(h.key[x])
		h.link(x, j, t)
	}
}

// Insert is Sec. 4.2. Item ids are small non-negative ints chosen by the caller.
func (h *MRH) Insert(x, kappa int) error {
	if kappa < h.Mu || kappa > h.Mu+h.C {
		return fmt.Errorf("mrh: key %d outside [mu, mu+C] = [%d, %d] (I5)", kappa, h.Mu, h.Mu+h.C)
	}
	h.grow(x)
	if h.lvl[x] >= 0 {
		return fmt.Errorf("mrh: item %d already in the heap", x)
	}
	h.key[x] = kappa
	j, t, ok := h.find(kappa)
	if !ok {
		h.rebase()
		j, t, _ = h.find(kappa)
	}
	h.link(x, j, t)
	h.n++
	return nil
}

// DecreaseKey is Sec. 4.3: mu <= kappa' < K(x); the item moves to an equal or finer level.
func (h *MRH) DecreaseKey(x, kappa int) error {
	if x >= len(h.lvl) || h.lvl[x] < 0 {
		return fmt.Errorf("mrh: item %d not in the heap", x)
	}
	if kappa < h.Mu || kappa >= h.key[x] {
		return fmt.Errorf("mrh: decrease-key %d not in [mu, K(x)) = [%d, %d)", kappa, h.Mu, h.key[x])
	}
	h.unlink(x)
	h.key[x] = kappa
	j, t, _ := h.find(kappa)
	h.link(x, j, t)
	return nil
}

// ExtractMin is Sec. 4.4: the least key, FIFO among equal keys; mu becomes that key.
func (h *MRH) ExtractMin() (x, kappa int, ok bool) {
	if h.n == 0 {
		return -1, 0, false
	}
	for {
		for t := h.Alpha[0]; t < h.R[0]; t++ {
			if h.head[0][t] >= 0 {
				h.Alpha[0] = t
				x = h.head[0][t]
				h.unlink(x)
				h.n--
				h.Mu = h.key[x]
				return x, h.Mu, true
			}
		}
		h.expand()
	}
}

// expand: the least level j* >= 1 with a nonempty bucket past its active one becomes the active
// path; that bucket's items are redistributed into the finer levels (each moves strictly finer).
func (h *MRH) expand() {
	for js := 1; js < len(h.R); js++ {
		for t := h.Alpha[js] + 1; t < h.R[js]; t++ {
			if h.head[js][t] < 0 {
				continue
			}
			h.Alpha[js] = t
			base := h.L[js] + t*h.Pi[js]
			for i := js - 1; i >= 0; i-- {
				h.L[i], h.Alpha[i] = base, 0
			}
			var items []int
			for y := h.head[js][t]; y >= 0; y = h.next[y] {
				items = append(items, y)
			}
			for _, y := range items {
				h.unlink(y)
				j, b, _ := h.find(h.key[y])
				h.link(y, j, b)
			}
			return
		}
	}
	panic("mrh: expand found no item although Len > 0")
}

// Validate checks invariants I1-I4 of Def. 4 for the current configuration.
func (h *MRH) Validate() error {
	k := len(h.R)
	for j := 1; j < k; j++ {
		if h.L[j-1] != h.L[j]+h.Alpha[j]*h.Pi[j] {
			return fmt.Errorf("I3: L_%d = %d, want L_%d + alpha_%d pi_%d = %d", j-1, h.L[j-1], j, j, j, h.L[j]+h.Alpha[j]*h.Pi[j])
		}
	}
	count := 0
	for j := 0; j < k; j++ {
		for t := 0; t < h.R[j]; t++ {
			for x := h.head[j][t]; x >= 0; x = h.next[x] {
				count++
				kx := h.key[x]
				if kx < h.Mu {
					return fmt.Errorf("I1: item %d key %d < mu %d", x, kx, h.Mu)
				}
				lo := h.L[j] + t*h.Pi[j]
				if kx < lo || kx >= lo+h.Pi[j] {
					return fmt.Errorf("I2: item %d key %d not in B_%d[%d] = [%d, %d)", x, kx, j, t, lo, lo+h.Pi[j])
				}
				if lj, _, _ := h.find(kx); lj != j {
					return fmt.Errorf("I4: item %d key %d at level %d, least level is %d", x, kx, j, lj)
				}
			}
		}
	}
	if count != h.n {
		return fmt.Errorf("mrh: %d items linked, Len %d", count, h.n)
	}
	return nil
}

// PopOrder inserts keys[i] as item i (in index order) and extracts them all: the MRH pop order.
func PopOrder(keys []int, r []int, C int) ([]int, error) {
	h, err := NewMRH(r, C)
	if err != nil {
		return nil, err
	}
	for i, k := range keys {
		if err := h.Insert(i, k); err != nil {
			return nil, err
		}
	}
	out := make([]int, 0, len(keys))
	for h.Len() > 0 {
		x, _, _ := h.ExtractMin()
		out = append(out, x)
	}
	return out, nil
}

// gridRadices is the radix tuple of a grid axis of n cells as this run's MRH builds its levels:
// --mrh dynamic orders the factorisation finest first (RadicesFor's order, without padding to the
// key universe); --mrh paper keeps paper_v2's coarsest-first factorisation. The BNZ4 channel map
// records it, so a decoder rebuilds the same levels.
func gridRadices(n int) []int {
	f := radices(n)
	if MRHMode != "dynamic" {
		return f
	}
	r := make([]int, len(f))
	for i, x := range f {
		r[len(f)-1-i] = x
	}
	return r
}

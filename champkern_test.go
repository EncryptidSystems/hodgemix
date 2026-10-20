package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// HODGEMIX_DAWN=<dawn.exe> (WASMApollo's Dawn driver), HODGEMIX_CHAMP_KERNELS=<examples/bitnet/kernels>:
// Champ's flow kernels on the GPU against the CPU sums eval/flow.go makes. flow.wgsl's integer
// (row cell, input cell) currents must equal the exact sums of a_j (code - CENTER); cellsum.wgsl's
// cell sums must match to float rounding.
func TestChampFlowKernels(t *testing.T) {
	dawn, kdir := os.Getenv("HODGEMIX_DAWN"), os.Getenv("HODGEMIX_CHAMP_KERNELS")
	if dawn == "" || kdir == "" {
		t.Skip("HODGEMIX_DAWN and HODGEMIX_CHAMP_KERNELS not set")
	}
	dir := t.TempDir()
	rng := rand.New(rand.NewSource(7))
	const K, N, NC, NR = 256, 96, 8, 6 // 32 inputs and 16 outputs a cell
	np := N / 4
	codes := make([]int, N*K) // 0..2, ternary limb 0
	for i := range codes {
		codes[i] = rng.Intn(3)
	}
	w := make([]byte, np*K)
	for r := 0; r < np; r++ {
		for j := 0; j < K; j++ {
			var b byte
			for i := 0; i < 4; i++ {
				b |= byte(codes[(i*np+r)*K+j]) << (2 * i)
			}
			w[r*K+j] = b
		}
	}
	xq := make([]byte, K)
	a := make([]int, K)
	for j := range a {
		a[j] = rng.Intn(255) - 127
		xq[j] = byte(int8(a[j]))
	}
	colCell, rowCell := make([]uint32, K), make([]uint32, N)
	for i, j := range rng.Perm(K) {
		colCell[j] = uint32(i * NC / K)
	}
	for i, r := range rng.Perm(N) {
		rowCell[r] = uint32(i * NR / N)
	}
	want := make([]int64, NR*NC)
	for i := 0; i < N; i++ {
		for j := 0; j < K; j++ {
			want[int(rowCell[i])*NC+int(colCell[j])] += int64(a[j] * (codes[i*K+j] - 1))
		}
	}
	u32s := func(v []uint32) []byte {
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], x)
		}
		return b
	}
	put := func(name string, b []byte) string {
		p := filepath.Join(dir, name+".bin")
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	cmd := strings.Join([]string{
		"prog " + filepath.Join(kdir, "flow.wgsl"),
		"load xq " + put("xq", xq),
		"load w " + put("w", w),
		"load colCell " + put("colCell", u32s(colCell)),
		"load rowCell " + put("rowCell", u32s(rowCell)),
		"load acc " + put("acc", make([]byte, 4*NR*NC)),
		fmt.Sprintf("run main -d %d,1,1 K=%d NP=%d NC=%d CENTER=1", (np+7)/8, K, np, NC),
		"save acc " + filepath.Join(dir, "acc.out"),
	}, " ; ")
	out, err := exec.Command(dawn, cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("dawn: %v\n%s", err, out)
	}
	got, err := os.ReadFile(filepath.Join(dir, "acc.out"))
	if err != nil || len(got) < 4*NR*NC {
		t.Fatalf("acc.out: %v (%d bytes)\n%s", err, len(got), out)
	}
	for i, w := range want {
		g := int64(int32(binary.LittleEndian.Uint32(got[4*i:])))
		if g != w {
			t.Fatalf("flow.wgsl cell (%d,%d): GPU %d, CPU %d\n%s", i/NC, i%NC, g, w, out)
		}
	}

	// cellsum: MODE 0 and the relu^2 hidden (MODE 1).
	const n, cells = 192, 12
	av, bv := make([]float32, n), make([]float32, n)
	for i := range av {
		av[i], bv[i] = float32(rng.NormFloat64()), float32(rng.NormFloat64())
	}
	order := make([]uint32, n)
	for i, j := range rng.Perm(n) {
		order[i] = uint32(j)
	}
	f32s := func(v []float32) []byte {
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
		}
		return b
	}
	for mode := 0; mode < 2; mode++ {
		cmd := strings.Join([]string{
			"prog " + filepath.Join(kdir, "cellsum.wgsl"),
			"load a " + put("a", f32s(av)), "load b " + put("b", f32s(bv)),
			"load order " + put("order", u32s(order)), "load out " + put("out", make([]byte, 4*cells)),
			fmt.Sprintf("run main -d 1,1,1 NCELL=%d M=%d MODE=%d ACT=0", cells, n/cells, mode),
			"save out " + filepath.Join(dir, "cs.out"),
		}, " ; ")
		out, err := exec.Command(dawn, cmd).CombinedOutput()
		if err != nil {
			t.Fatalf("dawn cellsum: %v\n%s", err, out)
		}
		got, _ := os.ReadFile(filepath.Join(dir, "cs.out"))
		for c := 0; c < cells; c++ {
			var s, abs float64
			for k := 0; k < n/cells; k++ {
				j := order[c*(n/cells)+k]
				v := float64(av[j])
				if mode == 1 {
					g := math.Max(v, 0)
					v = g * g * float64(bv[j])
				}
				s, abs = s+v, abs+math.Abs(v)
			}
			g := float64(math.Float32frombits(binary.LittleEndian.Uint32(got[4*c:])))
			if math.Abs(g-s) > 1e-5*math.Max(abs, 1) {
				t.Fatalf("cellsum MODE %d cell %d: GPU %v, CPU %v", mode, c, g, s)
			}
		}
	}
}

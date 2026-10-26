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

	"hodgemix/eval"
)

// packChampWeights is bitlinear.wgsl's layout: byte (r, k) holds the 2-bit codes
// of output rows r, r+NP, r+2NP, r+3NP. digits is row-major [n][k].
func packChampWeights(digits []uint8, n, k int) []byte {
	np := n / 4
	w := make([]byte, np*k)
	for r := 0; r < np; r++ {
		for j := 0; j < k; j++ {
			var b byte
			for i := 0; i < 4; i++ {
				b |= digits[(i*np+r)*k+j] << (2 * i)
			}
			w[r*k+j] = b
		}
	}
	return w
}

// packedAcc is the integer sum kernels/neuro.wgsl computes from that packing.
func packedAcc(w []byte, q []int8, n, k, center int) []int32 {
	np := n / 4
	out := make([]int32, n)
	for r := 0; r < np; r++ {
		var acc [4]int32
		for j := 0; j < k; j++ {
			if q[j] == 0 {
				continue
			}
			b := w[r*k+j]
			for i := 0; i < 4; i++ {
				d := int32((b>>(2*i))&3) - int32(center)
				acc[i] += d * int32(q[j])
			}
		}
		for i := 0; i < 4; i++ {
			out[i*np+r] = acc[i]
		}
	}
	return out
}

func f32bytes(v float32) []byte {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, math.Float32bits(v))
	return b
}

// TestNeuroKernel runs kernels/neuro.wgsl on Dawn. compact builds the graded
// event list; apply writes the same f32s as the dense integer sum times the
// two scales. A second apply with ACCUM adds a second copy.
func TestNeuroKernel(t *testing.T) {
	dawn := os.Getenv("HODGEMIX_DAWN")
	if dawn == "" {
		t.Skip("HODGEMIX_DAWN not set")
	}
	const n, k = 40, 128 // two workgroups, and more than one event per lane
	const center = 1
	const xs, ws float32 = 0.5, 1.25
	rng := rand.New(rand.NewSource(9))
	digits := make([]uint8, n*k)
	for i := range digits {
		digits[i] = uint8(rng.Intn(3))
	}
	q := make([]int8, k)
	for i := range q {
		q[i] = int8(rng.Intn(256) - 128)
	}
	q[1], q[2], q[8], q[20] = 0, 0, 0, 0
	acc := eval.EventAcc(digits, n, k, center, q)
	w := packChampWeights(digits, n, k)
	if got := packedAcc(w, q, n, k, center); !sameI32(got, acc) {
		t.Fatalf("packed layout acc %v, graded %v", got, acc)
	}
	dir := t.TempDir()
	put := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	xq := make([]byte, k)
	for i, v := range q {
		xq[i] = byte(v)
	}
	shader := filepath.Join("kernels", "neuro.wgsl")
	np := n / 4
	cmd := strings.Join([]string{
		"prog " + shader,
		"load xq " + put("xq.bin", xq),
		"load xs " + put("xs.bin", f32bytes(xs)),
		"load w " + put("w.bin", w),
		"load ws " + put("ws.bin", f32bytes(ws)),
		"load o " + put("o.bin", make([]byte, 4*n)),
		"load ev " + put("ev.bin", make([]byte, 4*k)),
		"load evn " + put("evn.bin", make([]byte, 4)),
		fmt.Sprintf("run compact -d 1,1,1 K=%d NP=%d CENTER=%d", k, np, center),
		fmt.Sprintf("run apply -d %d,1,1 K=%d NP=%d CENTER=%d", (np+7)/8, k, np, center),
		"save o " + filepath.Join(dir, "o.out"),
	}, " ; ")
	out, err := exec.Command(dawn, cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("dawn: %v\n%s", err, out)
	}
	got, err := os.ReadFile(filepath.Join(dir, "o.out"))
	if err != nil || len(got) < 4*n {
		t.Fatalf("o.out: %v (%d bytes)\n%s", err, len(got), out)
	}
	for i, a := range acc {
		want := float32(a) * xs * ws
		g := math.Float32frombits(binary.LittleEndian.Uint32(got[4*i:]))
		if g != want {
			t.Fatalf("row %d: GPU %v, CPU %v\n%s", i, g, want, out)
		}
	}

	// The same events, added again. The kernel adds f32(acc)*scale onto o.
	cmd = strings.Join([]string{
		"prog " + shader,
		"load xq " + put("xq.bin", xq),
		"load xs " + put("xs.bin", f32bytes(xs)),
		"load w " + put("w.bin", w),
		"load ws " + put("ws.bin", f32bytes(ws)),
		"load o " + filepath.Join(dir, "o.out"),
		"load ev " + put("ev.bin", make([]byte, 4*k)),
		"load evn " + put("evn.bin", make([]byte, 4)),
		fmt.Sprintf("run compact -d 1,1,1 K=%d NP=%d CENTER=%d", k, np, center),
		fmt.Sprintf("run apply -d %d,1,1 K=%d NP=%d CENTER=%d ACCUM=1", (np+7)/8, k, np, center),
		"save o " + filepath.Join(dir, "o2.out"),
	}, " ; ")
	out, err = exec.Command(dawn, cmd).CombinedOutput()
	if err != nil {
		t.Fatalf("dawn accum: %v\n%s", err, out)
	}
	got, err = os.ReadFile(filepath.Join(dir, "o2.out"))
	if err != nil || len(got) < 4*n {
		t.Fatalf("o2.out: %v (%d bytes)\n%s", err, len(got), out)
	}
	for i, a := range acc {
		once := float32(a) * xs * ws
		want := once + once
		g := math.Float32frombits(binary.LittleEndian.Uint32(got[4*i:]))
		if g != want {
			t.Fatalf("accum row %d: GPU %v, CPU %v\n%s", i, g, want, out)
		}
	}
}

func sameI32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

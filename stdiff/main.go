// stdiff compares two safetensors files tensor by tensor, as a model card states the difference
// between a hodgemix output and its base.
//
//	stdiff -a BASE.safetensors -b OUT.safetensors [-json report.json]
//
// Both files are named; nothing is looked up. For every tensor in both: dtype and shape, and
//   - U8 (BitNet packed ternary, four 2-bit codes a byte, code = field - 1): codes that changed,
//     split into sign flips (-1 <-> +1), 0 -> ±1 and ±1 -> 0;
//   - BF16 / F16 / F32: elements that differ, the largest absolute difference, and for a
//     one-element tensor (weight_scale) the ratio b / a;
//   - anything else: whether the bytes are equal.
// Tensors in only one file are listed with their dtype and shape. Each file's sha256 is printed.
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

type entry struct {
	Dtype   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

type stFile struct {
	f       *os.File
	base    int64
	tensors map[string]entry
	meta    map[string]string
}

func open(path string) (*stFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var n [8]byte
	if _, err := io.ReadFull(f, n[:]); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	size := binary.LittleEndian.Uint64(n[:])
	if size > 1<<30 {
		return nil, fmt.Errorf("%s: header of %d bytes", path, size)
	}
	h := make([]byte, size)
	if _, err := io.ReadFull(f, h); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(h, &raw); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	st := &stFile{f: f, base: 8 + int64(size), tensors: map[string]entry{}, meta: map[string]string{}}
	for k, v := range raw {
		if k == "__metadata__" {
			json.Unmarshal(v, &st.meta)
			continue
		}
		var e entry
		if err := json.Unmarshal(v, &e); err != nil {
			return nil, fmt.Errorf("%s: %s: %v", path, k, err)
		}
		st.tensors[k] = e
	}
	return st, nil
}

func (s *stFile) read(name string) ([]byte, error) {
	e := s.tensors[name]
	b := make([]byte, e.Offsets[1]-e.Offsets[0])
	_, err := s.f.ReadAt(b, s.base+e.Offsets[0])
	return b, err
}

func fileSHA(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

type tensorDiff struct {
	Name  string `json:"name"`
	Dtype string `json:"dtype"`
	Shape []int  `json:"shape"`
	Equal bool   `json:"equal"`
	// packed ternary
	Codes      int64 `json:"codes,omitempty"`
	Changed    int64 `json:"changed,omitempty"`
	Flips      int64 `json:"sign_flips,omitempty"`
	ZeroToSign int64 `json:"zero_to_sign,omitempty"`
	SignToZero int64 `json:"sign_to_zero,omitempty"`
	ZerosA     int64 `json:"zeros_a,omitempty"`
	ZerosB     int64 `json:"zeros_b,omitempty"`
	// float
	Elements  int64   `json:"elements,omitempty"`
	Differ    int64   `json:"differ,omitempty"`
	MaxAbs    float64 `json:"max_abs_diff,omitempty"`
	A         float64 `json:"a,omitempty"`
	B         float64 `json:"b,omitempty"`
	Ratio     float64 `json:"ratio,omitempty"`
	Different string  `json:"note,omitempty"`
}

func floatAt(dtype string, b []byte, i int) float64 {
	switch dtype {
	case "BF16":
		return float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*i:])) << 16))
	case "F16":
		return f16(binary.LittleEndian.Uint16(b[2*i:]))
	case "F32":
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:])))
	}
	return math.NaN()
}

func f16(h uint16) float64 {
	sign := 1.0
	if h&0x8000 != 0 {
		sign = -1
	}
	exp := int(h>>10) & 31
	frac := float64(h & 1023)
	switch exp {
	case 0:
		return sign * frac * math.Pow(2, -24)
	case 31:
		if frac == 0 {
			return sign * math.Inf(1)
		}
		return math.NaN()
	}
	return sign * (1 + frac/1024) * math.Pow(2, float64(exp-15))
}

func width(dtype string) int {
	switch dtype {
	case "BF16", "F16":
		return 2
	case "F32":
		return 4
	}
	return 0
}

func compare(name string, ea entry, a, b []byte) tensorDiff {
	d := tensorDiff{Name: name, Dtype: ea.Dtype, Shape: ea.Shape}
	d.Equal = string(a) == string(b)
	switch {
	case ea.Dtype == "U8":
		for i := range a {
			for k := 0; k < 4; k++ {
				ca := int((a[i]>>(2*k))&3) - 1
				cb := int((b[i]>>(2*k))&3) - 1
				d.Codes++
				if ca == 0 {
					d.ZerosA++
				}
				if cb == 0 {
					d.ZerosB++
				}
				if ca == cb {
					continue
				}
				d.Changed++
				switch {
				case ca == 0:
					d.ZeroToSign++
				case cb == 0:
					d.SignToZero++
				default:
					d.Flips++
				}
			}
		}
	case width(ea.Dtype) > 0:
		w := width(ea.Dtype)
		n := len(a) / w
		d.Elements = int64(n)
		for i := 0; i < n; i++ {
			x, y := floatAt(ea.Dtype, a, i), floatAt(ea.Dtype, b, i)
			if x != y {
				d.Differ++
				d.MaxAbs = math.Max(d.MaxAbs, math.Abs(x-y))
			}
		}
		if n == 1 {
			d.A, d.B = floatAt(ea.Dtype, a, 0), floatAt(ea.Dtype, b, 0)
			if d.A != 0 {
				d.Ratio = d.B / d.A
			}
		}
	default:
		if !d.Equal {
			d.Different = "bytes differ"
		}
	}
	return d
}

func sameShape(x, y []int) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

func run(pa, pb, out string) error {
	if pa == "" || pb == "" {
		return errors.New("-a and -b are both required")
	}
	A, err := open(pa)
	if err != nil {
		return err
	}
	B, err := open(pb)
	if err != nil {
		return err
	}
	shaA, err := fileSHA(pa)
	if err != nil {
		return err
	}
	shaB, err := fileSHA(pb)
	if err != nil {
		return err
	}
	var names, onlyA, onlyB []string
	for k := range A.tensors {
		if _, ok := B.tensors[k]; ok {
			names = append(names, k)
		} else {
			onlyA = append(onlyA, k)
		}
	}
	for k := range B.tensors {
		if _, ok := A.tensors[k]; !ok {
			onlyB = append(onlyB, k)
		}
	}
	sort.Strings(names)
	sort.Strings(onlyA)
	sort.Strings(onlyB)
	var diffs []tensorDiff
	var mismatch []string
	var tot tensorDiff
	equal := 0
	for _, k := range names {
		ea, eb := A.tensors[k], B.tensors[k]
		if ea.Dtype != eb.Dtype || !sameShape(ea.Shape, eb.Shape) {
			mismatch = append(mismatch, fmt.Sprintf("%s: %s%v -> %s%v", k, ea.Dtype, ea.Shape, eb.Dtype, eb.Shape))
			continue
		}
		a, err := A.read(k)
		if err != nil {
			return fmt.Errorf("%s: %s: %v", pa, k, err)
		}
		b, err := B.read(k)
		if err != nil {
			return fmt.Errorf("%s: %s: %v", pb, k, err)
		}
		d := compare(k, ea, a, b)
		if d.Equal {
			equal++
		}
		tot.Codes += d.Codes
		tot.Changed += d.Changed
		tot.Flips += d.Flips
		tot.ZeroToSign += d.ZeroToSign
		tot.SignToZero += d.SignToZero
		tot.ZerosA += d.ZerosA
		tot.ZerosB += d.ZerosB
		tot.Elements += d.Elements
		tot.Differ += d.Differ
		diffs = append(diffs, d)
	}
	fmt.Printf("a %s\n  sha256 %s, %d tensors\n", pa, shaA, len(A.tensors))
	fmt.Printf("b %s\n  sha256 %s, %d tensors\n", pb, shaB, len(B.tensors))
	fmt.Printf("shared %d: %d byte-identical, %d dtype/shape changed\n", len(names), equal, len(mismatch))
	fmt.Printf("ternary codes: %d, changed %d (sign flips %d, 0->±1 %d, ±1->0 %d); zeros a %d, b %d\n",
		tot.Codes, tot.Changed, tot.Flips, tot.ZeroToSign, tot.SignToZero, tot.ZerosA, tot.ZerosB)
	fmt.Printf("float elements: %d, differ %d\n", tot.Elements, tot.Differ)
	for _, d := range diffs {
		if d.Equal {
			continue
		}
		switch {
		case d.Dtype == "U8":
			fmt.Printf("  %s: %d of %d codes changed (flips %d, 0->±1 %d, ±1->0 %d)\n", d.Name, d.Changed, d.Codes, d.Flips, d.ZeroToSign, d.SignToZero)
		case d.Elements == 1:
			fmt.Printf("  %s: %g -> %g (x%.6f)\n", d.Name, d.A, d.B, d.Ratio)
		case d.Elements > 0:
			fmt.Printf("  %s: %d of %d differ, max |diff| %g\n", d.Name, d.Differ, d.Elements, d.MaxAbs)
		default:
			fmt.Printf("  %s: %s\n", d.Name, d.Different)
		}
	}
	for _, m := range mismatch {
		fmt.Println("  changed:", m)
	}
	group := func(list []string, s *stFile) map[string]string {
		g := map[string]string{}
		for _, k := range list {
			e := s.tensors[k]
			role := k
			if i := strings.Index(k, "layers."); i >= 0 {
				j := strings.Index(k[i+7:], ".")
				if j > 0 {
					role = k[:i+7] + "N" + k[i+7+j:]
				}
			}
			if _, ok := g[role]; !ok {
				g[role] = fmt.Sprintf("%s%v", e.Dtype, e.Shape)
			}
		}
		return g
	}
	if len(onlyA) > 0 {
		fmt.Printf("only in a: %d %v\n", len(onlyA), group(onlyA, A))
	}
	if len(onlyB) > 0 {
		fmt.Printf("only in b: %d %v\n", len(onlyB), group(onlyB, B))
	}
	if out != "" {
		rep := map[string]any{
			"a": map[string]any{"path": pa, "sha256": shaA, "tensors": len(A.tensors), "metadata": A.meta},
			"b": map[string]any{"path": pb, "sha256": shaB, "tensors": len(B.tensors), "metadata": B.meta},
			"shared": len(names), "byte_identical": equal, "dtype_shape_changed": mismatch,
			"total": tot, "tensors": diffs, "only_a": onlyA, "only_b": onlyB,
		}
		raw, _ := json.MarshalIndent(rep, "", "  ")
		if err := os.WriteFile(out, raw, 0o644); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	a := flag.String("a", "", "first safetensors (the base)")
	b := flag.String("b", "", "second safetensors (the output)")
	out := flag.String("json", "", "write the full comparison here")
	flag.Parse()
	if err := run(*a, *b, *out); err != nil {
		fmt.Fprintln(os.Stderr, "stdiff:", err)
		os.Exit(1)
	}
}

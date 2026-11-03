package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
)

type info struct {
	Dtype string  `json:"dtype"`
	Shape []int   `json:"shape"`
	Off   []int64 `json:"data_offsets"`
}

type st struct {
	f    *os.File
	base int64
	h    map[string]info
	meta map[string]string
}

func openST(path string) (*st, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var nbuf [8]byte
	if _, err := io.ReadFull(f, nbuf[:]); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	n := binary.LittleEndian.Uint64(nbuf[:])
	if n > 1<<26 {
		f.Close()
		return nil, fmt.Errorf("%s: header %d is too large", path, n)
	}
	hb := make([]byte, n)
	if _, err := io.ReadFull(f, hb); err != nil {
		f.Close()
		return nil, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(hb, &raw); err != nil {
		f.Close()
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	s := &st{f: f, base: 8 + int64(n), h: map[string]info{}, meta: map[string]string{}}
	for k, v := range raw {
		if k == "__metadata__" {
			json.Unmarshal(v, &s.meta)
			continue
		}
		var t info
		if err := json.Unmarshal(v, &t); err != nil || len(t.Off) != 2 {
			f.Close()
			return nil, fmt.Errorf("%s: %s", path, k)
		}
		s.h[k] = t
	}
	return s, nil
}

func (s *st) close() { s.f.Close() }

func (s *st) has(k string) bool { _, ok := s.h[k]; return ok }

func numel(sh []int) int {
	n := 1
	for _, d := range sh {
		n *= d
	}
	return n
}

func (s *st) readAt(t info, p []byte) error {
	_, err := s.f.ReadAt(p, s.base+t.Off[0])
	return err
}

func (s *st) i32(k string) ([]int32, []int, error) {
	t, ok := s.h[k]
	if !ok {
		return nil, nil, fmt.Errorf("no %s", k)
	}
	if t.Dtype != "I32" {
		return nil, nil, fmt.Errorf("%s is %s, want I32", k, t.Dtype)
	}
	n := numel(t.Shape)
	b := make([]byte, n*4)
	if err := s.readAt(t, b); err != nil {
		return nil, nil, err
	}
	o := make([]int32, n)
	for i := range o {
		o[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return o, t.Shape, nil
}

func (s *st) f32(k string) ([]float32, []int, error) {
	t, ok := s.h[k]
	if !ok {
		return nil, nil, fmt.Errorf("no %s", k)
	}
	if t.Dtype != "F32" {
		return nil, nil, fmt.Errorf("%s is %s, want F32", k, t.Dtype)
	}
	n := numel(t.Shape)
	b := make([]byte, n*4)
	if err := s.readAt(t, b); err != nil {
		return nil, nil, err
	}
	o := make([]float32, n)
	for i := range o {
		o[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return o, t.Shape, nil
}

func f32of(dtype string, b []byte) (float32, error) {
	switch dtype {
	case "F32":
		if len(b) < 4 {
			return 0, io.ErrUnexpectedEOF
		}
		return math.Float32frombits(binary.LittleEndian.Uint32(b)), nil
	case "BF16":
		if len(b) < 2 {
			return 0, io.ErrUnexpectedEOF
		}
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b)) << 16), nil
	default:
		return 0, fmt.Errorf("dtype %s is not a float cochain", dtype)
	}
}

func elemSize(dtype string) (int, error) {
	switch dtype {
	case "F32", "I32":
		return 4, nil
	case "BF16", "F16":
		return 2, nil
	default:
		return 0, fmt.Errorf("dtype %s", dtype)
	}
}

// writeI32 writes tensors and string metadata. Each tensor is I32, row-major.
func writeI32(path string, meta map[string]string, tensors map[string]tensor) error {
	names := make([]string, 0, len(tensors))
	for k := range tensors {
		names = append(names, k)
	}
	// Stable order: anchors, cochain, top, then the rest.
	order := []string{"cochain", "anchors", "top"}
	var keys []string
	seen := map[string]bool{}
	for _, k := range order {
		if _, ok := tensors[k]; ok {
			keys = append(keys, k)
			seen[k] = true
		}
	}
	for _, k := range names {
		if !seen[k] {
			keys = append(keys, k)
		}
	}
	off := int64(0)
	header := map[string]any{}
	var blobs [][]byte
	for _, k := range keys {
		t := tensors[k]
		b := make([]byte, 4*len(t.v))
		for i, v := range t.v {
			binary.LittleEndian.PutUint32(b[4*i:], uint32(v))
		}
		header[k] = info{Dtype: "I32", Shape: t.shape, Off: []int64{off, off + int64(len(b))}}
		off += int64(len(b))
		blobs = append(blobs, b)
	}
	header["__metadata__"] = meta
	hb, err := json.Marshal(header)
	if err != nil {
		return err
	}
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil && filepath.Dir(path) != "." {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var nbuf [8]byte
	binary.LittleEndian.PutUint64(nbuf[:], uint64(len(hb)))
	if _, err := f.Write(nbuf[:]); err != nil {
		return err
	}
	if _, err := f.Write(hb); err != nil {
		return err
	}
	for _, b := range blobs {
		if _, err := f.Write(b); err != nil {
			return err
		}
	}
	return nil
}

type ftensor struct {
	shape []int
	v     []float32
}

// writeFloat writes F32 tensors in the given order, then any key it skipped.
func writeFloat(path string, meta map[string]string, order []string, tensors map[string]ftensor) error {
	seen := map[string]bool{}
	var keys []string
	for _, k := range order {
		if _, ok := tensors[k]; ok {
			keys = append(keys, k)
			seen[k] = true
		}
	}
	var rest []string
	for k := range tensors {
		if !seen[k] {
			rest = append(rest, k)
		}
	}
	sort.Strings(rest)
	keys = append(keys, rest...)
	off := int64(0)
	header := map[string]any{}
	var blobs [][]byte
	for _, k := range keys {
		t := tensors[k]
		b := make([]byte, 4*len(t.v))
		for i, v := range t.v {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		header[k] = info{Dtype: "F32", Shape: t.shape, Off: []int64{off, off + int64(len(b))}}
		off += int64(len(b))
		blobs = append(blobs, b)
	}
	header["__metadata__"] = meta
	hb, err := json.Marshal(header)
	if err != nil {
		return err
	}
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil && filepath.Dir(path) != "." {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var nbuf [8]byte
	binary.LittleEndian.PutUint64(nbuf[:], uint64(len(hb)))
	if _, err := f.Write(nbuf[:]); err != nil {
		return err
	}
	if _, err := f.Write(hb); err != nil {
		return err
	}
	for _, b := range blobs {
		if _, err := f.Write(b); err != nil {
			return err
		}
	}
	return nil
}

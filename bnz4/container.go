// Vendored from /workspace/bitnet-compress/wgsl/bnz (Gregory's BNZ4 reference encoder, byte-identical to the
// WGSL encoder and decoded by bnz-loader/js/bnz4dec.js). Only change: embedding always row-range segmented (Plan).
package bnz4

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"sort"
	"sync"
)

const (
	MRaw          = 0
	MZstd         = 1
	MTern         = 2
	MBF16         = 3
	Magic         = "BNZ4"
	EmbChunkBytes = 8192 * 2560 * 2
)

type Segment struct {
	Name   string `json:"name"`
	Method int    `json:"method"`
	RawOff int64  `json:"raw_off"`
	RawLen int64  `json:"raw_len"`
	Off    int64  `json:"off"`
	Len    int64  `json:"len"`
	Shape  []int  `json:"shape"`
	Sha256 string `json:"sha256"`
	dtype  string
}

type Index struct {
	OrigSize int64     `json:"orig_size"`
	Sha256   string    `json:"sha256"`
	Segments []Segment `json:"segments"`
}

func mmap(path string) ([]byte, func(), error) { return mmapImpl(path) }

func Plan(data []byte) ([]Segment, error) {
	if len(data) < 8 {
		return nil, errors.New("not safetensors")
	}
	n := int64(binary.LittleEndian.Uint64(data))
	if 8+n > int64(len(data)) {
		return nil, errors.New("bad header")
	}
	var h map[string]json.RawMessage
	if err := json.Unmarshal(data[8:8+n], &h); err != nil {
		return nil, err
	}
	type ent struct {
		Dtype   string  `json:"dtype"`
		Shape   []int   `json:"shape"`
		Offsets []int64 `json:"data_offsets"`
		name    string
	}
	var items []ent
	for k, v := range h {
		if k == "__metadata__" {
			continue
		}
		var e ent
		if err := json.Unmarshal(v, &e); err != nil {
			return nil, err
		}
		e.name = k
		items = append(items, e)
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].Offsets[0] != items[j].Offsets[0] {
			return items[i].Offsets[0] < items[j].Offsets[0]
		}
		return items[i].name < items[j].name
	})
	base := 8 + n
	segs := []Segment{{Name: "header", RawOff: 0, RawLen: base}}
	pos := base
	for _, it := range items {
		s, e := base+it.Offsets[0], base+it.Offsets[1]
		if s > pos {
			segs = append(segs, Segment{Name: "gap", RawOff: pos, RawLen: s - pos})
		}
		// hodgemix: the embedding is always written as row-range segments "name[r0:r1]" (even when it fits in
		// one chunk), because bnz-loader's BitNetRT.plan() only schedules embedding segments named that way.
		if it.Dtype == "BF16" && len(it.Shape) == 2 && (e-s > EmbChunkBytes || it.name == "model.embed_tokens.weight") {
			rb := int64(2 * it.Shape[1])
			rows := max(1, EmbChunkBytes/rb)
			for r0 := int64(0); r0 < int64(it.Shape[0]); r0 += rows {
				r1 := min(int64(it.Shape[0]), r0+rows)
				segs = append(segs, Segment{Name: fmt.Sprintf("%s[%d:%d]", it.name, r0, r1), dtype: "BF16",
					Shape: []int{int(r1 - r0), it.Shape[1]}, RawOff: s + r0*rb, RawLen: (r1 - r0) * rb})
			}
		} else {
			segs = append(segs, Segment{Name: it.name, dtype: it.Dtype, Shape: it.Shape, RawOff: s, RawLen: e - s})
		}
		pos = e
	}
	if pos < int64(len(data)) {
		segs = append(segs, Segment{Name: "tail", RawOff: pos, RawLen: int64(len(data)) - pos})
	}
	return segs, nil
}

// ForceRANS (hodgemix test hook, default off): use methods 4/5 whenever they apply, even when larger, so tiny
// test files exercise bnz-loader's WGSL decode path. Output stays a valid BNZ4 file.
var ForceRANS bool

func encodeSeg(sg *Segment, raw []byte) []byte {
	best, bm := raw, MRaw
	try := func(m int, c []byte) {
		if c != nil && (len(c) < len(best) || (ForceRANS && m >= MTern4)) {
			best, bm = c, m
		}
	}
	if len(raw) > 64 {
		try(MZstd, Z(raw))
	}
	if sg.dtype == "U8" && len(sg.Shape) == 2 {
		try(MTern4, Engine.EncTern(raw, sg.Shape[0], sg.Shape[1]))
	}
	if sg.dtype == "BF16" && (len(raw) >= 4096 || (ForceRANS && len(raw) >= 4)) {
		try(MBF164, Engine.EncBF16(raw, sg.Shape))
	}
	sg.Method = bm
	if bm != MTern4 && bm != MBF164 {
		sg.Shape = nil
	}
	return best
}

func DecodeSeg(sg *Segment, c []byte) ([]byte, error) {
	var r []byte
	var err error
	switch sg.Method {
	case MRaw:
		r = c
	case MZstd:
		r, err = UZ(c)
	case MTern:
		if len(sg.Shape) != 2 {
			return nil, errors.New("tern: bad shape")
		}
		r, err = DecTern(c, sg.Shape[0], sg.Shape[1])
	case MBF16:
		r, err = DecBF16(c, sg.Shape)
	case MTern4:
		if len(sg.Shape) != 2 {
			return nil, errors.New("tern4: bad shape")
		}
		r, err = Engine.DecTern(c, sg.Shape[0], sg.Shape[1])
	case MBF164:
		r, err = Engine.DecBF16(c, sg.Shape)
	default:
		err = fmt.Errorf("unknown method %d", sg.Method)
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", sg.Name, err)
	}
	h := sha256.Sum256(r)
	if int64(len(r)) != sg.RawLen || hex.EncodeToString(h[:]) != sg.Sha256 {
		return nil, fmt.Errorf("%s: sha256/length mismatch", sg.Name)
	}
	return r, nil
}

// Compress encodes in; payloads go to a temp file in segment order, then the final file
// is magic | u64 index_len | JSON index | payloads.
func Compress(in, out string, log io.Writer) (int64, error) {
	data, unmap, err := mmap(in)
	if err != nil {
		return 0, err
	}
	defer unmap()
	segs, err := Plan(data)
	if err != nil {
		return 0, err
	}
	whole := make(chan string, 1)
	go func() { h := sha256.Sum256(data); whole <- hex.EncodeToString(h[:]) }()

	tmp, err := os.Create(out + ".payload.tmp")
	if err != nil {
		return 0, err
	}
	defer os.Remove(tmp.Name())
	results := make([][]byte, len(segs))
	done := make([]chan struct{}, len(segs))
	for i := range done {
		done[i] = make(chan struct{})
	}
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				sg := &segs[i]
				raw := data[sg.RawOff : sg.RawOff+sg.RawLen]
				h := sha256.Sum256(raw)
				sg.Sha256 = hex.EncodeToString(h[:])
				results[i] = encodeSeg(sg, raw)
				close(done[i])
			}
		}()
	}
	// big segments first for load balance
	order := make([]int, len(segs))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return segs[order[a]].RawLen > segs[order[b]].RawLen })
	go func() {
		for _, i := range order {
			jobs <- i
		}
		close(jobs)
	}()
	var off int64
	bw := bufio.NewWriterSize(tmp, 1<<20)
	for i := range segs {
		<-done[i]
		segs[i].Off, segs[i].Len = off, int64(len(results[i]))
		off += segs[i].Len
		if _, err := bw.Write(results[i]); err != nil {
			return 0, err
		}
		if log != nil {
			fmt.Fprintf(log, "{\"name\":%q,\"method\":%d,\"raw\":%d,\"comp\":%d}\n", segs[i].Name, segs[i].Method, segs[i].RawLen, segs[i].Len)
		}
		results[i] = nil
	}
	wg.Wait()
	bw.Flush()
	ij, _ := json.Marshal(Index{OrigSize: int64(len(data)), Sha256: <-whole, Segments: segs})
	f, err := os.Create(out)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	hdr := make([]byte, 12)
	copy(hdr, Magic)
	binary.LittleEndian.PutUint64(hdr[4:], uint64(len(ij)))
	f.Write(hdr)
	f.Write(ij)
	tmp.Seek(0, 0)
	if _, err := io.Copy(f, tmp); err != nil {
		return 0, err
	}
	tmp.Close()
	return 12 + int64(len(ij)) + off, f.Sync()
}

func ReadIndex(b []byte) (*Index, int64, error) {
	if len(b) < 12 || string(b[:4]) != Magic {
		return nil, 0, errors.New("not a BNZ4 file")
	}
	n := int64(binary.LittleEndian.Uint64(b[4:]))
	if 12+n > int64(len(b)) {
		return nil, 0, errors.New("truncated index")
	}
	var ix Index
	if err := json.Unmarshal(b[12:12+n], &ix); err != nil {
		return nil, 0, err
	}
	return &ix, 12 + n, nil
}

// Decompress rebuilds the original file, verifying per-segment and whole-file sha256.
func Decompress(in, out string) (string, bool, error) {
	data, unmap, err := mmap(in)
	if err != nil {
		return "", false, err
	}
	defer unmap()
	ix, base, err := ReadIndex(data)
	if err != nil {
		return "", false, err
	}
	f, err := os.Create(out)
	if err != nil {
		return "", false, err
	}
	if err := f.Truncate(ix.OrigSize); err != nil {
		return "", false, err
	}
	order := make([]int, len(ix.Segments))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return ix.Segments[order[a]].RawLen > ix.Segments[order[b]].RawLen })
	jobs := make(chan int)
	var mu sync.Mutex
	var firstErr error
	var wg sync.WaitGroup
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				var err error
				sg := &ix.Segments[i]
				if base+sg.Off+sg.Len > int64(len(data)) || sg.RawOff+sg.RawLen > ix.OrigSize {
					err = fmt.Errorf("%s: out of range", sg.Name)
				} else {
					var r []byte
					r, err = DecodeSeg(sg, data[base+sg.Off:base+sg.Off+sg.Len])
					if err == nil {
						_, err = f.WriteAt(r, sg.RawOff)
					}
				}
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					mu.Unlock()
				}
			}
		}()
	}
	for _, i := range order {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	if firstErr != nil {
		f.Close()
		return "", false, firstErr
	}
	f.Seek(0, 0)
	h := sha256.New()
	if _, err := io.Copy(h, bufio.NewReaderSize(f, 1<<22)); err != nil {
		return "", false, err
	}
	f.Close()
	d := hex.EncodeToString(h.Sum(nil))
	return d, d == ix.Sha256, nil
}

// Codec is the engine for the chunked methods: CPU (Go reference) or GPU (WGSL).
type Codec interface {
	EncTern(raw []byte, R, C int) []byte
	EncBF16(raw []byte, shape []int) []byte
	DecTern(b []byte, R, C int) ([]byte, error)
	DecBF16(b []byte, shape []int) ([]byte, error)
}

type CPU struct{}

func (CPU) EncTern(raw []byte, R, C int) []byte           { return EncTern4(raw, R, C) }
func (CPU) EncBF16(raw []byte, shape []int) []byte        { return EncBF164(raw, shape) }
func (CPU) DecTern(b []byte, R, C int) ([]byte, error)    { return DecTern4(b, R, C) }
func (CPU) DecBF16(b []byte, shape []int) ([]byte, error) { return DecBF164(b, shape) }

var Engine Codec = CPU{}

func SegDtype(s *Segment) string { return s.dtype }

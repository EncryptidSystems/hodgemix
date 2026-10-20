package bnz

// BNZ3 container (Go port of bnz.py's BNZ2 layout; magic BNZ3 because the rANS coder is this package's own,
// not constriction's). Streams with offset reads (ReadAt): no whole-file mmap, works on Windows.

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
	"sync/atomic"
)

const (
	MRaw          = 0
	MZstd         = 1
	MTern         = 2
	MBF16         = 3
	Magic         = "BNZ3"
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

// Progress is called with (bytes done, bytes total, current segment name).
type Progress func(done, total int64, name string)

// Paper: no equation (helper: I/O, progress or utility)
func Plan(f *os.File) ([]Segment, int64, error) {
	st, err := f.Stat()
	if err != nil {
		return nil, 0, err
	}
	size := st.Size()
	var hb [8]byte
	if _, err := f.ReadAt(hb[:], 0); err != nil {
		return nil, 0, errors.New("not safetensors")
	}
	n := int64(binary.LittleEndian.Uint64(hb[:]))
	if 8+n > size {
		return nil, 0, errors.New("bad header")
	}
	hj := make([]byte, n)
	if _, err := f.ReadAt(hj, 8); err != nil {
		return nil, 0, err
	}
	var h map[string]json.RawMessage
	if err := json.Unmarshal(hj, &h); err != nil {
		return nil, 0, err
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
			return nil, 0, err
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
		if it.Dtype == "BF16" && len(it.Shape) == 2 && e-s > EmbChunkBytes {
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
	if pos < size {
		segs = append(segs, Segment{Name: "tail", RawOff: pos, RawLen: size - pos})
	}
	return segs, size, nil
}

// Paper: no equation (helper: I/O, progress or utility)
func encodeSeg(sg *Segment, raw []byte) []byte {
	best, bm := raw, MRaw
	try := func(m int, c []byte) {
		if c != nil && len(c) < len(best) {
			best, bm = c, m
		}
	}
	if len(raw) > 64 {
		try(MZstd, Z(raw))
	}
	if sg.dtype == "U8" && len(sg.Shape) == 2 {
		try(MTern, EncTern(raw, sg.Shape[0], sg.Shape[1]))
	}
	if sg.dtype == "BF16" && len(raw) >= 4096 {
		try(MBF16, EncBF16(raw, sg.Shape))
	}
	sg.Method = bm
	if bm != MTern && bm != MBF16 {
		sg.Shape = nil
	}
	return best
}

// Paper: no equation (helper: I/O, progress or utility)
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

// Paper: no equation (helper: I/O, progress or utility)
func fileSha(f *os.File) (string, error) {
	h := sha256.New()
	if _, err := io.Copy(h, io.NewSectionReader(f, 0, 1<<62)); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Compress: workers ReadAt their own segment, encode, results written in order to a temp payload file.
// Paper: Contrib.5 / Sec.4 storage: BNZ3 lossless container (no equation in paper)
func Compress(in, out string, log io.Writer, prog Progress) (int64, error) {
	f, err := os.Open(in)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	segs, size, err := Plan(f)
	if err != nil {
		return 0, err
	}
	whole := make(chan string, 1)
	go func() { s, _ := fileSha(f); whole <- s }()
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
	var bytesDone int64
	var errOnce sync.Once
	var rerr error
	jobs := make(chan int)
	nw := runtime.NumCPU()
	for w := 0; w < nw; w++ {
		go func() {
			for i := range jobs {
				sg := &segs[i]
				raw := make([]byte, sg.RawLen)
				if _, err := f.ReadAt(raw, sg.RawOff); err != nil {
					errOnce.Do(func() { rerr = err })
				}
				h := sha256.Sum256(raw)
				sg.Sha256 = hex.EncodeToString(h[:])
				results[i] = encodeSeg(sg, raw)
				d := atomic.AddInt64(&bytesDone, sg.RawLen)
				if prog != nil {
					prog(d, size, sg.Name)
				}
				close(done[i])
			}
		}()
	}
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
	if rerr != nil {
		return 0, rerr
	}
	bw.Flush()
	ij, _ := json.Marshal(Index{OrigSize: size, Sha256: <-whole, Segments: segs})
	o, err := os.Create(out)
	if err != nil {
		return 0, err
	}
	defer o.Close()
	hdr := make([]byte, 12)
	copy(hdr, Magic)
	binary.LittleEndian.PutUint64(hdr[4:], uint64(len(ij)))
	o.Write(hdr)
	o.Write(ij)
	tmp.Seek(0, 0)
	if _, err := io.Copy(o, tmp); err != nil {
		return 0, err
	}
	tmp.Close()
	return 12 + int64(len(ij)) + off, o.Sync()
}

// Paper: no equation (helper: I/O, progress or utility)
func ReadIndex(f *os.File) (*Index, int64, error) {
	var hb [12]byte
	if _, err := f.ReadAt(hb[:], 0); err != nil || string(hb[:4]) != Magic {
		return nil, 0, errors.New("not a BNZ3 file")
	}
	n := int64(binary.LittleEndian.Uint64(hb[4:]))
	ij := make([]byte, n)
	if _, err := f.ReadAt(ij, 12); err != nil {
		return nil, 0, errors.New("truncated index")
	}
	var ix Index
	if err := json.Unmarshal(ij, &ix); err != nil {
		return nil, 0, err
	}
	return &ix, 12 + n, nil
}

// Decompress rebuilds the original file, verifying per-segment and whole-file sha256.
// Paper: Contrib.5 / Sec.4 storage: BNZ3 round-trip (no equation in paper)
func Decompress(in, out string, prog Progress) (string, bool, error) {
	f, err := os.Open(in)
	if err != nil {
		return "", false, err
	}
	defer f.Close()
	ix, base, err := ReadIndex(f)
	if err != nil {
		return "", false, err
	}
	o, err := os.Create(out)
	if err != nil {
		return "", false, err
	}
	if err := o.Truncate(ix.OrigSize); err != nil {
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
	var bytesDone int64
	for w := 0; w < runtime.NumCPU(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				sg := &ix.Segments[i]
				c := make([]byte, sg.Len)
				_, err := f.ReadAt(c, base+sg.Off)
				if err == nil {
					var r []byte
					r, err = DecodeSeg(sg, c)
					if err == nil {
						_, err = o.WriteAt(r, sg.RawOff)
					}
				}
				if err != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = err
					}
					mu.Unlock()
				}
				d := atomic.AddInt64(&bytesDone, sg.RawLen)
				if prog != nil {
					prog(d, ix.OrigSize, sg.Name)
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
		o.Close()
		return "", false, firstErr
	}
	d, err := fileSha(o)
	o.Close()
	if err != nil {
		return "", false, err
	}
	return d, d == ix.Sha256, nil
}

package main

// Integer cochain of an instrumented model, and the sub-topology of a text.
//
// The float weights carry entropy below the cell cochain the instrumented map
// records (one sum per residual cell). Strip keeps the integer cochain on that
// grid, quantum 1/4096, and drops the rest. A submitted text is the chain of
// its words. Each word's tensor is the sum of those cochain rows, and that
// tensor anchors the word on the cells it occupies. The top cochain is the
// coboundary along the chain: the difference of each successive anchor. One
// word's top cochain is the anchor itself.

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"hodgemix/eval"
)

// quantum is the integer grid. An entry is round(x * quantum). Anything smaller
// is the entropy the strip drops.
const quantum = 4096

type tensor struct {
	shape []int
	v     []int32
}

type Word struct {
	Text  string `json:"text"`
	IDs   []int  `json:"ids"`
	Cells []int  `json:"cells"`
}

type Edge struct {
	From  int   `json:"from"`
	To    int   `json:"to"`
	Cells []int `json:"cells"`
}

type TextCochain struct {
	Text    string     `json:"text"`
	SHA256  string     `json:"sha256"`
	Cells   int        `json:"cells"`
	Words   []Word     `json:"words"`
	Edges   []Edge     `json:"edges"`
	Hodge   [3]float64 `json:"hodge"`
	Anchors [][]int32  `json:"anchors"`
	Top     [][]int32  `json:"top"`
}

func quant(x float64) (int32, error) {
	y := math.Round(x * quantum)
	if y > math.MaxInt32 || y < math.MinInt32 {
		return 0, fmt.Errorf("cochain entry %g does not fit in int32", y)
	}
	return int32(y), nil
}

// stripFloat writes the integer cochain of a float tensor [rows, cols].
func stripFloat(s *st, name, out, source string) (int, int, error) {
	t, ok := s.h[name]
	if !ok {
		return 0, 0, fmt.Errorf("%s has no %s", source, name)
	}
	if len(t.Shape) != 2 {
		return 0, 0, fmt.Errorf("%s is %v, want a matrix", name, t.Shape)
	}
	es, err := elemSize(t.Dtype)
	if err != nil {
		return 0, 0, err
	}
	rows, cols := t.Shape[0], t.Shape[1]
	co := make([]int32, rows*cols)
	const chunk = 1 << 20
	buf := make([]byte, chunk)
	off := t.Off[0]
	n := rows * cols
	for i := 0; i < n; {
		want := (n - i) * es
		if want > len(buf) {
			want = len(buf) / es * es
		}
		if _, err := s.f.ReadAt(buf[:want], s.base+off); err != nil {
			return 0, 0, err
		}
		off += int64(want)
		for p := 0; p < want; p += es {
			x, err := f32of(t.Dtype, buf[p:p+es])
			if err != nil {
				return 0, 0, err
			}
			q, err := quant(float64(x))
			if err != nil {
				return 0, 0, err
			}
			co[i] = q
			i++
		}
	}
	err = writeI32(out, map[string]string{
		"format": "pt", "quantum": "1/4096", "cochain_of": source, "tensor": name,
	}, map[string]tensor{"cochain": {[]int{rows, cols}, co}})
	return rows, cols, err
}

// stripEmbed writes the integer cell-sum cochain of an embedding [V, D]
// using cell.D from the instrumented volume.
func stripEmbed(embed *st, embedName string, cell []int32, cells int, out, source string) (int, int, error) {
	t, ok := embed.h[embedName]
	if !ok {
		return 0, 0, fmt.Errorf("%s has no %s", source, embedName)
	}
	if len(t.Shape) != 2 || t.Shape[1] != len(cell) {
		return 0, 0, fmt.Errorf("%s is %v, cell.D has %d channels", embedName, t.Shape, len(cell))
	}
	es, err := elemSize(t.Dtype)
	if err != nil {
		return 0, 0, err
	}
	V, D := t.Shape[0], t.Shape[1]
	co := make([]int32, V*cells)
	row := make([]byte, D*es)
	acc := make([]float64, cells)
	for r := 0; r < V; r++ {
		if _, err := embed.f.ReadAt(row, embed.base+t.Off[0]+int64(r*D*es)); err != nil {
			return 0, 0, err
		}
		for i := range acc {
			acc[i] = 0
		}
		for j := 0; j < D; j++ {
			g := int(cell[j])
			if g < 0 || g >= cells {
				return 0, 0, fmt.Errorf("cell.D[%d]=%d is outside 0..%d", j, g, cells-1)
			}
			x, err := f32of(t.Dtype, row[j*es:(j+1)*es])
			if err != nil {
				return 0, 0, err
			}
			acc[g] += float64(x)
		}
		for g, v := range acc {
			q, err := quant(v)
			if err != nil {
				return 0, 0, err
			}
			co[r*cells+g] = q
		}
	}
	err = writeI32(out, map[string]string{
		"format": "pt", "quantum": "1/4096", "cochain_of": source, "tensor": embedName,
	}, map[string]tensor{"cochain": {[]int{V, cells}, co}})
	return V, cells, err
}

func cellAxis(volumePath string) ([]int32, int, error) {
	s, err := openST(volumePath)
	if err != nil {
		return nil, 0, err
	}
	defer s.close()
	c, shape, err := s.i32("cell.D")
	if err != nil {
		return nil, 0, fmt.Errorf("%s: %v", volumePath, err)
	}
	if len(shape) != 1 {
		return nil, 0, fmt.Errorf("cell.D shape %v", shape)
	}
	n := 0
	seen := map[int]int{}
	for _, g := range c {
		if g < 0 {
			return nil, 0, fmt.Errorf("cell.D has a negative cell")
		}
		seen[int(g)]++
		if int(g)+1 > n {
			n = int(g) + 1
		}
	}
	for g := 0; g < n; g++ {
		if seen[g] == 0 {
			return nil, 0, fmt.Errorf("residual cell %d holds no channel", g)
		}
	}
	return c, n, nil
}

// StripDir writes model.cochain.safetensors beside an instrumented model folder.
// A vocab.float.safetensors cochain is already the cell sum. Otherwise the
// embedding is summed onto cell.D from volume.safetensors.
func StripDir(dir, out string) (int, int, error) {
	vol := filepath.Join(dir, "volume.safetensors")
	vocab := filepath.Join(dir, "vocab.float.safetensors")
	if out == "" {
		out = filepath.Join(dir, "model.cochain.safetensors")
	}
	if st, err := os.Stat(vocab); err == nil && !st.IsDir() {
		s, err := openST(vocab)
		if err != nil {
			return 0, 0, err
		}
		defer s.close()
		if s.has("cochain") {
			return stripFloat(s, "cochain", out, vocab)
		}
	}
	if _, err := os.Stat(vol); err != nil {
		return 0, 0, fmt.Errorf("%s has no volume.safetensors and no vocab cochain", dir)
	}
	cell, n, err := cellAxis(vol)
	if err != nil {
		return 0, 0, err
	}
	embedPath, err := findEmbed(dir)
	if err != nil {
		return 0, 0, err
	}
	s, err := openST(embedPath)
	if err != nil {
		return 0, 0, err
	}
	defer s.close()
	name := "model.embed_tokens.weight"
	if !s.has(name) {
		name = "model.language_model.embed_tokens.weight"
	}
	return stripEmbed(s, name, cell, n, out, embedPath)
}

func findEmbed(dir string) (string, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var cands []string
	for _, e := range ents {
		n := e.Name()
		if e.IsDir() || !strings.HasSuffix(n, ".safetensors") {
			continue
		}
		if n == "volume.safetensors" || n == "vocab.float.safetensors" || n == "model.cochain.safetensors" {
			continue
		}
		cands = append(cands, filepath.Join(dir, n))
	}
	sort.Slice(cands, func(i, j int) bool {
		ni, nj := strings.Contains(cands[i], ".neck."), strings.Contains(cands[j], ".neck.")
		if ni != nj {
			return ni
		}
		return cands[i] < cands[j]
	})
	for _, p := range cands {
		s, err := openST(p)
		if err != nil {
			continue
		}
		ok := s.has("model.embed_tokens.weight") || s.has("model.language_model.embed_tokens.weight")
		s.close()
		if ok {
			return p, nil
		}
	}
	return "", fmt.Errorf("%s has no embedding safetensors", dir)
}

func loadCochain(path string) ([]int32, int, int, error) {
	s, err := openST(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer s.close()
	v, shape, err := s.i32("cochain")
	if err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %v", path, err)
	}
	if len(shape) != 2 {
		return nil, 0, 0, fmt.Errorf("cochain shape %v", shape)
	}
	if s.meta["quantum"] != "" && s.meta["quantum"] != "1/4096" {
		return nil, 0, 0, fmt.Errorf("%s quantum %s, this tool reads 1/4096", path, s.meta["quantum"])
	}
	return v, shape[0], shape[1], nil
}

func nonzero(row []int32) []int {
	var o []int
	for i, v := range row {
		if v != 0 {
			o = append(o, i)
		}
	}
	return o
}

func rowsOf(v []int32, n, C int) [][]int32 {
	o := make([][]int32, n)
	for i := 0; i < n; i++ {
		o[i] = append([]int32(nil), v[i*C:(i+1)*C]...)
	}
	return o
}

// Text builds the sub-topology of text on an integer cochain and writes it
// when out is set. The sha256 is the anchor chain, which is the cochain of
// this text in this model.
func Text(cochainPath, tokPath, cfgPath, text, out string) (*TextCochain, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text is empty")
	}
	co, V, C, err := loadCochain(cochainPath)
	if err != nil {
		return nil, err
	}
	sess, err := eval.OpenTok(tokPath, cfgPath)
	if err != nil {
		return nil, err
	}
	res, anchors, err := buildText(co, V, C, sess, text)
	if err != nil {
		return nil, err
	}
	if out != "" {
		topRows := len(res.Top)
		top := make([]int32, topRows*C)
		for i := 0; i < topRows; i++ {
			copy(top[i*C:(i+1)*C], res.Top[i])
		}
		meta := map[string]string{
			"format": "pt", "quantum": "1/4096", "text": text, "sha256": res.SHA256,
			"cochain_of": cochainPath,
		}
		if err := writeI32(out, meta, map[string]tensor{
			"anchors": {[]int{len(res.Words), C}, anchors},
			"top":     {[]int{topRows, C}, top},
		}); err != nil {
			return nil, err
		}
	}
	return res, nil
}

// buildText is the cochain of one text. anchors is the flat [words, cells] chain.
func buildText(co []int32, V, C int, sess *eval.Session, text string) (*TextCochain, []int32, error) {
	if strings.TrimSpace(text) == "" {
		return nil, nil, fmt.Errorf("text is empty")
	}
	segs := sess.Segments(text)
	if len(segs) == 0 {
		return nil, nil, fmt.Errorf("text has no word")
	}
	anchors, err := sumAnchors(co, V, C, segs)
	if err != nil {
		return nil, nil, err
	}
	top, err := coboundary(anchors, len(segs), C)
	if err != nil {
		return nil, nil, err
	}
	sum := sha256.Sum256(i32bytes(anchors))
	res := &TextCochain{
		Text: text, SHA256: hex.EncodeToString(sum[:]), Cells: C,
		Anchors: rowsOf(anchors, len(segs), C),
	}
	topRows := 1
	if len(segs) > 1 {
		topRows = len(segs) - 1
	}
	res.Top = rowsOf(top, topRows, C)
	g, c, h := hodgeEnergy(top, topRows, C)
	res.Hodge = [3]float64{g, c, h}
	for i, ids := range segs {
		w := Word{IDs: ids, Text: strings.TrimSpace(sess.Decode(ids)), Cells: nonzero(anchors[i*C : (i+1)*C])}
		res.Words = append(res.Words, w)
	}
	if len(segs) > 1 {
		for i := 0; i < len(segs)-1; i++ {
			res.Edges = append(res.Edges, Edge{From: i, To: i + 1, Cells: nonzero(top[i*C : (i+1)*C])})
		}
	}
	return res, anchors, nil
}

// Work is one text in a corpus.
type Work struct {
	Name string `json:"name"`
	Text string `json:"text"`
}

// CorpusWork is one text placed on the corpus map. Cells is the sum of its
// word anchors, in float, one entry per residual cell.
type CorpusWork struct {
	Name   string    `json:"name"`
	SHA256 string    `json:"sha256"`
	Words  int       `json:"words"`
	Cells  []float64 `json:"cells"`
}

// CorpusEdge is the cochain of the next text minus this one.
type CorpusEdge struct {
	From  int       `json:"from"`
	To    int       `json:"to"`
	Same  bool      `json:"same"`
	L1    float64   `json:"l1"`
	Cells []float64 `json:"cells"`
}

// CorpusMap is the cochain of every text, and the difference to the next one.
type CorpusMap struct {
	Cells int          `json:"cells"`
	Works []CorpusWork `json:"works"`
	Edges []CorpusEdge `json:"edges"`
	Path  string       `json:"path,omitempty"`
}

// Corpus loads the model once and places each text on the residual cells.
// Two texts with the same anchor chain share a sha256. The edge between
// neighbors is the difference of their cell sums. out, when set, is a float
// weight file with one row per text.
func Corpus(cochainPath, tokPath, cfgPath string, items []Work, out string) (*CorpusMap, error) {
	if len(items) == 0 {
		return nil, fmt.Errorf("the corpus is empty")
	}
	co, V, C, err := loadCochain(cochainPath)
	if err != nil {
		return nil, err
	}
	sess, err := eval.OpenTok(tokPath, cfgPath)
	if err != nil {
		return nil, err
	}
	m := &CorpusMap{Cells: C}
	var weight []float32
	for i, it := range items {
		name := strings.TrimSpace(it.Name)
		if name == "" {
			name = fmt.Sprintf("text %d", i+1)
		}
		res, anchors, err := buildText(co, V, C, sess, it.Text)
		if err != nil {
			return nil, fmt.Errorf("%s: %v", name, err)
		}
		cells := make([]float64, C)
		for w := 0; w < len(res.Words); w++ {
			for g := 0; g < C; g++ {
				cells[g] += float64(anchors[w*C+g]) / quantum
			}
		}
		m.Works = append(m.Works, CorpusWork{Name: name, SHA256: res.SHA256, Words: len(res.Words), Cells: cells})
		for _, v := range cells {
			weight = append(weight, float32(v))
		}
	}
	for i := 0; i+1 < len(m.Works); i++ {
		d := make([]float64, C)
		var l1 float64
		for g := 0; g < C; g++ {
			d[g] = m.Works[i+1].Cells[g] - m.Works[i].Cells[g]
			l1 += math.Abs(d[g])
		}
		m.Edges = append(m.Edges, CorpusEdge{
			From: i, To: i + 1,
			Same: m.Works[i].SHA256 == m.Works[i+1].SHA256,
			L1:   l1, Cells: d,
		})
	}
	if out != "" {
		if err := writeCorpus(out, cochainPath, m, weight); err != nil {
			return nil, err
		}
		m.Path = out
	}
	return m, nil
}

func writeCorpus(path, cochainPath string, m *CorpusMap, weight []float32) error {
	n := len(m.Works)
	C := m.Cells
	names := make([]string, n)
	shas := make([]string, n)
	meta := map[string]string{
		"format": "pt", "quantum": "1/4096", "cochain_of": cochainPath, "lift": "cells",
	}
	for i, w := range m.Works {
		names[i] = w.Name
		shas[i] = w.SHA256
		meta[fmt.Sprintf("text.%d", i)] = w.Name
		meta[fmt.Sprintf("sha.%d", i)] = w.SHA256
	}
	nb, err := json.Marshal(names)
	if err != nil {
		return err
	}
	sb, err := json.Marshal(shas)
	if err != nil {
		return err
	}
	meta["texts"] = string(nb)
	meta["sha256"] = string(sb)
	tensors := map[string]ftensor{
		"weight": {[]int{n, C}, weight},
	}
	order := []string{"weight"}
	if len(m.Edges) > 0 {
		diff := make([]float32, len(m.Edges)*C)
		for i, e := range m.Edges {
			for g, v := range e.Cells {
				diff[i*C+g] = float32(v)
			}
		}
		tensors["diff"] = ftensor{[]int{len(m.Edges), C}, diff}
		order = append(order, "diff")
	}
	return writeFloat(path, meta, order, tensors)
}

// Extracted is the floating-point weight of a text. Row i of Weight is word i.
type Extracted struct {
	Path    string       `json:"path"`
	Words   int          `json:"words"`
	Dim     int          `json:"dim"`
	Lifted  bool         `json:"lifted"`
	Cochain *TextCochain `json:"cochain"`
}

// Extract writes a float weight file for the words of text. Each row is that
// word's anchor divided by the quantum. When the model folder has cell.D, the
// row is lifted to one float per residual channel, the same equal share the
// vocab float uses. The integer cochain stays exact in the cochain file.
func Extract(cochainPath, tokPath, cfgPath, volumePath, text, out string) (*Extracted, error) {
	res, err := Text(cochainPath, tokPath, cfgPath, text, "")
	if err != nil {
		return nil, err
	}
	m := len(res.Words)
	C := res.Cells
	anchors := make([]int32, m*C)
	for i := 0; i < m; i++ {
		copy(anchors[i*C:(i+1)*C], res.Anchors[i])
	}
	cellFloat := make([]float32, m*C)
	for i, v := range anchors {
		cellFloat[i] = float32(float64(v) / quantum)
	}
	weight := cellFloat
	dim := C
	lifted := false
	if volumePath == "" {
		volumePath = filepath.Join(filepath.Dir(cochainPath), "volume.safetensors")
		if _, err := os.Stat(volumePath); err != nil {
			volumePath = ""
		}
	}
	if volumePath != "" {
		cell, n, err := cellAxis(volumePath)
		if err != nil {
			return nil, err
		}
		if n != C {
			return nil, fmt.Errorf("cell.D has %d cells, the cochain has %d", n, C)
		}
		liftedW, err := liftWeight(cellFloat, C, cell)
		if err != nil {
			return nil, err
		}
		weight = liftedW
		dim = len(cell)
		lifted = true
	}
	if out == "" {
		out = filepath.Join(filepath.Dir(cochainPath), "text.weight.safetensors")
	}
	words := make([]string, m)
	tensors := map[string]ftensor{
		"weight": {[]int{m, dim}, weight},
	}
	order := []string{"weight"}
	if lifted {
		tensors["cochain"] = ftensor{[]int{m, C}, cellFloat}
		order = append(order, "cochain")
	}
	for i, w := range res.Words {
		words[i] = w.Text
		key := fmt.Sprintf("word.%d", i)
		row := append([]float32(nil), weight[i*dim:(i+1)*dim]...)
		tensors[key] = ftensor{[]int{dim}, row}
		order = append(order, key)
	}
	names, err := json.Marshal(words)
	if err != nil {
		return nil, err
	}
	lift := "cells"
	if lifted {
		lift = "channels"
	}
	meta := map[string]string{
		"format": "pt", "quantum": "1/4096", "text": text, "sha256": res.SHA256,
		"cochain_of": cochainPath, "words": string(names), "lift": lift,
	}
	for i, w := range words {
		meta[fmt.Sprintf("word.%d", i)] = w
	}
	if err := writeFloat(out, meta, order, tensors); err != nil {
		return nil, err
	}
	return &Extracted{Path: out, Words: m, Dim: dim, Lifted: lifted, Cochain: res}, nil
}

// liftWeight spreads each cell's float across the channels in that cell.
// Every channel in a cell receives an equal share, so the cell sums of the
// result are the cochain.
func liftWeight(co []float32, C int, cell []int32) ([]float32, error) {
	if C == 0 || len(cell) == 0 || len(co)%C != 0 {
		return nil, fmt.Errorf("cannot lift a %d-cell cochain onto %d channels", C, len(cell))
	}
	count := make([]int, C)
	for _, c := range cell {
		g := int(c)
		if g < 0 || g >= C {
			return nil, fmt.Errorf("cell.D entry %d is outside %d cells", g, C)
		}
		count[g]++
	}
	m := len(co) / C
	D := len(cell)
	o := make([]float32, m*D)
	for t := 0; t < m; t++ {
		for j := 0; j < D; j++ {
			g := int(cell[j])
			if count[g] == 0 {
				return nil, fmt.Errorf("residual cell %d holds no channel", g)
			}
			o[t*D+j] = float32(float64(co[t*C+g]) / float64(count[g]))
		}
	}
	return o, nil
}

func sumAnchors(co []int32, V, C int, segs [][]int) ([]int32, error) {
	o := make([]int32, len(segs)*C)
	for i, ids := range segs {
		if len(ids) == 0 {
			return nil, fmt.Errorf("word %d has no token", i)
		}
		acc := make([]int64, C)
		for _, id := range ids {
			if id < 0 || id >= V {
				return nil, fmt.Errorf("token %d is outside the cochain (%d rows)", id, V)
			}
			row := co[id*C : (id+1)*C]
			for g, v := range row {
				acc[g] += int64(v)
			}
		}
		for g, v := range acc {
			if v > math.MaxInt32 || v < math.MinInt32 {
				return nil, fmt.Errorf("word %d cell %d does not fit in int32", i, g)
			}
			o[i*C+g] = int32(v)
		}
	}
	return o, nil
}

func coboundary(anchors []int32, m, C int) ([]int32, error) {
	if m <= 1 {
		return append([]int32(nil), anchors...), nil
	}
	top := make([]int32, (m-1)*C)
	for i := 0; i < m-1; i++ {
		for g := 0; g < C; g++ {
			d := int64(anchors[(i+1)*C+g]) - int64(anchors[i*C+g])
			if d > math.MaxInt32 || d < math.MinInt32 {
				return nil, fmt.Errorf("top cochain word %d cell %d does not fit in int32", i, g)
			}
			top[i*C+g] = int32(d)
		}
	}
	return top, nil
}

func i32bytes(v []int32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

// hodgeEnergy is the paper's split of one cochain matrix: gradient on the
// zero row or column, harmonic on the four corner bins, curl on the rest.
// G+C+H equals the squared Euclidean norm.
func hodgeEnergy(M []int32, R, C int) (g, curl, h float64) {
	if R == 0 || C == 0 {
		return 0, 0, 0
	}
	norm := float64(R * C)
	for kr := 0; kr < R; kr++ {
		for kc := 0; kc < C; kc++ {
			var re, im float64
			for r := 0; r < R; r++ {
				for c := 0; c < C; c++ {
					ang := -2 * math.Pi * (float64(kr*r)/float64(R) + float64(kc*c)/float64(C))
					v := float64(M[r*C+c])
					re += v * math.Cos(ang)
					im += v * math.Sin(ang)
				}
			}
			e := (re*re + im*im) / norm
			if kr == 0 || kc == 0 {
				g += e
			} else if (kr == 1 || kr == R-1) && (kc == 1 || kc == C-1) {
				h += e
			} else {
				curl += e
			}
		}
	}
	return g, curl, h
}

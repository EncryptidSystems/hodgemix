package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"hodgemix/eval"
)

func TestQuant(t *testing.T) {
	q, err := quant(1.0 / quantum)
	if err != nil || q != 1 {
		t.Fatalf("1/4096 → %d %v", q, err)
	}
	q, err = quant(0.5 / quantum)
	if err != nil || q != 1 {
		t.Fatalf("half → %d %v", q, err)
	}
	q, err = quant(-1.5 / quantum)
	if err != nil || q != -2 {
		t.Fatalf("neg → %d %v", q, err)
	}
}

func TestHodgeParseval(t *testing.T) {
	M := []int32{1, -2, 3, 4, 0, -1, 2, 2, 5}
	g, c, h := hodgeEnergy(M, 3, 3)
	var l2 float64
	for _, v := range M {
		l2 += float64(v * v)
	}
	if math.Abs(g+c+h-l2) > 1e-6 {
		t.Fatalf("G+C+H=%g l2=%g", g+c+h, l2)
	}
}

func TestStripAndText(t *testing.T) {
	dir := t.TempDir()
	tok := writeByteTok(t, dir)
	// Rows are the byte ids. 'A'=65, 'b'=98, 'C'=67, 'd'=100.
	const C = 4
	co := make([]int32, 256*C)
	put := func(b byte, row ...int32) {
		copy(co[int(b)*C:(int(b)+1)*C], row)
	}
	put('A', 3, 0, 0, 0)
	put('b', 0, 4, 0, 0)
	put('C', 0, 0, 5, 0)
	put('d', 0, 0, 0, 6)
	// Float cochain is the integer divided by quantum, so the strip returns it.
	fl := make([]float32, len(co))
	for i, v := range co {
		fl[i] = float32(v) / quantum
	}
	vocab := filepath.Join(dir, "vocab.float.safetensors")
	if err := writeF32(vocab, "cochain", []int{256, C}, fl); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "model.cochain.safetensors")
	rows, cells, err := StripDir(dir, out)
	if err != nil {
		t.Fatal(err)
	}
	if rows != 256 || cells != C {
		t.Fatalf("shape %d×%d", rows, cells)
	}
	got, V, CC, err := loadCochain(out)
	if err != nil {
		t.Fatal(err)
	}
	if V != 256 || CC != C {
		t.Fatal(V, CC)
	}
	for i := range co {
		if got[i] != co[i] {
			t.Fatalf("cochain[%d]=%d want %d", i, got[i], co[i])
		}
	}
	res, err := Text(out, tok, "", "Ab Cd", filepath.Join(dir, "text.cochain.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Words) != 2 || res.Words[0].Text != "Ab" || res.Words[1].Text != "Cd" {
		t.Fatalf("words %+v", res.Words)
	}
	if res.Words[0].Cells[0] != 0 || res.Words[0].Cells[1] != 1 {
		t.Fatalf("Ab cells %v", res.Words[0].Cells)
	}
	wantTop := []int32{-3, -4, 5, 6}
	if len(res.Top) != 1 || len(res.Top[0]) != 4 {
		t.Fatalf("top %#v", res.Top)
	}
	for i, v := range wantTop {
		if res.Top[0][i] != v {
			t.Fatalf("top %v want %v", res.Top[0], wantTop)
		}
	}
	if len(res.Edges) != 1 || res.Edges[0].From != 0 || res.Edges[0].To != 1 {
		t.Fatalf("edges %+v", res.Edges)
	}
	again, err := Text(out, tok, "", "Ab Cd", "")
	if err != nil {
		t.Fatal(err)
	}
	if again.SHA256 != res.SHA256 {
		t.Fatal("sha changed")
	}
	other, err := Text(out, tok, "", "Ab", "")
	if err != nil {
		t.Fatal(err)
	}
	if other.SHA256 == res.SHA256 {
		t.Fatal("different text, same cochain")
	}
	if len(other.Top) != 1 || other.Top[0][0] != 3 || other.Top[0][1] != 4 {
		t.Fatalf("one-word top %v", other.Top)
	}
	sess, err := eval.OpenTok(tok, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(sess.Segments("Ab Cd")) != 2 {
		t.Fatal(sess.Segments("Ab Cd"))
	}
	// Two spaces leave a whitespace pre-token. pieces keeps it; Segments does not.
	gap := sess.Encode("Ab  Cd", false, false)
	segs := sess.Segments("Ab  Cd")
	if len(segs) != 2 {
		t.Fatalf("segments %v", segs)
	}
	var n int
	for _, s := range segs {
		n += len(s)
	}
	if len(gap) <= n {
		t.Fatalf("pieces dropped the whitespace token: %v segments %v", gap, segs)
	}
}

func TestExtract(t *testing.T) {
	dir := t.TempDir()
	tok := writeByteTok(t, dir)
	const C = 4
	co := make([]int32, 256*C)
	copy(co[int('A')*C:], []int32{3, 0, 0, 0})
	copy(co[int('b')*C:], []int32{0, 4, 0, 0})
	copy(co[int('C')*C:], []int32{0, 0, 5, 0})
	copy(co[int('d')*C:], []int32{0, 0, 0, 6})
	path := filepath.Join(dir, "model.cochain.safetensors")
	if err := writeI32(path, map[string]string{"format": "pt", "quantum": "1/4096"}, map[string]tensor{
		"cochain": {[]int{256, C}, co},
	}); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "text.weight.safetensors")
	ex, err := Extract(path, tok, "", "", "Ab Cd", out)
	if err != nil {
		t.Fatal(err)
	}
	if ex.Lifted || ex.Words != 2 || ex.Dim != C {
		t.Fatalf("unlifted %+v", ex)
	}
	got := openWeight(t, out)
	if g, _, err := got.f32("weight"); err != nil || len(g) != 8 || g[0] != float32(3)/quantum || g[1] != float32(4)/quantum {
		t.Fatalf("cell weight %v %v", g, err)
	}
	var words []string
	if err := json.Unmarshal([]byte(got.meta["words"]), &words); err != nil || len(words) != 2 || words[0] != "Ab" || words[1] != "Cd" {
		t.Fatalf("words %s %v", got.meta["words"], err)
	}
	if got.meta["word.0"] != "Ab" || got.meta["word.1"] != "Cd" {
		t.Fatalf("word meta %v", got.meta)
	}

	cell := []int32{0, 0, 1, 1, 2, 2, 3, 3}
	vol := filepath.Join(dir, "volume.safetensors")
	if err := writeI32(vol, map[string]string{"format": "pt"}, map[string]tensor{
		"cell.D": {[]int{len(cell)}, cell},
	}); err != nil {
		t.Fatal(err)
	}
	ex, err = Extract(path, tok, "", "", "Ab Cd", out)
	if err != nil {
		t.Fatal(err)
	}
	if !ex.Lifted || ex.Dim != 8 {
		t.Fatalf("lifted %+v", ex)
	}
	got = openWeight(t, out)
	w, shape, err := got.f32("weight")
	if err != nil || len(shape) != 2 || shape[0] != 2 || shape[1] != 8 {
		t.Fatalf("weight shape %v %v", shape, err)
	}
	// Ab is cells [3,4,0,0]. Two channels share each cell.
	share := func(q int32) float32 { return float32(float64(q) / quantum / 2) }
	want := []float32{share(3), share(3), share(4), share(4), 0, 0, 0, 0}
	for i, v := range want {
		if w[i] != v {
			t.Fatalf("Ab[%d]=%g want %g", i, w[i], v)
		}
	}
	one, _, err := got.f32("word.0")
	if err != nil || len(one) != 8 || one[0] != w[0] {
		t.Fatalf("word.0 %v %v", one, err)
	}
	cf, cs, err := got.f32("cochain")
	if err != nil || cs[0] != 2 || cs[1] != C || cf[0] != float32(3)/quantum || cf[1] != float32(4)/quantum {
		t.Fatalf("cochain %v %v %v", cs, cf, err)
	}
	// Cell sums of the lift recover the float cochain.
	for g := 0; g < C; g++ {
		var s float32
		s += w[g*2]
		s += w[g*2+1]
		if s != cf[g] {
			t.Fatalf("cell %d sum %g want %g", g, s, cf[g])
		}
	}
}

func TestCorpusDiff(t *testing.T) {
	dir := t.TempDir()
	tok := writeByteTok(t, dir)
	const C = 4
	co := make([]int32, 256*C)
	copy(co[int('A')*C:], []int32{3, 0, 0, 0})
	copy(co[int('b')*C:], []int32{0, 4, 0, 0})
	copy(co[int('C')*C:], []int32{0, 0, 5, 0})
	copy(co[int('d')*C:], []int32{0, 0, 0, 6})
	path := filepath.Join(dir, "model.cochain.safetensors")
	if err := writeI32(path, map[string]string{"format": "pt", "quantum": "1/4096"}, map[string]tensor{
		"cochain": {[]int{256, C}, co},
	}); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "corpus.weight.safetensors")
	m, err := Corpus(path, tok, "", []Work{
		{Name: "one", Text: "Ab Cd"},
		{Name: "two", Text: "Ab"},
		{Name: "again", Text: "Ab Cd"},
	}, out)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Works) != 3 || len(m.Edges) != 2 {
		t.Fatalf("map %+v", m)
	}
	if m.Works[0].SHA256 == m.Works[1].SHA256 || m.Works[0].SHA256 != m.Works[2].SHA256 {
		t.Fatalf("sha %s %s %s", m.Works[0].SHA256, m.Works[1].SHA256, m.Works[2].SHA256)
	}
	if m.Edges[0].Same || m.Edges[0].L1 == 0 {
		t.Fatalf("first edge should differ: %+v", m.Edges[0])
	}
	// Ab minus Ab Cd is [0,0,-5,-6] / quantum.
	want := []float64{0, 0, -5 / float64(quantum), -6 / float64(quantum)}
	for i, v := range want {
		if m.Edges[0].Cells[i] != v {
			t.Fatalf("diff %v want %v", m.Edges[0].Cells, want)
		}
	}
	same, err := Corpus(path, tok, "", []Work{{Name: "a", Text: "Ab"}, {Name: "b", Text: "Ab"}}, "")
	if err != nil {
		t.Fatal(err)
	}
	if !same.Edges[0].Same || same.Edges[0].L1 != 0 {
		t.Fatalf("identical texts: %+v", same.Edges[0])
	}
	got := openWeight(t, out)
	if _, shape, err := got.f32("weight"); err != nil || shape[0] != 3 || shape[1] != C {
		t.Fatalf("weight %v %v", shape, err)
	}
	if _, shape, err := got.f32("diff"); err != nil || shape[0] != 2 || shape[1] != C {
		t.Fatalf("diff %v %v", shape, err)
	}
}

func openWeight(t *testing.T, path string) *st {
	t.Helper()
	s, err := openST(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.close)
	return s
}

func writeByteTok(t *testing.T, dir string) string {
	t.Helper()
	n := 0
	vocab := map[string]int{}
	for b := 0; b < 256; b++ {
		var r rune
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE) {
			r = rune(b)
		} else {
			r = rune(256 + n)
			n++
		}
		vocab[string(r)] = b
	}
	body := map[string]any{
		"model": map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
	}
	b, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	p := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func writeF32(path, name string, shape []int, v []float32) error {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		put := math.Float32bits(x)
		b[4*i] = byte(put)
		b[4*i+1] = byte(put >> 8)
		b[4*i+2] = byte(put >> 16)
		b[4*i+3] = byte(put >> 24)
	}
	header := map[string]any{
		name:           info{Dtype: "F32", Shape: shape, Off: []int64{0, int64(len(b))}},
		"__metadata__": map[string]string{"format": "pt"},
	}
	hb, err := json.Marshal(header)
	if err != nil {
		return err
	}
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var nbuf [8]byte
	nbuf[0] = byte(len(hb))
	nbuf[1] = byte(len(hb) >> 8)
	nbuf[2] = byte(len(hb) >> 16)
	nbuf[3] = byte(len(hb) >> 24)
	if _, err := f.Write(nbuf[:]); err != nil {
		return err
	}
	if _, err := f.Write(hb); err != nil {
		return err
	}
	_, err = f.Write(b)
	return err
}

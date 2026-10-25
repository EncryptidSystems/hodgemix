package main

// Union vocabulary (default; --vocab base keeps the base's): the output is a new model, so it gets its own
// tokenizer.json, tokenizer_config.json and config.json, over the union of the base's and every
// source's vocabulary.
//
//   - Ids: the base's tokens keep their ids (every base input still encodes the same); each source token
//     whose string the base lacks is appended, in the source's id order, then the source's added
//     (special) tokens the base lacks.
//   - Merges: the base's, then each source's merges whose result is in the union and that the base does
//     not have, so BPE reaches every appended token; pre-tokenizer, normalizer and decoder stay the base's.
//   - Embeddings: base rows as they are; an appended token's row is its source embedding carried into
//     the base's residual space by the residual gauge map M of the shared vocabulary: both models embed
//     the same token string, so the least-squares map over the shared strings,
//         E_base[t] - mu_b ~ (E_src[t] - mu_s) M,
//     is the change of basis between the two residual streams (tied heads: the same map at the output).
//     It is reported with its held-out R^2.
// No paper equation.

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"

	"hodgemix/eval"
)

// VocabMode is --vocab: union | base.
var VocabMode = "union"

type addedTok struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type tokFile struct {
	raw    map[string]json.RawMessage
	vocab  map[string]int
	merges []string // "a b"
	pairs  bool     // merges stored as [a, b] pairs
	added  []addedTok
	addRaw []json.RawMessage
}

func readTok(p string) (*tokFile, error) {
	if p == "" {
		return nil, fmt.Errorf("no tokenizer set (set it in the run file)")
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	t := &tokFile{}
	if err := json.Unmarshal(b, &t.raw); err != nil {
		return nil, fmt.Errorf("%s: %v", p, err)
	}
	var model struct {
		Type   string          `json:"type"`
		Vocab  map[string]int  `json:"vocab"`
		Merges json.RawMessage `json:"merges"`
	}
	if err := json.Unmarshal(t.raw["model"], &model); err != nil || model.Type != "BPE" || len(model.Vocab) == 0 {
		return nil, fmt.Errorf("%s: not a BPE tokenizer.json", p)
	}
	t.vocab = model.Vocab
	if json.Unmarshal(model.Merges, &t.merges) != nil {
		var mp [][2]string
		if err := json.Unmarshal(model.Merges, &mp); err != nil {
			return nil, fmt.Errorf("%s: merges: %v", p, err)
		}
		t.pairs = true
		for _, m := range mp {
			t.merges = append(t.merges, m[0]+" "+m[1])
		}
	}
	if a, ok := t.raw["added_tokens"]; ok {
		if err := json.Unmarshal(a, &t.added); err != nil {
			return nil, fmt.Errorf("%s: added_tokens: %v", p, err)
		}
		if err := json.Unmarshal(a, &t.addRaw); err != nil {
			return nil, fmt.Errorf("%s: added_tokens: %v", p, err)
		}
	}
	return t, nil
}

// ids is 1 + the largest id the tokenizer can produce.
func (t *tokFile) ids() int {
	n := 0
	for _, v := range t.vocab {
		n = max(n, v+1)
	}
	for _, a := range t.added {
		n = max(n, a.ID+1)
	}
	return n
}

type vocabUnion struct {
	tok, tokCfg []byte
	emb         []float32 // [V, D]
	V, D        int
	appended    int
	added       int
	r2          []float64
	shared      []int
}

// residualMap is M (Ds x Db) and the two means for the shared strings of two vocabularies, fitted on a
// random 90% and scored (R^2) on the rest.
func residualMap(Eb []float32, Db int, vb map[string]int, Es []float32, Ds int, vs map[string]int) (M, mb, ms []float64, r2 float64, n int) {
	type pr struct{ b, s int }
	var ps []pr
	for k, i := range vs {
		if j, ok := vb[k]; ok {
			ps = append(ps, pr{j, i})
		}
	}
	sort.Slice(ps, func(a, b int) bool { return ps[a].b < ps[b].b })
	rand.New(rand.NewSource(1)).Shuffle(len(ps), func(a, b int) { ps[a], ps[b] = ps[b], ps[a] })
	n = len(ps)
	fit := ps[:n*9/10]
	mb, ms = make([]float64, Db), make([]float64, Ds)
	for _, p := range fit {
		for j := 0; j < Db; j++ {
			mb[j] += float64(Eb[p.b*Db+j])
		}
		for j := 0; j < Ds; j++ {
			ms[j] += float64(Es[p.s*Ds+j])
		}
	}
	for j := range mb {
		mb[j] /= float64(len(fit))
	}
	for j := range ms {
		ms[j] /= float64(len(fit))
	}
	// Normal equations A M = B, A = Xs^T Xs (Ds x Ds), B = Xs^T Xb (Ds x Db), centred.
	A, B := make([]float64, Ds*Ds), make([]float64, Ds*Db)
	parFor(Ds, func(lo, hi int) {
		xs, xb := make([]float64, Ds), make([]float64, Db)
		for _, p := range fit {
			for j := 0; j < Ds; j++ {
				xs[j] = float64(Es[p.s*Ds+j]) - ms[j]
			}
			for j := 0; j < Db; j++ {
				xb[j] = float64(Eb[p.b*Db+j]) - mb[j]
			}
			for r := lo; r < hi; r++ {
				a := xs[r]
				if a == 0 {
					continue
				}
				ar, br := A[r*Ds:(r+1)*Ds], B[r*Db:(r+1)*Db]
				for c := 0; c < Ds; c++ {
					ar[c] += a * xs[c]
				}
				for c := 0; c < Db; c++ {
					br[c] += a * xb[c]
				}
			}
		}
	})
	var tr float64
	for i := 0; i < Ds; i++ {
		tr += A[i*Ds+i]
	}
	for i := 0; i < Ds; i++ { // a ridge of 1e-6 of the mean variance keeps the solve definite
		A[i*Ds+i] += 1e-6 * tr / float64(Ds)
	}
	M = cholSolve(A, B, Ds, Db)
	var ss, st float64
	var mu sync.Mutex
	test := ps[n*9/10:]
	parFor(len(test), func(lo, hi int) {
		var a, b float64
		xs := make([]float64, Ds)
		for _, p := range test[lo:hi] {
			for j := 0; j < Ds; j++ {
				xs[j] = float64(Es[p.s*Ds+j]) - ms[j]
			}
			for c := 0; c < Db; c++ {
				var y float64
				for j := 0; j < Ds; j++ {
					y += xs[j] * M[j*Db+c]
				}
				d := float64(Eb[p.b*Db+c]) - mb[c]
				a += (d - y) * (d - y)
				b += d * d
			}
		}
		mu.Lock()
		ss, st = ss+a, st+b
		mu.Unlock()
	})
	return M, mb, ms, 1 - ss/math.Max(st, 1e-30), n
}

// cholSolve solves A X = B for symmetric positive definite A (n x n), B (n x m).
func cholSolve(A, B []float64, n, m int) []float64 {
	Lm := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s := A[i*n+j]
			for k := 0; k < j; k++ {
				s -= Lm[i*n+k] * Lm[j*n+k]
			}
			if i == j {
				Lm[i*n+i] = math.Sqrt(math.Max(s, 1e-300))
			} else {
				Lm[i*n+j] = s / Lm[j*n+j]
			}
		}
	}
	X := append([]float64(nil), B...)
	parFor(m, func(lo, hi int) {
		y := make([]float64, n)
		for c := lo; c < hi; c++ {
			for i := 0; i < n; i++ {
				s := X[i*m+c]
				for k := 0; k < i; k++ {
					s -= Lm[i*n+k] * y[k]
				}
				y[i] = s / Lm[i*n+i]
			}
			for i := n - 1; i >= 0; i-- {
				s := y[i]
				for k := i + 1; k < n; k++ {
					s -= Lm[k*n+i] * X[k*m+c]
				}
				X[i*m+c] = s / Lm[i*n+i]
			}
		}
	})
	return X
}

// buildUnion makes the union tokenizer and embedding; nil when no source adds a token.
func buildUnion(base *ST, basePath string, srcs []*ST) (*vocabUnion, error) {
	bt, err := readTok(eval.FilesOf(basePath).Tokenizer)
	if err != nil {
		return nil, err
	}
	if !base.Has("model.embed_tokens.weight") {
		return nil, fmt.Errorf("base has no model.embed_tokens.weight")
	}
	esh := base.LogicalShape("model.embed_tokens.weight")
	Vb, Db := esh[0], esh[1]
	if n := bt.ids(); n > Vb {
		return nil, fmt.Errorf("the tokenizer set for the base has ids up to %d but the base embeds %d rows: it is not the base's tokenizer", n-1, Vb)
	}
	Eb, err := base.F32("model.embed_tokens.weight")
	if err != nil {
		return nil, err
	}
	u := &vocabUnion{D: Db}
	vocab := map[string]int{}
	for k, v := range bt.vocab {
		vocab[k] = v
	}
	have := map[string]bool{}
	next := Vb
	for k, v := range bt.vocab {
		have[k] = true
		next = max(next, v+1)
	}
	for _, a := range bt.added {
		have[a.Content] = true
		next = max(next, a.ID+1)
	}
	merges := append([]string(nil), bt.merges...)
	haveMerge := map[string]bool{}
	for _, m := range merges {
		haveMerge[m] = true
	}
	added := append([]json.RawMessage(nil), bt.addRaw...)
	var newAdded []addedTok
	type newRow struct{ src, sid, uid int } // source, its token id, the union id
	var rows []newRow
	for si, st := range srcs {
		ek := embedKey(st)
		if ek == "" {
			continue // a shard without the embedding adds no tokens
		}
		stk, err := readTok(eval.FilesOf(st.Path).Tokenizer)
		if err != nil {
			return nil, fmt.Errorf("source %d (%s): %v", si+1, st.Path, err)
		}
		if n, rows := stk.ids(), st.LogicalShape(ek)[0]; n > rows {
			return nil, fmt.Errorf("source %d (%s): its tokenizer has ids up to %d but it embeds %d rows: it is another model's tokenizer", si+1, st.Path, n-1, rows)
		}
		type kv struct {
			s  string
			id int
		}
		var ord []kv
		for k, v := range stk.vocab {
			ord = append(ord, kv{k, v})
		}
		sort.Slice(ord, func(a, b int) bool { return ord[a].id < ord[b].id })
		for _, x := range ord {
			if !have[x.s] {
				have[x.s] = true
				vocab[x.s] = next
				rows = append(rows, newRow{si, x.id, next})
				next++
			}
		}
		for _, a := range stk.added {
			if !have[a.Content] {
				have[a.Content] = true
				newAdded = append(newAdded, addedTok{next, a.Content, a.Special})
				rows = append(rows, newRow{si, a.ID, next})
				next++
			}
		}
		for _, m := range stk.merges {
			if haveMerge[m] {
				continue
			}
			parts := strings.SplitN(m, " ", 2)
			if len(parts) != 2 {
				continue
			}
			_, a := vocab[parts[0]]
			_, b := vocab[parts[1]]
			_, c := vocab[parts[0]+parts[1]]
			if a && b && c {
				merges = append(merges, m)
				haveMerge[m] = true
			}
		}
	}
	if len(rows) == 0 {
		return nil, nil
	}
	u.V = next
	u.appended = len(rows) - len(newAdded)
	u.added = len(newAdded)
	// Residual maps, one per source that adds rows.
	emb := make([]float32, u.V*Db)
	copy(emb, Eb)
	bySrc := map[int][]newRow{}
	for _, r := range rows {
		bySrc[r.src] = append(bySrc[r.src], r)
	}
	for si := range srcs {
		rs := bySrc[si]
		if len(rs) == 0 {
			continue
		}
		st := srcs[si]
		ek := embedKey(st)
		Ds := st.LogicalShape(ek)[1]
		Es, err := st.F32(ek)
		if err != nil {
			return nil, err
		}
		stk, err := readTok(eval.FilesOf(st.Path).Tokenizer)
		if err != nil {
			return nil, err
		}
		M, mb, ms, r2, n := residualMap(Eb, Db, bt.vocab, Es, Ds, stk.vocab)
		u.r2, u.shared = append(u.r2, r2), append(u.shared, n)
		var jobs []newRow
		for _, r := range rs {
			if (r.sid+1)*Ds > len(Es) {
				return nil, fmt.Errorf("source %d: token id %d has no embedding row", si+1, r.sid)
			}
			jobs = append(jobs, r)
		}
		parFor(len(jobs), func(lo, hi int) {
			x := make([]float64, Ds)
			for _, j := range jobs[lo:hi] {
				for k := 0; k < Ds; k++ {
					x[k] = float64(Es[j.sid*Ds+k]) - ms[k]
				}
				row := emb[j.uid*Db : (j.uid+1)*Db]
				for c := 0; c < Db; c++ {
					y := mb[c]
					for k := 0; k < Ds; k++ {
						y += x[k] * M[k*Db+c]
					}
					row[c] = float32(y)
				}
			}
		})
	}
	u.emb = emb
	// tokenizer.json: the base's file with the union vocab, merges and added tokens.
	var model map[string]json.RawMessage
	json.Unmarshal(bt.raw["model"], &model)
	model["vocab"], _ = json.Marshal(vocab)
	if bt.pairs {
		mp := make([][2]string, len(merges))
		for i, m := range merges {
			p := strings.SplitN(m, " ", 2)
			mp[i] = [2]string{p[0], p[1]}
		}
		model["merges"], _ = json.Marshal(mp)
	} else {
		model["merges"], _ = json.Marshal(merges)
	}
	out := map[string]json.RawMessage{}
	for k, v := range bt.raw {
		out[k] = v
	}
	out["model"], _ = json.Marshal(model)
	for _, a := range newAdded {
		b, _ := json.Marshal(map[string]any{"id": a.ID, "content": a.Content, "single_word": false, "lstrip": false,
			"rstrip": false, "normalized": false, "special": a.Special})
		added = append(added, b)
	}
	out["added_tokens"], _ = json.Marshal(added)
	u.tok, err = json.Marshal(out)
	if err != nil {
		return nil, err
	}
	// tokenizer_config.json: the base's, with the new added tokens in added_tokens_decoder.
	cp := eval.FilesOf(basePath).TokenizerConfig
	b, err := os.ReadFile(cp)
	if err != nil {
		return nil, fmt.Errorf("the base's tokenizer_config: %v", err)
	}
	var tc map[string]json.RawMessage
	if err := json.Unmarshal(b, &tc); err != nil {
		return nil, fmt.Errorf("%s: %v", cp, err)
	}
	dec := map[string]json.RawMessage{}
	if raw, ok := tc["added_tokens_decoder"]; ok {
		if err := json.Unmarshal(raw, &dec); err != nil {
			return nil, fmt.Errorf("%s: added_tokens_decoder: %v", cp, err)
		}
	}
	for _, a := range newAdded {
		dec[itoa(a.ID)], _ = json.Marshal(map[string]any{"content": a.Content, "lstrip": false, "normalized": false,
			"rstrip": false, "single_word": false, "special": a.Special})
	}
	tc["added_tokens_decoder"], _ = json.Marshal(dec)
	if u.tokCfg, err = json.MarshalIndent(tc, "", "  "); err != nil {
		return nil, err
	}
	return u, nil
}

// spm2json writes a SentencePiece BPE tokenizer.model as the byte-level BPE tokenizer.json that
// hodgemix (eval/eval.go loadTok) and Champ (go++ stdlib/tokenizers) read.
//
//	spm2json -model tokenizer.model -config tokenizer_config.json -out tokenizer.json
//
// Both files are named; nothing is looked up beside them. Every piece keeps its id, so the
// embedding rows line up:
//   - a normal or user-defined piece is its text ("▁" as a space) in the byte-level alphabet;
//   - a byte-fallback piece <0xNN> is byte NN's letter, unless a normal piece already is that byte
//     (then it keeps its own name and is never produced);
//   - control and unknown pieces, and every id the tokenizer_config's added_tokens_decoder lists,
//     are added tokens with that config's special flag.
//
// Merges are made as transformers' SentencePiece extractor makes them: every split of a piece into
// two pieces, ranked by the piece's score (then the halves' lengths), here at byte boundaries, so a
// two-byte character that is a piece is merged from its bytes. A longer character has no
// two-piece split and encodes as its bytes. The pre-tokenizer is the Llama 3 Split + ByteLevel and
// the post-processor puts bos_token first when add_bos_token is set.
//
// Refused: a model that is not BPE with byte fallback, or whose normalizer changes text (a
// precompiled charsmap, a dummy prefix, extra-whitespace removal); a tokenizer_config whose
// tokens are not this model's pieces.
package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Piece types (sentencepiece_model.proto).
const (
	typeNormal      = 1
	typeUnknown     = 2
	typeControl     = 3
	typeUserDefined = 4
	typeUnused      = 5
	typeByte        = 6
)

const modelBPE = 2

// llama3Split is the Split regex Champ's tokenizer runs (go++ stdlib/tokenizers).
const llama3Split = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

type piece struct {
	text  string
	score float32
	typ   int
}

type spModel struct {
	pieces       []piece
	modelType    int
	byteFallback bool
	normName     string
	charsmap     int // bytes of precompiled_charsmap
	dummyPrefix  bool
	removeSpaces bool
	escapeSpaces bool
}

// ---- protobuf, the few fields read

type pbuf struct {
	b []byte
	i int
}

func (p *pbuf) varint() (uint64, error) {
	var v uint64
	for s := 0; s < 64; s += 7 {
		if p.i >= len(p.b) {
			return 0, errors.New("truncated varint")
		}
		c := p.b[p.i]
		p.i++
		v |= uint64(c&0x7f) << s
		if c < 0x80 {
			return v, nil
		}
	}
	return 0, errors.New("varint too long")
}

// next is the next field's number, wire type and payload (a varint's value, or the bytes of a
// length-delimited or fixed field).
func (p *pbuf) next() (field int, wire int, val uint64, data []byte, err error) {
	key, err := p.varint()
	if err != nil {
		return
	}
	field, wire = int(key>>3), int(key&7)
	switch wire {
	case 0:
		val, err = p.varint()
	case 1:
		if p.i+8 > len(p.b) {
			return 0, 0, 0, nil, errors.New("truncated fixed64")
		}
		data = p.b[p.i : p.i+8]
		p.i += 8
	case 2:
		var n uint64
		if n, err = p.varint(); err != nil {
			return
		}
		if n > uint64(len(p.b)-p.i) {
			return 0, 0, 0, nil, errors.New("truncated field")
		}
		data = p.b[p.i : p.i+int(n)]
		p.i += int(n)
	case 5:
		if p.i+4 > len(p.b) {
			return 0, 0, 0, nil, errors.New("truncated fixed32")
		}
		data = p.b[p.i : p.i+4]
		p.i += 4
	default:
		err = fmt.Errorf("wire type %d at byte %d", wire, p.i)
	}
	return
}

func parsePiece(b []byte) (piece, error) {
	pc := piece{typ: typeNormal}
	p := &pbuf{b: b}
	for p.i < len(b) {
		f, w, v, d, err := p.next()
		if err != nil {
			return pc, err
		}
		switch {
		case f == 1 && w == 2:
			pc.text = string(d)
		case f == 2 && w == 5:
			pc.score = math.Float32frombits(binary.LittleEndian.Uint32(d))
		case f == 3 && w == 0:
			pc.typ = int(v)
		}
	}
	return pc, nil
}

func parseModel(b []byte) (*spModel, error) {
	m := &spModel{modelType: 1, dummyPrefix: true, removeSpaces: true, escapeSpaces: true}
	p := &pbuf{b: b}
	for p.i < len(b) {
		f, w, _, d, err := p.next()
		if err != nil {
			return nil, err
		}
		if w != 2 {
			continue
		}
		switch f {
		case 1:
			pc, err := parsePiece(d)
			if err != nil {
				return nil, fmt.Errorf("piece %d: %v", len(m.pieces), err)
			}
			m.pieces = append(m.pieces, pc)
		case 2:
			q := &pbuf{b: d}
			for q.i < len(d) {
				f, w, v, _, err := q.next()
				if err != nil {
					return nil, fmt.Errorf("trainer_spec: %v", err)
				}
				if w == 0 && f == 3 {
					m.modelType = int(v)
				} else if w == 0 && f == 35 {
					m.byteFallback = v != 0
				}
			}
		case 3:
			q := &pbuf{b: d}
			for q.i < len(d) {
				f, w, v, d, err := q.next()
				if err != nil {
					return nil, fmt.Errorf("normalizer_spec: %v", err)
				}
				switch {
				case f == 1 && w == 2:
					m.normName = string(d)
				case f == 2 && w == 2:
					m.charsmap = len(d)
				case f == 3 && w == 0:
					m.dummyPrefix = v != 0
				case f == 4 && w == 0:
					m.removeSpaces = v != 0
				case f == 5 && w == 0:
					m.escapeSpaces = v != 0
				}
			}
		}
	}
	if len(m.pieces) == 0 {
		return nil, errors.New("no pieces: not a SentencePiece model")
	}
	return m, nil
}

// ---- tokenizer_config.json

type addedDecl struct {
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type tokConfig struct {
	AddBOS  *bool                `json:"add_bos_token"`
	AddEOS  *bool                `json:"add_eos_token"`
	BOS     json.RawMessage      `json:"bos_token"`
	EOS     json.RawMessage      `json:"eos_token"`
	Decoder map[string]addedDecl `json:"added_tokens_decoder"`
}

// tokenText is a tokenizer_config token, written as a string or as {"content": ...}.
func tokenText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var o struct {
		Content string `json:"content"`
	}
	json.Unmarshal(raw, &o)
	return o.Content
}

// ---- conversion

func byteAlphabet() [256]string {
	var out [256]string
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
			r = rune(256 + n)
			n++
		}
		out[b] = string(r)
	}
	return out
}

type added struct {
	id      int
	content string
	special bool
}

type result struct {
	keys   []string // id -> vocabulary key
	merges [][2]string
	added  []added
	bos    string
	bosID  int
	eos    string
	eosID  int
	addBOS bool
	addEOS bool
	shadow int // byte pieces a normal piece already covers
}

func convert(m *spModel, tc *tokConfig) (*result, error) {
	var bad []string
	if m.modelType != modelBPE {
		bad = append(bad, fmt.Sprintf("model_type %d is not BPE (2)", m.modelType))
	}
	if !m.byteFallback {
		bad = append(bad, "byte_fallback is off: some text has no pieces")
	}
	if m.charsmap > 0 {
		bad = append(bad, fmt.Sprintf("normalizer %q has a precompiled charsmap", m.normName))
	}
	if m.dummyPrefix {
		bad = append(bad, "normalizer adds a dummy prefix")
	}
	if m.removeSpaces {
		bad = append(bad, "normalizer removes extra whitespace")
	}
	if len(bad) > 0 {
		return nil, errors.New("not expressible as byte-level BPE: " + strings.Join(bad, "; "))
	}

	alpha := byteAlphabet()
	text := func(s string) string {
		if m.escapeSpaces {
			s = strings.ReplaceAll(s, "▁", " ")
		}
		var b strings.Builder
		for i := 0; i < len(s); i++ {
			b.WriteString(alpha[s[i]])
		}
		return b.String()
	}

	r := &result{keys: make([]string, len(m.pieces))}
	owner := map[string]int{}
	claim := func(id int, key string) error {
		if prev, ok := owner[key]; ok {
			return fmt.Errorf("pieces %d %q and %d %q are the same text", prev, m.pieces[prev].text, id, m.pieces[id].text)
		}
		owner[key] = id
		r.keys[id] = key
		return nil
	}
	// Text pieces first, so a byte piece gives way to a normal piece of the same byte.
	for id, pc := range m.pieces {
		if pc.typ == typeByte {
			continue
		}
		key := pc.text
		if pc.typ == typeNormal || pc.typ == typeUserDefined || pc.typ == typeUnused {
			key = text(pc.text)
		}
		if err := claim(id, key); err != nil {
			return nil, err
		}
	}
	for id, pc := range m.pieces {
		if pc.typ != typeByte {
			continue
		}
		if len(pc.text) != 6 || !strings.HasPrefix(pc.text, "<0x") || !strings.HasSuffix(pc.text, ">") {
			return nil, fmt.Errorf("byte piece %d is %q, not <0xNN>", id, pc.text)
		}
		v, err := strconv.ParseUint(pc.text[3:5], 16, 8)
		if err != nil {
			return nil, fmt.Errorf("byte piece %d is %q, not <0xNN>", id, pc.text)
		}
		key := alpha[v]
		if _, taken := owner[key]; taken {
			key = pc.text
			r.shadow++
		}
		if err := claim(id, key); err != nil {
			return nil, err
		}
	}
	for b := 0; b < 256; b++ {
		if _, ok := owner[alpha[b]]; !ok {
			return nil, fmt.Errorf("byte %#02x has no piece", b)
		}
	}

	// Added tokens: control and unknown pieces, user-defined pieces, and what the config declares.
	decl := map[int]addedDecl{}
	for k, d := range tc.Decoder {
		id, err := strconv.Atoi(k)
		if err != nil || id < 0 || id >= len(m.pieces) {
			return nil, fmt.Errorf("tokenizer_config: added token %q is not an id of this model (%d pieces)", k, len(m.pieces))
		}
		if d.Content != m.pieces[id].text {
			return nil, fmt.Errorf("tokenizer_config: id %d is %q, this model's piece %d is %q", id, d.Content, id, m.pieces[id].text)
		}
		decl[id] = d
	}
	for id, pc := range m.pieces {
		d, declared := decl[id]
		special := pc.typ == typeControl || pc.typ == typeUnknown
		if declared {
			special = d.Special
		} else if pc.typ != typeControl && pc.typ != typeUnknown && pc.typ != typeUserDefined {
			continue
		}
		content := pc.text
		if pc.typ == typeUserDefined && m.escapeSpaces {
			content = strings.ReplaceAll(content, "▁", " ")
		}
		r.added = append(r.added, added{id: id, content: content, special: special})
	}
	addedID := map[string]int{}
	for _, a := range r.added {
		addedID[a.content] = a.id
	}

	r.addBOS = tc.AddBOS != nil && *tc.AddBOS
	r.addEOS = tc.AddEOS != nil && *tc.AddEOS
	r.bos, r.eos = tokenText(tc.BOS), tokenText(tc.EOS)
	var ok bool
	if r.eosID, ok = addedID[r.eos]; !ok {
		return nil, fmt.Errorf("tokenizer_config: eos_token %q is not an added token of this model", r.eos)
	}
	if r.bosID, ok = addedID[r.bos]; r.addBOS && !ok {
		return nil, fmt.Errorf("tokenizer_config: add_bos_token is set and bos_token %q is not an added token of this model", r.bos)
	}

	// Merges: each piece BPE can reach, split into two vocabulary pieces.
	type cand struct {
		l, r  string
		score float32
	}
	var all []cand
	for id, pc := range m.pieces {
		if pc.typ != typeNormal && pc.typ != typeUserDefined {
			continue
		}
		w := []rune(r.keys[id])
		var local []cand
		for k := 1; k < len(w); k++ {
			l, rt := string(w[:k]), string(w[k:])
			if _, ok := owner[l]; !ok {
				continue
			}
			if _, ok := owner[rt]; !ok {
				continue
			}
			local = append(local, cand{l, rt, pc.score})
		}
		sort.SliceStable(local, func(i, j int) bool {
			if owner[local[i].l] != owner[local[j].l] {
				return owner[local[i].l] < owner[local[j].l]
			}
			return owner[local[i].r] < owner[local[j].r]
		})
		all = append(all, local...)
	}
	sort.SliceStable(all, func(i, j int) bool {
		a, b := all[i], all[j]
		if a.score != b.score {
			return a.score > b.score
		}
		if la, lb := utf8.RuneCountInString(a.l), utf8.RuneCountInString(b.l); la != lb {
			return la > lb
		}
		return utf8.RuneCountInString(a.r) > utf8.RuneCountInString(b.r)
	})
	r.merges = make([][2]string, len(all))
	for i, c := range all {
		r.merges[i] = [2]string{c.l, c.r}
	}
	return r, nil
}

// ---- tokenizer.json

func js(v any) string {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.Encode(v)
	return strings.TrimRight(b.String(), "\n")
}

func write(r *result, path string) error {
	var b bytes.Buffer
	b.WriteString("{\n  \"version\": \"1.0\",\n  \"truncation\": null,\n  \"padding\": null,\n  \"added_tokens\": [")
	for i, a := range r.added {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "\n    {\"id\": %d, \"content\": %s, \"single_word\": false, \"lstrip\": false, \"rstrip\": false, \"normalized\": false, \"special\": %t}", a.id, js(a.content), a.special)
	}
	b.WriteString("\n  ],\n  \"normalizer\": null,\n")
	b.WriteString(`  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": ` + js(llama3Split) + `}, "behavior": "Isolated", "invert": false}, {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},` + "\n")

	type tmpl = map[string]any
	special := func(t string, typ int) tmpl { return tmpl{"SpecialToken": tmpl{"id": t, "type_id": typ}} }
	seq := func(s string, typ int) tmpl { return tmpl{"Sequence": tmpl{"id": s, "type_id": typ}} }
	var single, pair []tmpl
	specials := tmpl{}
	if r.addBOS {
		single = append(single, special(r.bos, 0))
		specials[r.bos] = tmpl{"id": r.bos, "ids": []int{r.bosID}, "tokens": []string{r.bos}}
	}
	single = append(single, seq("A", 0))
	if r.addEOS {
		single = append(single, special(r.eos, 0))
		specials[r.eos] = tmpl{"id": r.eos, "ids": []int{r.eosID}, "tokens": []string{r.eos}}
	}
	pair = append(pair, single...)
	if r.addBOS {
		pair = append(pair, special(r.bos, 1))
	}
	pair = append(pair, seq("B", 1))
	if r.addEOS {
		pair = append(pair, special(r.eos, 1))
	}
	b.WriteString(`  "post_processor": ` + js(tmpl{"type": "TemplateProcessing", "single": single, "pair": pair, "special_tokens": specials}) + ",\n")
	b.WriteString(`  "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},` + "\n")
	b.WriteString("  \"model\": {\n    \"type\": \"BPE\",\n    \"dropout\": null,\n    \"unk_token\": null,\n    \"continuing_subword_prefix\": null,\n    \"end_of_word_suffix\": null,\n    \"fuse_unk\": false,\n    \"byte_fallback\": false,\n    \"ignore_merges\": false,\n    \"vocab\": {")
	for id, k := range r.keys {
		if id > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "\n      %s: %d", js(k), id)
	}
	b.WriteString("\n    },\n    \"merges\": [")
	for i, mg := range r.merges {
		if i > 0 {
			b.WriteString(",")
		}
		fmt.Fprintf(&b, "\n      [%s, %s]", js(mg[0]), js(mg[1]))
	}
	b.WriteString("\n    ]\n  }\n}\n")
	return os.WriteFile(path, b.Bytes(), 0o644)
}

func run(modelPath, configPath, outPath string) error {
	if modelPath == "" || configPath == "" || outPath == "" {
		return errors.New("-model, -config and -out are all required")
	}
	raw, err := os.ReadFile(modelPath)
	if err != nil {
		return err
	}
	m, err := parseModel(raw)
	if err != nil {
		return fmt.Errorf("%s: %v", modelPath, err)
	}
	cb, err := os.ReadFile(configPath)
	if err != nil {
		return err
	}
	var tc tokConfig
	if err := json.Unmarshal(cb, &tc); err != nil {
		return fmt.Errorf("%s: %v", configPath, err)
	}
	r, err := convert(m, &tc)
	if err != nil {
		return fmt.Errorf("%s: %v", modelPath, err)
	}
	if err := write(r, outPath); err != nil {
		return err
	}
	fmt.Printf("%s: %d ids, %d merges, %d added tokens (%d special), bos %q first: %t, eos %q\n",
		outPath, len(r.keys), len(r.merges), len(r.added), countSpecial(r.added), r.bos, r.addBOS, r.eos)
	if r.shadow > 0 {
		fmt.Printf("  %d byte pieces keep their <0xNN> name: a normal piece is already that byte\n", r.shadow)
	}
	return nil
}

func countSpecial(a []added) int {
	n := 0
	for _, x := range a {
		if x.special {
			n++
		}
	}
	return n
}

func main() {
	model := flag.String("model", "", "SentencePiece tokenizer.model")
	config := flag.String("config", "", "the same model's tokenizer_config.json")
	out := flag.String("out", "", "tokenizer.json to write")
	flag.Parse()
	if err := run(*model, *config, *out); err != nil {
		fmt.Fprintln(os.Stderr, "spm2json:", err)
		os.Exit(1)
	}
}

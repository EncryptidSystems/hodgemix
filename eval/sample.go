package eval

// Session keeps one model and its tokenizer open across many prompts, and samples instead of taking
// the argmax. A dialog on a session keeps its KV cache, so a follow-up turn feeds only its own tokens.
// Generate stays the greedy reference; this is for harvesting several different replies from one
// prompt (Moonshine's vocabulary probe).

import (
	"math"
	"math/rand"
	"sort"
	"strings"
)

// Session is an opened model and tokenizer.
type Session struct {
	m  *model
	tk *tok
}

// Open loads modelPath with tokPath (or the tokenizer set by SetFiles).
func Open(modelPath, tokPath string) (*Session, error) {
	m, tk, err := open(modelPath, tokPath)
	if err != nil {
		return nil, err
	}
	return &Session{m: m, tk: tk}, nil
}

// OpenTok is a session with only the tokenizer: Encode and Decode work, the model calls do not.
func OpenTok(tokPath, cfgPath string) (*Session, error) {
	tk, err := loadTok(tokPath, cfgPath)
	if err != nil {
		return nil, err
	}
	return &Session{tk: tk}, nil
}

// Rows is the embedding's row count.
func (s *Session) Rows() int { return s.m.nEmb }

// EOT is the turn end token text, empty when the tokenizer config set none.
func (s *Session) EOT() string { return s.tk.eot }

// Encode is the ids of text: the chat encoder (added tokens kept whole) when chat, with the
// post-processor's BOS when bos.
func (s *Session) Encode(text string, chat, bos bool) []int {
	var ids []int
	if bos {
		ids = append(ids, s.tk.bos...)
	}
	if chat {
		return append(ids, s.tk.chatPieces(text)...)
	}
	return append(ids, s.tk.pieces(text)...)
}

// Segments is the BPE ids of each non-whitespace pre-token, in order. A whitespace
// pre-token is not an anchor. The ids are the same merges pieces uses inside one pre-token.
func (s *Session) Segments(text string) [][]int {
	var out [][]int
	for _, w := range pre.FindAllString(text, -1) {
		if strings.TrimSpace(w) == "" {
			continue
		}
		out = append(out, s.tk.bpe(w))
	}
	return out
}

// Decode is the text of ids.
func (s *Session) Decode(ids []int) string {
	var b []byte
	for _, id := range ids {
		b = append(b, s.tk.decode(id)...)
	}
	return string(b)
}

// Logits runs ids from position 0 and returns the logits after each one.
func (s *Session) Logits(ids []int) [][]float32 {
	q := &seq{}
	out := make([][]float32, len(ids))
	for i, id := range ids {
		out[i] = s.m.step(q, id, i)
	}
	return out
}

// Dialog is one conversation in this tokenizer's chat format (Llama 3 headers, Qwen
// im_start turns, or the 2B4T "Role: content" line).
type Dialog struct {
	s     *Session
	q     *seq
	pos   int
	lg    []float32
	Fed   []int // every id fed so far, prompt and replies
	sys   string
	first bool
}

// NewDialog starts a conversation with system as its system message (none when empty).
func (s *Session) NewDialog(system string) *Dialog {
	c := &Dialog{s: s, q: &seq{}, sys: system, first: true}
	c.feed(s.tk.bos)
	if pre := s.tk.systemPrefix(system); pre != "" {
		c.feed(s.tk.chatPieces(pre))
	}
	return c
}

// feed runs ids through the layers. Only the last one goes through the LM head: the logits of the
// others are never read.
func (c *Dialog) feed(ids []int) {
	for i, id := range ids {
		if i == len(ids)-1 {
			c.lg = c.s.m.step(c.q, id, c.pos)
		} else {
			c.s.m.feed(c.q, id, c.pos)
		}
		c.Fed = append(c.Fed, id)
		c.pos++
	}
}

// Pos is the next position.
func (c *Dialog) Pos() int { return c.pos }

// User feeds a user turn and the assistant's prompt.
func (c *Dialog) User(text string) {
	sys := ""
	if c.first && c.s.tk.systemPrefix(c.sys) == "" {
		sys = c.sys
	}
	c.first = false
	c.feed(c.s.tk.chatPieces(c.s.tk.promptTurn(sys, text)))
}

// Assistant feeds a reply the model did not sample (a worked example) and the turn end.
func (c *Dialog) Assistant(text string) {
	c.feed(c.s.tk.chatPieces(text + c.s.tk.closeTurn()))
}

// Fork is a copy of the dialog that shares its cache so far. What either one feeds next stays its own:
// the cache slices are capped at their length, so an append copies instead of writing into the other.
func (c *Dialog) Fork() *Dialog {
	q := &seq{phi: append([]float64(nil), c.q.phi...)}
	if c.q.kc != nil {
		q.kc, q.vc = make([][][]float32, len(c.q.kc)), make([][][]float32, len(c.q.vc))
		for li := range c.q.kc {
			k, v := c.q.kc[li], c.q.vc[li]
			q.kc[li], q.vc[li] = k[:len(k):len(k)], v[:len(v):len(v)]
		}
	}
	return &Dialog{s: c.s, q: q, pos: c.pos, lg: c.lg, Fed: append([]int(nil), c.Fed...), sys: c.sys, first: c.first}
}

// Score is the log-probability that the assistant replies text and then ends the turn, and the
// number of tokens that covers. It runs on a fork, so the dialog itself is unchanged.
func (c *Dialog) Score(text string) (float64, int) {
	ids := c.s.tk.chatPieces(text + c.s.tk.closeTurn())
	f := c.Fork()
	lp := 0.0
	for i, id := range ids {
		lp += logsoft(f.lg, id)
		if i < len(ids)-1 {
			f.feed([]int{id})
		}
	}
	return lp, len(ids)
}

// Sampling is the next-token rule: temperature, then nucleus (top-p). Temp 0 is the argmax.
type Sampling struct {
	Temp float64
	TopP float64
	Rng  *rand.Rand
}

// Reply samples up to n tokens, stopping at the turn end or when stop (given the reply's ids so far)
// says so. The turn end is fed, so a following User continues the same cache. It returns the reply's
// ids without the turn end, and whether the model ended the turn itself.
func (c *Dialog) Reply(n int, sp Sampling, stop func(ids []int) bool) ([]int, bool) {
	var ids []int
	for j := 0; j < n; j++ {
		nx := sample(c.lg, sp)
		if c.s.tk.ended(nx) {
			c.feed([]int{nx})
			if tail := c.s.tk.afterStop(); tail != "" {
				c.feed(c.s.tk.chatPieces(tail))
			}
			return ids, true
		}
		ids = append(ids, nx)
		c.feed([]int{nx})
		if stop != nil && stop(ids) {
			break
		}
	}
	if end := c.s.tk.closeTurn(); end != "" {
		c.feed(c.s.tk.chatPieces(end))
	}
	return ids, false
}

func sample(lg []float32, sp Sampling) int {
	if sp.Temp <= 0 || sp.Rng == nil {
		return argmax(lg)
	}
	mx := lg[argmax(lg)]
	idx := make([]int, len(lg))
	p := make([]float64, len(lg))
	sum := 0.0
	for i, v := range lg {
		idx[i] = i
		p[i] = math.Exp(float64(v-mx) / sp.Temp)
		sum += p[i]
	}
	sort.Slice(idx, func(a, b int) bool { return p[idx[a]] > p[idx[b]] })
	top := sp.TopP
	if top <= 0 || top > 1 {
		top = 1
	}
	keep, mass := 0, 0.0
	for keep < len(idx) {
		mass += p[idx[keep]] / sum
		keep++
		if mass >= top {
			break
		}
	}
	r := sp.Rng.Float64() * mass
	for _, i := range idx[:keep] {
		r -= p[i] / sum
		if r <= 0 {
			return i
		}
	}
	return idx[keep-1]
}

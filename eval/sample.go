package eval

// Session keeps one model and its tokenizer open across many prompts, and samples instead of taking
// the argmax. A dialog on a session keeps its KV cache, so a follow-up turn feeds only its own tokens.
// Generate stays the greedy reference; this is for harvesting several different replies from one
// prompt (Moonshine's vocabulary probe).

import (
	"math"
	"math/rand"
	"sort"
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

// Dialog is one conversation in the runtime's "Role: content<eot>" format.
type Dialog struct {
	s   *Session
	q   *seq
	pos int
	lg  []float32
	Fed []int // every id fed so far, prompt and replies
}

// NewDialog starts a conversation with system as its system message (none when empty).
func (s *Session) NewDialog(system string) *Dialog {
	c := &Dialog{s: s, q: &seq{}}
	c.feed(s.tk.bos)
	if system != "" {
		c.feed(s.tk.chatPieces("System: " + system + s.tk.eot))
	}
	return c
}

func (c *Dialog) feed(ids []int) {
	for _, id := range ids {
		c.lg = c.s.m.step(c.q, id, c.pos)
		c.Fed = append(c.Fed, id)
		c.pos++
	}
}

// Pos is the next position.
func (c *Dialog) Pos() int { return c.pos }

// User feeds a user turn and the assistant's prompt.
func (c *Dialog) User(text string) {
	c.feed(c.s.tk.chatPieces("User: " + text + c.s.tk.eot + "Assistant: "))
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
	eot, hasEot := c.s.tk.special[c.s.tk.eot]
	var ids []int
	for j := 0; j < n; j++ {
		nx := sample(c.lg, sp)
		if hasEot && nx == eot {
			c.feed([]int{eot})
			return ids, true
		}
		ids = append(ids, nx)
		c.feed([]int{nx})
		if stop != nil && stop(ids) {
			break
		}
	}
	if hasEot {
		c.feed([]int{eot})
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

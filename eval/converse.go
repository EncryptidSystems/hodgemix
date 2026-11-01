package eval

import (
	"fmt"
	"strings"
)

// Suite is the conversation suite: each entry is one conversation's user turns, sent in the page's
// chat format after its system prompt (ChatSystem). It covers follow-ups and memory across turns, the
// page's tool calls, code, Chinese, arithmetic, summarising and translating.
var Suite = [][]string{
	{"What is the capital of France?", "And which river runs through it?"},
	{"My name is Sam and I live in Vermont.", "What's my name, and where do I live?"},
	{"Write a haiku about winter."},
	{"Explain what a black hole is in two sentences."},
	{"Search the web for today's weather in Boston."},
	{"Save a note that says buy milk and eggs."},
	{"Write a Python function that returns the factorial of n."},
	{"请用中文简单介绍一下长城。"},
	{"What is 12 times 13?"},
	{"Summarize in one sentence: the meeting covered budget cuts, a new hire for the design team, and moving the office in June."},
	{"Translate 'good morning, how are you?' into Spanish."},
	{"Give me three tips for sleeping better.", "Which of those is the most important?"},
}

// Turn is one exchange of a conversation.
type Turn struct {
	User  string `json:"user"`
	Reply string `json:"reply"`
}

// Converse runs the suite greedily (up to n tokens a reply, stopping at the turn end) and returns each
// conversation's turns. The system prompt's cache is computed once and shared.
func Converse(modelPath, tokPath string, n int) ([][]Turn, error) {
	m, tk, err := open(modelPath, tokPath)
	if err != nil {
		return nil, err
	}
	if tk.eot == "" && tk.format() == fmtPlain {
		return nil, fmt.Errorf("%s: no turn-end token (set the tokenizer_config whose eos_token ends a turn)", tokPath)
	}
	sysText := tk.chatSystem()
	pre := tk.systemPrefix(sysText)
	var sys []int
	if pre == "" {
		sys = append([]int(nil), tk.bos...)
	} else {
		sys = tk.encodeChat(pre)
	}
	root := &seq{}
	for i, id := range sys {
		m.feed(root, id, i)
	}
	var out [][]Turn
	for _, conv := range Suite {
		s := &seq{kc: make([][][]float32, m.L), vc: make([][][]float32, m.L)}
		for l := 0; l < m.L; l++ {
			s.kc[l] = append([][]float32(nil), root.kc[l]...)
			s.vc[l] = append([][]float32(nil), root.vc[l]...)
		}
		pos := len(sys)
		var turns []Turn
		for i, u := range conv {
			turnSys := ""
			if i == 0 && pre == "" {
				turnSys = sysText
			}
			ids := tk.chatPieces(tk.promptTurn(turnSys, u)) // no second BOS
			var lg []float32
			for k, id := range ids {
				if k < len(ids)-1 {
					m.feed(s, id, pos)
				} else {
					lg = m.step(s, id, pos)
				}
				pos++
			}
			var sb strings.Builder
			ended := false
			for j := 0; j < n && !ended; j++ {
				nx := argmax(lg)
				if ended = tk.ended(nx); ended {
					m.feed(s, nx, pos)
					pos++
					if tail := tk.afterStop(); tail != "" {
						for _, id := range tk.chatPieces(tail) {
							m.feed(s, id, pos)
							pos++
						}
					}
					break
				}
				sb.WriteString(tk.decode(nx))
				lg = m.step(s, nx, pos)
				pos++
			}
			if !ended { // a reply cut at n tokens still closes its turn
				for _, id := range tk.chatPieces(tk.closeTurn()) {
					m.feed(s, id, pos)
					pos++
				}
			}
			turns = append(turns, Turn{u, strings.TrimSpace(sb.String())})
		}
		out = append(out, turns)
	}
	return out, nil
}

package main

import (
	"sort"
	"strings"
	"unicode"
)

// functionWords are the words a count skips.
var functionWords = map[string]bool{}

func init() {
	for _, w := range strings.Fields(functionList) {
		functionWords[w] = true
	}
}

const functionList = "" +
	"a an the and or but if then else when where what which who whom whose why how " +
	"this that these those he she it they we you i me him her them us my your his its our their " +
	"mine yours hers ours theirs of to in on at by for with from into over under about as than " +
	"too very not no nor so just also be is am are was were been being do does did done " +
	"have has had having will would shall should can could may might must " +
	"said says say saying there here up down out off again further once only own same other such " +
	"both each few more most some any all between through during before after above below " +
	"because while until although though whether per via onto upon within without among across " +
	"don't doesnt didn't can't won't isn't aren't wasn't weren't hasn't haven't hadn't " +
	"it's that's there's he's she's they're we're you're i'm i've i'd i'll " +
	"mr mrs ms dr sir madam "

// DescriptiveWord is one of the words the model is allowed to see.
type DescriptiveWord struct {
	Word  string
	Count int
}

// TopDescriptive returns up to n words, most frequent first.
// Function words are left out. Ties break on the lowercase word.
func TopDescriptive(text string, n int) []DescriptiveWord {
	if n < 1 {
		return nil
	}
	type tally struct {
		n       int
		surface map[string]int
	}
	got := map[string]*tally{}
	for _, raw := range scanWords(text) {
		key := strings.ToLower(raw)
		key = strings.Trim(key, "'")
		key = strings.TrimSuffix(key, "'s")
		if key == "" || functionWords[key] || len([]rune(key)) < 2 {
			continue
		}
		t := got[key]
		if t == nil {
			t = &tally{surface: map[string]int{}}
			got[key] = t
		}
		t.n++
		t.surface[raw]++
	}
	out := make([]DescriptiveWord, 0, len(got))
	for key, t := range got {
		out = append(out, DescriptiveWord{Word: pickSurface(key, t.surface), Count: t.n})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return strings.ToLower(out[i].Word) < strings.ToLower(out[j].Word)
	})
	if len(out) > n {
		out = out[:n]
	}
	return out
}

func pickSurface(key string, surface map[string]int) string {
	best, n := key, -1
	for s, c := range surface {
		if c > n || (c == n && s < best) {
			best, n = s, c
		}
	}
	return best
}

func scanWords(text string) []string {
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		s := strings.Trim(b.String(), "'")
		b.Reset()
		if s != "" {
			out = append(out, s)
		}
	}
	for _, r := range text {
		if unicode.IsLetter(r) || r == '\'' {
			b.WriteRune(r)
			continue
		}
		flush()
	}
	flush()
	return out
}

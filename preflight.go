package main

// preflight checks, before anything is read in bulk or written, everything a run needs. Whatever would
// otherwise fail later, or quietly make a different model, stops the run here with every reason listed.
// No paper equation.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"hodgemix/eval"
)

// tokCheck checks that tok is a byte-level BPE tokenizer whose ids all have a row in st's embedding;
// cfg, when set, must name an eos_token that is one of its added tokens (the turn end of a chat).
func tokCheck(what, tok, cfg string, st *ST, needCfg bool) []string {
	var bad []string
	if tok == "" {
		return []string{what + ": no tokenizer set (run file: tokenizer)"}
	}
	t, err := readTok(tok)
	if err != nil {
		return []string{fmt.Sprintf("%s: %v", what, err)}
	}
	if !st.Has("model.embed_tokens.weight") {
		return []string{what + ": has no model.embed_tokens.weight"}
	}
	if n, rows := t.ids(), st.LogicalShape("model.embed_tokens.weight")[0]; n > rows {
		bad = append(bad, fmt.Sprintf("%s: %s has ids up to %d but the model embeds %d rows: it is another model's tokenizer", what, tok, n-1, rows))
	}
	if cfg == "" {
		if needCfg {
			bad = append(bad, what+": no tokenizer_config set (run file: tokenizer_config; its eos_token ends a chat turn)")
		}
		return bad
	}
	b, err := os.ReadFile(cfg)
	if err != nil {
		return append(bad, fmt.Sprintf("%s: %v", what, err))
	}
	var tc struct {
		EOS json.RawMessage `json:"eos_token"`
	}
	if err := json.Unmarshal(b, &tc); err != nil {
		return append(bad, fmt.Sprintf("%s: %s: %v", what, cfg, err))
	}
	var eos string
	if json.Unmarshal(tc.EOS, &eos) != nil {
		var o struct {
			Content string `json:"content"`
		}
		json.Unmarshal(tc.EOS, &o)
		eos = o.Content
	}
	found := false
	for _, a := range t.added {
		found = found || a.Content == eos
	}
	if !found {
		bad = append(bad, fmt.Sprintf("%s: %s's eos_token %q is not an added token of %s: they are not one model's", what, cfg, eos, tok))
	}
	return bad
}

// preflight returns the models' topologies; any problem stops the run.
func preflight(c runCfg, basePath string, all []*ST, baseOnly bool) []modelMeta {
	metas, bad := preflightCheck(c, basePath, all, baseOnly)
	if len(bad) > 0 {
		die("preflight: the run would fail or make a broken model:\n  %s", strings.Join(bad, "\n  "))
	}
	logf("PREFLIGHT  ok: %d models' configs, tokenizers and tensors; the output config runs in Champ", len(all))
	return metas
}

// preflightCheck is every reason the run would fail or make a broken model.
func preflightCheck(c runCfg, basePath string, all []*ST, baseOnly bool) ([]modelMeta, []string) {
	var bad []string
	add := func(f string, a ...any) { bad = append(bad, fmt.Sprintf(f, a...)) }
	name := func(i int) string {
		if i == 0 {
			return "base (" + all[0].Path + ")"
		}
		return fmt.Sprintf("source %d (%s)", i, all[i].Path)
	}
	base := all[0]
	metas := make([]modelMeta, len(all))
	for i, st := range all {
		if eval.FilesOf(st.Path).Config == "" {
			add("%s: no config set (run file: config)", name(i))
			continue
		}
		m, err := metaOf(st)
		if err != nil {
			add("%s: %v", name(i), err)
		}
		metas[i] = m
	}
	bf := eval.FilesOf(basePath)
	bad = append(bad, tokCheck(name(0), bf.Tokenizer, bf.TokenizerConfig, base, true)...)
	if c.evalTok != "" && c.evalTok != bf.Tokenizer {
		bad = append(bad, tokCheck("--eval-tokenizer", c.evalTok, "", base, false)...)
	}
	if VocabMode == "union" && baseOnly {
		for i, st := range all[1:] {
			if st.Has("model.embed_tokens.weight") { // a shard without the embedding adds no tokens
				f := eval.FilesOf(st.Path)
				bad = append(bad, tokCheck(name(i+1), f.Tokenizer, f.TokenizerConfig, st, false)...)
			}
		}
	}
	// The output's head is the tied embedding: a base with its own lm_head would lose it.
	if base.Has("lm_head.weight") {
		a, e1 := base.Raw("lm_head.weight")
		b, e2 := base.Raw("model.embed_tokens.weight")
		if e1 != nil || e2 != nil || !bytes.Equal(a, b) {
			add("%s: its lm_head.weight is its own (untied); Champ runs a tied LM head", name(0))
		}
	}
	if c.rotate {
		add("--rotate on unties the LM head (Eq.24e); Champ runs a tied LM head")
	}
	// Every tensor of every model is mixed, folded, or one hodgemix keeps from the base by design
	// (norms, embeddings); anything else would be dropped.
	idxRoles := map[string]bool{}
	for _, r := range c.roles {
		idxRoles[r] = true
	}
	for i, st := range all {
		drop := map[string][]string{}
		for _, k := range st.Keys {
			sh := st.LogicalShape(k)
			r := roleOf(k)
			why := ""
			switch {
			case strings.HasSuffix(k, ".bias"):
				why = "biases (BitLinear and attention.wgsl have none)"
			case i > 0 && !AttnOn && (strings.Contains(k, "q_norm") || strings.Contains(k, "k_norm")):
				why = "per-head q/k norms (--attn off does not fold them into q/k)"
			case len(sh) == 2 && r == "qkv":
				why = "a fused qkv projection (Champ and the target run separate q/k/v)"
			case i > 0 && len(sh) == 2 && r == "" && k != "model.embed_tokens.weight" && k != "lm_head.weight" &&
				!strings.HasPrefix(k, "bnzmix."):
				why = "2-D tensors with no BitLinear role"
			}
			if why != "" {
				drop[why] = append(drop[why], k)
			}
		}
		for why, ks := range drop {
			add("%s: %d tensors would be dropped: %s, e.g. %s", name(i), len(ks), why, ks[0])
		}
	}
	// The config Champ will load: the base's (or --target-config) with the output's format keys.
	if metas[0].Heads > 0 {
		var cb []byte
		var err error
		if c.targetCfg != "" {
			cb, err = os.ReadFile(c.targetCfg)
		} else {
			g := metas[0]
			if g.Theta == 0 {
				g.Theta = SrcRopeTheta
			}
			cb, _, err = configFor(g, "", basePath, 0)
		}
		cfg := map[string]any{}
		if err == nil {
			err = json.Unmarshal(cb, &cfg)
		}
		if err != nil {
			add("output config.json: %v", err)
		} else {
			g := metas[0]
			for _, b := range champRuntime(cfg, g.D, g.F, g.Heads, g.KV, g.HD) {
				add("output %s", b)
			}
		}
	}
	return metas, bad
}

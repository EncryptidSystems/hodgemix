package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"hodgemix/eval"
)

type tinyModel struct {
	D, F, H, KV, HD, L, V int
	Qwen                  bool // q_norm / k_norm, no attn_sub_norm / ffn_sub_norm (Qwen3 layout)
}

// writeTiny writes a BitNet-layout float model (BF16) with config.json into dir, set as its config as a
// run file would.
func writeTiny(t *testing.T, dir string, m tinyModel, seed int64) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(seed))
	p := filepath.Join(dir, "model.safetensors")
	W, err := NewSTW(p, nil)
	if err != nil {
		t.Fatal(err)
	}
	mat := func(name string, r, c int) {
		a := make([]float32, r*c)
		for i := range a {
			a[i] = float32(rng.NormFloat64() * 0.05)
		}
		W.AddBF16(name, []int{r, c}, a)
	}
	vec := func(name string, n int) {
		a := make([]float32, n)
		for i := range a {
			a[i] = float32(1 + 0.1*rng.NormFloat64())
		}
		W.AddBF16(name, []int{n}, a)
	}
	if m.Qwen { // a stored tied head, as Qwen3-0.6B ships it
		e := make([]float32, m.V*m.D)
		for i := range e {
			e[i] = float32(rng.NormFloat64() * 0.05)
		}
		W.AddBF16("model.embed_tokens.weight", []int{m.V, m.D}, e)
		W.AddBF16("lm_head.weight", []int{m.V, m.D}, e)
	} else {
		mat("model.embed_tokens.weight", m.V, m.D)
	}
	vec("model.norm.weight", m.D)
	for l := 0; l < m.L; l++ {
		p := "model.layers." + itoa(l) + "."
		mat(p+"self_attn.q_proj.weight", m.H*m.HD, m.D)
		mat(p+"self_attn.k_proj.weight", m.KV*m.HD, m.D)
		mat(p+"self_attn.v_proj.weight", m.KV*m.HD, m.D)
		mat(p+"self_attn.o_proj.weight", m.D, m.H*m.HD)
		mat(p+"mlp.gate_proj.weight", m.F, m.D)
		mat(p+"mlp.up_proj.weight", m.F, m.D)
		mat(p+"mlp.down_proj.weight", m.D, m.F)
		vec(p+"input_layernorm.weight", m.D)
		vec(p+"post_attention_layernorm.weight", m.D)
		if m.Qwen {
			vec(p+"self_attn.q_norm.weight", m.HD)
			vec(p+"self_attn.k_norm.weight", m.HD)
		} else {
			vec(p+"self_attn.attn_sub_norm.weight", m.H*m.HD)
			vec(p+"mlp.ffn_sub_norm.weight", m.F)
		}
	}
	if err := W.Close(nil); err != nil {
		t.Fatal(err)
	}
	cfg := map[string]any{"model_type": "bitnet", "hidden_size": m.D, "intermediate_size": m.F, "num_attention_heads": m.H,
		"num_key_value_heads": m.KV, "head_dim": m.HD, "num_hidden_layers": m.L, "vocab_size": m.V, "tie_word_embeddings": true,
		"max_position_embeddings": 4096}
	b, _ := json.Marshal(cfg)
	_ = os.WriteFile(filepath.Join(dir, "config.json"), b, 0o644)
	setFile(p, func(f *eval.ModelFiles) { f.Config = filepath.Join(dir, "config.json") })
	writeTok(t, dir, 0, 0) // the 256 byte tokens and the turn end: V must be at least 257
	return p
}

// setFile changes one of model's files as a run file would set it.
func setFile(model string, set func(*eval.ModelFiles)) {
	f := eval.FilesOf(model)
	set(&f)
	eval.SetFiles(model, f)
}

func tinyCfg(topology string) runCfg {
	return runCfg{roles: []string{"q", "k", "v", "o", "gate", "up", "down", "qkv"}, baseW: 1, residual: true, FD: "BF16", topology: topology,
		sp: SynthParams{K: KParams{GTheta: math.Ldexp(1, -GThetaShift), Weak: math.Ldexp(1, -(GOmegaShift - GThetaShift)), WLock: WLock,
			NLost: NLost, RTol: RefineTol}, Lock: "dtft", D: DParams{SeqAxis: "auto", OmegaUpdate: true}}}
}

func TestPlanTargetSizes(t *testing.T) {
	bitnet := modelMeta{D: 2560, F: 6912, Heads: 20, KV: 5, HD: 128, Layers: 30}
	qwen3 := modelMeta{D: 1024, F: 3072, Heads: 16, KV: 8, HD: 128, Layers: 28}
	g, _ := planTarget("married", []modelMeta{bitnet, qwen3})
	if g.D != 5120 || g.F != 27648 || g.KV != 40 || g.Heads != 160 || g.Layers != 30 {
		t.Fatalf("married %+v", g)
	}
	g, _ = planTarget("max", []modelMeta{bitnet, qwen3})
	if g.D != 2560 || g.F != 6912 || g.KV != 8 || g.Heads != 32 || g.Layers != 30 {
		t.Fatalf("max %+v", g)
	}
}

// Two different tiny topologies married into one: every tensor gets the married shape and the
// config says so.
func TestMarriedTopologyRun(t *testing.T) {
	dir := tempDir(t)
	a := writeTiny(t, filepath.Join(dir, "a"), tinyModel{D: 16, F: 24, H: 4, KV: 2, HD: 4, L: 2, V: 260}, 1)
	b := writeTiny(t, filepath.Join(dir, "b"), tinyModel{D: 12, F: 18, H: 3, KV: 1, HD: 4, L: 3, V: 260}, 2)
	out := filepath.Join(dir, "out", "mix.bnz")
	run(tinyCfg("married"), a, []string{b}, out, "", "", true)
	st, err := OpenST(out + ".ternary.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	// lcm: D 48, F 72, kv 2, G = lcm(2,3) = 6 -> 12 heads of 4 = 48; layers max(2,3) = 3
	want := map[string][]int{
		"model.layers.2.self_attn.q_proj.weight":        {48, 48},
		"model.layers.2.self_attn.k_proj.weight":        {8, 48},
		"model.layers.0.self_attn.o_proj.weight":        {48, 48},
		"model.layers.1.mlp.gate_proj.weight":           {72, 48},
		"model.layers.1.mlp.down_proj.weight":           {48, 72},
		"model.embed_tokens.weight":                     {260, 48},
		"model.norm.weight":                             {48},
		"model.layers.2.self_attn.attn_sub_norm.weight": {48},
		"model.layers.2.mlp.ffn_sub_norm.weight":        {72},
	}
	for name, sh := range want {
		if !st.Has(name) {
			t.Fatalf("missing %s", name)
		}
		if got := st.LogicalShape(name); !eqShape(got, sh) {
			t.Fatalf("%s: %v, want %v", name, got, sh)
		}
	}
	cb, err := os.ReadFile(filepath.Join(dir, "out", "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg map[string]float64
	_ = json.Unmarshal(cb, &cfg)
	if cfg["hidden_size"] != 48 || cfg["intermediate_size"] != 72 || cfg["num_attention_heads"] != 12 || cfg["num_key_value_heads"] != 2 || cfg["num_hidden_layers"] != 3 {
		t.Fatalf("config %v", cfg)
	}
}

// A model married with itself in a married topology is the model with its residual stream in one
// married order: every D-reading column and D-writing row are permuted by the same permutation.
func TestMarriedTopologySelfIsOnePermutation(t *testing.T) {
	dir := tempDir(t)
	m := tinyModel{D: 16, F: 24, H: 4, KV: 2, HD: 4, L: 2, V: 260}
	a := writeTiny(t, filepath.Join(dir, "a"), m, 3)
	out := filepath.Join(dir, "out", "self.bnz")
	run(tinyCfg("married"), a, []string{a}, out, "", "", true)
	in, _ := OpenST(a)
	mo, err := OpenST(out + ".ternary.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	codes := func(st *ST, name string) ([]int8, []int) {
		if st.isPacked(name) { // limb 0's codes (LoadDense adds any limbs)
			b, err := st.Raw(name)
			if err != nil {
				t.Fatal(err)
			}
			sh := st.LogicalShape(name)
			return unpackTrits(b, sh[0], sh[1]), sh
		}
		W, sh, err := st.LoadDense(name)
		if err != nil {
			t.Fatal(err)
		}
		s := absmean(W)
		q := make([]int8, len(W))
		for i, v := range W {
			q[i] = int8(ternQ(float64(v), s))
		}
		return q, sh
	}
	// The D permutation read off the embedding (every target D column is one base D column).
	E0, _ := in.F32("model.embed_tokens.weight")
	E1, _ := mo.F32("model.embed_tokens.weight")
	perm := make([]int, m.D)
	for i := range perm {
		perm[i] = -1
		for j := 0; j < m.D && perm[i] < 0; j++ {
			same := true
			for v := 0; v < m.V && same; v++ {
				same = math.Abs(float64(E1[v*m.D+i]-E0[v*m.D+j])) < 1e-6
			}
			if same {
				perm[i] = j
			}
		}
		if perm[i] < 0 {
			t.Fatalf("target D channel %d is not a base D channel", i)
		}
	}
	// q_proj columns and o_proj rows follow the same D permutation.
	q0, _ := codes(in, "model.layers.0.self_attn.q_proj.weight")
	q1, sh := codes(mo, "model.layers.0.self_attn.q_proj.weight")
	o0, _ := codes(in, "model.layers.0.self_attn.o_proj.weight")
	o1, _ := codes(mo, "model.layers.0.self_attn.o_proj.weight")
	R := sh[0]
	colsMatch := func(a, b []int8, rows, cols, i, j int, rowPerm func(int) int) bool {
		for r := 0; r < rows; r++ {
			if a[rowPerm(r)*cols+j] != b[r*cols+i] {
				return false
			}
		}
		return true
	}
	// q rows (Q axis) are permuted too; find each target q row's base row by matching codes on the D-permuted columns.
	qrow := make([]int, R)
	for r := 0; r < R; r++ {
		qrow[r] = -1
		for s := 0; s < R && qrow[r] < 0; s++ {
			ok := true
			for i := 0; i < m.D && ok; i++ {
				ok = q1[r*m.D+i] == q0[s*m.D+perm[i]]
			}
			if ok {
				qrow[r] = s
			}
		}
		if qrow[r] < 0 {
			t.Fatalf("target q row %d is no base q row under the D permutation", r)
		}
	}
	for i := 0; i < m.D; i++ {
		if !colsMatch(q0, q1, R, m.D, i, perm[i], func(r int) int { return qrow[r] }) {
			t.Fatalf("q column %d does not follow the D permutation", i)
		}
	}
	// o rows follow the same D permutation, and o columns the q rows' (Q axis) permutation.
	for i := 0; i < m.D; i++ {
		for c := 0; c < R; c++ {
			if o1[i*R+c] != o0[perm[i]*R+qrow[c]] {
				t.Fatalf("o[%d,%d] does not follow the D and Q permutations", i, c)
			}
		}
	}
}

// tempDir is a scratch folder removed best-effort: run() keeps its input files open for the life of
// the process, which Windows will not delete under t.TempDir's strict cleanup.
func tempDir(t *testing.T) string {
	d, err := os.MkdirTemp("", "hodgemix-test-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(d) })
	return d
}

// A base whose folder holds another model's config.json, with its own set as "config (1).json": the set
// one is read (the folder's is not) and the output's config.json takes the output's sizes. Set to
// another model's config, or none, the heads are unknown (the tensors admit several head_dims) and the
// topology read says so instead of guessing.
func TestConfigFollowsTensors(t *testing.T) {
	dir := tempDir(t)
	a := writeTiny(t, filepath.Join(dir, "a"), tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 260}, 3)
	if err := os.Rename(filepath.Join(dir, "a", "config.json"), filepath.Join(dir, "a", "config (1).json")); err != nil {
		t.Fatal(err)
	}
	setFile(a, func(f *eval.ModelFiles) { f.Config = filepath.Join(dir, "a", "config (1).json") })
	wrong := map[string]any{"model_type": "bitnet", "hidden_act": "relu2", "hidden_size": 16, "intermediate_size": 24,
		"num_attention_heads": 4, "num_key_value_heads": 2, "num_hidden_layers": 3, "vocab_size": 32, "rope_theta": 500000.0}
	b, _ := json.Marshal(wrong)
	_ = os.WriteFile(filepath.Join(dir, "a", "config.json"), b, 0o644)
	out := filepath.Join(dir, "out", "mix.bnz")
	SrcRopeTheta = 1e6
	defer func() { SrcRopeTheta = 0 }()
	run(tinyCfg("base"), a, []string{a}, out, "", "", true)
	if bad, _ := champCheck(out+".ternary.safetensors", champConfigPath(out)); len(bad) > 0 {
		t.Fatalf("champCheck: %v", bad)
	}
	var cfg map[string]any
	cb, _ := os.ReadFile(champConfigPath(out))
	json.Unmarshal(cb, &cfg)
	want := map[string]float64{"hidden_size": 128, "intermediate_size": 192, "num_attention_heads": 2, "num_key_value_heads": 1,
		"num_hidden_layers": 2, "vocab_size": 260, "rope_theta": 1e6}
	for k, v := range want {
		if cfg[k] != v {
			t.Fatalf("%s = %v, want %v", k, cfg[k], v)
		}
	}
	if cfg["model_type"] != "bitnet" || cfg["hidden_act"] != "relu2" {
		t.Fatalf("config is not one Champ loads: %v", cfg)
	}
	if _, ok := cfg["quantization_config"]; ok {
		t.Fatalf("invented quantization_config: %v", cfg["quantization_config"])
	}
	var rep map[string]any
	if rb, _ := os.ReadFile(out + ".report.json"); json.Unmarshal(rb, &rep) != nil || rep["synth"] == nil {
		t.Fatalf("report.json empty or unreadable (%d keys)", len(rep))
	}

	// Another model's config set, or none: no head count is guessed.
	st, err := OpenST(a)
	if err != nil {
		t.Fatal(err)
	}
	for cp, want := range map[string]string{filepath.Join(dir, "a", "config.json"): "another model's config", "": "run file"} {
		setFile(a, func(f *eval.ModelFiles) { f.Config = cp })
		if m, err := metaOf(st); err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("config %q: heads %d / %d, err %v", cp, m.Heads, m.KV, err)
		}
	}
}

// A Qwen3-layout base (QK-norm, no sub-norms) comes out in the layout Champ loads: q_norm / k_norm
// folded and dropped, attn_sub_norm / ffn_sub_norm present, every q head at QK-norm's unit rms.
func TestQwenBaseChampLayout(t *testing.T) {
	dir := tempDir(t)
	m := tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 260, Qwen: true}
	a := writeTiny(t, filepath.Join(dir, "a"), m, 4)
	out := filepath.Join(dir, "out", "mix.bnz")
	run(tinyCfg("base"), a, []string{a}, out, "", "", true)
	if bad, _ := champCheck(out+".ternary.safetensors", champConfigPath(out)); len(bad) > 0 {
		t.Fatalf("champCheck: %v", bad)
	}
	mo, err := OpenST(out + ".ternary.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range mo.Keys {
		if strings.Contains(k, "q_norm") || strings.Contains(k, "k_norm") {
			t.Fatalf("%s still in the output", k)
		}
	}
}

// --bisect's runs write the ternary safetensors only, and a --roles subset keeps every other packed
// tensor loadable (weight and weight_scale copied from the base).
func TestNoBNZRolesSubset(t *testing.T) {
	dir := tempDir(t)
	a := writeTiny(t, filepath.Join(dir, "a"), tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 260}, 5)
	full := filepath.Join(dir, "full", "mix.bnz")
	run(tinyCfg("base"), a, []string{a}, full, "", "", true) // a ternary base to mix into
	c := tinyCfg("base")
	c.roles, c.noBNZ = []string{"q"}, true
	out := filepath.Join(dir, "out", "mix.bnz")
	run(c, full+".ternary.safetensors", []string{a}, out, "", "", true)
	if _, err := os.Stat(out); err == nil {
		t.Fatalf("BNZ written with noBNZ")
	}
	if bad, _ := champCheck(out+".ternary.safetensors", champConfigPath(out)); len(bad) > 0 {
		t.Fatalf("champCheck: %v", bad)
	}
}

// writeTok writes a byte-level BPE tokenizer.json with tokens t<from>..t<to-1> (ids from 0), one merge
// per token pair "t<i> t<i+1>" whose result is listed, and the added turn end <|eot|>, with a
// tokenizer_config.json naming it: both set as dir's model.safetensors' files.
func writeTok(t *testing.T, dir string, from, to int) {
	t.Helper()
	vocab := map[string]int{}
	var merges []string
	name := func(i int) string { return fmt.Sprintf("t%03d", i) }
	for i := from; i < to; i++ {
		vocab[name(i)] = i - from
	}
	for i := from; i+1 < to; i += 50 {
		a, b := name(i), name(i+1)
		vocab[a+b] = len(vocab)
		merges = append(merges, a+" "+b)
	}
	// Every byte's token (GPT-2 byte-level symbols), so any text encodes.
	n := 0
	for c := 0; c < 256; c++ {
		r := rune(c)
		if !((c >= '!' && c <= '~') || (c >= 0xA1 && c <= 0xAC) || c >= 0xAE) {
			r = rune(256 + n)
			n++
		}
		if _, ok := vocab[string(r)]; !ok {
			vocab[string(r)] = len(vocab)
		}
	}
	tk := map[string]any{"version": "1.0", "model": map[string]any{"type": "BPE", "vocab": vocab, "merges": merges},
		"added_tokens": []any{map[string]any{"id": len(vocab), "content": "<|eot|>", "special": true}}, "pre_tokenizer": nil}
	b, _ := json.Marshal(tk)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{"eos_token": "<|eot|>"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	setFile(filepath.Join(dir, "model.safetensors"), func(f *eval.ModelFiles) {
		f.Tokenizer, f.TokenizerConfig = filepath.Join(dir, "tokenizer.json"), filepath.Join(dir, "tokenizer_config.json")
	})
}

// The union vocabulary keeps the base's ids, appends the source's other tokens with embeddings, and
// the written tokenizer.json, config.json and embedding agree on the size.
func TestUnionVocab(t *testing.T) {
	dir := tempDir(t)
	a := writeTiny(t, filepath.Join(dir, "a"), tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 600}, 6)
	b := writeTiny(t, filepath.Join(dir, "b"), tinyModel{D: 128, F: 192, H: 2, KV: 1, HD: 64, L: 2, V: 600}, 7)
	writeTok(t, filepath.Join(dir, "a"), 0, 300)   // 300 tokens + 6 merged
	writeTok(t, filepath.Join(dir, "b"), 100, 400) // 200 shared; 100 new + 2 new merged strings (4 merged ones the base has)
	out := filepath.Join(dir, "out", "mix.bnz")
	run(tinyCfg("base"), a, []string{b}, out, "", "", true)
	tk, err := readTok(filepath.Join(dir, "out", "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	if tk.vocab["t000"] != 0 || tk.vocab["t299"] != 299 {
		t.Fatalf("base ids moved: t000=%d t299=%d", tk.vocab["t000"], tk.vocab["t299"])
	}
	if _, ok := tk.vocab["t399"]; !ok {
		t.Fatalf("source token t399 not in the union")
	}
	if !strings.Contains(strings.Join(tk.merges, ","), "t350 t351") {
		t.Fatalf("source merge t350 t351 missing")
	}
	st, err := OpenST(out + ".ternary.safetensors")
	if err != nil {
		t.Fatal(err)
	}
	rows := st.LogicalShape("model.embed_tokens.weight")[0]
	var cfg map[string]any
	cb, _ := os.ReadFile(champConfigPath(out))
	json.Unmarshal(cb, &cfg)
	if want := 600 + 100 + 2; rows != want || int(cfg["vocab_size"].(float64)) != want {
		t.Fatalf("embedding rows %d, config vocab_size %v, want %d", rows, cfg["vocab_size"], want)
	}
	if bad, _ := champCheck(out+".ternary.safetensors", champConfigPath(out)); len(bad) > 0 {
		t.Fatalf("champCheck: %v", bad)
	}
}

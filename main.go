// Command hodgemix: shared spline index + Hodge-Kuramoto synthesis of new ternary models from safetensors
// (paper: "Ternary Conversion and Mixing of Open-Weight Models via Hodge-Kuramoto Alignment"),
// output: BitNet-layout ternary safetensors compressed to BNZ.
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"hodgemix/bnz4"
	"hodgemix/eval"
)

// Paper: no equation (helper: eval output)
func logScore(tag string, r *eval.Result) {
	if tag != "" {
		tag += ": "
	}
	for _, g := range r.Gens {
		logf("  %sgreedy %q -> %q", tag, g[0], g[1])
	}
	logf("  %smean NLL %.3f nats/token over %d tokens (ppl %.1f)", tag, r.MeanNLL, r.Tokens, r.PPL)
}

// qkv is tested first: a fused query_key_value must not be taken for q. GPT-2's c_attn/c_fc/c_proj are
// not recognized: they are Conv1D weights stored [in, out] (transposed against every other role here),
// and attn.c_proj is the attention output, not down. A bare "gate" is not gate_proj: in MoE models
// (mlp.gate, block_sparse_moe.gate) it is the router.
var roleOrder = []string{"qkv", "q", "k", "v", "o", "gate", "up", "down", "embed", "lm_head"}
var rolePat = map[string]*regexp.Regexp{
	"q": regexp.MustCompile(`(q_proj|\bwq\b|\bquery\b|\.q\b)`), "k": regexp.MustCompile(`(k_proj|\bwk\b|\bkey\b|\.k\b)`),
	"v": regexp.MustCompile(`(v_proj|\bwv\b|\bvalue\b|\.v\b)`), "o": regexp.MustCompile(`(o_proj|\bwo\b|out_proj|attn\.dense|attention\.output\.dense)`),
	"gate": regexp.MustCompile(`(gate_proj|\bw1\b)`), "up": regexp.MustCompile(`(up_proj|\bw3\b|\bfc1\b|dense_h_to_4h)`),
	"down": regexp.MustCompile(`(down_proj|\bw2\b|\bfc2\b|dense_4h_to_h)`), "qkv": regexp.MustCompile(`(qkv_proj|\bqkv\b|\bWqkv\b|query_key_value)`),
	"embed": regexp.MustCompile(`(embed_tokens|wte|word_embeddings|tok_embeddings)`), "lm_head": regexp.MustCompile(`(lm_head|output\.weight$)`),
}
var layerRe = regexp.MustCompile(`(?:layers?|h|blocks?|block)\.(\d+)\.`)

// Paper: no equation (helper: I/O, progress or utility)
func roleOf(n string) string {
	for _, r := range roleOrder {
		if rolePat[r].MatchString(n) {
			return r
		}
	}
	return ""
}

// languageModel is true when the checkpoint nests its text stack under
// model.language_model (Gemma 4). The vision tower and the audio tower sit
// beside it; they are part of the file, and they are not the text model the
// config's text_config describes.
func languageModel(st *ST) bool {
	for _, k := range st.Keys {
		if strings.Contains(k, "model.language_model.") {
			return true
		}
	}
	return false
}

// towerTensor is an audio or vision weight. It is not a text-mix role.
func towerTensor(name string) bool {
	return strings.Contains(name, "audio_tower") || strings.Contains(name, "vision_tower") ||
		strings.Contains(name, "embed_audio") || strings.Contains(name, "embed_vision")
}

// asideTensor keeps a tensor out of the text BitLinear mix and out of the output:
// the audio and vision towers, their embedding projections, the per-layer input
// embeddings, and a contributor's language-model token embedding. The output's
// towers are the L-A towers writeTowers builds (bnzmix.vision_tower,
// bnzmix.audio_tower, bnzmix.embed_vision, bnzmix.embed_audio), so no source
// tensor is written under its own bytes.
func asideTensor(name string) bool {
	if strings.HasPrefix(name, "bnzmix.") || name == "model.embed_tokens.weight" || name == "lm_head.weight" {
		return false
	}
	if towerTensor(name) {
		return true
	}
	return strings.Contains(name, "embed_tokens") || strings.Contains(name, "per_layer_") ||
		strings.HasSuffix(name, "layer_scalar") || strings.Contains(name, "post_per_layer_input_norm")
}

// embedKey is the token embedding. Gemma 4 stores it on the language model.
func embedKey(st *ST) string {
	for _, k := range []string{"model.embed_tokens.weight", "model.language_model.embed_tokens.weight"} {
		if st.Has(k) {
			return k
		}
	}
	return ""
}

// Paper: no equation (helper: I/O, progress or utility)
func layerOf(n string) int {
	m := layerRe.FindStringSubmatch(n)
	if m == nil {
		return -1
	}
	v, _ := strconv.Atoi(m[1])
	return v
}

// Paper: no equation (helper: I/O, progress or utility)
func stripUpdate(args []string) ([]string, string) {
	if len(args) == 0 {
		return args, ""
	}
	out := []string{args[0]}
	var u string
	for i := 1; i < len(args); i++ {
		if args[i] == "--update" {
			if i+1 >= len(args) {
				die("--update needs a safetensors file")
			}
			u = args[i+1]
			i++
			continue
		}
		out = append(out, args[i])
	}
	return out, u
}

func useUpdate(path string) {
	if path == "" || len(os.Args) < 3 {
		return
	}
	eval.SetUpdate(os.Args[2], path)
	logf("  update %s", path)
}

func die(f string, a ...any) {
	endInner()
	endOuter()
	fmt.Fprintf(os.Stderr, "hodgemix: "+f+"\n", a...)
	removeTemps()
	os.Exit(1)
}

// onInterrupt removes the run's temp files on Ctrl+C, so an interrupted run leaves no .tmp behind.
func onInterrupt() {
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt)
	go func() {
		<-ch
		removeTemps()
		fmt.Fprintln(os.Stderr, "\nhodgemix: interrupted; temp files removed")
		os.Exit(130)
	}()
}

type srcList []string

// Paper: no equation (helper: I/O, progress or utility)
func (s *srcList) String() string { return strings.Join(*s, ",") }

// Paper: no equation (helper: I/O, progress or utility)
func (s *srcList) Set(v string) error { *s = append(*s, v); return nil }

// Paper: no equation (CLI). Every default is the paper_v2 derived value (Appendix A); flags only override.
func main() {
	onInterrupt()
	var updatePath string
	os.Args, updatePath = stripUpdate(os.Args)
	// hodgemix --config RUN.yaml [probe command]: every model's files from the run file (runfile.go).
	if len(os.Args) > 1 && (os.Args[1] == "--config" || strings.HasPrefix(os.Args[1], "--config=")) {
		path := strings.TrimPrefix(os.Args[1], "--config=")
		rest := os.Args[2:]
		if os.Args[1] == "--config" {
			if len(os.Args) < 3 {
				die("--config needs a run file: hodgemix --config RUN.yaml")
			}
			path = os.Args[2]
			rest = os.Args[3:]
		}
		args, err := loadRunFile(path)
		if err != nil {
			die("%v", err)
		}
		if len(rest) > 0 {
			os.Args = append([]string{os.Args[0]}, rest...)
		} else {
			os.Args = append([]string{os.Args[0]}, args...)
		}
	}
	if updatePath != "" && (len(os.Args) < 2 || (os.Args[1] != "--generate" && os.Args[1] != "--chat" && os.Args[1] != "--flow")) {
		die("--update is used with --generate, --chat, or --flow")
	}
	if len(os.Args) > 1 && os.Args[1] == "--selftest" {
		selftest()
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "--dtft-tests" {
		dtftFalsify(os.Args[2:])
		return
	}
	if len(os.Args) > 3 && os.Args[1] == "--trace" { // hodgemix --trace BASE OUT [tokenizer.json]
		tok := eval.FilesOf(os.Args[2]).Tokenizer
		if len(os.Args) > 4 {
			tok = os.Args[4]
		}
		text, chat := eval.Texts[0], os.Getenv("HODGEMIX_TRACE_CHAT") != ""
		if chat {
			text = "" // eval.Trace builds the page's chat prompt with the tokenizer's own turn end
		}
		rows, posCos, err := eval.Trace(os.Args[2], os.Args[3], tok, text, chat)
		if err != nil {
			die("trace: %v", err)
		}
		var buckets []string
		for lo := 0; lo < len(posCos); lo += max(1, len(posCos)/8) {
			hi := min(len(posCos), lo+max(1, len(posCos)/8))
			var m float64
			for _, c := range posCos[lo:hi] {
				m += c
			}
			buckets = append(buckets, fmt.Sprintf("%d-%d:%.3f", lo, hi-1, m/float64(hi-lo)))
		}
		logf("  last-layer cosine by position: %s", strings.Join(buckets, " "))
		logf("TRACE  %s vs %s on %q", os.Args[2], os.Args[3], text[:min(len(text), 80)])
		logf("  %-5s %-6s %8s %8s %8s  %s", "layer", "stage", "cos", "norm", "peak", "top channels of the difference (share)")
		for _, r := range rows {
			if r.Stage == "hidden" {
				logf("  %-5d %-6s %8s %8s %8.2f", r.Layer, r.Stage, "", "", r.Peak)
			} else {
				logf("  %-5d %-6s %8.4f %8.3f %8s  %v (%.0f%%)  base largest %v (>= %.0fx median)", r.Layer, r.Stage, r.Cos, r.Norm, "", r.Top, 100*r.TopFr, r.Big, r.BigX)
			}
		}
		return
	}
	if len(os.Args) > 3 && os.Args[1] == "--map" { // hodgemix --map BASE SRC: the shared-vocabulary residual map's fit
		b, err1 := OpenST(os.Args[2])
		s, err2 := OpenST(os.Args[3])
		if err1 != nil || err2 != nil {
			die("map: %v %v", err1, err2)
		}
		bt, err1 := readTok(eval.FilesOf(b.Path).Tokenizer)
		stk, err2 := readTok(eval.FilesOf(s.Path).Tokenizer)
		if err1 != nil || err2 != nil {
			die("map: %v %v", err1, err2)
		}
		Eb, err1 := b.F32("model.embed_tokens.weight")
		Es, err2 := s.F32("model.embed_tokens.weight")
		if err1 != nil || err2 != nil {
			die("map: %v %v", err1, err2)
		}
		Db, Ds := b.LogicalShape("model.embed_tokens.weight")[1], s.LogicalShape("model.embed_tokens.weight")[1]
		_, _, _, r2, n := residualMap(Eb, Db, bt.vocab, Es, Ds, stk.vocab)
		logf("  residual map %s (D=%d) -> %s (D=%d): held-out R^2 %.3f over %d shared token strings", filepath.Base(s.Path), Ds, filepath.Base(b.Path), Db, r2, n)
		return
	}
	if len(os.Args) > 3 && os.Args[1] == "--ov-chart" { // hodgemix --ov-chart BASE OUT.ternary.safetensors: in place
		g, err := metaFromConfigOrTensors(os.Args[2])
		if err != nil {
			die("ov-chart: %v", err)
		}
		rep, err := ovChart(os.Args[2], os.Args[3], g.Heads, g.KV, g.HD)
		if err != nil {
			die("ov-chart: %v", err)
		}
		logf("  ov-chart: %d layers: v_proj = the base's chart, o_proj = first-order circuit (cross term removed: %.3g of ||O_b V_b||)", rep.Layers, rep.Cross)
		return
	}
	if len(os.Args) > 3 && os.Args[1] == "--repair" { // hodgemix --repair BASE OUT.ternary.safetensors [tokenizer.json]: in place
		tok := eval.FilesOf(os.Args[2]).Tokenizer
		if len(os.Args) > 4 {
			tok = os.Args[4]
		}
		if _, err := repair(os.Args[2], os.Args[3], tok); err != nil {
			die("repair: %v", err)
		}
		return
	}
	if len(os.Args) > 3 && os.Args[1] == "--surgery" { // hodgemix --surgery BASE OUT.ternary.safetensors [tokenizer.json]: in place
		tok := eval.FilesOf(os.Args[2]).Tokenizer
		if len(os.Args) > 4 {
			tok = os.Args[4]
		}
		rep, err := surgery(os.Args[2], os.Args[3], tok)
		if err != nil {
			die("surgery: %v", err)
		}
		logSurgery(rep)
		return
	}
	if len(os.Args) > 2 && os.Args[1] == "--converse" { // hodgemix --converse MODEL [tokenizer.json]: the conversation suite
		tok := eval.FilesOf(os.Args[2]).Tokenizer
		if len(os.Args) > 3 {
			tok = os.Args[3]
		}
		convs, err := eval.Converse(os.Args[2], tok, 96)
		if err != nil {
			die("converse: %v", err)
		}
		for i, c := range convs {
			for _, t := range c {
				logf("  [%d] user: %s", i+1, t.User)
				logf("  [%d] champ: %s", i+1, t.Reply)
			}
		}
		if b, err := json.MarshalIndent(convs, "", " "); err == nil {
			out := strings.TrimSuffix(filepath.Base(os.Args[2]), ".safetensors") + ".converse.json" // in the working directory
			os.WriteFile(out, b, 0o644)
			logf("  transcripts: %s", out)
		}
		return
	}
	if len(os.Args) > 2 && os.Args[1] == "--chat" { // hodgemix --chat MODEL [tokenizer.json]: the page's first answer
		useUpdate(updatePath)
		tok := eval.FilesOf(os.Args[2]).Tokenizer
		if len(os.Args) > 3 {
			tok = os.Args[3]
		}
		cells, cx, fam, err := manifoldSetup(os.Args[2])
		if err != nil {
			die("chat: %v", err)
		}
		r, n, err := eval.Generate(os.Args[2], tok, "", 24, true, cells, func(pos, fed, next int, piece string, layers []eval.FlowLayer) {
			if next < 0 {
				return
			}
			line, err := manifoldLine(fam, pos, piece, cells, cx, layers)
			if err != nil {
				die("chat: %v", err)
			}
			logf("%s", line)
		})
		if err != nil {
			die("chat: %v", err)
		}
		logf("  chat (page system prompt, %d-token context) %q -> %q", n, eval.ChatUser, r)
		return
	}
	if len(os.Args) > 3 && os.Args[1] == "--generate" { // hodgemix --generate MODEL "text" [n]
		useUpdate(updatePath)
		n := 8
		if len(os.Args) > 4 {
			var err error
			n, err = strconv.Atoi(os.Args[4])
			if err != nil || n < 0 {
				die("generate: n is %q", os.Args[4])
			}
		}
		tok := eval.FilesOf(os.Args[2]).Tokenizer
		cells, cx, fam, err := manifoldSetup(os.Args[2])
		if err != nil {
			die("generate: %v", err)
		}
		r, prompt, err := eval.Generate(os.Args[2], tok, os.Args[3], n, false, cells, func(pos, fed, next int, piece string, layers []eval.FlowLayer) {
			if next < 0 {
				return
			}
			line, err := manifoldLine(fam, pos, piece, cells, cx, layers)
			if err != nil {
				die("generate: %v", err)
			}
			logf("%s", line)
		})
		if err != nil {
			die("generate: %v", err)
		}
		logf("  generate (%d-token prompt) %q -> %q", prompt, os.Args[3], r)
		return
	}
	if len(os.Args) > 4 && os.Args[1] == "--flow" { // hodgemix --config RUN.yaml --flow MODEL OUT.safetensors "text" (cochain.go)
		useUpdate(updatePath)
		rep, err := flowRun(os.Args[2], os.Args[3], os.Args[4])
		if err != nil {
			die("flow: %v", err)
		}
		logFlow(rep)
		return
	}
	if len(os.Args) > 3 && os.Args[1] == "--neck" { // hodgemix --config RUN.yaml --neck TERNARY OUT.safetensors
		if runBase == "" {
			die("--neck needs --config RUN.yaml whose base is the float Llama 3 checkpoint")
		}
		cp := eval.FilesOf(runBase).Config
		if cp == "" {
			die("neck: the run file sets no config for the base")
		}
		ok, mt, err := llama3Config(cp)
		if err != nil {
			die("neck: %v", err)
		}
		if !ok {
			if mt == "" {
				mt = "unset"
			}
			die("neck pack runs when the base config is a llama3 arch (rope_type llama3); this base is %s", mt)
		}
		nLimbs, tol := llama3Budget(MaxLimbs, LimbTol)
		logf("LLAMA3  neck pack (%d limbs, residual tol %.3f)", nLimbs, tol)
		if err := packNeck(os.Args[2], runBase, os.Args[3], nLimbs, tol); err != nil {
			die("neck: %v", err)
		}
		return
	}
	if len(os.Args) > 3 && os.Args[1] == "--sing" { // hodgemix --config RUN.yaml --sing PACK.safetensors OUT.safetensors
		if runBase == "" {
			die("--sing needs --config RUN.yaml whose base is the float checkpoint")
		}
		logf("neck limbs (%d) on the singular set the body limbs left at center", NeckLimbs)
		if err := appendSing(os.Args[2], runBase, os.Args[3]); err != nil {
			die("sing: %v", err)
		}
		return
	}
	if len(os.Args) > 2 && os.Args[1] == "--score" { // hodgemix --score MODEL.safetensors [tokenizer.json]
		tok := eval.FilesOf(os.Args[2]).Tokenizer
		if len(os.Args) > 3 {
			tok = os.Args[3]
		}
		r, err := eval.Run(os.Args[2], tok)
		if err != nil {
			die("score: %v", err)
		}
		logScore("", r)
		return
	}
	fs := flag.NewFlagSet("hodgemix", flag.ExitOnError)
	var srcs srcList
	base := fs.String("base", "", "base model .safetensors (BitNet b1.58 2B4T, Champ packed ternary)")
	fs.Var(&srcs, "src", "source model PATH[:WEIGHT], repeatable")
	out := fs.String("out", "", "output .bnz")
	baseW := fs.Float64("base-weight", 1.0, "weight of the base (as --src :WEIGHT)")
	targetCfg := fs.String("target-config", "", "config.json of the output architecture (default: base tensor list)")
	outTern := fs.String("out-ternary-st", "", "ternary safetensors path (default OUT.ternary.safetensors, kept)")
	outMerged := fs.String("out-merged-st", "", "also write the merged (pre-ternary, post-rotation) model as safetensors")
	noVerify := fs.Bool("no-verify", false, "skip BNZ round-trip check")
	roles := fs.String("roles", "q,k,v,o,gate,up,down,qkv", "Eq.(1)/App.A: BitLinear roles (BitNet ternarizes every attention/FFN linear)")
	rot := fs.String("rotate", "off", "Eq.(23)-(24): randomized Hadamard residual rotation before absmean {on|off}; default off because Champ requires a tied LM head (C8)")
	rotSeed := fs.Int64("rot-seed", 0, "Eq.(23): Rademacher sign seed (reproducibility only)")
	resid := fs.String("base-residual", "on", "Eq.(22): keep base detail (I - P R) finer than the index {on|off}")
	fdt := fs.String("float-dtype", "bf16", "App.A: dtype of non-ternary tensors (BitNet 2B4T stores them in BF16)")
	gth := fs.Int("g-theta", GThetaShift, "Eq.(15): coupling gain shift (KTE g_theta)")
	gom := fs.Int("g-omega", GOmegaShift, "Eq.(15): weak gain shift (KTE g_omega)")
	nlost := fs.Int("n-lost", NLost, "Eq.(16)-(18): KTE N_lost ticks")
	forceRANS := fs.Bool("bnz-force-rans", false, "TEST ONLY: store every U8/BF16 tensor with BNZ4 methods 4/5 even when larger (exercises the WGSL decoder on tiny files)")
	signSeed := fs.String("sign-seed", "", "Ed25519 seed file (bnz4sign keygen format): write OUT.sig over header+index for bnz-loader")
	lock := fs.String("lock", "dtft", "dtft_lock Alg.D selective lock {dtft} or paper_v2 omega=0 lock {v2} (comparison only)")
	bandInc := fs.Bool("band-inclusive", false, "dtft_lock (H2) tuning: use lambda <= lambda_gap (adds lowest bins of the coarsest long digit); paper Eq.(H2) is strict <")
	seqAxis := fs.String("seq-axis", "auto", "dtft_lock tuning item 'sequence axis': auto (axis of the coarsest long cycle carries the modes) | rows | cols")
	omUpd := fs.String("omega-update", "on", "dtft_lock Alg.D step 5: R-KTE-1 g_omega frequency update {on|off}")
	pers := fs.String("persistence", "on", "dtft_lock Sec.2.5 heap-key persistence filtration {on|off}")
	persB := fs.Int64("persist-budget", 0, "dtft_lock tuning item 'H1 persistence cost': max Z2 column additions, 0 = unlimited")
	mrhMode := fs.String("mrh", "dynamic", "multi-radix heap: dynamic = MRH(r;C) with each axis' own radices (pop order, buckets, grid levels) | paper = paper_v2's 16-bucket levels and 256 grid")
	gridMax := fs.Int("grid-max", GridMax, "Eq.(6): most index cells per axis (whole radix levels); paper_v2 is 256")
	topo := fs.String("topology", "base", "target topology: base = the base model's | max = the larger size per axis | married = lcm per axis (every model's radix hierarchy embeds exactly); heads keep the base head_dim, layers the deeper model's count")
	attn := fs.String("attn", "on", "attention after marriage: fold a contributor's QK-norm, marry RoPE frequencies, match each layer's logit temperature to the base {on|off}")
	srcTheta := fs.Float64("src-rope-theta", 0, "rope_theta of any model (base or source) whose config.json does not give one (Qwen3: 1000000)")
	indexMode := fs.String("index", "married", "channel index: married = all contributors' channels in one MRH per axis, coordinates from the joint pop order | own = each model's own pop order (paper_v2 Eq.8)")
	evalOn := fs.String("eval", "on", "after writing, score base and output with the CPU reference forward (greedy text + NLL on fixed sentences) {on|off}; needs a 2B4T-shape output")
	evalTok := fs.String("eval-tokenizer", "", "tokenizer.json for --eval (default: the base's tokenizer from the run file)")
	seedAl := fs.String("seed-align", "auto", "per-tensor seed alignment (Eq.12-13 shift + Re-Basin assignment) of each contributor's grid; auto = on (off measured ppl 1.3e7 on 2B4T + Qwen3: the married index alone does not align the models) {auto|on|off}")
	vocabM := fs.String("vocab", "union", "output vocabulary: union = the base's tokens (same ids) + every source's other tokens, embeddings by the shared-vocabulary residual map, its own tokenizer.json (vocab.go) | base {union|base}")
	limbsN := fs.Int("limbs", MaxLimbs, "body MRH limbs per weight, including limb 0: 1 = ternary body only. The singular neck is coded on its own limbs either way (limbs.go)")
	limbTol := fs.Float64("limb-tol", LimbTol, "stop adding limbs once the residual is below this fraction of ||W||")
	ovM := fs.String("ov", "chart", "attention OV circuits: chart = keep the base's v_proj (the head's chart) and carry v's change into o_proj to first order, no dO dV cross term (ovchart.go) | mix {chart|mix}")
	pinM := fs.String("pin-mean", "on", "keep every mixed BitLinear's response to its mean input (the base forward's) equal to the base's: dW mu = 0 (pinmean.go) {on|off}")
	repairM := fs.String("repair", "on", "after surgery, set every RMSNorm gain so each channel's spread matches the base's on the calibration tokens (REPAIR; surgery paper step 7 on every channel; repair.go). Passes run until the mean |log2| spread is under the stop in repair.go, short of a ratio of 1 {on|off}")
	surg := fs.String("surgery", "on", "after the mix, put the base's codes back on the singular set of its forward: massive residual channels and the FFN spike units that write them (surgery.go; needs the 2B4T-shape reference forward) {on|off}")
	surgRatio := fs.Float64("surgery-ratio", SurgeryRatio, "a residual channel / FFN unit is singular at this many times the median's rms / peak")
	bisectOn := fs.Bool("bisect", false, "mix each of --roles alone (every other tensor stays the base's), score each against the base with the reference forward, print role -> ppl; outputs are deleted, the table is kept in OUT.bisect.json")
	fs.Parse(os.Args[1:])
	bnz4.ForceRANS = *forceRANS
	switch *mrhMode {
	case "dynamic", "paper":
		MRHMode = *mrhMode
	default:
		die("--mrh must be dynamic or paper")
	}
	if *gridMax < 2 {
		die("--grid-max must be at least 2")
	}
	GridMax = *gridMax
	if *indexMode != "married" && *indexMode != "own" {
		die("--index must be married or own")
	}
	IndexMode = *indexMode
	switch *seedAl {
	case "auto":
		SeedAlign = true
	case "on", "off":
		SeedAlign = *seedAl == "on"
	default:
		die("--seed-align must be auto, on or off")
	}
	AttnOn = *attn == "on"
	SurgeryOn, SurgeryRatio = *surg == "on", *surgRatio
	PinMean = *pinM == "on"
	RepairOn = *repairM == "on"
	if *limbsN < 1 {
		die("--limbs must be at least 1")
	}
	MaxLimbs, LimbTol = *limbsN, *limbTol
	if *ovM != "chart" && *ovM != "mix" {
		die("--ov must be chart or mix")
	}
	OVMode = *ovM
	if *vocabM != "union" && *vocabM != "base" {
		die("--vocab must be union or base")
	}
	VocabMode = *vocabM
	SrcRopeTheta = *srcTheta
	if *base == "" || len(srcs) == 0 || *out == "" {
		die("need --base, --src and --out (all other terms have paper_v2 derived defaults)")
	}
	FD := map[string]string{"bf16": "BF16", "f16": "F16", "f32": "F32"}[*fdt]
	if FD == "" {
		die("--float-dtype must be bf16, f16 or f32")
	}
	sp := SynthParams{K: KParams{GTheta: math.Ldexp(1, -*gth), Weak: math.Ldexp(1, -(*gom - *gth)), WLock: WLock,
		NLost: *nlost, RTol: RefineTol}, Lock: *lock,
		D: DParams{BandInclusive: *bandInc, SeqAxis: *seqAxis, OmegaUpdate: *omUpd == "on", Persist: *pers == "on", PersistBudget: *persB}}
	logDParams(sp)
	cfg := runCfg{roles: strings.Split(*roles, ","), residual: *resid == "on", FD: FD, sp: sp,
		baseW: *baseW, targetCfg: *targetCfg, rotate: *rot == "on", rotSeed: *rotSeed, signSeed: *signSeed, topology: *topo}
	if *topo != "base" && *topo != "max" && *topo != "married" {
		die("--topology must be base, max or married")
	}
	if *evalOn == "on" {
		cfg.evalTok = *evalTok
		if cfg.evalTok == "" {
			cfg.evalTok = eval.FilesOf(*base).Tokenizer
		}
	}
	logf("paper_v2 parameters: g_theta=2^-%d weak=2^-%d W_lock=1/16 turn N_lost=%d refine tol=2^-8 key=12-bit rotate=%v residual=%v",
		*gth, *gom-*gth, *nlost, cfg.rotate, cfg.residual)
	if MRHMode == "dynamic" {
		logf("MRH dynamic: MRH(r; C=%d), r = each axis' factorisation finest first (e.g. 2560 -> %v); pop order, buckets and grid levels from it; grid <= %d and <= the shortest contributor axis",
			keyUniverse, RadicesFor(2560, keyUniverse), GridMax)
	} else {
		logf("MRH paper: 16-bucket levels, bucket = top 4-bit digit, grid <= 256")
	}
	if *bisectOn {
		tok := *evalTok
		if tok == "" {
			tok = eval.FilesOf(*base).Tokenizer
		}
		bisect(cfg, *base, srcs, *out, tok)
		return
	}
	run(cfg, *base, srcs, *out, *outTern, *outMerged, *noVerify)
}

type runCfg struct {
	roles            []string
	baseW            float64
	residual, rotate bool
	rotSeed          int64
	FD, targetCfg    string
	signSeed         string
	evalTok          string // "" = no eval
	noBNZ            bool   // write the ternary safetensors only (--bisect)
	topology         string // base | max | married
	sp               SynthParams
}

type entry struct {
	src        int
	name, role string
	u          float64
	shape      []int
	kr, kc     []int
	nl         int
	scale      float64
	xr, xc     []float64
	bucket     int
}

type target struct {
	name, role string
	layer      int
	shape      []int
}

// Paper: no equation (helper: I/O, progress or utility)
func nLayers(names []string) int {
	m := -1
	for _, n := range names {
		m = max(m, layerOf(n))
	}
	if m < 0 {
		return 1
	}
	return m + 1
}

// Paper: Sec.4 ternarize (absmean) and pack 4 trits per U8 (BitNet b1.58 layout)
func packTrits(q []int8, R, C int) []byte {
	R4 := R / 4
	o := make([]byte, R4*C)
	for i := 0; i < R4*C; i++ {
		var b byte
		for s := 0; s < 4; s++ {
			b |= byte(q[s*R4*C+i]+1) << (2 * s)
		}
		o[i] = b
	}
	return o
}

// Paper: Sec.4 pipeline: align, resize, mix, ternarize; Contrib.5 channel-map chunk (stored as index tensors)
func run(c runCfg, basePath string, srcArgs []string, out, outTern, outMerged string, noVerify bool) {
	logf("hodgemix: %d CPU cores, GOOS=%s", runtime.NumCPU(), runtime.GOOS)
	logf("loading base header %s", basePath)
	baseST, err := OpenST(basePath)
	if err != nil {
		die("%v", err)
	}
	all := []*ST{baseST}
	ws := []float64{c.baseW}
	wre := regexp.MustCompile(`^(.*):([-+]?[0-9.eE+-]+)$`)
	for _, s := range srcArgs {
		path, w := s, 1.0
		if m := wre.FindStringSubmatch(s); m != nil && !regexp.MustCompile(`^[A-Za-z]$`).MatchString(m[1]) {
			path = m[1]
			w, _ = strconv.ParseFloat(m[2], 64)
		}
		logf("loading source header %s (weight %g)", path, w)
		st, err := OpenST(path)
		if err != nil {
			die("%v", err)
		}
		all = append(all, st)
		ws = append(ws, w)
	}
	if outTern == "" {
		outTern = out + ".ternary.safetensors"
	}
	// Surgery, mean pinning, repair, the OV chart and the union vocabulary work in the base's topology.
	baseOnly := c.targetCfg == "" && c.topology == "base"
	if !baseOnly {
		logf("  --topology %s / --target-config: surgery, pin-mean, repair, ov chart and the union vocabulary apply to --topology base only and are not run", c.topology)
	}
	preflight(c, basePath, all, baseOnly)
	if err := os.MkdirAll(filepath.Dir(absP(out)), 0o755); err != nil {
		die("%v", err)
	}
	// Calibration: one reference forward of the base gives the singular set (surgery.go) and every
	// BitLinear's mean input (pinmean.go); repair compares against the same forward.
	var calib *eval.Singular
	if (SurgeryOn || PinMean || RepairOn) && baseOnly {
		logf("CALIBRATE  base reference forward (calibration texts + the page's chat prompt)")
		if calib, err = eval.FindSingular(basePath, eval.FilesOf(basePath).Tokenizer, SurgeryRatio); err != nil {
			die("calibration: %v", err)
		}
		logf("  %d tokens; %d massive residual channels", calib.Tokens, len(calib.Massive))
	}
	// Union vocabulary (vocab.go): the base's tokens keep their ids, every source's other tokens are
	// appended with embeddings carried over by the shared-vocabulary residual map.
	var U *vocabUnion
	if VocabMode == "union" && baseOnly && len(all) > 1 {
		logf("VOCAB  union of the base's and the sources' vocabularies")
		if U, err = buildUnion(baseST, basePath, all[1:]); err != nil {
			die("vocab: %v", err)
		}
		if U == nil {
			logf("  no source adds a token: the base's vocabulary is the output's")
		} else {
			logf("  %d tokens: %d appended from the sources, %d added (special); residual map held-out R^2 %v over %v shared strings",
				U.V, U.appended, U.added, U.r2, U.shared)
		}
	}
	srcJ, _ := json.Marshal(append([]string{basePath}, srcPaths(all[1:])...))
	wsJ, _ := json.Marshal(ws)
	meta := append([][2]string{}, baseST.Meta...)
	meta = setMeta(meta, "format", "pt")
	meta = setMeta(meta, "bnzmix_method", "index")
	meta = setMeta(meta, "bnzmix_sources", string(srcJ))
	meta = setMeta(meta, "bnzmix_weights", string(wsJ))
	W, err := NewSTW(outTern, meta)
	if err != nil {
		die("%v", err)
	}
	var MW *STW
	if outMerged != "" {
		if MW, err = NewSTW(outMerged, meta); err != nil {
			die("%v", err)
		}
	}
	idxRoles := map[string]bool{}
	for _, r := range c.roles {
		idxRoles[r] = true
	}
	report := map[string]any{}
	if U != nil {
		report["vocab"] = map[string]any{"size": U.V, "appended": U.appended, "added": U.added, "map_r2": U.r2, "shared": U.shared}
	}
	var ternarized []string
	var keptBase [][2]string

	// ---------- STAGE 1 ----------
	logf("STAGE 1/6  index entries + MRH keys (Eq.7) (stream every 2-D tensor of every model, one at a time)")
	var E []*entry
	for si, st := range all {
		var names []string
		lm := languageModel(st)
		for _, k := range st.Keys {
			if asideTensor(k) || (lm && !strings.Contains(k, "model.language_model.")) {
				continue
			}
			if !strings.HasSuffix(k, "weight_scale") && !isLimbTensor(k) { // a limbed weight is one entry (LoadDense sums its limbs)
				names = append(names, k)
			}
		}
		nl := nLayers(names)
		logf("  model %d/%d: %s (%d tensors, %d layers)", si, len(all)-1, st.Path, len(names), nl)
		var todo []string
		for _, k := range names {
			r := roleOf(k)
			sh := st.LogicalShape(k)
			dt := st.H[k].Dtype
			if !idxRoles[r] || len(sh) != 2 {
				continue // norms and embeddings (preflight: nothing else is left out)
			}
			if !(dt == "F32" || dt == "F16" || dt == "BF16" || dt == "U8") {
				die("model %d: %s is %s, not a weight dtype hodgemix reads", si, k, dt)
			}
			todo = append(todo, k)
		}
		if si > 0 && len(todo) == 0 {
			die("model %d (%s) has no tensors with a recognized role; nothing of it would be mixed", si, st.Path)
		}
		setOuter(fmt.Sprintf("stage1 model %d", si), float64(len(todo)))
		for i, k := range todo {
			t := time.Now()
			Wd, sh, err := st.LoadDense(k)
			if err != nil {
				die("%s: %v", k, err)
			}
			l := layerOf(k)
			u := 0.5
			if l >= 0 {
				u = float64(l) / float64(max(1, nl-1))
			}
			e := &entry{src: si, name: k, role: roleOf(k), u: u, shape: sh, scale: absmean(Wd), nl: nl}
			e.kr = heapKey(Wd, sh[0], sh[1], 0) // Eq.(7)
			e.kc = heapKey(Wd, sh[0], sh[1], 1)
			E = append(E, e)
			logf("    [%d/%d] load+dequant %s %v (%s) -> MRH keys rows/cols  %.2fs", i+1, len(todo), k, sh, st.H[k].Dtype, time.Since(t).Seconds())
			outerSet(float64(i + 1))
		}
		endOuter()
	}
	logf("  %d index entries", len(E))
	// ---------- STAGE 2 ----------
	logf("STAGE 2/6  shared channel index: MRH pop order -> channel coordinates (Eq.8), tensor buckets (Eq.9)")
	setOuter("index", float64(len(E)))
	for i, e := range E {
		e.xr, e.xc = chanCoords(e.kr), chanCoords(e.kc)
		e.bucket = bucketOf(e.kr)
		outerSet(float64(i + 1))
	}
	endOuter()
	roleSeen := map[string]bool{}
	for _, e := range E {
		roleSeen[e.role] = true
	}
	// ---------- targets ----------
	var RT *rotPlan
	var T []target
	TL := 1
	// Topology (topology.go): each model's axis graph, a married index per axis node, and the
	// target sizes both models coexist in (--topology base|max|married).
	var tp *topology
	if MRHMode == "dynamic" && IndexMode == "married" {
		var err error
		if tp, err = newTopology(c.topology, all, E, ws); err != nil {
			die("topology: %v", err)
		}
	} else if c.topology != "base" {
		die("--topology %s needs --mrh dynamic --index married", c.topology)
	}
	if tp != nil {
		for m, mm := range tp.metas {
			src := "heads from q_norm"
			if mm.Config != "" {
				src = mm.Config
			}
			logf("  topology model %d: D=%d F=%d heads=%d kv=%d head_dim=%d layers=%d rope_theta=%g (%s)", m, mm.D, mm.F, mm.Heads, mm.KV, mm.HD, mm.Layers, mm.Theta, src)
		}
		v := 0
		if baseST.Has("model.embed_tokens.weight") {
			v = baseST.LogicalShape("model.embed_tokens.weight")[0]
		}
		g := tp.tgt
		logf("  target topology (%s): D=%d F=%d heads=%d kv=%d head_dim=%d layers=%d, %.2fB parameters (%.2f GB ternary)",
			c.topology, g.D, g.F, g.Heads, g.KV, g.HD, g.Layers, g.params(v)/1e9, g.params(v)*2/8/1e9)
		report["topology"] = map[string]any{"policy": c.topology, "models": tp.metas, "target": g}
	}
	cal := newAttnCal(tp, baseST)
	if c.targetCfg != "" {
		T, TL = targetsFromConfig(c.targetCfg)
	} else if tp != nil && c.topology != "base" {
		if c.rotate {
			die("--rotate needs --topology base")
		}
		T, TL = tp.targets(baseST)
	} else {
		var names []string
		for _, k := range baseST.Keys {
			if !(strings.HasSuffix(k, "weight_scale") && baseST.Has(strings.TrimSuffix(k, "_scale"))) && !isLimbTensor(k) {
				names = append(names, k)
			}
		}
		TL = nLayers(names)
		for _, k := range names {
			sh := baseST.LogicalShape(k)
			r := "norm"
			if len(sh) == 2 {
				r = roleOf(k)
			}
			T = append(T, target{k, r, layerOf(k), sh})
		}
	}
	if c.targetCfg == "" {
		var notes []string
		if T, notes = champLayout(T, tp, baseST); len(notes) > 0 {
			logf("  Champ layout: %s", strings.Join(notes, ", "))
		}
	}
	if c.rotate {
		RT = planRotation(T, baseST, c.rotSeed, &T)
	}
	sw := append([]float64{c.baseW}, ws[1:]...)
	nsyn := 0
	for _, t := range T {
		if roleSeen[t.role] && idxRoles[t.role] && len(t.shape) == 2 {
			nsyn++
		}
	}
	// ---------- STAGE 3 ----------
	logf("STAGE 3/6  synthesize %d tensors (load -> spline on index -> seed -> spectral Hodge -> TIES -> Kuramoto lock -> merge -> ternarize -> pack), %d target tensors total", nsyn, len(T))
	setOuter("tensors", float64(nsyn))
	isyn := 0
	var synthRep []map[string]any
	var limbRep []map[string]any
	var tensorSecs []float64
	for _, t := range T {
		if U != nil && t.name == "model.embed_tokens.weight" {
			W.AddFloat(t.name, []int{U.V, U.D}, U.emb, c.FD)
			if MW != nil {
				MW.AddFloat(t.name, []int{U.V, U.D}, U.emb, c.FD)
			}
			keptBase = append(keptBase, [2]string{t.name, "base rows + union rows (vocab.go)"})
			logf("  %s [%d %d]: base rows + %d union rows", t.name, U.V, U.D, U.V-t.shape[0])
			continue
		}
		if !(roleSeen[t.role] && idxRoles[t.role] && len(t.shape) == 2) {
			if tp != nil && c.topology != "base" {
				tp.copyResampled(t, baseST, W, MW, c.FD, &keptBase)
				continue
			}
			copyNonIndex(t, baseST, W, MW, c.FD, &keptBase, RT)
			continue
		}
		tt := time.Now()
		u := 0.5
		if t.layer >= 0 {
			u = float64(t.layer) / float64(max(1, TL-1))
		}
		R, C := t.shape[0], t.shape[1]
		type cand struct {
			ei int
			dw float64
		}
		var cs []cand
		for ei, e := range E {
			if e.role != t.role {
				continue
			}
			dw := sw[e.src] * depthHat(e.u, u, e.nl) // Eq.(11)
			if dw <= 0 {
				continue
			}
			cs = append(cs, cand{ei, dw})
		}
		isyn++
		if len(cs) == 0 {
			keptBase = append(keptBase, [2]string{t.name, "no index entries for role"})
			outerSet(float64(isyn))
			continue
		}
		// Eq.(6): the index keeps whole radix levels of the target axes, no finer than the
		// shortest contributor axis (--mrh dynamic) or at most 256 cells (--mrh paper).
		fineR, fineC := R, C
		for _, cd := range cs {
			fineR, fineC = min(fineR, E[cd.ei].shape[0]), min(fineC, E[cd.ei].shape[1])
		}
		G := [2]int{gridSizeFor(R, fineR), gridSizeFor(C, fineC)}
		// Eq.(9): every contributor's bucket in the target rows' radix tuple, so buckets compare.
		rowRadices := RadicesFor(R, keyUniverse)
		buckets := make([]int, len(cs))
		for j, cd := range cs {
			buckets[j] = bucketMRH(E[cd.ei].kr, rowRadices)
		}
		// Married index (--index married): every contributor's rows go into one heap and its columns
		// into another, and each channel's coordinate is its place in the joint pop order, so all the
		// models of this target sit on one coherent index per axis (married.go).
		cxr, cxc := make([][]float64, len(cs)), make([][]float64, len(cs))
		for j, cd := range cs {
			cxr[j], cxc[j] = E[cd.ei].xr, E[cd.ei].xc
		}
		baseJ := -1
		for j, cd := range cs {
			if E[cd.ei].src == 0 && E[cd.ei].name == t.name {
				baseJ = j
			}
		}
		// yr, yc: where the target's own channels sit on the married axes (nil: base's own coords).
		var yr, yc []float64
		if rk, ck, ok := roleAxes(t.role); ok && tp != nil {
			// One married index per axis node, shared by every tensor on that node (topology.go).
			mgR, err1 := tp.marriageFor(rk, t.layer)
			mgC, err2 := tp.marriageFor(ck, t.layer)
			if err1 != nil || err2 != nil {
				die("%s: married index: %v %v", t.name, err1, err2)
			}
			for j, cd := range cs {
				e := E[cd.ei]
				l := layerOf(e.name)
				xr, okr := mgR.coords[instFor(e.src, rk, l)]
				xc, okc := mgC.coords[instFor(e.src, ck, l)]
				if !okr || !okc || len(xr) != e.shape[0] || len(xc) != e.shape[1] {
					die("%s: contributor %s is not on the married %s/%s axes", t.name, e.name, rk, ck)
				}
				cxr[j], cxc[j] = xr, xc
			}
			if len(mgR.target) == R && len(mgC.target) == C {
				yr, yc = mgR.target, mgC.target
			}
			logf("      married axes: rows %s@%d r=%v, cols %s@%d r=%v", rk, t.layer, mgR.radix, ck, t.layer, mgC.radix)
		} else if IndexMode == "married" && MRHMode == "dynamic" && len(cs) > 1 {
			kr, kc := make([][]int, len(cs)), make([][]int, len(cs))
			for j, cd := range cs {
				kr[j], kc[j] = E[cd.ei].kr, E[cd.ei].kc
			}
			mr, _, err1 := MarriedCoords(kr, keyUniverse)
			mc, _, err2 := MarriedCoords(kc, keyUniverse)
			if err1 != nil || err2 != nil {
				die("%s: married index: %v %v", t.name, err1, err2)
			}
			cxr, cxc = mr, mc
		}
		logf("  [%d/%d] %s %v role=%s depth u=%.3f grid=%v contributors=%d", isyn, nsyn, t.name, t.shape, t.role, u, G, len(cs))
		t1 := time.Now()
		setInner("load+spline", float64(len(cs)), "")
		ents := make([]Entry, len(cs))
		// A new topology keeps every contributor's normalized tensor for its fine detail (Eq.22).
		newTopo := tp != nil && c.topology != "base" && c.residual
		dens, dshape := make([][]float32, len(cs)), make([][]int, len(cs))
		attnNotes := make([]string, len(cs))
		sem := make(chan struct{}, max(1, runtime.NumCPU()/2))
		errc := make(chan error, len(cs))
		var dn int
		fin := make(chan int, len(cs))
		for j, cd := range cs {
			sem <- struct{}{}
			go func(j int, cd cand) {
				defer func() { <-sem }()
				e := E[cd.ei]
				Wd, sh, err := all[e.src].LoadDense(e.name)
				if err != nil {
					errc <- err
					fin <- j
					return
				}
				scale := e.scale
				if w2, notes := tp.prepQK(e.src, all[e.src], e.name, e.role, Wd, sh); notes != nil {
					Wd, scale = w2, absmean(w2)
					attnNotes[j] = e.name + ": " + strings.Join(notes, ", ")
				}
				if s := scale; s > 0 {
					for i := range Wd {
						Wd[i] = float32(float64(Wd[i]) / s)
					}
				}
				ents[j] = Entry{W: cd.dw, Grid: splineGrid(Wd, sh[0], sh[1], cxr[j], cxc[j], G, 0), Bucket: buckets[j], IsBase: e.src == 0} // Eq.(10)
				if newTopo {
					dens[j], dshape[j] = Wd, sh
				}
				fin <- j
			}(j, cd)
		}
		for range cs {
			<-fin
			dn++
			innerSet(float64(dn), 0)
		}
		endInner()
		select {
		case err := <-errc:
			die("%v", err)
		default:
		}
		logf("      load+dequant+spline on index grid %v for %d contributors: %.2fs", G, len(cs), time.Since(t1).Seconds())
		for _, n := range attnNotes {
			if n != "" {
				logf("      attention: %s", n)
			}
		}
		grid, rec := synth(ents, G, c.sp)
		t2 := time.Now()
		inBase := baseST.Has(t.name) && eqShape(baseST.LogicalShape(t.name), t.shape)
		var baseScale float64 // the base tensor's absmean (after the QK-norm fold, if any)
		var xr, xc []float64
		var eb *entry
		if inBase {
			for _, e := range E {
				if e.src == 0 && e.name == t.name {
					eb = e
				}
			}
			if eb == nil {
				die("%s: base tensor has no index entry", t.name)
			}
			baseScale = eb.scale
			if (t.role == "q" || t.role == "k") && tp != nil && AttnOn && tp.metas[0].QKNorm {
				Bd, bsh, err := baseST.LoadDense(t.name)
				if err != nil {
					die("%s: %v", t.name, err)
				}
				if w2, notes := tp.prepQK(0, baseST, t.name, t.role, Bd, bsh); notes != nil {
					baseScale = absmean(w2)
				}
			}
			xr, xc = eb.xr, eb.xc
			if baseJ >= 0 { // the base's place on the married index
				xr, xc = cxr[baseJ], cxc[baseJ]
			}
		} else {
			xr, xc = make([]float64, R), make([]float64, C)
			for i := range xr {
				xr[i] = (float64(i) + .5) / float64(R)
			}
			for i := range xc {
				xc[i] = (float64(i) + .5) / float64(C)
			}
		}
		if yr != nil {
			xr, xc = yr, yc // the target's channels on the married axes
		}
		Wn := sampleGrid(grid, G, xr, xc)
		if newTopo {
			// Eq.(22) for a new topology: every contributor's detail finer than the index,
			// (I - P R) M_i, carried onto the target channels along the married axes and mixed
			// with the contributors' weights, so both models' detail lives in the new tensor.
			var wsum float64
			for _, cd := range cs {
				wsum += cd.dw
			}
			for j, cd := range cs {
				Ri, Ci := dshape[j][0], dshape[j][1]
				sm := sampleGrid(ents[j].Grid, G, cxr[j], cxc[j])
				det := make([]float64, Ri*Ci)
				for i := range det {
					det[i] = float64(dens[j][i]) - sm[i]
				}
				det = newAxisInterp(cxr[j], xr).rows(det, Ci)
				det = newAxisInterp(cxc[j], xc).cols(det, R, Ci)
				f := cd.dw / wsum
				for i := range Wn {
					Wn[i] += f * det[i]
				}
				dens[j] = nil
			}
		} else if inBase && c.residual {
			Bd, bsh, err := baseST.LoadDense(t.name)
			if err != nil {
				die("%s: %v", t.name, err)
			}
			if w2, notes := tp.prepQK(0, baseST, t.name, t.role, Bd, bsh); notes != nil {
				Bd, baseScale = w2, absmean(w2) // the base's QK-norm folded, as in its index entry
			}
			if s := baseScale; s > 0 {
				for i := range Bd {
					Bd[i] = float32(float64(Bd[i]) / s)
				}
			}
			sm := sampleGrid(splineGrid(Bd, R, C, xr, xc, G, 0), G, xr, xc) // Eq.(22) P R W_base
			for i := range Wn {
				Wn[i] += float64(Bd[i]) - sm[i]
			}
		}
		// Magnitude of the new tensor. A tensor the base has keeps the base's absmean: the base's norms,
		// residual stream and attention temperature are tuned to it, and a source's absmean belongs to
		// that source's own norm gains (Qwen3's are ~40x smaller than 2B4T's), so averaging them shrank
		// every BitLinear output (measured: weight_scale 0.27-0.39x the base, ppl 18 -> 60 from the
		// scales alone). Tensors the base lacks (--target-config) still take the weighted source mean.
		var scale float64
		if inBase {
			scale = baseScale
		} else if bs := baseScaleFor(E, t, tp); bs > 0 {
			scale = bs // new topology: the base's same-role tensor at the nearest depth
		} else {
			var num, den float64
			for _, cd := range cs {
				num += cd.dw * E[cd.ei].scale
				den += cd.dw
			}
			scale = num / den
		}
		Wf := make([]float32, len(Wn))
		for i := range Wn {
			Wf[i] = float32(Wn[i] * scale)
		}
		logf("      synth: index grid -> new tensor %v (sample at target channel coords, base residual=%v): %.2fs", t.shape, c.residual, time.Since(t2).Seconds())
		if PinMean && calib != nil && inBase && RT == nil {
			note, err := pinMean(t, Wf, baseST, calib)
			if err != nil {
				die("pin-mean %s: %v", t.name, err)
			}
			if note != "" {
				logf("      pin-mean %s", note)
			}
		}
		if RT != nil { // Eq.(24)
			RT.apply(t, Wf)
		}
		if MW != nil {
			MW.AddFloat(t.name, t.shape, Wf, c.FD)
		}
		t3 := time.Now()
		s := absmean(Wf) // Eq.(25): absmean picks the codes, RoundClip(W/s)
		q := make([]int8, len(Wf))
		var cnt [3]int
		var nzSum float64
		for i, v := range Wf {
			q[i] = int8(ternQ(float64(v), s))
			cnt[q[i]+1]++
			if q[i] != 0 {
				nzSum += math.Abs(float64(v))
			}
		}
		// weight_scale is the magnitude the runtime gives each nonzero code (bitlinear.wgsl: y = (W_t x) * ws[0]).
		// absmean s is the right threshold but the wrong magnitude for a tensor that was not trained with it:
		// re-ternarizing the base's own (t-1)*sc gives s = sc * (fraction nonzero), shrinking an unchanged
		// tensor by ~0.6. The least-squares magnitude for fixed codes, mean |W| over nonzero codes, gives
		// back sc exactly for an unchanged base tensor.
		ws := s
		if nz := cnt[0] + cnt[2]; nz > 0 {
			ws = nzSum / float64(nz)
		}
		if R%4 == 0 && s > 0 {
			W.Add(t.name, "U8", []int{R / 4, C}, packTrits(q, R, C))
			W.AddBF16(scaleKey(t.name), []int{1}, []float32{float32(ws)})
			cal.add(t, q, ws, W)
			ternarized = append(ternarized, t.name)
			// Body limbs follow --limbs (default 1: no Kuramoto body). The singular neck is
			// always coded on its own limbs (residualLimbs, NeckLimbs). That is not the
			// --limbs budget, and it does not re-lock.
			limbs, rel := residualLimbs(Wf, q, ws, MaxLimbs, LimbTol, R, C)
			if len(limbs) > 0 {
				writeLimbs(W, t.name, R, C, limbs)
				radices := make([]int, len(limbs))
				for i, lb := range limbs {
					radices[i] = lb.r
				}
				if MaxLimbs <= 1 {
					logf("      neck-limbs: %d rel %.3f -> %.3f (radices %v)", len(limbs), rel[0], rel[len(rel)-1], radices)
				} else {
					logf("      limbs: residual after each %.3f (radices beyond limb 0: %v)", rel, radices)
				}
				limbRep = append(limbRep, map[string]any{"tensor": t.name, "radices": radices, "residual": rel})
			}
			bws := 0.0
			if inBase && baseST.Has(scaleKey(t.name)) { // a float base (not packed) has no weight_scale
				if sc, err := baseST.F32(scaleKey(t.name)); err == nil && len(sc) == 1 {
					bws = float64(sc[0])
				}
			}
			logf("      ternarize (absmean threshold s=%.4g, -1/0/+1 = %d/%d/%d) + pack 4 trits/U8 + BF16 weight_scale %.4g (base %.4g): %.2fs",
				s, cnt[0], cnt[1], cnt[2], ws, bws, time.Since(t3).Seconds())
		} else {
			W.AddFloat(t.name, t.shape, Wf, c.FD)
			logf("      rows %d not divisible by 4: stored %s", R, c.FD)
		}
		{ // Sec.6: BNZ4 signed invertible channel-map chunk
			cm := &ChanMap{Name: t.name, Rout: uint32(R), Cout: uint32(C), Gr: uint32(G[0]), Gc: uint32(G[1]),
				ProR: hatWeights(xr, G[0]), ProC: hatWeights(xc, G[1])}
			for _, r := range gridRadices(G[0]) {
				cm.RadR = append(cm.RadR, uint8(r))
			}
			for _, r := range gridRadices(G[1]) {
				cm.RadC = append(cm.RadC, uint8(r))
			}
			for j, cd := range cs {
				sh := make([]uint8, len(rec.Shifts[j]))
				for i, x := range rec.Shifts[j] {
					sh[i] = uint8(x)
				}
				cm.C = append(cm.C, Contrib{Src: uint16(E[cd.ei].src), Bucket: uint8(buckets[j]), W: float32(cd.dw),
					Shift: sh, PR: signedPerm(rec.PermR[j]), PC: signedPerm(rec.PermC[j])})
			}
			S := NewSpectrum(G, 0)
			for i, h := range S.Harm {
				if h {
					cm.Bins = append(cm.Bins, uint32(i))
					cm.Re = append(cm.Re, float32(real(rec.DFT[i])))
					cm.Im = append(cm.Im, float32(imag(rec.DFT[i])))
				}
			}
			if d := rec.D; d != nil && d.Axis >= 0 && len(d.Modes) > 0 && len(d.Nu) == len(cs) {
				dt := &DtftChunk{Axis: uint8(d.Axis), LGap: float32(d.LGap), CPS: float32(d.CPS)}
				for _, b := range d.Modes {
					dt.Modes = append(dt.Modes, uint32(b))
				}
				nl := 0
				if len(d.Windings) > 0 {
					nl = len(d.Windings[0]) / len(d.Modes)
				}
				dt.NLoops = uint8(nl)
				for c := range cs {
					var nu, ph []uint16
					var qq [][]int8
					var pp [][]uint16
					for k := range d.Modes {
						nu = append(nu, bf16(float32(d.Nu[c][k])))
						ph = append(ph, uint16(int(math.Floor(4096*(d.Phi[c][k]-math.Floor(d.Phi[c][k]))))&4095))
						var q1 []int8
						var p1 []uint16
						for l := 0; l < nl; l++ {
							q1 = append(q1, int8(d.Windings[c][k*nl+l]))
							p1 = append(p1, bf16(float32(d.Periods[c][k*nl+l])))
						}
						qq, pp = append(qq, q1), append(pp, p1)
					}
					dt.Nu, dt.Phi, dt.Flags, dt.Q, dt.P = append(dt.Nu, nu), append(dt.Phi, ph), append(dt.Flags, d.Flags[c]), append(dt.Q, qq), append(dt.P, pp)
				}
				cm.DT = dt
			}
			blob := cm.Encode()
			if _, err := DecodeChanMap(blob); err != nil {
				die("%s: %v", t.name, err)
			}
			W.Add("bnzmix.bnz4."+t.name, "U8", []int{len(blob)}, blob)
			logf("      BNZ4 channel-map chunk: %d B (dense float32 permutation+weights would be %d B)", len(blob), 4*(len(cs)*(G[0]*G[0]+G[1]*G[1])))
		}
		sec := time.Since(tt).Seconds()
		tensorSecs = append(tensorSecs, sec)
		synthRep = append(synthRep, map[string]any{"tensor": t.name, "shape": t.shape, "grid": G, "n_contrib": len(cs),
			"passes": rec.Passes, "dtft": rec.D, "energy_grad_curl_harm": rec.Energy, "spectral_recon_err": rec.RecErr, "secs": sec})
		logf("      done %s in %.2fs", t.name, sec)
		outerSet(float64(isyn))
	}
	endOuter()
	aside := 0
	for _, st := range all {
		for _, k := range st.Keys {
			if asideTensor(k) {
				aside++
			}
		}
	}
	if aside > 0 {
		logf("TOWERS  %d encoder and per-layer tensors of the inputs are not written; the output's towers are the L-A towers", aside)
		report["not_written"] = aside
	}
	tw, err := writeTowers(W, baseST, basePath)
	if err != nil {
		die("towers: %v", err)
	}
	if tw != nil {
		report["towers"] = tw
	}
	var ent []int32
	for _, e := range E {
		ent = append(ent, int32(e.src), int32(e.shape[0]), int32(e.shape[1]), int32(e.bucket))
	}
	W.AddI32("bnzmix.index.entries", []int{len(E), 4}, ent)
	var ej [][]any
	for _, e := range E {
		ej = append(ej, []any{e.src, e.name, e.role, math.Round(e.u*1e4) / 1e4})
	}
	b, _ := json.Marshal(ej)
	W.Meta = append(W.Meta, [2]string{"bnzmix_entries", string(b)})
	// ---------- STAGE 4 ----------
	logf("STAGE 4/6  writing ternary safetensors %s", outTern)
	setInner("write", 1, "B")
	if err := W.Close(func(d, t int64) { innerSet(float64(d), float64(t)) }); err != nil {
		die("%v", err)
	}
	endInner()
	if MW != nil {
		logf("  writing merged safetensors %s", outMerged)
		setInner("write merged", 1, "B")
		if err := MW.Close(func(d, t int64) { innerSet(float64(d), float64(t)) }); err != nil {
			die("%v", err)
		}
		endInner()
	}
	of, err := copyAux(basePath, out, c.targetCfg, report)
	if err != nil {
		die("%v", err)
	}
	vocab := 0
	if baseST.Has("model.embed_tokens.weight") {
		vocab = baseST.LogicalShape("model.embed_tokens.weight")[0]
	}
	if U != nil {
		vocab = U.V
		od := filepath.Dir(absP(out))
		of.Tokenizer = filepath.Join(od, "tokenizer.json")
		if err := os.WriteFile(of.Tokenizer, U.tok, 0o644); err != nil {
			die("%v", err)
		}
		if U.tokCfg != nil {
			of.TokenizerConfig = filepath.Join(od, "tokenizer_config.json")
			if err := os.WriteFile(of.TokenizerConfig, U.tokCfg, 0o644); err != nil {
				die("%v", err)
			}
		}
		logf("  wrote the union tokenizer.json and tokenizer_config.json (%d tokens)", U.V)
	}
	if c.targetCfg == "" {
		cp := champConfigPath(out)
		var g modelMeta
		if tp != nil {
			g = tp.tgt
		} else {
			if g, err = metaOf(baseST); err != nil {
				die("output config.json: %v", err)
			}
			if g.Theta == 0 {
				g.Theta = SrcRopeTheta
			}
		}
		b, changed, err := configFor(g, cp, basePath, vocab)
		if err != nil {
			die("output config.json: %v", err)
		}
		if err := os.WriteFile(cp, b, 0o644); err != nil {
			die("%v", err)
		}
		logf("  wrote %s with the output's sizes (%s topology): %s", cp, c.topology, strings.Join(changed, ", "))
	}
	if n := tokenizerVocab(of.Tokenizer); n > vocab {
		die("the output's tokenizer.json has ids up to %d but its embedding has %d rows", n-1, vocab)
	}
	eval.SetFiles(outTern, of)
	if err := writeRunFile(out+".yaml", outTern, of); err != nil {
		die("%v", err)
	}
	if RT != nil {
		report["rotation"] = map[string]any{"H": RT.H, "block": RT.Q.B, "seed": c.rotSeed}
	}
	if OVMode == "chart" && baseOnly {
		g, err := metaOf(baseST)
		if err != nil {
			die("ov-chart: %v", err)
		}
		rep, err := ovChart(basePath, outTern, g.Heads, g.KV, g.HD)
		if err != nil {
			die("ov-chart: %v", err)
		}
		logf("OV CHART  %d layers: v_proj = the base's chart, o_proj = the first-order circuit (cross term removed: %.3g of ||O_b V_b||)", rep.Layers, rep.Cross)
		report["ov_chart"] = rep
	}
	if SurgeryOn && baseOnly {
		logf("SURGERY  singular set of the base forward -> base codes there (surgery.go)")
		rep, err := surgeryWith(calib, basePath, outTern, eval.FilesOf(basePath).Tokenizer)
		if err != nil {
			die("surgery: %v", err)
		}
		logSurgery(rep)
		report["surgery"] = rep
	}
	if RepairOn && baseOnly {
		logf("REPAIR  per-channel statistics back to the base's with the BF16 gains (repair.go), until mean |log2| spread is under %.3g", repairStop)
		rr, err := repair(basePath, outTern, eval.FilesOf(basePath).Tokenizer)
		if err != nil {
			die("repair: %v", err)
		}
		report["repair"] = rr
	}
	if ch, err := stampHodgeFile(champConfigPath(out)); err != nil {
		die("config names: %v", err)
	} else if len(ch) > 0 {
		logf("  config.json: %s", strings.Join(ch, ", "))
	}
	bad, notes := champCheck(outTern, champConfigPath(out))
	if len(bad) > 0 {
		die("the output does not run in Champ:\n  %s", strings.Join(bad, "\n  "))
	}
	logf("  Champ compatibility: OK")
	for _, n := range notes {
		logf("  note: %s", n)
	}
	report["champ_compatible"], report["champ_notes"] = true, notes
	bnzSrc := outTern
	if ok, err := baseLlama3(basePath); err != nil {
		die("llama3: %v", err)
	} else if ok {
		nLimbs, tol := llama3Budget(MaxLimbs, LimbTol)
		neck := strings.TrimSuffix(out, ".bnz") + ".neck.safetensors"
		logf("LLAMA3  neck pack (%d limbs, residual tol %.3f) -> %s", nLimbs, tol, neck)
		if err := packNeck(outTern, basePath, neck, nLimbs, tol); err != nil {
			die("neck: %v", err)
		}
		bad, notes = champCheck(neck, champConfigPath(out))
		if len(bad) > 0 {
			die("the neck pack does not run in Champ:\n  %s", strings.Join(bad, "\n  "))
		}
		for _, n := range notes {
			logf("  note: %s", n)
		}
		bnzSrc = neck
		report["neck"] = neck
	}
	// ---------- STAGE 5 ----------
	if c.noBNZ { // --bisect: only the ternary safetensors is scored
		logf("STAGE 5/6, 6/6  not run (--bisect scores the ternary safetensors only)")
		noVerify = true
	} else {
		logf("STAGE 5/6  BNZ compress -> %s", out)
		lf, err := os.Create(out + ".log.jsonl")
		if err != nil {
			die("%v", err)
		}
		setInner("bnz compress", 1, "B")
		n, err := bnz4.Compress(bnzSrc, out, lf)
		lf.Close()
		endInner()
		if err != nil {
			die("compress: %v", err)
		}
		ts := fsize(bnzSrc)
		logf("  %d B -> %d B (%.3f)", ts, n, float64(n)/float64(ts))
		report["bnz"], report["bnz_bytes"], report["ternary_bytes"] = out, n, ts
	}
	// ---------- STAGE 6 ----------
	if !noVerify {
		logf("STAGE 6/6  verify BNZ round-trip")
		rt := out + ".roundtrip.safetensors"
		setInner("bnz decompress", 1, "B")
		_, ok, err := bnz4.Decompress(out, rt)
		endInner()
		if err == nil && ok {
			ok = sameFile(rt, bnzSrc)
		}
		os.Remove(rt)
		report["roundtrip_ok"] = ok
		logf("  round-trip %s", map[bool]string{true: "OK", false: "FAILED"}[ok])
		if !ok {
			die("BNZ round-trip FAILED %v", err)
		}
	}
	// Every output gets its volume (volume.go), from the ternary weights the BNZ holds, beside it as
	// volume.safetensors: the name Champ's page reads from the model's folder, so the map is always
	// the served weights'.
	if !c.noBNZ {
		vol := filepath.Join(filepath.Dir(out), "volume.safetensors")
		if err := volumes(bnzSrc, vol); err != nil {
			die("volume: %v", err)
		}
		logf("  volume: %s", vol)
		report["volume"] = vol
	}
	if c.signSeed != "" && !c.noBNZ {
		if err := signBNZ4(c.signSeed, out); err != nil {
			die("sign: %v", err)
		}
		logf("  signed: %s.sig (Ed25519 over BNZ4 header+index)", out)
		report["signature"] = out + ".sig"
	}
	if c.evalTok != "" {
		logf("EVAL  base vs output: CPU reference forward, %d prompts, %d texts, and the page's first answer", len(eval.Prompts), len(eval.Texts))
		rb, err := eval.Run(basePath, c.evalTok)
		if err != nil {
			die("eval base: %v", err)
		}
		logScore("base", rb)
		ro, err := eval.Run(outTern, c.evalTok)
		if err != nil {
			die("eval output: %v", err)
		}
		logScore("output", ro)
		verdict := "not worse than base"
		if ro.MeanNLL > rb.MeanNLL {
			verdict = "WORSE than base"
		}
		logf("  output ppl %.1f vs base %.1f (x%.2f): %s", ro.PPL, rb.PPL, ro.PPL/rb.PPL, verdict)
		ev := map[string]any{"base": rb, "output": ro, "ppl_ratio": ro.PPL / rb.PPL, "verdict": verdict}
		// The page's first answer: its system prompt in chat format (~334 tokens), where the short texts
		// above do not reach (long-context failures show only here).
		cb, n, err := eval.Chat(basePath, c.evalTok, 24)
		if err != nil {
			die("eval base chat: %v", err)
		}
		co, _, err := eval.Chat(outTern, c.evalTok, 24)
		if err != nil {
			die("eval output chat: %v", err)
		}
		logf("  chat (page system prompt, %d tokens) %q: base %q, output %q", n, eval.ChatUser, cb, co)
		ev["chat"] = map[string]any{"tokens": n, "question": eval.ChatUser, "base": cb, "output": co}
		report["eval"] = ev
	}
	report["ternarized"], report["kept_base"], report["synth"], report["tensor_secs"] = ternarized, keptBase, synthRep, tensorSecs
	if limbRep != nil {
		report["limbs"] = limbRep
	}
	rp := out + ".report.json"
	rbs, err := json.MarshalIndent(jsonSafe(reflect.ValueOf(report)), "", " ")
	if err != nil {
		die("report: %v", err)
	}
	if err := os.WriteFile(rp, rbs, 0o644); err != nil {
		die("%v", err)
	}
	logf("DONE  %s  (%d ternarized tensors)  report: %s", out, len(ternarized), rp)
}

// Paper: no equation (helper: I/O, progress or utility)
func copyNonIndex(t target, base *ST, W, MW *STW, FD string, kept *[][2]string, RT *rotPlan) {
	if RT != nil && RT.handles(t) {
		a := RT.nonIndex(t, base)
		W.AddFloat(t.name, t.shape, a, FD)
		if MW != nil {
			MW.AddFloat(t.name, t.shape, a, FD)
		}
		*kept = append(*kept, [2]string{t.name, "rotation-folded (Eq.24)"})
		logf("  %s %v: rotation-folded (Eq.24) -> %s", t.name, t.shape, FD)
		return
	}
	if base.Has(t.name) && eqShape(base.LogicalShape(t.name), t.shape) {
		dt := base.H[t.name].Dtype
		if dt != "F32" && dt != "F16" && dt != "BF16" {
			raw := func(k string) []byte {
				b, err := base.Raw(k)
				if err != nil {
					die("%s: %v", k, err)
				}
				return b
			}
			W.Add(t.name, dt, base.H[t.name].Shape, raw(t.name))
			// A packed tensor left unmixed (--roles subset) needs its weight_scale too: the target
			// list leaves *_scale out because the mixing path writes it.
			if sk := scaleKey(t.name); base.Has(sk) {
				W.Add(sk, base.H[sk].Dtype, base.H[sk].Shape, raw(sk))
				for li := 1; base.Has(limbName(t.name, li)); li++ { // and its dynamic MRH limbs
					for _, ln := range []string{limbName(t.name, li), limbName(t.name, li) + "_scale", limbName(t.name, li) + "_radix"} {
						W.Add(ln, base.H[ln].Dtype, base.H[ln].Shape, raw(ln))
					}
				}
			}
			logf("  copy %s (%s) verbatim from base", t.name, dt)
			return
		}
		a, err := base.F32(t.name)
		if err != nil {
			die("%s: %v", t.name, err)
		}
		W.AddFloat(t.name, t.shape, a, FD)
		if MW != nil {
			MW.AddFloat(t.name, t.shape, a, FD)
		}
		*kept = append(*kept, [2]string{t.name, "copied/derived from base"})
		logf("  copy %s %v from base -> %s", t.name, t.shape, FD)
		return
	}
	var a []float32
	if t.role == "embed" && base.Has("model.embed_tokens.weight") {
		b, _ := base.F32("model.embed_tokens.weight")
		sh := base.H["model.embed_tokens.weight"].Shape
		a = resize2(b, sh[0], sh[1], t.shape[0], t.shape[1])
		logf("  derive %s %v: Whitney P0 (linear) prolongation of base embedding %v (Eq.10)", t.name, t.shape, sh)
	} else {
		a = make([]float32, numel(t.shape))
		if t.role == "norm" {
			for i := range a {
				a[i] = 1
			}
		}
		logf("  init %s %v (%s)", t.name, t.shape, map[bool]string{true: "ones", false: "zeros"}[t.role == "norm"])
	}
	W.AddFloat(t.name, t.shape, a, FD)
	if MW != nil {
		MW.AddFloat(t.name, t.shape, a, FD)
	}
	*kept = append(*kept, [2]string{t.name, "copied/derived from base"})
}

// Paper: no equation (helper: I/O, progress or utility)
func resize2(a []float32, R, C, nR, nC int) []float32 { // HM_resize along axis 0 then 1 (np.interp)
	xp := func(m int) []float64 {
		o := make([]float64, m)
		for i := range o {
			o[i] = float64(i)
		}
		return o
	}
	x := func(n, m int) []float64 {
		o := make([]float64, n)
		for i := range o {
			o[i] = (float64(i)+.5)/float64(n)*float64(m) - .5
		}
		return o
	}
	b := a
	if nR != R {
		b = make([]float32, nR*C)
		xs, p := x(nR, R), xp(R)
		col := make([]float64, R)
		for c := 0; c < C; c++ {
			for r := 0; r < R; r++ {
				col[r] = float64(a[r*C+c])
			}
			for i, v := range xs {
				b[i*C+c] = float32(interp(v, p, col))
			}
		}
	}
	if nC == C {
		return b
	}
	o := make([]float32, nR*nC)
	xs, p := x(nC, C), xp(C)
	row := make([]float64, C)
	for r := 0; r < nR; r++ {
		for c := 0; c < C; c++ {
			row[c] = float64(b[r*C+c])
		}
		for i, v := range xs {
			o[r*nC+i] = float32(interp(v, p, row))
		}
	}
	return o
}

// Paper: no equation (helper: I/O, progress or utility)
func targetsFromConfig(p string) ([]target, int) {
	b, err := os.ReadFile(p)
	if err != nil {
		die("%v", err)
	}
	var all map[string]any
	if err := json.Unmarshal(b, &all); err != nil {
		die("%s: %v", p, err)
	}
	cfg := map[string]float64{}
	for k, v := range all {
		if f, ok := v.(float64); ok {
			cfg[k] = f
		}
	}
	H, I, L := int(cfg["hidden_size"]), int(cfg["intermediate_size"]), int(cfg["num_hidden_layers"])
	nh := 1
	if v, ok := cfg["num_attention_heads"]; ok {
		nh = int(v)
	}
	kvh := nh
	if v, ok := cfg["num_key_value_heads"]; ok {
		kvh = int(v)
	}
	kv := kvh * (H / nh)
	V := 32
	if v, ok := cfg["vocab_size"]; ok {
		V = int(v)
	}
	T := []target{{"model.embed_tokens.weight", "embed", -1, []int{V, H}}}
	for l := 0; l < L; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		T = append(T, target{p + "input_layernorm.weight", "norm", l, []int{H}}, target{p + "post_attention_layernorm.weight", "norm", l, []int{H}},
			target{p + "self_attn.attn_sub_norm.weight", "norm", l, []int{H}}, target{p + "mlp.ffn_sub_norm.weight", "norm", l, []int{I}})
		for _, x := range []struct {
			n, r string
			s    []int
		}{{"self_attn.q_proj", "q", []int{H, H}}, {"self_attn.k_proj", "k", []int{kv, H}}, {"self_attn.v_proj", "v", []int{kv, H}},
			{"self_attn.o_proj", "o", []int{H, H}}, {"mlp.gate_proj", "gate", []int{I, H}}, {"mlp.up_proj", "up", []int{I, H}}, {"mlp.down_proj", "down", []int{H, I}}} {
			T = append(T, target{p + x.n + ".weight", x.r, l, x.s})
		}
	}
	T = append(T, target{"model.norm.weight", "norm", -1, []int{H}})
	return T, L
}

// copyAux copies the base's files set in the run file beside the output under their standard names and
// returns them as the output's files (the output's config.json is always configFor's).
func copyAux(basePath, out, tcfg string, report map[string]any) (eval.ModelFiles, error) {
	bf := eval.FilesOf(basePath)
	od := filepath.Dir(absP(out))
	of := eval.ModelFiles{Config: champConfigPath(out)}
	var copied []string
	for _, x := range []struct {
		src, name string
		dst       *string
	}{{bf.Config, "config.json", new(string)}, {bf.GenerationConfig, "generation_config.json", &of.GenerationConfig},
		{bf.Tokenizer, "tokenizer.json", &of.Tokenizer}, {bf.TokenizerConfig, "tokenizer_config.json", &of.TokenizerConfig},
		{bf.SpecialTokensMap, "special_tokens_map.json", &of.SpecialTokensMap}} {
		if x.src == "" || (x.name == "config.json" && tcfg != "") {
			continue
		}
		dp := filepath.Join(od, x.name)
		if absP(x.src) != absP(dp) {
			if err := copyFile(x.src, dp); err != nil {
				return of, err
			}
		}
		*x.dst = dp
		copied = append(copied, x.src+" -> "+x.name)
	}
	if tcfg != "" {
		if err := copyFile(tcfg, champConfigPath(out)); err != nil {
			return of, err
		}
		copied = append(copied, tcfg+" -> config.json (target)")
	}
	if len(copied) > 0 {
		logf("  copied %v next to output", copied)
	}
	report["copied"] = copied
	return of, nil
}

// Paper: no equation (helper: I/O, progress or utility)
func copyFile(a, b string) error {
	s, err := os.Open(a)
	if err != nil {
		return err
	}
	defer s.Close()
	d, err := os.Create(b)
	if err != nil {
		return err
	}
	if _, err := io.Copy(d, s); err != nil {
		d.Close()
		return err
	}
	return d.Close()
}

// Paper: no equation (helper: I/O, progress or utility)
func sameFile(a, b string) bool {
	fa, e1 := os.Open(a)
	fb, e2 := os.Open(b)
	if e1 != nil || e2 != nil {
		return false
	}
	defer fa.Close()
	defer fb.Close()
	ba, bb := make([]byte, 1<<22), make([]byte, 1<<22)
	for {
		na, ea := io.ReadFull(fa, ba)
		nb, _ := io.ReadFull(fb, bb)
		if na != nb || string(ba[:na]) != string(bb[:nb]) {
			return false
		}
		if ea != nil {
			return true
		}
	}
}

// Paper: no equation (helper: I/O, progress or utility)
func setMeta(m [][2]string, k, v string) [][2]string {
	for i := range m {
		if m[i][0] == k {
			m[i][1] = v
			return m
		}
	}
	return append(m, [2]string{k, v})
}

// Paper: no equation (helper: I/O, progress or utility)
func srcPaths(s []*ST) []string {
	var o []string
	for _, x := range s {
		o = append(o, x.Path)
	}
	return o
}

// Paper: no equation (helper: I/O, progress or utility)
func absP(p string) string { a, _ := filepath.Abs(p); return a }

// Paper: no equation (helper: I/O, progress or utility)
func fsize(p string) int64 {
	st, err := os.Stat(p)
	if err != nil {
		return 0
	}
	return st.Size()
}

// Paper: no equation (helper: I/O, progress or utility)
func eqShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Paper: no equation (helper: I/O, progress or utility)
func f32s(v []float64) []float32 {
	o := make([]float32, len(v))
	for i, x := range v {
		o[i] = float32(x)
	}
	return o
}

// Paper: no equation (helper: I/O, progress or utility)
func absC(c complex128) float64 { return math.Hypot(real(c), imag(c)) }

var _ = binary.LittleEndian

// Paper_v2 Eq.(11): depth weight = Whitney hat in normalized depth with the source's layer spacing
func depthHat(us, ut float64, L int) float64 {
	if L <= 1 {
		return 1
	}
	// Adjacent layers of an equal-depth model sit exactly on the hat's zero, but l/(L-1)*(L-1) rounds to
	// 1 +- 1 ulp, which left a ~1e-16 weight on both neighbours. Eq.(21) divides by the weight of the
	// agreeing contributors only, so where the real contributors' deltas are zero such a neighbour sets
	// the whole delta: a base mixed with itself changed 10 % of its codes (ppl 12.0 -> 16.5).
	v := 1 - math.Abs(us-ut)*float64(L-1)
	if v < 1e-9 {
		return 0
	}
	return v
}

// jsonSafe is v with every NaN/±Inf float as the string "NaN"/"+Inf"/"-Inf" (encoding/json refuses
// them, which left the whole report empty); structs become maps by their json tags.
func jsonSafe(v reflect.Value) any {
	switch v.Kind() {
	case reflect.Invalid:
		return nil
	case reflect.Interface, reflect.Pointer:
		if v.IsNil() {
			return nil
		}
		return jsonSafe(v.Elem())
	case reflect.Float32, reflect.Float64:
		f := v.Float()
		if math.IsNaN(f) || math.IsInf(f, 0) {
			return strconv.FormatFloat(f, 'g', -1, 64)
		}
		return f
	case reflect.Map:
		m := make(map[string]any, v.Len())
		for it := v.MapRange(); it.Next(); {
			m[fmt.Sprint(it.Key().Interface())] = jsonSafe(it.Value())
		}
		return m
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			return nil
		}
		s := make([]any, v.Len())
		for i := range s {
			s[i] = jsonSafe(v.Index(i))
		}
		return s
	case reflect.Struct:
		m := map[string]any{}
		for i := 0; i < v.NumField(); i++ {
			f := v.Type().Field(i)
			if !f.IsExported() {
				continue
			}
			name, _, _ := strings.Cut(f.Tag.Get("json"), ",")
			if name == "-" {
				continue
			}
			if name == "" {
				name = f.Name
			}
			m[name] = jsonSafe(v.Field(i))
		}
		return m
	}
	return v.Interface()
}

// metaFromConfigOrTensors is a model's topology for the CLI passes (--ov-chart).
func metaFromConfigOrTensors(p string) (modelMeta, error) {
	st, err := OpenST(p)
	if err != nil {
		return modelMeta{}, err
	}
	return metaOf(st)
}

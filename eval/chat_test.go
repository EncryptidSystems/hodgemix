package eval

import (
	"math"
	"testing"
)

func TestPromptFormats(t *testing.T) {
	role := &tok{eot: "<|eot_id|>", special: map[string]int{"<|eot_id|>": 1}}
	if got, want := role.promptTurn("SYS", "Hi"), "System: SYS<|eot_id|>User: Hi<|eot_id|>Assistant: "; got != want {
		t.Fatalf("role %q", got)
	}
	llama := &tok{eot: "<|eot_id|>", special: map[string]int{
		"<|start_header_id|>": 1, "<|end_header_id|>": 2, "<|eot_id|>": 3,
	}}
	want := "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	if got := llama.promptTurn("You are helpful.", "Hi"); got != want {
		t.Fatalf("llama3 %q", got)
	}
	qwen := &tok{eot: "<|im_end|>", special: map[string]int{"<|im_start|>": 1, "<|im_end|>": 2}}
	want = "<|im_start|>system\n" + ImageSystem + "<|im_end|>\n<|im_start|>user\na red cube<|im_end|>\n<|im_start|>assistant\n"
	if got := qwen.promptTurn(ImageSystem, "a red cube"); got != want {
		t.Fatalf("qwen %q", got)
	}
	if qwen.format() != fmtQwen || llama.format() != fmtLlama3 || role.format() != fmtRole {
		t.Fatalf("formats qwen %v llama %v role %v", qwen.format(), llama.format(), role.format())
	}
}

func TestRopeParameters(t *testing.T) {
	params := map[string]any{"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
		"high_freq_factor": 4.0, "original_max_position_embeddings": 8192.0}
	a, _, err := ropeFreqs(map[string]any{"rope_parameters": params}, 500000, 128)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := ropeFreqs(map[string]any{"rope_scaling": params}, 500000, 128)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != len(b) || a[0] != b[0] || a[63] != b[63] {
		t.Fatalf("rope_parameters %v rope_scaling %v", a[63], b[63])
	}
	only := map[string]any{"rope_parameters": map[string]any{"rope_type": "llama3", "rope_theta": 500000.0, "factor": 32.0,
		"low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192.0}}
	c, _, err := ropeFreqs(only, 10000, 64)
	if err != nil {
		t.Fatal(err)
	}
	d, _, err := ropeFreqs(map[string]any{"rope_scaling": params}, 500000, 64)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(c[31]-d[31]) > 1e-12 {
		t.Fatalf("theta from rope_parameters %v want %v", c[31], d[31])
	}
}

func TestMropeAccepted(t *testing.T) {
	cfg := map[string]any{"rope_scaling": map[string]any{"type": "mrope", "mrope_section": []any{16.0, 24.0, 24.0}}}
	f, scale, err := ropeFreqs(cfg, 1000000, 128)
	if err != nil {
		t.Fatal(err)
	}
	if scale != 1 || f[0] != 1 {
		t.Fatalf("mrope f0 %v scale %v", f[0], scale)
	}
	q, err := Classify(map[string]any{"model_type": "qwen3_vl", "num_hidden_layers": 2.0, "image_token_id": 151655.0}, 2)
	if err != nil || q.Family != FamQwen {
		t.Fatalf("qwen3_vl %+v %v", q, err)
	}
}

func TestEosList(t *testing.T) {
	inv := map[int]string{128001: "<|end_of_text|>", 128008: "<|eom_id|>", 128009: "<|eot_id|>"}
	special := map[string]int{"<|eot_id|>": 128009, "<|eom_id|>": 128008, "<|end_of_text|>": 128001}
	got := eosText([]byte(`[128001,128008,128009]`), special, inv)
	if got != "<|eot_id|>" {
		t.Fatalf("eos list %q", got)
	}
}

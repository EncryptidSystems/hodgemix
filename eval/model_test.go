package eval

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The BOS comes from the tokenizer's own post-processor: a llama3 template gives <|begin_of_text|>,
// a plain ByteLevel post-processor gives none.
func TestTemplateBOS(t *testing.T) {
	llama := `{"type":"Sequence","processors":[{"type":"ByteLevel"},{"type":"TemplateProcessing",
		"single":[{"SpecialToken":{"id":"<|begin_of_text|>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
		"special_tokens":{"<|begin_of_text|>":{"id":"<|begin_of_text|>","ids":[128000]}}}]}`
	if b := templateBOS(json.RawMessage(llama)); len(b) != 1 || b[0] != 128000 {
		t.Fatalf("llama3 template: %v", b)
	}
	if b := templateBOS(json.RawMessage(`{"type":"ByteLevel"}`)); len(b) != 0 {
		t.Fatalf("ByteLevel: %v", b)
	}
}

func TestUnitGain(t *testing.T) {
	if !unitGain([]float32{1, 1, 1}) {
		t.Fatal("ones")
	}
	if unitGain(nil) || unitGain([]float32{1, 0.5}) {
		t.Fatal("not a unit gain")
	}
}

// llama3 rope_scaling: high frequencies kept, low ones divided by the factor, the band between blended.
func TestRopeLlama3(t *testing.T) {
	cfg := map[string]any{"rope_scaling": map[string]any{"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
		"high_freq_factor": 4.0, "original_max_position_embeddings": 8192.0}}
	f, scale, err := ropeFreqs(cfg, 500000, 128)
	if err != nil {
		t.Fatal(err)
	}
	if scale != 1 || f[0] != 1 {
		t.Fatalf("f[0]=%v scale=%v", f[0], scale)
	}
	last := math.Pow(500000, -126.0/128)
	if math.Abs(f[63]-last/32) > 1e-18 {
		t.Fatalf("f[63]=%v want %v", f[63], last/32)
	}
	if _, _, err := ropeFreqs(map[string]any{"rope_scaling": map[string]any{"rope_type": "yarn"}}, 1e4, 64); err == nil {
		t.Fatal("yarn without factor accepted")
	}
}

// The config read is the one set; another model's config is refused, and none set is an error.
func TestConfigSet(t *testing.T) {
	dir := t.TempDir()
	bitnet, llama := filepath.Join(dir, "config.json"), filepath.Join(dir, "config (2).json")
	os.WriteFile(bitnet, []byte(`{"hidden_size":2560,"num_attention_heads":20,"num_key_value_heads":5}`), 0o644)
	os.WriteFile(llama, []byte(`{"hidden_size":2048,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":64,"num_hidden_layers":16}`), 0o644)
	sh := Shape{D: 2048, L: 16, F: 8192}
	if _, err := config(llama, sh, 2048, 512); err != nil {
		t.Fatal(err)
	}
	if _, err := config(bitnet, sh, 2048, 512); err == nil {
		t.Fatal("another model's config accepted")
	}
	if _, err := config("", sh, 2048, 512); err == nil {
		t.Fatal("no config set, none should be read")
	}
}

func TestF16(t *testing.T) {
	for h, want := range map[uint16]float32{0x3c00: 1, 0xc000: -2, 0x7bff: 65504, 0x0001: 5.960464477539063e-08, 0x8000: 0} {
		if g := f16(h); g != want {
			t.Fatalf("f16(%#x)=%v want %v", h, g, want)
		}
	}
}

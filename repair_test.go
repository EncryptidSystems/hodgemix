package main

import (
	"math"
	"testing"
)

// Three successive clamps of one binade walk a unit gain to 1/8. The base tap is the
// raw sublayer; the output tap is an RMSNorm. The ratio is not a shared gain, and
// repair must leave the inserted gain at 1. The repair loop itself has no pass count.
func TestRepairLeavesInsertedSubnorm(t *testing.T) {
	g := 1.0
	for i := 0; i < 3; i++ {
		g *= repairRatio(0.02, 1)
	}
	if math.Abs(g-0.125) > 1e-12 {
		t.Fatalf("three clamps walk a unit gain to %g, want 0.125", g)
	}
	base := &ST{H: map[string]TInfo{
		"model.layers.0.input_layernorm.weight":          {},
		"model.layers.0.post_attention_layernorm.weight": {},
		"model.norm.weight":                              {},
	}}
	for _, n := range []string{
		"model.layers.0.input_layernorm.weight",
		"model.layers.0.post_attention_layernorm.weight",
		"model.norm.weight",
	} {
		if !baseOwnsGain(base, n) {
			t.Fatalf("repair would skip %s, which the base stores", n)
		}
	}
	for _, n := range []string{
		"model.layers.0.self_attn.attn_sub_norm.weight",
		"model.layers.0.mlp.ffn_sub_norm.weight",
	} {
		if baseOwnsGain(base, n) {
			t.Fatalf("repair would scale %s, which Champ inserted", n)
		}
	}
	vg := &ST{H: map[string]TInfo{
		"model.layers.0.pre_feedforward_layernorm.weight": {},
	}}
	if got := normTensor(vg, "0/in_ffn"); got != "model.layers.0.pre_feedforward_layernorm.weight" {
		t.Fatalf("vaultgemma FFN pre-norm %s", got)
	}
	if got := normTensor(base, "0/in_ffn"); got != "model.layers.0.post_attention_layernorm.weight" {
		t.Fatalf("llama FFN pre-norm %s", got)
	}
}

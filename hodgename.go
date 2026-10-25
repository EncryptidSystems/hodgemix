package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// The output's config.json names the Hodge block, whatever the base called it. The base
// may be a ternary 2B4T checkpoint (model_type "bitnet"); hkruntime reads only these names.
const (
	hodgeModelType = "hodge"
	hodgeArch      = "HodgeForCausalLM"
	hodgeQuant     = "hodge"
	hodgeLinear    = "hodgelinear"
)

// ternaryFamily is a model_type of the ternary 2B4T block: the base's name or the output's.
func ternaryFamily(mt string) bool {
	return mt == "bitnet" || mt == hodgeModelType
}

// stampHodge renames the 2B4T block's model_type, architectures and quantization_config to
// the Hodge names, and drops an auto_map that points at another library's classes.
func stampHodge(cfg map[string]any) []string {
	var changed []string
	mt, _ := cfg["model_type"].(string)
	if mt == "bitnet" {
		cfg["model_type"] = hodgeModelType
		changed = append(changed, "model_type bitnet -> "+hodgeModelType)
	}
	if ternaryFamily(mt) {
		cfg["architectures"] = []any{hodgeArch}
		changed = append(changed, "architectures = "+hodgeArch)
	}
	if q, ok := cfg["quantization_config"].(map[string]any); ok && q != nil {
		if q["quant_method"] == "bitnet" || q["linear_class"] == "autobitlinear" {
			q["quant_method"], q["linear_class"] = hodgeQuant, hodgeLinear
			changed = append(changed, "quantization_config = "+hodgeQuant+" / "+hodgeLinear)
		}
	}
	if am, ok := cfg["auto_map"].(map[string]any); ok {
		for _, v := range am {
			if s, _ := v.(string); strings.Contains(strings.ToLower(s), "bitnet") {
				delete(cfg, "auto_map")
				changed = append(changed, "auto_map dropped")
				break
			}
		}
	}
	return changed
}

// stampHodgeFile applies stampHodge to a config.json in place.
func stampHodgeFile(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	cfg := map[string]any{}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	changed := stampHodge(cfg)
	if len(changed) == 0 {
		return nil, nil
	}
	out, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return nil, err
	}
	return changed, os.WriteFile(path, out, 0o644)
}

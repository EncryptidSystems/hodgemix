package main

// Run file: hodgemix --config RUN.yaml. Every model's files are set here (nothing is looked up beside a model),
// with the output and the run's settings:
//
//	base:
//	  model: C:\models\Llama-3.2-1B\model.safetensors   # or a list: one model's shards
//	  tokenizer: C:\models\Llama-3.2-1B\tokenizer.json
//	  tokenizer_config: C:\models\Llama-3.2-1B\tokenizer_config.json
//	  config: C:\models\Llama-3.2-1B\config.json
//	  special_tokens_map: ...                           # optional, copied beside the output
//	  generation_config: ...                            # optional, copied beside the output
//	sources:
//	  - model: C:\models\Llama-3.2-3B\model-00001-of-00002.safetensors
//	    weight: 1                                       # optional, default 1
//	    tokenizer: ...                                  # the union vocabulary reads it
//	    config: ...
//	out: C:\models\hodge-1.bnz
//	settings:                                           # optional: any option by name
//	  limbs: 1
//	models:                                             # optional: other models' files (an output's),
//	  - model: ...                                      # for the probe commands
//
// hodgemix --config RUN.yaml --score MODEL (or --chat, --converse, --trace, ...) runs a probe command with the
// files the run file sets for MODEL. A run writes OUT.yaml with its output's files.

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"hodgemix/eval"

	"gopkg.in/yaml.v3"
)

type runEntry struct {
	Model            any      `yaml:"model"`
	Weight           *float64 `yaml:"weight,omitempty"`
	Tokenizer        string   `yaml:"tokenizer,omitempty"`
	TokenizerConfig  string   `yaml:"tokenizer_config,omitempty"`
	Config           string   `yaml:"config,omitempty"`
	SpecialTokensMap string   `yaml:"special_tokens_map,omitempty"`
	GenerationConfig string   `yaml:"generation_config,omitempty"`
}

type runFile struct {
	Base     *runEntry      `yaml:"base,omitempty"`
	Sources  []runEntry     `yaml:"sources,omitempty"`
	Out      string         `yaml:"out,omitempty"`
	Settings map[string]any `yaml:"settings,omitempty"`
	Models   []runEntry     `yaml:"models,omitempty"`
}

// register records e's files (paths relative to dir) and returns its model path (its first file).
func (e *runEntry) register(dir, what string) (string, error) {
	abs := func(p string) string {
		if p == "" || filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(dir, p)
	}
	var shards []string
	switch m := e.Model.(type) {
	case string:
		shards = []string{m}
	case []any:
		for _, x := range m {
			s, ok := x.(string)
			if !ok {
				return "", fmt.Errorf("%s: model: every shard is a path", what)
			}
			shards = append(shards, s)
		}
	}
	if len(shards) == 0 || shards[0] == "" {
		return "", fmt.Errorf("%s: no model", what)
	}
	for i := range shards {
		shards[i] = abs(shards[i])
	}
	f := eval.ModelFiles{Tokenizer: abs(e.Tokenizer), TokenizerConfig: abs(e.TokenizerConfig), Config: abs(e.Config),
		SpecialTokensMap: abs(e.SpecialTokensMap), GenerationConfig: abs(e.GenerationConfig)}
	if len(shards) > 1 {
		f.Shards = shards
	}
	for _, p := range append(append([]string{}, shards...), f.Tokenizer, f.TokenizerConfig, f.Config, f.SpecialTokensMap, f.GenerationConfig) {
		if p == "" {
			continue
		}
		if _, err := os.Stat(p); err != nil {
			return "", fmt.Errorf("%s: %v", what, err)
		}
	}
	eval.SetFiles(shards[0], f)
	return shards[0], nil
}

// runBase is the base checkpoint recorded by the last --config, for --neck.
var runBase string

// loadRunFile reads a run file, records every model's files, and returns the run's options.
func loadRunFile(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rf runFile
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	if err := dec.Decode(&rf); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	dir := filepath.Dir(absP(path))
	var args []string
	if rf.Base != nil {
		m, err := rf.Base.register(dir, "base")
		if err != nil {
			return nil, fmt.Errorf("%s: %v", path, err)
		}
		args = append(args, "--base", m)
		runBase = m
	}
	for i := range rf.Sources {
		m, err := rf.Sources[i].register(dir, fmt.Sprintf("sources[%d]", i))
		if err != nil {
			return nil, fmt.Errorf("%s: %v", path, err)
		}
		if w := rf.Sources[i].Weight; w != nil {
			m += fmt.Sprintf(":%g", *w)
		}
		args = append(args, "--src", m)
	}
	for i := range rf.Models {
		if _, err := rf.Models[i].register(dir, fmt.Sprintf("models[%d]", i)); err != nil {
			return nil, fmt.Errorf("%s: %v", path, err)
		}
	}
	if rf.Out != "" {
		o := rf.Out
		if !filepath.IsAbs(o) {
			o = filepath.Join(dir, o)
		}
		args = append(args, "--out", o)
	}
	keys := make([]string, 0, len(rf.Settings))
	for k := range rf.Settings {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		args = append(args, "--"+strings.TrimLeft(k, "-")+"="+fmt.Sprint(rf.Settings[k]))
	}
	return args, nil
}

// writeRunFile writes path, a run file whose models entry is the output's files, for the probe
// commands (hodgemix --config OUT.yaml --score OUT.ternary.safetensors).
func writeRunFile(path, model string, f eval.ModelFiles) error {
	b, err := yaml.Marshal(runFile{Models: []runEntry{{Model: model, Tokenizer: f.Tokenizer, TokenizerConfig: f.TokenizerConfig,
		Config: f.Config, SpecialTokensMap: f.SpecialTokensMap, GenerationConfig: f.GenerationConfig}}})
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

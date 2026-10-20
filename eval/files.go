package eval

import "path/filepath"

// ModelFiles is a model's files, exactly as a run file (hodgemix run.yaml) sets them. Nothing is
// looked up beside a model: a file not set is not read.
type ModelFiles struct {
	Shards           []string // the safetensors files, when the model is more than one
	Tokenizer        string
	TokenizerConfig  string
	Config           string
	SpecialTokensMap string
	GenerationConfig string
}

var given = map[string]ModelFiles{}

// updates is an optional second feed-forward, admitted per token by the cochain.
var updates = map[string]string{}

func key(model string) string {
	if a, err := filepath.Abs(model); err == nil {
		return filepath.Clean(a)
	}
	return filepath.Clean(model)
}

// SetFiles records model's files (model is its first safetensors file).
func SetFiles(model string, f ModelFiles) { given[key(model)] = f }

// SetUpdate attaches an improvement checkpoint to model. Its feed-forward is a second expert
// on the same cells. The forward admits it on a token only when it keeps the base's residual cell.
func SetUpdate(model, update string) {
	if update == "" {
		delete(updates, key(model))
		return
	}
	updates[key(model)] = update
}

// UpdateOf is the improvement checkpoint set for model, or empty.
func UpdateOf(model string) string { return updates[key(model)] }

// FilesOf is the files set for model; empty fields were not set.
func FilesOf(model string) ModelFiles { return given[key(model)] }

// ShardsOf is model's safetensors files: the shards set for it, else model itself.
func ShardsOf(model string) []string {
	if s := given[key(model)].Shards; len(s) > 0 {
		return s
	}
	return []string{model}
}

package eval

import "strings"

// Chat turns follow the checkpoint's own tokens, the same formats hkruntime/chatfmt renders.
// A Llama 3.2 Instruct file is the header turn. A Qwen file, including the Qwen-Image text
// encoder, is the im_start / im_end turn. The 2B4T page stays "Role: content" plus the turn end.

const roleTemplate = `{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['role'] | capitalize + ': '+ message['content'] | trim + '<|eot_id|>' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}`

// ImageSystem is the system line Qwen-Image (and its turbo checkpoint) puts in front of a prompt.
// The text encoder was trained on this string, not on a chat assistant prompt.
const ImageSystem = "Comprehend and analyze the provided prompt."

type chatFmt int

const (
	fmtRole chatFmt = iota
	fmtLlama3
	fmtGemma
	fmtQwen
	fmtHarmony
	fmtPlain
)

func (t *tok) has(s string) bool {
	_, ok := t.special[s]
	return ok
}

func (t *tok) format() chatFmt {
	if t == nil {
		return fmtPlain
	}
	if t.template == roleTemplate {
		return fmtRole
	}
	if t.has("<|start_header_id|>") && t.has("<|end_header_id|>") && t.has("<|eot_id|>") {
		return fmtLlama3
	}
	if t.has("<start_of_turn>") && t.has("<end_of_turn>") {
		return fmtGemma
	}
	if t.has("<|im_start|>") && t.has("<|im_end|>") {
		return fmtQwen
	}
	if t.has("<|start|>") && t.has("<|message|>") && t.has("<|channel|>") {
		return fmtHarmony
	}
	if t.has("<|eot_id|>") || t.eot != "" {
		return fmtRole
	}
	return fmtPlain
}

// systemPrefix is the system turn fed once and shared. Gemma folds it into the first user turn.
func (t *tok) systemPrefix(system string) string {
	system = strings.TrimSpace(system)
	if system == "" {
		return ""
	}
	switch t.format() {
	case fmtLlama3:
		return "<|start_header_id|>system<|end_header_id|>\n\n" + system + "<|eot_id|>"
	case fmtQwen:
		return "<|im_start|>system\n" + system + "<|im_end|>\n"
	case fmtHarmony:
		return "<|start|>system<|message|>" + system + "<|end|>"
	case fmtGemma, fmtPlain:
		return ""
	default:
		return "System: " + system + t.eot
	}
}

// promptTurn is one user turn plus the assistant prefix the model continues.
// system is included only when this format has no separate system prefix (Gemma, plain).
func (t *tok) promptTurn(system, user string) string {
	system = strings.TrimSpace(system)
	user = strings.TrimSpace(user)
	switch t.format() {
	case fmtLlama3:
		var b strings.Builder
		if system != "" {
			b.WriteString(t.systemPrefix(system))
		}
		b.WriteString("<|start_header_id|>user<|end_header_id|>\n\n")
		b.WriteString(user)
		b.WriteString("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
		return b.String()
	case fmtQwen:
		var b strings.Builder
		if system != "" {
			b.WriteString(t.systemPrefix(system))
		}
		b.WriteString("<|im_start|>user\n")
		b.WriteString(user)
		b.WriteString("<|im_end|>\n<|im_start|>assistant\n")
		return b.String()
	case fmtGemma:
		var b strings.Builder
		b.WriteString("<start_of_turn>user\n")
		if system != "" {
			b.WriteString(system)
			b.WriteString("\n\n")
		}
		b.WriteString(user)
		b.WriteString("<end_of_turn>\n<start_of_turn>model\n")
		return b.String()
	case fmtHarmony:
		var b strings.Builder
		if system != "" {
			b.WriteString(t.systemPrefix(system))
		}
		b.WriteString("<|start|>user<|message|>")
		b.WriteString(user)
		b.WriteString("<|end|><|start|>assistant<|channel|>final<|message|>")
		return b.String()
	case fmtPlain:
		if system != "" {
			return system + "\n\n" + user + "\n"
		}
		return user + "\n"
	default:
		s := ""
		if system != "" {
			s = t.systemPrefix(system)
		}
		return s + "User: " + user + t.eot + "Assistant: "
	}
}

// closeTurn is the text that ends an assistant turn the model did not close itself.
func (t *tok) closeTurn() string {
	switch t.format() {
	case fmtLlama3:
		return "<|eot_id|>"
	case fmtQwen:
		return "<|im_end|>\n"
	case fmtGemma:
		return "<end_of_turn>\n"
	case fmtHarmony:
		return "<|return|>"
	case fmtPlain:
		return ""
	default:
		return t.eot
	}
}

func (t *tok) ended(id int) bool {
	if t == nil {
		return false
	}
	switch t.format() {
	case fmtLlama3, fmtRole:
		if id == t.special["<|eot_id|>"] && t.has("<|eot_id|>") {
			return true
		}
		if id == t.special["<|eom_id|>"] && t.has("<|eom_id|>") {
			return true
		}
	case fmtQwen:
		if id == t.special["<|im_end|>"] && t.has("<|im_end|>") {
			return true
		}
	case fmtGemma:
		if id == t.special["<end_of_turn>"] && t.has("<end_of_turn>") {
			return true
		}
		if id == t.special["<eos>"] && t.has("<eos>") {
			return true
		}
	case fmtHarmony:
		for _, s := range []string{"<|return|>", "<|end|>", "<|call|>"} {
			if id == t.special[s] && t.has(s) {
				return true
			}
		}
	}
	if t.eot != "" && t.has(t.eot) && id == t.special[t.eot] {
		return true
	}
	return false
}

// afterStop is the template text that follows a stop token the model just emitted.
func (t *tok) afterStop() string {
	switch t.format() {
	case fmtQwen, fmtGemma:
		return "\n"
	default:
		return ""
	}
}

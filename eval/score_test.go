package eval

import (
	"math"
	"testing"
)

func TestLogsoftTokenizer(t *testing.T) {
	x := []float32{0, 0, 20}
	full := logsoft(x, 0)
	cut := logsoftTok(x, 0, []bool{true, true, false})
	if math.Abs(cut-math.Log(0.5)) > 1e-6 {
		t.Fatalf("tokenizer softmax %g, want log 1/2", cut)
	}
	if cut <= full {
		t.Fatalf("full head %g hid the token that the tokenizer softmax keeps at %g", full, cut)
	}
}

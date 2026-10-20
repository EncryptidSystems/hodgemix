package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"hodgemix/eval"
)

// HODGEMIX_SHARD_PROBE=<shard files, path-list separated>: per layer, which shard holds q_proj and down_proj and their
// absmean / max |w| / fraction of non-finite values, read through OpenST's sharded path.
func TestShardProbe(t *testing.T) {
	fs := filepath.SplitList(os.Getenv("HODGEMIX_SHARD_PROBE"))
	if len(fs) == 0 {
		t.Skip()
	}
	p := fs[0]
	eval.SetFiles(p, eval.ModelFiles{Shards: fs})
	st, err := OpenST(p)
	if err != nil {
		t.Fatal(err)
	}
	for l := 0; ; l++ {
		found := false
		for _, r := range []string{"self_attn.q_proj", "mlp.down_proj"} {
			k := fmt.Sprintf("model.layers.%d.%s.weight", l, r)
			if !st.Has(k) {
				continue
			}
			found = true
			W, sh, err := st.LoadDense(k)
			if err != nil {
				t.Fatal(err)
			}
			var am, mx float64
			bad := 0
			for _, v := range W {
				f := float64(v)
				if math.IsNaN(f) || math.IsInf(f, 0) {
					bad++
					continue
				}
				am += math.Abs(f)
				mx = math.Max(mx, math.Abs(f))
			}
			fmt.Printf("L%-2d %-16s %v shard=%s absmean=%.5f max=%.3f nonfinite=%d\n", l, r, sh, st.owner[k].Path[len(st.owner[k].Path)-31:], am/float64(len(W)), mx, bad)
		}
		if !found {
			break
		}
	}
}

package main

import (
	"os"
	"strings"
	"testing"
)

// HODGEMIX_LIMB_PROBE=base.safetensors HODGEMIX_LIMB_PROBE_OUT=out.safetensors go test -run LimbProbe
// writes the base with every packed weight moved to a radix-4 limb 1 (limb 0 scale 0): the same model,
// so it must score exactly like the base if limbs are written and read right.
func TestLimbProbe(t *testing.T) {
	src, dst := os.Getenv("HODGEMIX_LIMB_PROBE"), os.Getenv("HODGEMIX_LIMB_PROBE_OUT")
	if src == "" || dst == "" {
		t.Skip()
	}
	st, err := OpenST(src)
	if err != nil {
		t.Fatal(err)
	}
	W, err := NewSTW(dst, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range st.Keys {
		b, err := st.Raw(k)
		if err != nil {
			t.Fatal(err)
		}
		ti := st.H[k]
		if strings.HasSuffix(k, "weight_scale") && st.isPacked(strings.TrimSuffix(k, "_scale")) {
			W.AddBF16(k, []int{1}, []float32{0}) // limb 0 contributes nothing
			continue
		}
		W.Add(k, ti.Dtype, ti.Shape, b)
		if st.isPacked(k) {
			sh := st.LogicalShape(k)
			sc, _ := st.F32(scaleKey(k))
			d := unpackDigits(b, sh[0], sh[1]) // ternary codes 0..2 = radix-4 digits meaning -1..1
			writeLimbs(W, k, sh[0], sh[1], []limbData{{4, d, float64(sc[0])}})
		}
	}
	if err := W.Close(nil); err != nil {
		t.Fatal(err)
	}
}

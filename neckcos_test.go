package main

import (
	"math"
	"os"
	"testing"

	"hodgemix/eval"
)

func TestNeckTrace(t *testing.T) {
	if os.Getenv("HODGEMIX_NECKTRACE") == "" {
		t.Skip()
	}
	dir := `C:\Users\grego\models\prod\llama`
	floatM := dir + `\model.safetensors`
	neck := dir + `\model.neck.safetensors`
	tok := dir + `\tokenizer.json`
	eval.SetFiles(floatM, eval.ModelFiles{Tokenizer: tok, TokenizerConfig: dir + `\tokenizer_config.json`, Config: dir + `\base.config.json`})
	eval.SetFiles(neck, eval.ModelFiles{Tokenizer: tok, TokenizerConfig: dir + `\tokenizer_config.json`, Config: dir + `\config.json`})
	rows, pos, err := eval.Trace(floatM, neck, tok, "The capital of France is", false)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range rows {
		if r.Stage == "hidden" {
			t.Logf("L%d hidden peak neck/float %.3f", r.Layer, r.Peak)
			continue
		}
		if r.Stage == "attn" || r.Stage == "mlp" || r.Stage == "in_o" || r.Stage == "in_down" || r.Layer == 0 {
			t.Logf("L%d %-8s cos %.3f norm %.3f", r.Layer, r.Stage, r.Cos, r.Norm)
		}
	}
	t.Logf("pos cos %v", pos)
}

func TestNeckCosine(t *testing.T) {
	if os.Getenv("HODGEMIX_NECKCOS") == "" {
		t.Skip()
	}
	float, err := OpenST(`C:\Users\grego\models\prod\llama\model.safetensors`)
	if err != nil {
		t.Fatal(err)
	}
	neck, err := OpenST(`C:\Users\grego\models\prod\llama\model.neck.safetensors`)
	if err != nil {
		t.Fatal(err)
	}
	names := []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
		"model.layers.15.self_attn.o_proj.weight",
		"model.layers.15.mlp.down_proj.weight",
	}
	for _, k := range names {
		W, _, err := float.LoadDense(k)
		if err != nil {
			t.Fatal(err)
		}
		R, rows, cols := reconNeck(neck, k)
		L0, _, err := neck.LoadDense(k)
		if err != nil {
			t.Fatal(err)
		}
		ab0, aa0, bb0 := dotf(W, L0)
		t.Logf("  limb0 cos %.4f rel %.3f", ab0/math.Sqrt(aa0*bb0), math.Sqrt(aa0+bb0-2*ab0)/math.Sqrt(aa0))
		var maxd float64
		nz := 0
		for i := range R {
			d := math.Abs(float64(R[i] - L0[i]))
			if d > maxd {
				maxd = d
			}
			if d > 0 {
				nz++
			}
		}
		for _, lb := range neck.limbsOf(k) {
			n := 0
			c := (lb.r - 1) / 2
			for _, d := range lb.d {
				if int(d) != c {
					n++
				}
			}
			t.Logf("  limb r=%d scale %.6g nonzero %d/%d", lb.r, lb.s, n, len(lb.d))
		}
		ab, aa, bb := dotf(W, R)
		rel := math.Sqrt(aa+bb-2*ab) / math.Sqrt(aa)
		t.Logf("%s %dx%d cos %.6f rel %.6f limbs %d  |recon-limb0| max %.6g nonzero %d", k, rows, cols, ab/math.Sqrt(aa*bb), rel, len(neck.limbsOf(k)), maxd, nz)
		// singularNeck cuts the limb-0 gap. Those entries stay at the limb-0 trit in every later limb.
		b0, err := neck.Raw(k)
		if err != nil {
			t.Fatal(err)
		}
		sc0, err := neck.F32(scaleKey(k))
		if err != nil || len(sc0) != 1 {
			t.Fatal(k)
		}
		q0 := unpackTrits(b0, rows, cols)
		gap := make([]float32, len(W))
		var gAll, gNeck, eAll, eNeck float64
		for i, v := range q0 {
			gap[i] = W[i] - float32(float64(sc0[0])*float64(v))
			gAll += float64(gap[i]) * float64(gap[i])
		}
		mask := singularNeck(gap)
		ns := 0
		for i := range W {
			e := float64(W[i]-R[i]) * float64(W[i]-R[i])
			eAll += e
			if mask != nil && mask[i] {
				ns++
				gNeck += float64(gap[i]) * float64(gap[i])
				eNeck += e
			}
		}
		t.Logf("  neck %d/%d  limb0-gap energy in neck %.1f%%  final ||err||^2 in neck %.1f%%", ns, len(W), 100*gNeck/gAll, 100*eNeck/eAll)
		if rows == cols {
			T := make([]float32, len(R))
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					T[c*rows+r] = R[r*cols+c]
				}
			}
			t.Logf("  transpose cos %.4f", cosf(W, T))
		}
		err2 := 0.0
		var top []float64
		for i := range W {
			e := float64(W[i]-R[i]) * float64(W[i]-R[i])
			err2 += e
			top = append(top, e)
		}
		// partial select of the largest 64
		for i := 0; i < 64 && i < len(top); i++ {
			m := i
			for j := i + 1; j < len(top); j++ {
				if top[j] > top[m] {
					m = j
				}
			}
			top[i], top[m] = top[m], top[i]
		}
		sum := 0.0
		for i := 0; i < 64 && i < len(top); i++ {
			sum += top[i]
		}
		t.Logf("  top64 entries hold %.1f%% of ||err||^2", 100*sum/err2)
		hit, n := 0, 8
		step := rows / n
		for s := 0; s < n; s++ {
			r := s * step
			row := R[r*cols : (r+1)*cols]
			best, at := -2.0, -1
			for i := 0; i < rows; i++ {
				c := cosf(row, W[i*cols:(i+1)*cols])
				if c > best {
					best, at = c, i
				}
			}
			if at == r {
				hit++
			}
			if s < 3 {
				t.Logf("  row %d best match row %d cos %.3f", r, at, best)
			}
		}
		t.Logf("  %d/%d sample rows match their own index", hit, n)
	}
}

func reconNeck(st *ST, k string) ([]float32, int, int) {
	sh := st.LogicalShape(k)
	R, C := sh[0], sh[1]
	b, err := st.Raw(k)
	if err != nil {
		panic(err)
	}
	sc, err := st.F32(scaleKey(k))
	if err != nil || len(sc) != 1 {
		panic(k)
	}
	q := unpackTrits(b, R, C)
	o := make([]float32, R*C)
	for i, v := range q {
		o[i] = float32(float64(sc[0]) * float64(v))
	}
	for _, lb := range st.limbsOf(k) {
		c := (lb.r - 1) / 2
		for i, d := range lb.d {
			o[i] += float32(lb.s * float64(int(d)-c))
		}
	}
	return o, R, C
}

func dotf(a, b []float32) (ab, aa, bb float64) {
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		ab += x * y
		aa += x * x
		bb += y * y
	}
	return
}

func cosf(a, b []float32) float64 {
	ab, aa, bb := dotf(a, b)
	return ab / math.Sqrt(aa*bb)
}

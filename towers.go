package main

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"hodgemix/tower"
)

// copyTowers writes every carried audio, vision, and embedding tensor from the
// contributors into the output under the same name and the same bytes.
// bnzmix.* is left to writeTowers. already names are already in the writer.
// Later models win on a repeated name.
func copyTowers(models []*ST, W, MW *STW, already map[string]bool) (int, error) {
	seen := map[string]bool{}
	for k := range already {
		seen[k] = true
	}
	n := 0
	for i := len(models) - 1; i >= 0; i-- {
		st := models[i]
		for _, k := range st.Keys {
			if !carriedTensor(k) || seen[k] {
				continue
			}
			b, err := st.Raw(k)
			if err != nil {
				return n, err
			}
			info := st.H[k]
			sh := append([]int(nil), info.Shape...)
			// An encoder's ternary copy is assimilateGemmaTowers'; the merged file keeps its bytes.
			if !gemmaTowerTensor(k) {
				W.Add(k, info.Dtype, sh, b)
			}
			if MW != nil {
				MW.Add(k, info.Dtype, append([]int(nil), sh...), b)
			}
			seen[k] = true
			n++
		}
	}
	return n, nil
}

// writeTowers adds the L-A vision and audio towers and their embedding layers.
// The projectors are least-squares maps from tower readouts onto base embedding
// rows (D6), ternarised with the same scale and limb path as the other weights
// (D7), and stored as bnzmix.vision_tower / audio_tower / embed_vision / embed_audio (D8).
func writeTowers(W *STW, base *ST, basePath string) (map[string]any, error) {
	key := embedKey(base)
	if key == "" {
		logf("TOWERS  skipped: the base has no token embedding, so the vision and audio embedding layers were not written")
		return map[string]any{"skipped": "no embedding"}, nil
	}
	if !tower.Ready() {
		return nil, fmt.Errorf("the L-A operator has no attracting fixed point at 1.6i")
	}
	emb, err := base.F32(key)
	if err != nil {
		return nil, err
	}
	sh := base.LogicalShape(key)
	if len(sh) != 2 || sh[0] < 1 || sh[1] < 1 {
		return nil, fmt.Errorf("%s shape %v", key, sh)
	}
	V, D := sh[0], sh[1]
	if len(emb) != V*D {
		return nil, fmt.Errorf("%s has %d values, shape %v", key, len(emb), sh)
	}
	sum, err := tensorHeaderSHA(basePath)
	if err != nil {
		return nil, err
	}
	logf("TOWERS  L-A depth %d, embedding layers into residual width %d", tower.Depth, D)
	nVis, nAud := probeCount(V, tower.VisFeat), probeCount(V, tower.AudFeat)
	t0 := time.Now()
	visX, err := tower.ProbeVision(nVis)
	if err != nil {
		return nil, err
	}
	logf("  vision probes %d in %.2fs", nVis, time.Since(t0).Seconds())
	t0 = time.Now()
	audX, err := tower.ProbeAudio(nAud)
	if err != nil {
		return nil, err
	}
	logf("  audio probes %d in %.2fs", nAud, time.Since(t0).Seconds())
	visRows := sampleRows(emb, V, D, nVis)
	audRows := sampleRows(emb, V, D, nAud)
	visP, visMF, visMY, err := fitProjector(visX, nVis, tower.VisFeat, visRows, D)
	if err != nil {
		return nil, fmt.Errorf("vision projector: %w", err)
	}
	audP, audMF, audMY, err := fitProjector(audX, nAud, tower.AudFeat, audRows, D)
	if err != nil {
		return nil, fmt.Errorf("audio projector: %w", err)
	}
	visRMS := meanRowRMS(visRows, nVis, D)
	audRMS := meanRowRMS(audRows, nAud, D)
	op := tower.Operator()
	writeTowerConst(W, tower.VisTower, op, sum)
	writeTowerConst(W, tower.AudTower, op, sum)
	writeEmbed(W, tower.VisEmbed, visP, D, tower.VisFeat, visMF, visMY, visRMS)
	writeEmbed(W, tower.AudEmbed, audP, D, tower.AudFeat, audMF, audMY, audRMS)
	logf("  embed_vision [%d,%d]  embed_audio [%d,%d]  rms %.4g / %.4g", D, tower.VisFeat, D, tower.AudFeat, visRMS, audRMS)
	sig := tower.Sigma()
	return map[string]any{
		"depth": tower.Depth, "hidden": D,
		"vision": []int{D, tower.VisFeat}, "audio": []int{D, tower.AudFeat},
		"sigma": []float64{real(sig), imag(sig)},
	}, nil
}

func probeCount(V, df int) int {
	n := df + 32
	if n > V {
		n = V
	}
	if n < 1 {
		return 1
	}
	return n
}

func sampleRows(emb []float32, V, D, n int) []float32 {
	o := make([]float32, n*D)
	for i := 0; i < n; i++ {
		r := 0
		if n > 1 && V > 1 {
			r = i * (V - 1) / (n - 1)
		}
		copy(o[i*D:(i+1)*D], emb[r*D:(r+1)*D])
	}
	return o
}

func meanRowRMS(rows []float32, n, D int) float64 {
	if n == 0 || D == 0 {
		return 0
	}
	var s float64
	for i := 0; i < n; i++ {
		var ss float64
		for j := 0; j < D; j++ {
			v := float64(rows[i*D+j])
			ss += v * v
		}
		s += math.Sqrt(ss / float64(D))
	}
	return s / float64(n)
}

// fitProjector solves the centred normal equations the residual gauge map uses
// (vocab.go residualMap / cholSolve). P is row-major [D, df].
func fitProjector(X []float64, n, df int, Y []float32, D int) (P []float32, mf, my []float64, err error) {
	if n < 1 || len(X) != n*df || len(Y) != n*D {
		return nil, nil, nil, fmt.Errorf("fit shape n=%d df=%d D=%d", n, df, D)
	}
	mf, my = make([]float64, df), make([]float64, D)
	for i := 0; i < n; i++ {
		for j := 0; j < df; j++ {
			mf[j] += X[i*df+j]
		}
		for c := 0; c < D; c++ {
			my[c] += float64(Y[i*D+c])
		}
	}
	inv := 1 / float64(n)
	for j := range mf {
		mf[j] *= inv
	}
	for c := range my {
		my[c] *= inv
	}
	A := make([]float64, df*df)
	B := make([]float64, df*D)
	parFor(df, func(lo, hi int) {
		xs := make([]float64, df)
		for i := 0; i < n; i++ {
			for j := 0; j < df; j++ {
				xs[j] = X[i*df+j] - mf[j]
			}
			for r := lo; r < hi; r++ {
				a := xs[r]
				if a == 0 {
					continue
				}
				ar := A[r*df : (r+1)*df]
				br := B[r*D : (r+1)*D]
				for c := 0; c < df; c++ {
					ar[c] += a * xs[c]
				}
				for c := 0; c < D; c++ {
					br[c] += a * (float64(Y[i*D+c]) - my[c])
				}
			}
		}
	})
	var tr float64
	for i := 0; i < df; i++ {
		tr += A[i*df+i]
	}
	for i := 0; i < df; i++ {
		A[i*df+i] += 1e-6 * tr / float64(df)
	}
	M := cholSolve(A, B, df, D)
	P = make([]float32, D*df)
	for c := 0; c < D; c++ {
		for j := 0; j < df; j++ {
			v := M[j*D+c]
			if math.IsNaN(v) || math.IsInf(v, 0) {
				err = fmt.Errorf("projector entry %d,%d is not finite", c, j)
				return nil, nil, nil, err
			}
			P[c*df+j] = float32(v)
		}
	}
	return P, mf, my, nil
}

func writeTowerConst(W *STW, prefix string, op *tower.LambertOp, sum []byte) {
	put := func(name string, z complex128) {
		W.AddF32(prefix+"."+name, []int{2}, []float32{float32(real(z)), float32(imag(z))})
	}
	put("sigma", op.Sigma)
	put("zstar", tower.ZStar)
	put("lambda", tower.Lambda())
	put("ell", tower.Generator())
	put("c", op.C)
	put("s", op.S)
	W.AddF32(prefix+".radius", []int{1}, []float32{float32(tower.Radius)})
	W.AddI32(prefix+".depth", []int{1}, []int32{int32(tower.Depth)})
	W.Add(prefix+".base_sha256", "U8", []int{32}, append([]byte(nil), sum...))
}

func writeEmbed(W *STW, name string, P []float32, D, df int, mf, my []float64, rms float64) {
	W.AddF32(name+".mean_feature", []int{df}, f64tof32(mf))
	W.AddF32(name+".mean_row", []int{D}, f64tof32(my))
	W.AddF32(name+".rms", []int{1}, []float32{float32(rms)})
	if D%4 != 0 {
		W.AddF32(name+".weight", []int{D, df}, P)
		logf("  %s stored F32 [%d,%d]", name, D, df)
		return
	}
	s := absmean(P)
	q := make([]int8, len(P))
	var nz int
	var nzSum float64
	for i, v := range P {
		q[i] = int8(ternQ(float64(v), s))
		if q[i] != 0 {
			nz++
			nzSum += math.Abs(float64(v))
		}
	}
	ws := s
	if nz > 0 {
		ws = nzSum / float64(nz)
	}
	W.Add(name+".weight", "U8", []int{D / 4, df}, packTrits(q, D, df))
	W.AddBF16(name+".weight_scale", []int{1}, []float32{float32(ws)})
	limbs, rel := residualLimbs(P, q, ws, MaxLimbs, LimbTol, D, df)
	if len(limbs) > 0 {
		writeLimbs(W, name+".weight", D, df, limbs)
		logf("  %s limbs %d, residual %.3f to %.3f", name, len(limbs), rel[0], rel[len(rel)-1])
	}
}

func f64tof32(a []float64) []float32 {
	o := make([]float32, len(a))
	for i, v := range a {
		o[i] = float32(v)
	}
	return o
}

// tensorHeaderSHA is the sha256 of the safetensors header, the identity the
// tower records as base_sha256 (D9).
func tensorHeaderSHA(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var n [8]byte
	if _, err := io.ReadFull(f, n[:]); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	hn := binary.LittleEndian.Uint64(n[:])
	if hn == 0 || hn > 64<<20 {
		return nil, fmt.Errorf("%s: header length %d", path, hn)
	}
	h := make([]byte, hn)
	if _, err := io.ReadFull(f, h); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if h[0] != '{' {
		return nil, fmt.Errorf("%s: not a safetensors header", path)
	}
	sum := sha256.Sum256(append(append([]byte{}, n[:]...), h...))
	return sum[:], nil
}

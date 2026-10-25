package main

import (
	"fmt"
	"math"
	"strings"
)

// Gemma 4's encoders assimilated into the output (D2, D7). Every linear of vision_tower,
// audio_tower, embed_vision and embed_audio is ternarised with residual limbs, the same as
// the text tensors; norms, convolutions, biases, clip bounds and the position table keep
// their bytes. embed_vision and embed_audio end on the source's 1536-wide residual axis;
// bnzmix.embed_bridge carries that axis onto the base's residual axis D: the residual gauge
// map of the source token embedding onto the base's over shared token strings
// (vocab.go residualMap), stored like the L-A projectors (writeEmbed).

const bridgeEmbed = "bnzmix.embed_bridge"

// gemmaTowerTensor is a tensor of a Gemma 4 encoder or its embedding projection.
func gemmaTowerTensor(name string) bool {
	for _, p := range []string{"model.vision_tower.", "model.audio_tower.", "model.embed_vision.", "model.embed_audio."} {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

// ternLinear is an encoder linear the runtime runs as packed ternary: a 2-D weight whose
// output rows pack four to a byte.
func ternLinear(name string, shape []int) bool {
	return gemmaTowerTensor(name) && strings.HasSuffix(name, ".weight") && len(shape) == 2 && shape[0]%4 == 0 && shape[1] > 0
}

// writeTernWeight stores P [R, C] as k (U8 [R/4, C]), k_scale and k's limbs, as writeEmbed
// does for a projector. rel is the relative residual after the last limb.
func writeTernWeight(W *STW, k string, P []float32, R, C int) (rel float64) {
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
	base := strings.TrimSuffix(k, ".weight")
	W.Add(k, "U8", []int{R / 4, C}, packTrits(q, R, C))
	W.AddBF16(base+".weight_scale", []int{1}, []float32{float32(ws)})
	limbs, rels := residualLimbs(P, q, ws, MaxLimbs, LimbTol, R, C)
	if len(limbs) > 0 {
		writeLimbs(W, k, R, C, limbs)
	}
	if len(rels) > 0 {
		return rels[len(rels)-1]
	}
	return 1
}

// assimilateGemmaTowers writes src's encoder tensors into W: linears ternarised, the rest
// as stored. already names are in W and skipped.
func assimilateGemmaTowers(W *STW, src *ST, already map[string]bool) (map[string]any, error) {
	tern, kept := 0, 0
	worst := 0.0
	for _, k := range src.Keys {
		if !gemmaTowerTensor(k) || already[k] {
			continue
		}
		info := src.H[k]
		sh := append([]int(nil), info.Shape...)
		if ternLinear(k, sh) {
			P, err := src.F32(k)
			if err != nil {
				return nil, err
			}
			if rel := writeTernWeight(W, k, P, sh[0], sh[1]); rel > worst {
				worst = rel
			}
			tern++
		} else {
			b, err := src.Raw(k)
			if err != nil {
				return nil, err
			}
			W.Add(k, info.Dtype, sh, b)
			kept++
		}
		already[k] = true
	}
	if tern == 0 && kept == 0 {
		return nil, nil
	}
	logf("  encoders: %d linears ternarised (worst residual %.3f), %d tensors kept as stored", tern, worst, kept)
	return map[string]any{"ternarised": tern, "kept": kept, "worst_residual": worst}, nil
}

// writeBridge fits the residual gauge map of src's token embedding (srcKey, the axis the
// encoders end on) onto base's over the token strings both tokenizers share, and stores it
// as bridgeEmbed [Db, Ds].
func writeBridge(W *STW, base *ST, baseTok string, src *ST, srcKey, srcTok string) (map[string]any, error) {
	bk := embedKey(base)
	if bk == "" || !src.Has(srcKey) {
		return nil, fmt.Errorf("bridge: no token embedding (%q, %q)", bk, srcKey)
	}
	bt, err := readTok(baseTok)
	if err != nil {
		return nil, err
	}
	stk, err := readTok(srcTok)
	if err != nil {
		return nil, err
	}
	bs, ss := base.LogicalShape(bk), src.LogicalShape(srcKey)
	V, Db, Ds := bs[0], bs[1], ss[1]
	if Db%4 != 0 {
		return nil, fmt.Errorf("bridge: residual width %d does not pack", Db)
	}
	Eb, err := base.F32(bk)
	if err != nil {
		return nil, err
	}
	Es, err := src.F32(srcKey)
	if err != nil {
		return nil, err
	}
	M, mb, ms, r2, n := residualMap(Eb, Db, bt.vocab, Es, Ds, stk.vocab)
	if n == 0 {
		return nil, fmt.Errorf("bridge: the tokenizers share no token strings")
	}
	P := make([]float32, Db*Ds)
	for c := 0; c < Db; c++ {
		for j := 0; j < Ds; j++ {
			P[c*Ds+j] = float32(M[j*Db+c])
		}
	}
	rms := meanRowRMS(Eb, V, Db)
	logf("  bridge %d -> %d: held-out R^2 %.3f over %d shared token strings", Ds, Db, r2, n)
	writeEmbed(W, bridgeEmbed, P, Db, Ds, ms, mb, rms)
	return map[string]any{"from": Ds, "to": Db, "r2": r2, "shared": n}, nil
}

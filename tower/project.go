package tower

import "math"

// Limb is one extra digit of a packed projector, in the same layout as
// limbs.go packDigits: four rows per byte, scale times (digit - center).
type Limb struct {
	Radix  int
	Digits []byte
	Scale  float32
}

// Dense rebuilds a [D, df] projector from its packed ternary limb and the
// extra limbs. D must be a multiple of 4.
func Dense(packed []byte, D, df int, scale float32, limbs []Limb) []float32 {
	R4 := D / 4
	o := make([]float32, D*df)
	for s := 0; s < 4; s++ {
		for i, v := range packed {
			o[s*R4*df+i] = float32(int((v>>(2*s))&3)-1) * scale
		}
	}
	for _, lb := range limbs {
		d := unpack(lb.Digits, D, df)
		c := (lb.Radix - 1) / 2
		for i := range o {
			o[i] += lb.Scale * float32(int(d[i])-c)
		}
	}
	return o
}

func unpack(b []byte, D, df int) []byte {
	R4 := D / 4
	d := make([]byte, D*df)
	for i := 0; i < R4*df; i++ {
		for s := 0; s < 4; s++ {
			d[s*R4*df+i] = (b[i] >> (2 * s)) & 3
		}
	}
	return d
}

// Project maps one tower feature into the residual stream:
// x = P (f - meanF) + meanRow, then scaled to the embedding rms (D3, D6).
// P is row-major [D, df].
func Project(P []float32, D, df int, meanF, meanRow []float32, rms float64, f []float32) ([]float32, error) {
	if len(f) != df || len(meanF) != df || len(meanRow) != D || len(P) != D*df {
		return nil, errf("projector shape D=%d df=%d, feature %d", D, df, len(f))
	}
	y := make([]float32, D)
	copy(y, meanRow)
	for c := 0; c < D; c++ {
		row := P[c*df:]
		var s float64
		for j := 0; j < df; j++ {
			s += float64(row[j]) * (float64(f[j]) - float64(meanF[j]))
		}
		y[c] += float32(s)
	}
	return matchRMS(y, rms), nil
}

func matchRMS(y []float32, target float64) []float32 {
	if target <= 0 || len(y) == 0 {
		return y
	}
	var ss float64
	for _, v := range y {
		ss += float64(v) * float64(v)
	}
	r := math.Sqrt(ss / float64(len(y)))
	if r == 0 {
		return y
	}
	sc := float32(target / r)
	for i := range y {
		y[i] *= sc
	}
	return y
}

// Phi is the RoPE extension angle (D22): argument of the first complex pair,
// which is the patch-mean readout of field w1, or the first audio band.
func Phi(f []float32) float64 {
	if len(f) < 2 {
		return 0
	}
	return math.Atan2(float64(f[1]), float64(f[0]))
}

// AudioPhi is the energy-weighted argument of the 80 band means (D22).
// The feature layout is 80 complex means followed by 80 log-mel energies.
func AudioPhi(f []float32) float64 {
	if len(f) < AudFeat {
		return Phi(f)
	}
	var re, im, w float64
	for b := 0; b < 80; b++ {
		e := math.Exp(float64(f[160+b]))
		re += e * float64(f[2*b])
		im += e * float64(f[2*b+1])
		w += e
	}
	if w == 0 {
		return 0
	}
	return math.Atan2(im, re)
}

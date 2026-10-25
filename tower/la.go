package tower

// L-A tower (decisions D13–D16). The operator, Lambert W, and the log-sine
// primitives are the tower_check sources. This file only calls them.

import (
	"math"
	"math/cmplx"
	"sync"
)

const (
	// Depth is the Koenigs readout depth for eps = 2^-8 (D16).
	Depth = 4
	// Radius is the seed-disc radius r_max/2 (D15).
	Radius = 0.175
	// VisFeat and AudFeat are the readout widths (D21).
	VisFeat = 46
	AudFeat = 240
)

// ZStar is the attracting fixed point on the chosen ray (D14).
const ZStar = complex(0, 1.6)

var (
	prepOnce sync.Once
	sigVal   complex128
	lamVal   complex128
	ready    bool
)

func prepare() {
	prepOnce.Do(func() {
		sig, _, ok := targetLA(ZStar)
		if !ok {
			return
		}
		f, fp, ok := lambertA(sig, 0, 0).Eval(ZStar)
		if !ok || cmplx.Abs(fp) >= 1 || cmplx.Abs(f-ZStar) > 1e-8 {
			return
		}
		sigVal, lamVal, ready = sig, fp, true
	})
}

// Ready reports that the closed-form bias lands on an attracting fixed point.
func Ready() bool { prepare(); return ready }

// Sigma is the closed-form L-A bias at ZStar (D4, D14).
func Sigma() complex128 { prepare(); return sigVal }

// Lambda is T'(ZStar).
func Lambda() complex128 { prepare(); return lamVal }

// Generator is log(lambda), the Koenigs-flow generator at the attractor.
func Generator() complex128 { return cmplx.Log(Lambda()) }

// Operator is the L-A layer used by both towers (D13).
func Operator() *LambertOp { return lambertA(Sigma(), 0, 0) }

// numLA is the numerator Log A_{c,0}(z) as the tower check evaluates it.
func numLA(z complex128) complex128 {
	W, _ := LambertW0(z)
	return wrapIm(LogSin(cA/W) + LogSin(z))
}

// targetLA is the closed-form sigma that places a fixed point at zs.
func targetLA(zs complex128) (complex128, complex128, bool) {
	q := numLA(zs) / zs
	if math.Abs(imag(q)) >= math.Pi {
		return 0, 0, false
	}
	sig := cmplx.Exp(q) - sA*cmplx.Sin(zs)
	return sig, q, true
}

// Seed maps a disc coordinate u, |u|<1, to z* + r u (D15).
func Seed(u complex128) complex128 { return ZStar + complex(Radius, 0)*u }

// Squash is u = w / (s + |w|), which lands in the open unit disc.
func Squash(w complex128, s float64) complex128 {
	a := cmplx.Abs(w)
	if s+a == 0 {
		return 0
	}
	return w / complex(s+a, 0)
}

// Iterate applies the L-A layer n times.
func Iterate(z complex128, n int) (complex128, bool) {
	op := Operator()
	for k := 0; k < n; k++ {
		f, _, ok := op.Eval(z)
		if !ok || !finite(f) {
			return 0, false
		}
		z = f
	}
	return z, true
}

// Embed is the Koenigs embedding of a seed: R_n = lambda^{-n}(z_n - z*) (D5, D16).
func Embed(z0 complex128) (complex128, bool) {
	z, ok := Iterate(z0, Depth)
	if !ok {
		return 0, false
	}
	return (z - ZStar) / cmplx.Pow(Lambda(), complex(float64(Depth), 0)), true
}

// Micro is the continuous embedding at fractional depth alpha in [0, 1):
// T^alpha(z) = z* + exp(alpha log lambda) (z - z*), the Koenigs flow
// linearised at the attractor.
func Micro(z complex128, alpha float64) complex128 {
	return ZStar + cmplx.Exp(complex(alpha, 0)*Generator())*(z-ZStar)
}

// Nano is the first-order embedding z + eps X(z), with X(z) = log(lambda) (z - z*).
func Nano(z complex128, eps float64) complex128 {
	return z + complex(eps, 0)*Generator()*(z-ZStar)
}

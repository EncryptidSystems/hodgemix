package tower

import (
	"math"
	"math/cmplx"
)

// Op: holomorphic branch evaluation of T and T'. ok=false means z is in the
// singular set Sigma (pole, log singularity, branch point) or numerics overflowed.
type Op interface {
	Name() string
	Eval(z complex128) (f, fp complex128, ok bool)
}

// Lambert log-quotient family:
// T(z) = [Log( sin(c/(W0(z)+d)) * sin z  (+ sigma if SigNum) ) + 2 pi i k1]
//        / [Log( s*sin z + sigma ) + 2 pi i k2]
type LambertOp struct {
	Label  string
	C, D   complex128
	S      complex128
	Sigma  complex128
	SigNum bool
	K1, K2 int
}

func (o *LambertOp) Name() string { return o.Label }

func (o *LambertOp) Eval(z complex128) (complex128, complex128, bool) {
	W, ok := LambertW0(z)
	if !ok {
		return 0, 0, false
	}
	var Wp complex128
	if z == 0 {
		Wp = 1
	} else {
		if W+1 == 0 {
			return 0, 0, false
		}
		Wp = W / (z * (1 + W))
	}
	Ws := W + o.D
	if Ws == 0 {
		return 0, 0, false
	}
	u := o.C / Ws
	du := -o.C * Wp / (Ws * Ws)
	lsu := LogSin(u)
	lsz := LogSin(z)
	if !finite(lsu) || !finite(lsz) {
		return 0, 0, false
	}
	logA0 := wrapIm(lsu + lsz)
	dlogA0 := Cot(u)*du + Cot(z)
	var LA, dLA complex128
	if o.SigNum {
		if real(logA0) < 600 {
			A0 := cmplx.Exp(logA0)
			A := A0 + o.Sigma
			if A == 0 {
				return 0, 0, false
			}
			LA = cmplx.Log(A)
			dLA = dlogA0 * A0 / A
		} else {
			r := o.Sigma * cmplx.Exp(-logA0)
			LA = wrapIm(logA0 + cmplx.Log(1+r))
			dLA = dlogA0 / (1 + r)
		}
	} else {
		LA, dLA = logA0, dlogA0
	}
	LA += complex(0, twoPi*float64(o.K1))
	LB, dLB := LogAffineSin(o.S, o.Sigma, z)
	LB += complex(0, twoPi*float64(o.K2))
	if LB == 0 || !finite(LB) || !finite(LA) {
		return 0, 0, false
	}
	f := LA / LB
	fp := (dLA*LB - LA*dLB) / (LB * LB)
	if !finite(f) || !finite(fp) {
		return 0, 0, false
	}
	return f, fp, true
}

// Generic log-quotient T = Log(T1+s1)/Log(T2+s2) with closures giving value and derivative.
type GenOp struct {
	Label  string
	T1, T2 func(z complex128) (complex128, complex128, bool)
	Sigma1 complex128
	Sigma2 complex128
}

func (o *GenOp) Name() string { return o.Label }

func (o *GenOp) Eval(z complex128) (complex128, complex128, bool) {
	a, da, ok1 := o.T1(z)
	b, db, ok2 := o.T2(z)
	if !ok1 || !ok2 {
		return 0, 0, false
	}
	A := a + o.Sigma1
	B := b + o.Sigma2
	if A == 0 || B == 0 || !finite(A) || !finite(B) {
		return 0, 0, false
	}
	LA, LB := cmplx.Log(A), cmplx.Log(B)
	if LB == 0 {
		return 0, 0, false
	}
	f := LA / LB
	fp := (da/A*LB - LA*db/B) / (LB * LB)
	if !finite(f) || !finite(fp) {
		return 0, 0, false
	}
	return f, fp, true
}

func sinT(k complex128) func(z complex128) (complex128, complex128, bool) {
	return func(z complex128) (complex128, complex128, bool) {
		if math.Abs(imag(z)) > 700 {
			return 0, 0, false
		}
		return k * cmplx.Sin(z), k * cmplx.Cos(z), true
	}
}
func cosT(k complex128) func(z complex128) (complex128, complex128, bool) {
	return func(z complex128) (complex128, complex128, bool) {
		if math.Abs(imag(z)) > 700 {
			return 0, 0, false
		}
		return k * cmplx.Cos(z), -k * cmplx.Sin(z), true
	}
}
func cosSq(z complex128) (complex128, complex128, bool) {
	w := z * z
	if math.Abs(imag(w)) > 700 {
		return 0, 0, false
	}
	return cmplx.Cos(w), -2 * z * cmplx.Sin(w), true
}
func sinW(z complex128) (complex128, complex128, bool) {
	W, ok := LambertW0(z)
	if !ok || math.Abs(imag(W)) > 700 {
		return 0, 0, false
	}
	var Wp complex128 = 1
	if z != 0 {
		if W+1 == 0 {
			return 0, 0, false
		}
		Wp = W / (z * (1 + W))
	}
	return cmplx.Sin(W), cmplx.Cos(W) * Wp, true
}

var (
	cA     = cmplx.Exp(complex(0, math.Pi*math.E)) // e^{i pi e}
	sA     = complex(-math.Sin(1), 0)               // sin(e^{i pi}) = -sin 1
	cC     = complex(-1.5, 2.1)
	halfPi = math.Pi / 2
)

func lambertA(sig complex128, k1, k2 int) *LambertOp {
	return &LambertOp{Label: "L-A", C: cA, S: sA, Sigma: sig, SigNum: false, K1: k1, K2: k2}
}
func lambertB(sig complex128, k1, k2 int) *LambertOp {
	return &LambertOp{Label: "L-B", C: cA, S: sA, Sigma: sig, SigNum: true, K1: k1, K2: k2}
}
func lambertC(sig complex128) *LambertOp {
	return &LambertOp{Label: "L-C", C: cC, D: 1e-6, S: -1, Sigma: sig, SigNum: true}
}
func toroidal(sig complex128) *GenOp {
	return &GenOp{Label: "TOR", T1: sinT(complex(0.8, 0.5)), T2: cosT(complex(1.1, -0.2)), Sigma1: sig, Sigma2: sig}
}
func splineOp() *GenOp {
	return &GenOp{Label: "SPL", T1: sinW, T2: sinT(1), Sigma1: complex(0, halfPi), Sigma2: complex(0, halfPi)}
}
func wallpaper(sig complex128) *GenOp {
	return &GenOp{Label: "WAL", T1: cosSq, T2: cosT(1), Sigma1: sig, Sigma2: sig}
}

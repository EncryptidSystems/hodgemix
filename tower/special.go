package tower

import (
	"math"
	"math/cmplx"
)

const twoPi = 2 * math.Pi

func wrap(t float64) float64 { // to (-pi, pi]
	r := math.Remainder(t, twoPi)
	if r <= -math.Pi {
		r += twoPi
	}
	return r
}

func wrapIm(z complex128) complex128 { return complex(real(z), wrap(imag(z))) }

func finite(z complex128) bool {
	return !(math.IsNaN(real(z)) || math.IsNaN(imag(z)) || math.IsInf(real(z), 0) || math.IsInf(imag(z), 0))
}

// LambertW0: principal branch. Halley iteration from several initial guesses;
// a result is accepted only if it converged (small residual) AND lies in the
// range of the principal branch (Corless et al. 1996), which characterises W0 uniquely.
func LambertW0(z complex128) (complex128, bool) {
	if z == 0 {
		return 0, true
	}
	e := math.E
	guesses := []complex128{}
	if cmplx.Abs(z+complex(1/e, 0)) < 0.5 {
		pp := cmplx.Sqrt(2 * (complex(e, 0)*z + 1))
		guesses = append(guesses, -1+pp-pp*pp/3+11.0/72.0*pp*pp*pp)
	}
	if cmplx.Abs(z) < 0.3 {
		guesses = append(guesses, z*(1-z+1.5*z*z))
	}
	if z != -1 {
		guesses = append(guesses, cmplx.Log(1+z))
	}
	if cmplx.Abs(z) > 3 {
		l1 := cmplx.Log(z)
		l2 := cmplx.Log(l1)
		guesses = append(guesses, l1-l2+l2/l1)
	}
	guesses = append(guesses, complex(0.5, 0.5), complex(0.5, -0.5), complex(-0.5, 1.5), complex(-0.5, -1.5))
	for _, g := range guesses {
		if w, ok := halleyW(z, g); ok && inPrincipalRange(w) {
			return w, true
		}
	}
	return 0, false
}

func halleyW(z, w complex128) (complex128, bool) {
	for i := 0; i < 100; i++ {
		ew := cmplx.Exp(w)
		f := w*ew - z
		wp1 := w + 1
		if wp1 == 0 {
			return w, f == 0
		}
		dw := f / (ew*wp1 - (w+2)*f/(2*wp1))
		w -= dw
		if !finite(w) {
			return w, false
		}
		if cmplx.Abs(dw) <= 1e-15*(1+cmplx.Abs(w)) {
			break
		}
	}
	return w, cmplx.Abs(w*cmplx.Exp(w)-z) < 1e-12*(1+cmplx.Abs(z))
}

// principal range of W0: |Im w|<pi and Re w > -Im w cot(Im w)
func inPrincipalRange(w complex128) bool {
	x, y := real(w), imag(w)
	if math.Abs(y) >= math.Pi {
		return false
	}
	if math.Abs(y) < 1e-12 {
		return x >= -1-1e-9
	}
	return x > -y/math.Tan(y)-1e-9
}

// principal Log(sin u), computed without overflow for large |Im u|.
func LogSin(u complex128) complex128 {
	y := imag(u)
	x := real(u)
	if math.Abs(y) < 20 {
		s := cmplx.Sin(u)
		if s == 0 {
			return cmplx.Inf()
		}
		return cmplx.Log(s)
	}
	if y > 0 { // sin u = e^{-iu}(1-e^{2iu}) * (i/2)
		v := complex(y, -x) + complex(-math.Ln2, math.Pi/2) + cmplx.Log(1-cmplx.Exp(2i*u))
		return wrapIm(v)
	}
	v := complex(-y, x) + complex(-math.Ln2, -math.Pi/2) + cmplx.Log(1-cmplx.Exp(-2i*u))
	return wrapIm(v)
}

func Cot(u complex128) complex128 {
	y := imag(u)
	if y > 20 {
		return -1i
	}
	if y < -20 {
		return 1i
	}
	return cmplx.Cos(u) / cmplx.Sin(u)
}

// Log(a*sin z + sig) and its z-derivative, stable for large |Im z|.
func LogAffineSin(a, sig, z complex128) (complex128, complex128) {
	if math.Abs(imag(z)) < 600 {
		b := a*cmplx.Sin(z) + sig
		return cmplx.Log(b), a * cmplx.Cos(z) / b
	}
	ls := LogSin(z) + cmplx.Log(a)
	r := sig * cmplx.Exp(-ls)
	return wrapIm(ls + cmplx.Log(1+r)), Cot(z) / (1 + r)
}

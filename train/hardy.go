// Hardy lattice conformal weight (Hardy_lattice_conformal).
//
// Ω(x) = 2ρ / (ρ² + |x|²) on R^4. The lattice constant is
//
//	c_H = 2 - 4/ρ²,
//
// the exact infimum of (−Δ_L Ω) / Ω³ for ρ² ≥ (√97 − 7)/12.
// No constant is fitted.
package train

import "math"

// Rho is the BPST scale 13.275 = 531/40 used with the column Hessian.
const Rho = 531.0 / 40.0

// Rho0Sq is (√97 − 7)/12. Lemma 6 needs ρ² at least this.
func Rho0Sq() float64 { return (math.Sqrt(97) - 7) / 12 }

// CH is 2 − 4/ρ². The caller passes ρ, not ρ².
func CH(rho float64) float64 {
	if rho == 0 {
		return math.Inf(-1)
	}
	return 2 - 4/(rho*rho)
}

// Omega is the conformal factor of the unit 4-sphere in stereographic coordinates.
func Omega(rho float64, x [4]float64) float64 {
	q := rho*rho + x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
	return 2 * rho / q
}

// LapOmega is (Δ_L Ω)(x) on the unit lattice, four axes.
func LapOmega(rho float64, x [4]float64) float64 {
	c := Omega(rho, x)
	var s float64
	for mu := 0; mu < 4; mu++ {
		xp, xm := x, x
		xp[mu]++
		xm[mu]--
		s += Omega(rho, xp) + Omega(rho, xm) - 2*c
	}
	return s
}

// RatioClosed is (−Δ_L Ω)/Ω³ from the closed form (4.1).
func RatioClosed(rho float64, x [4]float64) float64 {
	q := rho*rho + x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
	s := q + 1
	var sum float64
	for mu := 0; mu < 4; mu++ {
		t := 4 * x[mu] * x[mu]
		sum += (s - t) / (s*s - t)
	}
	return (q * q / (2 * rho * rho)) * sum
}

// LowerF is the axis reduction F(y) of (5.1), y = |x|².
func LowerF(rho, y float64) float64 {
	R := rho * rho
	s := y + R + 1
	return ((s - 1) * (s - 1) / (2 * R)) * (3/s + (s-4*y)/(s*s-4*y))
}

// PolyP is the expanded numerator (6.1). It is positive for ρ² ≥ Rho0Sq and y ≥ 0.
func PolyP(y, R float64) float64 {
	return 4*(R+1)*(R+1)*(R+2) + 4*(6*R*R+7*R-2)*y + 4*(5*R-1)*y*y
}

// SiteIndex packs a point of the n^4 torus.
func SiteIndex(n, x0, x1, x2, x3 int) int {
	return ((x0*n+x1)*n+x2)*n + x3
}

// PiconeGap is the left side of Lemma 3 minus the right side, on the n^4 torus.
// w is positive at every site. A zero return is the identity.
func PiconeGap(n int, f, w []float64) float64 {
	N := n * n * n * n
	if len(f) != N || len(w) != N || n < 1 {
		return math.NaN()
	}
	lapW := make([]float64, N)
	var lhs, pot, rem float64
	for x0 := 0; x0 < n; x0++ {
		for x1 := 0; x1 < n; x1++ {
			for x2 := 0; x2 < n; x2++ {
				for x3 := 0; x3 < n; x3++ {
					x := [4]int{x0, x1, x2, x3}
					i := SiteIndex(n, x0, x1, x2, x3)
					for mu := 0; mu < 4; mu++ {
						y := x
						y[mu] = (y[mu] + 1) % n
						j := SiteIndex(n, y[0], y[1], y[2], y[3])
						df := f[i] - f[j]
						lhs += df * df
						lapW[i] += w[j] - w[i]
						lapW[j] += w[i] - w[j]
						gx := f[i] / w[i]
						gy := f[j] / w[j]
						dg := gx - gy
						rem += w[i] * w[j] * dg * dg
					}
				}
			}
		}
	}
	for i := range f {
		pot += f[i] * f[i] * (-lapW[i]) / w[i]
	}
	return lhs - (pot + rem)
}

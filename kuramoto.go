package main

// Kuramoto lock as the discrete KTE tick (paper_v2 Eq.(14)-(18)):
//   theta_i <- theta_i + sum_j K_ij sin(theta_j - theta_i),  K_ij = 2^-g_theta * c_ij / d_max   (omega_i = 0, Eq.(14b))
// sum_j K_ij sin(theta_j - theta_i) = cos(theta_i)(K sin theta)_i - sin(theta_i)(K cos theta)_i.

import (
	"fmt"
	"math"
	"math/cmplx"
)

type KParams struct {
	GTheta float64 // Eq.(15) strong gain 2^-4
	Weak   float64 // Eq.(15) weak/strong ratio 2^-4 (= 2^-(g_omega-g_theta))
	WLock  float64 // Eq.(16) 1/16 turn
	NLost  int     // Eq.(16)-(18) ticks per anneal stage / refine window / lost timeout
	RTol   float64 // Eq.(20)
	MaxWin int     // Eq.(18) refine window budget = levels + 2
	Anneal bool
}

type KResult struct {
	Theta     []float64
	R         float64
	Locked    []bool
	Labels    []int
	Steps     int
	Converged bool
	Windows   int
	DPhase    []float64 // window-averaged arg <e^{i(theta_k - theta_ref(k))}>
}

// Paper_v2 Eq.(16): wrap to (-pi, pi]
func wrap(x float64) float64 { return math.Remainder(x, 2*math.Pi) }

// Paper_v2 Eq.(15): K = 2^-g_theta * C / d_max with d_max = max_i sum_j C_ij (Gershgorin => Euler tick stable)
func couplingMatrix(C []float64, n int, gTheta float64) []float64 {
	dmax := 0.0
	for i := 0; i < n; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			s += C[i*n+j]
		}
		dmax = math.Max(dmax, s)
	}
	K := make([]float64, n*n)
	if dmax == 0 {
		return K
	}
	for i := range C {
		K[i] = gTheta * C[i] / dmax
	}
	return K
}

// Paper_v2 Eq.(14)-(18): anneal (Eq.18, stage s activates digit levels <= s, K ramps 0->K over NLost ticks),
// then refine windows of NLost ticks: locked iff |wrap(theta_k - theta_ref(k))| <= W_lock at window end;
// converged iff the locked set repeats and |<r> - <r>_prev| < RTol (Eq.20).
func kuramotoTick(theta []float64, K []float64, level []int, refOf []int, p KParams, tag string) KResult {
	n := len(theta)
	nst := 0
	for _, l := range level {
		nst = max(nst, l+1)
	}
	tot := p.NLost * p.MaxWin
	if p.Anneal {
		tot += nst * p.NLost
	}
	lbl := tag + " refine"
	if p.Anneal {
		lbl = tag + " anneal+refine"
	}
	setInner(lbl, float64(tot), "")
	sn, cs, ks, kc := make([]float64, n), make([]float64, n), make([]float64, n), make([]float64, n)
	act := make([]float64, n)
	tick := func(scale float64) {
		for i := range theta {
			sn[i], cs[i] = math.Sin(theta[i])*act[i], math.Cos(theta[i])*act[i]
		}
		parFor(n, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				row := K[i*n : (i+1)*n]
				var a, b float64
				for j, k := range row {
					a += k * sn[j]
					b += k * cs[j]
				}
				ks[i], kc[i] = a, b
			}
		})
		for i := range theta {
			theta[i] += act[i] * scale * (math.Cos(theta[i])*ks[i] - math.Sin(theta[i])*kc[i])
		}
	}
	rOf := func() float64 {
		var z complex128
		m := 0
		for i := range theta {
			if act[i] > 0 {
				z += cmplx.Exp(complex(0, theta[i]))
				m++
			}
		}
		return cmplx.Abs(z) / float64(max(1, m))
	}
	t := 0
	if p.Anneal {
		for s := 0; s < nst; s++ {
			for i := range act {
				act[i] = 0
				if level[i] <= s {
					act[i] = 1
				}
			}
			for q := 0; q < p.NLost; q++ {
				tick(float64(q+1) / float64(p.NLost))
				t++
				innerSet(float64(t), 0)
			}
			logf("        anneal stage %d/%d (digit level %d): r=%.6f active=%d", s+1, nst, s, rOf(), countAct(act))
		}
	}
	for i := range act {
		act[i] = 1
	}
	res := KResult{Labels: make([]int, n)}
	var prevL []bool
	prevR, have := 0.0, false
	for w := 0; w < p.MaxWin; w++ {
		zw := make([]complex128, n)
		rw := 0.0
		for q := 0; q < p.NLost; q++ {
			tick(1)
			t++
			for i := range zw {
				zw[i] += cmplx.Exp(complex(0, theta[i]-theta[refOf[i]]))
			}
			rw += rOf()
			innerSet(float64(t), 0)
		}
		rbar := rw / float64(p.NLost)
		L := make([]bool, n)
		nl := 0
		res.DPhase = make([]float64, n)
		for i := range L {
			res.DPhase[i] = cmplx.Phase(zw[i])
			L[i] = math.Abs(wrap(theta[i]-theta[refOf[i]])) <= p.WLock
			if L[i] {
				nl++
			}
		}
		res.Windows++
		pmu.Lock()
		if inner != nil {
			inner.label = fmt.Sprintf("%s <r>=%.5f locked=%d/%d", tag, rbar, nl, n)
		}
		pmu.Unlock()
		same := prevL != nil
		for i := range L {
			if same && L[i] != prevL[i] {
				same = false
			}
		}
		res.R, res.Locked = rbar, L
		if have && same && math.Abs(rbar-prevR) < p.RTol {
			res.Converged = true
			break
		}
		prevL, prevR, have = L, rbar, true
	}
	res.Theta, res.Steps = theta, t
	endInner()
	return res
}

// Paper: no equation (helper)
func countAct(a []float64) int {
	c := 0
	for _, v := range a {
		if v > 0 {
			c++
		}
	}
	return c
}

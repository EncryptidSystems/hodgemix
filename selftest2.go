package main

// paper_v2 numerical checks for the new operators (TRACE.md lists them as T1..T9).

import (
	"fmt"
	"math"
	"math/rand"
)

// Paper_v2 checks: Eq.(23)-(24), (26), (14)-(18), Sec.6, (6)-(9)
func selftestV2() {
	g := rand.New(rand.NewSource(7))
	// T1 Eq.(23): Q orthogonal: ||Q^T Q - I||_max for H in {32, 2560-like 40, 48}
	for _, H := range []int{32, 40, 48, 2560} {
		q := NewRot(H, 1)
		e := 0.0
		cols := make([][]float64, H)
		for i := 0; i < H; i++ {
			v := make([]float64, H)
			v[i] = 1
			q.RowTimesQ(v)
			cols[i] = v
		}
		step := 1
		if H > 64 {
			step = 37
		}
		for i := 0; i < H; i += step {
			for j := 0; j < H; j += step {
				d := dot(cols[i], cols[j])
				if i == j {
					d -= 1
				}
				e = math.Max(e, math.Abs(d))
			}
		}
		fmt.Printf("T1 hadamard H=%d block=%d max|QQ^T-I|=%.2e\n", H, q.B, e)
	}
	// T2 Eq.(24): RMSNorm-reader invariance: W diag(g) rms(x) == (W diag(g) Q) rms(Q^T x); writer: Q (Q^T W z) == W z
	{
		H, R := 32, 8
		q := NewRot(H, 3)
		x, gm := make([]float64, H), make([]float32, H)
		W := make([]float32, R*H)
		for i := range x {
			x[i], gm[i] = g.NormFloat64(), float32(.5+g.Float64())
		}
		for i := range W {
			W[i] = float32(g.NormFloat64())
		}
		rms := func(v []float64) []float64 {
			s := 0.0
			for _, a := range v {
				s += a * a
			}
			s = math.Sqrt(s / float64(len(v)))
			o := make([]float64, len(v))
			for i := range v {
				o[i] = v[i] / s
			}
			return o
		}
		y0 := make([]float64, R)
		nx := rms(x)
		for r := 0; r < R; r++ {
			for c := 0; c < H; c++ {
				y0[r] += float64(W[r*H+c]) * float64(gm[c]) * nx[c]
			}
		}
		Wr := append([]float32(nil), W...)
		q.Reader(Wr, R, gm)
		xr := append([]float64(nil), x...)
		q.QTTimes(xr)
		nxr := rms(xr)
		e := 0.0
		for r := 0; r < R; r++ {
			y := 0.0
			for c := 0; c < H; c++ {
				y += float64(Wr[r*H+c]) * nxr[c]
			}
			e = math.Max(e, math.Abs(y-y0[r]))
		}
		fmt.Printf("T2 rotation invariance (RMSNorm reader, folded gain): max|y_rot - y|=%.2e (float32 weights)\n", e)
	}
	// T3 Eq.(26): Whitney / de Rham: d P0 = P1 d, R0 P0 = I, R1 P1 = I, harmonic 1-cochain preserved
	{
		n := 12
		a, b := make([]float64, n), make([]float64, n)
		for i := range a {
			a[i], b[i] = g.NormFloat64(), g.NormFloat64()
		}
		mx := func(x, y []float64) float64 {
			e := 0.0
			for i := range x {
				e = math.Max(e, math.Abs(x[i]-y[i]))
			}
			return e
		}
		e1 := mx(dCycle(prolong0(a)), prolong1(dCycle(a)))
		e2 := mx(restrict0(prolong0(a)), a)
		e3 := mx(restrict1(prolong1(b)), b)
		e4 := mx(dCycle(restrict0(prolong0(a))), restrict1(prolong1(dCycle(a))))
		h := make([]float64, n)
		for i := range h {
			h[i] = 1.7
		}
		e5 := 0.0
		for _, v := range l1Cycle(prolong1(h)) {
			e5 = math.Max(e5, math.Abs(v))
		}
		fmt.Printf("T3 whitney/de Rham: |dP0-P1d|=%.1e |R0P0-I|=%.1e |R1P1-I|=%.1e |dR0-R1d|=%.1e |L1 P1 h|=%.1e (harmonic kept)\n", e1, e2, e3, e4, e5)
	}
	// T4 Eq.(5)-(6): factorisation, index size, harmonic bin count
	for _, n := range []int{8, 16, 24, 40, 2560, 6912} {
		G := gridSize(n, 0)
		fmt.Printf("T4 n=%d radices=%v index=%d radices(index)=%v\n", n, radices(n), G, radices(G))
	}
	for _, G := range [][2]int{{16, 16}, {160, 216}} {
		S := NewSpectrum(G, 0)
		nh := 0
		for _, h := range S.Harm {
			if h {
				nh++
			}
		}
		fmt.Printf("T4 grid %v low/harmonic bins=%d of %d levels=%d\n", G, nh, G[0]*G[1], S.NLevel)
	}
	// T5 Eq.(14)-(18): lock: phases within +-pi/4 lock; one oscillator at pi stays lost within budget? report
	for _, sp := range []float64{math.Pi / 4, math.Pi * 0.95} {
		n := 24
		th := make([]float64, n)
		lv, ref := make([]int, n), make([]int, n)
		C := make([]float64, n*n)
		for i := range th {
			th[i] = (2*g.Float64() - 1) * sp
			if i%8 == 0 {
				th[i] = 0
			}
			ref[i] = (i / 8) * 8
			lv[i] = i % 3
			for j := 0; j < n; j++ {
				if i != j {
					C[i*n+j] = 1
					if lv[i] != lv[j] {
						C[i*n+j] = 1.0 / 16
					}
				}
			}
		}
		K := couplingMatrix(C, n, 1.0/16)
		r := kuramotoTick(th, K, lv, ref, KParams{GTheta: 1.0 / 16, Weak: 1.0 / 16, WLock: WLock, NLost: NLost, RTol: RefineTol, MaxWin: 5, Anneal: true}, "selftest")
		nl := 0
		for _, x := range r.Locked {
			if x {
				nl++
			}
		}
		fmt.Printf("T5 kuramoto tick, initial spread +-%.2f rad: <r>=%.6f locked=%d/%d converged=%v ticks=%d\n", sp, r.R, nl, n, r.Converged, r.Steps)
	}
	// T6 Sec.6: BNZ4 chunk round trip, CRC, invertibility rejection
	{
		cm := &ChanMap{Name: "x.weight", Rout: 5, Cout: 3, Gr: 4, Gc: 2, RadR: []uint8{2, 2}, RadC: []uint8{2},
			C:    []Contrib{{Src: 1, Bucket: 3, W: .5, Shift: []uint8{1, 0, 1}, PR: []int32{2, -1, 4, 3}, PC: []int32{1, 2}}},
			ProR: hatWeights([]float64{.1, .3, .5, .7, .9}, 4), ProC: hatWeights([]float64{.2, .5, .8}, 2),
			Bins: []uint32{5}, Re: []float32{1.5}, Im: []float32{-2}}
		b := cm.Encode()
		d, err := DecodeChanMap(b)
		ok := err == nil && string(d.Encode()) == string(b)
		b[10] ^= 1
		_, e2 := DecodeChanMap(b)
		cm.C[0].PR[1] = 2
		_, e3 := DecodeChanMap(cm.Encode())
		fmt.Printf("T6 BNZ4 chunk: %d B round-trip=%v corrupted->%v duplicate-perm->%v\n", len(b), ok, e2, e3)
	}
	// T7 Eq.(10): hat weights reproduce sampleGrid exactly
	{
		G := [2]int{8, 1}
		S := make([]float64, 8)
		for i := range S {
			S[i] = g.NormFloat64()
		}
		x := []float64{.01, .2, .33, .5, .77, .99}
		v := sampleGrid(S, G, x, []float64{.5})
		pw := hatWeights(x, 8)
		e := 0.0
		for i, p := range pw {
			y := float64(p.WL) * S[p.Left]
			if p.WR != 0 {
				y += float64(p.WR) * S[p.Left+1]
			}
			e = math.Max(e, math.Abs(y-v[i]))
		}
		fmt.Printf("T7 BNZ4 prolongation weights vs sampleGrid: max err=%.1e (float32 weights)\n", e)
	}
	// T8 Eq.(7)-(9): keys, MRH order, bucket
	{
		W := []float32{1, 0.1, -1, 0.9, 0, 0, 2, -3}
		k := heapKey(W, 2, 4, 0)
		fmt.Printf("T8 MRH keys rows=%v coords=%v bucket=%d\n", k, chanCoords(k), bucketOf(k))
	}
}

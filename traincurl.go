package main

// trainCurl is the training step.
//
// Ricci–DeTurck is already the gradient. The energy is the Dirichlet–Hardy
// energy whose L2 gradient is −Δf + c_H Ω² f, and train.Flow is one explicit
// descent step of that energy, time step 1/8, neck cap included. There is no
// second gradient and no step-size search.
//
// The Hodge split decides which part of a weight grid that gradient is
// allowed to see:
//
//   - gradient bins are the DeTurck gauge and stay with the keep grid
//   - harmonic bins are the lock and stay with the keep grid
//   - the curl of the source is the residual Flow trains
//
// TrainCurl's keep grid is the reference, after Align. The mix's keep grid
// is the assembled tensor, so a locked harmonic is written back after the
// step. Neck sites stay the reference. Flow runs once.

import (
	"math"

	"hodgemix/train"
)

// TrainCurl aligns src to ref, then takes one curl step toward ref.
// The gauge and the harmonic stay with ref.
func TrainCurl(src, ref []float64, rows, cols int) train.Result {
	aligned, permR, permC, _, _ := train.Align(src, ref, rows, cols)
	empty := train.Result{PermR: permR, PermC: permC, CH: train.CH(train.Rho), Rho: train.Rho, DT: train.DT}
	if rows < 1 || cols < 1 || len(aligned) != rows*cols || len(ref) != len(aligned) {
		empty.Grid = append([]float64(nil), aligned...)
		return empty
	}
	g, neck := descend(aligned, ref, ref, rows, cols)
	empty.Grid = g
	empty.Neck = neck
	return empty
}

// descend is one training step. src supplies the curl. keep supplies the
// gauge and the harmonic, which are written back onto the reference after
// the step. A neck site stays the reference value.
func descend(src, ref, keep []float64, rows, cols int) ([]float64, int) {
	n := rows * cols
	if rows < 1 || cols < 1 || len(src) != n || len(ref) != n || len(keep) != n {
		return append([]float64(nil), src...), 0
	}
	S := NewSpectrum([2]int{rows, cols}, 0)
	ps := S.Hodge(src)
	pr := S.Hodge(ref)
	pk := S.Hodge(keep)
	inv := func(f []complex128) []float64 { return realPart(fftn(f, S.dims, true)) }
	curl := inv(ps.CurlF)
	refCurl := inv(pr.CurlF)
	keepGrad := inv(pk.GradF)
	keepHarm := inv(pk.HarmF)
	refGrad := inv(pr.GradF)
	refHarm := inv(pr.HarmF)
	residual := make([]float64, n)
	grid := make([]float64, n)
	for i := range residual {
		residual[i] = curl[i] - refCurl[i]
		grid[i] = ref[i] + residual[i]
	}
	lap := train.Lap2(residual, rows, cols)
	kappa := 6 / (train.Rho * train.Rho)
	stepped, neck := train.Flow(grid, ref, rows, cols)
	for i := range stepped {
		if math.Abs(lap[i]) >= kappa {
			continue
		}
		stepped[i] += (keepGrad[i] - refGrad[i]) + (keepHarm[i] - refHarm[i])
	}
	return stepped, neck
}

// hardyEnergy is the energy Flow descends on a residual.
// E(f) = ½ ⟨f, −Δf⟩ + (c_H/2) ⟨Ω² f, f⟩, with Δ and Ω from train.Flow.
func hardyEnergy(f []float64, rows, cols int) float64 {
	if rows < 1 || cols < 1 || len(f) != rows*cols {
		return math.NaN()
	}
	lap := train.Lap2(f, rows, cols)
	cH := train.CH(train.Rho)
	var e float64
	for i, v := range f {
		r := i / cols
		c := i % cols
		x := [4]float64{float64(r) - float64(rows)/2, float64(c) - float64(cols)/2, 0, 0}
		om := train.Omega(train.Rho, x)
		e += 0.5*v*(-lap[i]) + 0.5*cH*om*om*v*v
	}
	return e
}

// curlOf is the inverse DFT of the curl bins. The result is real for a real grid.
func curlOf(grid []float64, rows, cols int) []float64 {
	S := NewSpectrum([2]int{rows, cols}, 0)
	p := S.Hodge(grid)
	return realPart(fftn(p.CurlF, S.dims, true))
}

// gradOf is the inverse DFT of the gradient bins.
func gradOf(grid []float64, rows, cols int) []float64 {
	S := NewSpectrum([2]int{rows, cols}, 0)
	p := S.Hodge(grid)
	return realPart(fftn(p.GradF, S.dims, true))
}


// One Ricci–DeTurck step on the weight torus (discrete_ricci_flow_M4).
//
// The semi-discrete scheme is a single explicit step of the scalar principal
// part. The time step is the 4-dimensional mesh constant 1/8. The zeroth-order
// term is the Hardy potential c_H Ω². The step is not repeated and the step
// size is not searched.
//
// Uncertified fibres are already the cone cap before this step. A lattice neck
// is the vertex set of a flagged star; capping it writes the base value.
package train

import "math"

// DT is the explicit-step size. On the unit 4-lattice the symbol of Δ_L lies
// in [−16, 0], so 1/8 keeps 1 + dt λ inside [−1, 1] up to the Hardy term,
// which is O(1/ρ²).
const DT = 1.0 / 8.0

// FactorRadius2 is the exact S² factor in Proposition 7.7: R(t)² = R(0)² − 2t.
func FactorRadius2(r0, t float64) float64 { return r0*r0 - 2*t }

// Lap2 is the periodic lattice Laplacian on the row-column torus.
func Lap2(f []float64, rows, cols int) []float64 {
	o := make([]float64, len(f))
	if rows < 1 || cols < 1 || len(f) != rows*cols {
		return o
	}
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			i := r*cols + c
			rp := ((r+1)%rows)*cols + c
			rm := ((r-1+rows)%rows)*cols + c
			cp := r*cols + (c+1)%cols
			cm := r*cols + (c-1+cols)%cols
			o[i] = f[rp] + f[rm] + f[cp] + f[cm] - 4*f[i]
		}
	}
	return o
}

// StepResidual applies one step to a residual and returns the updated residual
// and how many neck sites were capped. kappa ≤ 0 skips the neck.
func StepResidual(f []float64, rows, cols int, kappa float64) (next []float64, neck int) {
	next = append([]float64(nil), f...)
	if rows < 1 || cols < 1 || len(f) != rows*cols {
		return next, 0
	}
	lap := Lap2(f, rows, cols)
	cH := CH(Rho)
	for i := range f {
		r := i / cols
		c := i % cols
		x := [4]float64{float64(r) - float64(rows)/2, float64(c) - float64(cols)/2, 0, 0}
		om := Omega(Rho, x)
		next[i] = f[i] + DT*lap[i] - DT*cH*om*om*f[i]
	}
	if kappa > 0 {
		for i := range lap {
			if math.Abs(lap[i]) >= kappa {
				next[i] = 0
				neck++
			}
		}
	}
	return next, neck
}

// Flow is one training step of a mixed grid toward the base grid.
// The neck threshold is the cylinder scale 6/ρ² of §8. A site whose
// discrete Laplacian meets that scale is capped to the base residual, 0.
func Flow(grid, ref []float64, rows, cols int) (out []float64, neck int) {
	if rows < 1 || cols < 1 || len(grid) != rows*cols || len(ref) != len(grid) {
		return append([]float64(nil), grid...), 0
	}
	f := make([]float64, len(grid))
	var scale, res float64
	for i := range f {
		f[i] = grid[i] - ref[i]
		if a := math.Abs(f[i]); a > res {
			res = a
		}
		if a := math.Abs(ref[i]); a > scale {
			scale = a
		}
	}
	if res <= 1e-12*(scale+1) {
		return append([]float64(nil), grid...), 0
	}
	kappa := 6 / (Rho * Rho)
	nxt, neck := StepResidual(f, rows, cols, kappa)
	out = make([]float64, len(grid))
	for i := range out {
		out[i] = ref[i] + nxt[i]
	}
	return out, neck
}

// Train aligns src to ref and takes one Ricci–DeTurck step of the residual.
type Result struct {
	Grid  []float64
	PermR []int
	PermC []int
	Neck  int
	CH    float64
	Rho   float64
	DT    float64
}

// Train is the closed training step: gap selection, then one Ricci–DeTurck step.
func Train(src, ref []float64, rows, cols int) Result {
	aligned, pr, pc, _, _ := Align(src, ref, rows, cols)
	g, neck := Flow(aligned, ref, rows, cols)
	return Result{Grid: g, PermR: pr, PermC: pc, Neck: neck, CH: CH(Rho), Rho: Rho, DT: DT}
}

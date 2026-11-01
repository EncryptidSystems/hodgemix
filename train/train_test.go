package train

import (
	"math"
	"math/rand"
	"testing"
)

func TestHardyConstant(t *testing.T) {
	R0 := Rho0Sq()
	if math.Abs(R0-((math.Sqrt(97)-7)/12)) > 1e-15 {
		t.Fatalf("rho0^2 %v", R0)
	}
	c := CH(Rho)
	// 2 - 4/(531/40)^2 = 1.977302...
	if math.Abs(c-1.977302) > 5e-7 {
		t.Fatalf("c_H %v", c)
	}
	if PolyP(1, R0) <= 0 {
		t.Fatal("P is not positive at the Lemma 6 threshold")
	}
	for _, y := range []float64{0, 0.5, 3, 40, 1000} {
		if PolyP(y, Rho*Rho) <= 0 {
			t.Fatalf("P(%v) not positive", y)
		}
		if LowerF(Rho, y) <= c {
			t.Fatalf("F(%v) = %v, c_H = %v", y, LowerF(Rho, y), c)
		}
	}
	// (6.1) matches F − L times the positive denominator.
	R := Rho * Rho
	L := 2 - 4/R
	for _, y := range []float64{0, 1, 7.5, 80} {
		s := y + R + 1
		den := 2 * R * s * (s*s - 4*y)
		got := (LowerF(Rho, y) - L) * den
		if math.Abs(got-PolyP(y, R)) > 1e-6*math.Abs(PolyP(y, R)) {
			t.Fatalf("P expansion y=%v got %v want %v", y, got, PolyP(y, R))
		}
	}
}

func TestRatioClosed(t *testing.T) {
	for _, x := range [][4]float64{{0.5, 0.5, 0.5, 0.5}, {10.5, 0.5, 0.5, 0.5}, {0, 0, 0, 0}, {3, -2, 1, 4}} {
		om := Omega(Rho, x)
		direct := -LapOmega(Rho, x) / (om * om * om)
		closed := RatioClosed(Rho, x)
		if math.Abs(direct-closed) > 1e-9*math.Abs(closed) {
			t.Fatalf("at %v direct %v closed %v", x, direct, closed)
		}
		if closed < CH(Rho)-1e-9 {
			t.Fatalf("ratio %v under c_H at %v", closed, x)
		}
	}
}

func TestPicone(t *testing.T) {
	const n = 3
	N := n * n * n * n
	f := make([]float64, N)
	w := make([]float64, N)
	g := rand.New(rand.NewSource(1))
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				for l := 0; l < n; l++ {
					id := SiteIndex(n, i, j, k, l)
					f[id] = g.NormFloat64()
					w[id] = Omega(Rho, [4]float64{float64(i), float64(j), float64(k), float64(l)})
				}
			}
		}
	}
	if d := PiconeGap(n, f, w); math.Abs(d) > 1e-8 {
		t.Fatalf("Picone gap %v", d)
	}
}

func TestAlignRecoversPermutation(t *testing.T) {
	const rows, cols = 6, 4
	ref := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		ref[i*cols+i%cols] = 1
		ref[i*cols+(i+1)%cols] = 0.25 * float64(i+1)
	}
	perm := []int{3, 0, 5, 1, 4, 2}
	A := make([]float64, len(ref))
	for i := 0; i < rows; i++ {
		copy(A[perm[i]*cols:(perm[i]+1)*cols], ref[i*cols:(i+1)*cols])
	}
	out, pr, _, _, _ := Align(A, ref, rows, cols)
	for i := range out {
		if out[i] != ref[i] {
			t.Fatalf("site %d got %v want %v", i, out[i], ref[i])
		}
	}
	for i := 0; i < rows; i++ {
		if pr[i] != perm[i] {
			t.Fatalf("row %d perm %d want %d", i, pr[i], perm[i])
		}
	}
}

func TestAlignCapsCollision(t *testing.T) {
	// Two equal source rows cannot be a bijection onto two separated base rows
	// when both sit on the same point. The whole fibre is uncertified if the
	// base rows themselves collide, and the cap is the base.
	ref := []float64{1, 0, 0, 1}
	A := []float64{0, 1, 1, 0}
	out, _, _, _, _ := Align(A, ref, 2, 2)
	// Distinct base rows, source is a swap: this one is certified.
	for i := range out {
		if out[i] != ref[i] {
			t.Fatalf("swap site %d %v", i, out[i])
		}
	}
	// Identical base rows: gap is 0, nothing is certified, output is the base.
	same := []float64{1, 2, 1, 2}
	src := []float64{9, 9, 8, 8}
	capped, _, _, _, _ := Align(src, same, 2, 2)
	for i := range capped {
		if capped[i] != same[i] {
			t.Fatalf("cap site %d got %v", i, capped[i])
		}
	}
}

func TestSelectAngles(t *testing.T) {
	dst := [][2]float64{{0, 0}, {1.2, -0.4}, {2.2, 0.7}, {-1, 1.5}}
	order := []int{2, 0, 3, 1}
	src := make([][2]float64, len(dst))
	for i, j := range order {
		src[j] = dst[i]
	}
	perm, hits := SelectAngles(src, dst)
	if hits != len(dst) {
		t.Fatalf("hits %d", hits)
	}
	for i, j := range order {
		if perm[i] != j {
			t.Fatalf("slot %d source %d want %d", i, perm[i], j)
		}
	}
}

func TestBranchAndChern(t *testing.T) {
	if BranchLevel(0, []int{3, 4, 5}) != 0 {
		t.Fatal("zero difference")
	}
	// diff = 1 is the finest digit: level 1.
	if BranchLevel(1, []int{3, 4, 5}) != 1 {
		t.Fatalf("finest %d", BranchLevel(1, []int{3, 4, 5}))
	}
	// place value of digit 2 (0-based index 2) is 3*4 = 12. Level 3.
	if BranchLevel(12, []int{3, 4, 5}) != 3 {
		t.Fatalf("coarsest %d", BranchLevel(12, []int{3, 4, 5}))
	}
	if CyclicDiff(1, 11, 12) != 2 {
		t.Fatal(CyclicDiff(1, 11, 12))
	}
	if Chern2(1, 1, 1, 0, 0) != -1 {
		t.Fatal(Chern2(1, 1, 1, 0, 0))
	}
	if Chern2(2, 1, 0, 0, 1) != 0 {
		t.Fatal("k2 l3 - k3 l2 = 0")
	}
	if Chern2(1, 1, 2, 0, 3) != -2 {
		t.Fatalf("table (1,2; 0,3) per sheet wants -2, got %d", Chern2(1, 1, 2, 0, 3))
	}
	if math.Abs(StripDelta([]float64{0.1, -0.2, 0.3})-0.6) > 1e-15 {
		t.Fatal(StripDelta([]float64{0.1, -0.2, 0.3}))
	}
	if ConeCap(2, 6, 0.25) != 3 {
		t.Fatal(ConeCap(2, 6, 0.25))
	}
}

func TestRicciStep(t *testing.T) {
	if FactorRadius2(3, 0.5) != 9-1 {
		t.Fatal(FactorRadius2(3, 0.5))
	}
	const rows, cols = 8, 8
	f := make([]float64, rows*cols)
	for r := 0; r < rows; r++ {
		s := math.Sin(2 * math.Pi * float64(r) / float64(rows))
		for c := 0; c < cols; c++ {
			f[r*cols+c] = s
		}
	}
	alpha := 2 * math.Pi / float64(rows)
	lambda := 2 * (math.Cos(alpha) - 1)
	got, neck := StepResidual(f, rows, cols, 0)
	if neck != 0 {
		t.Fatalf("neck %d with kappa 0", neck)
	}
	for i := range f {
		r := i / cols
		c := i % cols
		x := [4]float64{float64(r) - float64(rows)/2, float64(c) - float64(cols)/2, 0, 0}
		om := Omega(Rho, x)
		want := f[i] * (1 + DT*lambda - DT*CH(Rho)*om*om)
		if math.Abs(got[i]-want) > 1e-9 {
			t.Fatalf("site %d got %v want %v", i, got[i], want)
		}
	}
	// A flat residual is the identity. A one-site spike is a neck under 6/ρ²
	// because its laplacian is 4 at that scale, above 6/ρ² ≈ 0.034, and the
	// spike itself is capped.
	ref := make([]float64, rows*cols)
	grid := make([]float64, len(ref))
	grid[0] = 1
	out, n := Flow(grid, ref, rows, cols)
	if n < 1 {
		t.Fatal("spike was not capped")
	}
	if out[0] != 0 {
		t.Fatalf("spike residual %v", out[0])
	}
	same, n2 := Flow(ref, ref, rows, cols)
	if n2 != 0 || same[3] != ref[3] {
		t.Fatalf("flat flow neck %d", n2)
	}
}

func TestTrainPermutationIsBase(t *testing.T) {
	const rows, cols = 5, 3
	ref := make([]float64, rows*cols)
	g := rand.New(rand.NewSource(2))
	for i := range ref {
		ref[i] = g.NormFloat64()
	}
	perm := []int{4, 1, 0, 3, 2}
	src := make([]float64, len(ref))
	for i := 0; i < rows; i++ {
		copy(src[perm[i]*cols:(perm[i]+1)*cols], ref[i*cols:(i+1)*cols])
	}
	tr := Train(src, ref, rows, cols)
	for i := range ref {
		if math.Abs(tr.Grid[i]-ref[i]) > 1e-12 {
			t.Fatalf("trained site %d %v", i, tr.Grid[i])
		}
	}
	if tr.DT != DT || tr.Rho != Rho {
		t.Fatalf("record %+v", tr)
	}
}

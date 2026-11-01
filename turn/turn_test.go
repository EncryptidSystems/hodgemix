package turn

import (
	"math"
	"math/rand"
	"testing"
)

func rot2(theta float64) []float64 {
	c, s := math.Cos(theta), math.Sin(theta)
	return []float64{c, -s, s, c}
}

func closeM(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len %d want %d", len(got), len(want))
	}
	var s float64
	for i := range got {
		d := got[i] - want[i]
		s += d * d
	}
	if math.Sqrt(s) > tol {
		t.Fatalf("diff %g\ngot %v\nwant %v", math.Sqrt(s), got, want)
	}
}

func TestJacobiDiagonal(t *testing.T) {
	a := []float64{2, 0, 0, 0, 5, 0, 0, 0, 3}
	Q, lam := jacobi(a, 3)
	D := make([]float64, 9)
	for c := 0; c < 3; c++ {
		for r := 0; r < 3; r++ {
			D[r*3+c] = Q[r*3+c] * lam[c]
		}
	}
	back := mul(D, transpose(Q, 3, 3), 3, 3, 3)
	closeM(t, back, a, 1e-9)
}

func parents(t *testing.T) (mother, father Parent) {
	t.Helper()
	mother = Parent{
		Writer: []float64{1, 0.2, 0, 1}, Chart: []float64{1, 0, 0, 0, 1, 0},
		WriterLabel: "mother/writer", ChartLabel: "mother/chart",
		M: 2, K: 2, N: 3,
	}
	father = Parent{
		Writer: []float64{0.3, 1, 1, -0.4}, Chart: []float64{0, 1, 0, 1, 0, 0.5},
		WriterLabel: "father/writer", ChartLabel: "father/chart",
		M: 2, K: 2, N: 3,
	}
	return mother, father
}

func TestChildMatchesNeitherParent(t *testing.T) {
	mother, father := parents(t)
	got, ok := Child(mother, father)
	if !ok {
		t.Fatal("no child")
	}
	for _, tensor := range []struct {
		name string
		got  []float64
		ban  [][]float64
	}{
		{"writer", got.Writer, [][]float64{mother.Writer, father.Writer}},
		{"chart", got.Chart, [][]float64{mother.Chart, father.Chart}},
	} {
		for _, p := range tensor.ban {
			if sameTensor(tensor.got, p) {
				t.Fatalf("child %s matches a parent", tensor.name)
			}
		}
	}
	Cm := Circuit(mother.Writer, mother.Chart, mother.M, mother.K, mother.N)
	Cf := Circuit(father.Writer, father.Chart, father.M, father.K, father.N)
	P := Circuit(got.Writer, got.Chart, got.M, got.K, got.N)
	if sameTensor(P, Cm) || sameTensor(P, Cf) {
		t.Fatal("child circuit matches a parent circuit")
	}
	want, ok := inherit(Cm, Cf, mother.M, mother.N)
	if !ok {
		t.Fatal("inherit")
	}
	closeM(t, P, want, 1e-6)
	if energy(P)+1e-6 < energy(Cm) || energy(P)+1e-6 < energy(Cf) {
		t.Fatalf("child energy %g mother %g father %g", energy(P), energy(Cm), energy(Cf))
	}
	banned := map[string]bool{
		mother.WriterLabel: true, mother.ChartLabel: true,
		father.WriterLabel: true, father.ChartLabel: true,
	}
	if banned[got.WriterLabel] || banned[got.ChartLabel] || got.WriterLabel == got.ChartLabel {
		t.Fatalf("labels %q %q", got.WriterLabel, got.ChartLabel)
	}
}

func TestIdenticalParentsAreStillAChild(t *testing.T) {
	mother, _ := parents(t)
	father := mother
	father.WriterLabel = "father/writer"
	father.ChartLabel = "father/chart"
	got, ok := Child(mother, father)
	if !ok {
		t.Fatal("no child")
	}
	if sameTensor(got.Writer, mother.Writer) || sameTensor(got.Chart, mother.Chart) {
		t.Fatal("child tensors are the parent")
	}
	Cm := Circuit(mother.Writer, mother.Chart, mother.M, mother.K, mother.N)
	P := Circuit(got.Writer, got.Chart, got.M, got.K, got.N)
	if sameTensor(P, Cm) {
		t.Fatal("child circuit is the parent circuit")
	}
	if got.WriterLabel == mother.WriterLabel || got.ChartLabel == mother.ChartLabel {
		t.Fatal("child took a parent label")
	}
}

func TestQuarterTurnStillInherits(t *testing.T) {
	mother, _ := parents(t)
	R := rot2(math.Pi / 2)
	father := Parent{
		Writer:      mul(R, mother.Writer, 2, 2, 2),
		Chart:       append([]float64(nil), mother.Chart...),
		WriterLabel: "father/writer", ChartLabel: "father/chart",
		M: 2, K: 2, N: 3,
	}
	got, ok := Child(mother, father)
	if !ok {
		t.Fatal("a quarter turn produced no child")
	}
	if sameTensor(got.Writer, mother.Writer) || sameTensor(got.Writer, father.Writer) {
		t.Fatal("writer is a parent")
	}
	if sameTensor(got.Chart, mother.Chart) || sameTensor(got.Chart, father.Chart) {
		t.Fatal("chart is a parent")
	}
	P := Circuit(got.Writer, got.Chart, got.M, got.K, got.N)
	Cm := Circuit(mother.Writer, mother.Chart, 2, 2, 3)
	Cf := Circuit(father.Writer, father.Chart, 2, 2, 3)
	if energy(P)+1e-6 < energy(Cm) || energy(P)+1e-6 < energy(Cf) {
		t.Fatalf("energy %g mother %g father %g", energy(P), energy(Cm), energy(Cf))
	}
}

func TestOneParentIsNotAChild(t *testing.T) {
	mother, father := parents(t)
	father.Writer = make([]float64, len(father.Writer))
	if _, ok := Child(mother, father); ok {
		t.Fatal("empty father produced a child")
	}
}

func TestOccupiedLabelsAreLeftToTheParents(t *testing.T) {
	mother, father := parents(t)
	mother.WriterLabel = "child/writer"
	mother.ChartLabel = "child/chart"
	father.WriterLabel = "child/writer/2"
	father.ChartLabel = "child/chart/2"
	got, ok := Child(mother, father)
	if !ok {
		t.Fatal("no child")
	}
	banned := map[string]bool{
		"child/writer": true, "child/chart": true,
		"child/writer/2": true, "child/chart/2": true,
	}
	if banned[got.WriterLabel] || banned[got.ChartLabel] {
		t.Fatalf("labels %q %q", got.WriterLabel, got.ChartLabel)
	}
}

func TestRandomParents(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const m, k, n = 4, 3, 5
	makeP := func(wl, cl string) Parent {
		B := make([]float64, m*k)
		A := make([]float64, k*n)
		for i := range B {
			B[i] = rng.NormFloat64()
		}
		for i := range A {
			A[i] = rng.NormFloat64()
		}
		return Parent{Writer: B, Chart: A, WriterLabel: wl, ChartLabel: cl, M: m, K: k, N: n}
	}
	mother := makeP("layers.0.w", "layers.0.c")
	father := makeP("blocks.0.w", "blocks.0.c")
	got, ok := Child(mother, father)
	if !ok {
		t.Fatal("no child")
	}
	Cm := Circuit(mother.Writer, mother.Chart, m, k, n)
	Cf := Circuit(father.Writer, father.Chart, m, k, n)
	P := Circuit(got.Writer, got.Chart, got.M, got.K, got.N)
	want, ok := inherit(Cm, Cf, m, n)
	if !ok {
		t.Fatal("inherit")
	}
	closeM(t, P, want, 1e-5)
	if sameTensor(got.Writer, mother.Writer) || sameTensor(got.Writer, father.Writer) {
		t.Fatal("writer matches")
	}
	if sameTensor(got.Chart, mother.Chart) || sameTensor(got.Chart, father.Chart) {
		t.Fatal("chart matches")
	}
	if got.WriterLabel == mother.WriterLabel || got.WriterLabel == father.WriterLabel {
		t.Fatal("writer label")
	}
	if got.ChartLabel == mother.ChartLabel || got.ChartLabel == father.ChartLabel {
		t.Fatal("chart label")
	}
}

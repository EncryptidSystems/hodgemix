package main

// Dynamic MRH limbs (--limbs, --limb-tol): a weight is a digit string, not one ternary code.
//
// The output of a mix is a float tensor W that ternarization flattens: one code in {-1, 0, +1} per
// weight keeps what the base had and loses most of what the merge added. Following the multi-limb
// digits of unified_main.pdf (Sec. 10.1), W is written as
//     W ~ sum_l s_l (D_l - c_r),     D_l digits of radix r_l in {2, 3, 4} (2 bits), c_r = (r-1)/2,
// limb 0 the ternary weight (r = 3, the BitNet code every loader reads), each further limb the
// quantized residual of the earlier ones, its radix the one of {2, 3, 4} that leaves the smallest
// residual and its scale the least-squares one for its digits. A T1 neck (singularity surgery §5)
// is cut out of that residual and left capped at limb 0, so a spike cannot set the limb's step.
// Each further limb takes the Kuramoto lock against limb 0 (paper_v2 Eq.(14)–(20)); a harmonic bin
// that does not lock stays with limb 0. Body limbs are added while the residual is above
// --limb-tol of ||W|| and the count is below --limbs; a tensor the mix barely moved stays one
// body limb. The singular neck is then coded on its own limbs even when --limbs is 1, so a
// spike is not left as an uncoded trit and does not set the body scale. Limb l >= 1
// of tensor X.weight is X.weight_limb<l> (U8, packed like limb 0, raw digits),
// X.weight_limb<l>_scale (BF16 [1]) and X.weight_limb<l>_radix (I32 [1]).

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"strconv"
	"sync"

	"hodgemix/eval"
)

// MaxLimbs is --limbs; LimbTol is --limb-tol.
var (
	MaxLimbs = 1 // limbs carry the float mix faithfully; on 2B4T + Qwen3 that is ppl 4142 vs 14.4 (README)
	LimbTol  = 0.15
)

// A config with rope_scaling.rope_type llama3 (Llama 3, and the Champ file written from it)
// packs this many limbs and stops at this fraction of ||W||. Other runs keep --limbs and
// --limb-tol. The default stays 1: extra limbs on 2B4T + Qwen3 scored ppl 4142.
// NeckLimbs is not part of that budget. After the body limbs, this many limbs code the
// singular neck on its own scale. The body limbs stay at center there, so a spike does
// not set the body scale, and the neck is not left as an uncoded trit.
const (
	Llama3Limbs = 6
	Llama3Tol   = 0.02
	NeckLimbs   = 4
)

// llama3Arch is a Llama 3 run: rope_type llama3. A bare model_type llama is not enough
// (that name also covers earlier Llamas). A Champ file keeps the base rope, so the
// packed product is detected the same way.
func llama3Arch(cfg map[string]any) bool {
	return eval.RopeType(eval.RopeDict(cfg)) == "llama3"
}

// llama3Budget is the neck pack for a Llama 3 run. An explicit --limbs above Llama3Limbs
// or a --limb-tol below Llama3Tol is kept. A default run (1 limb, tol 0.15) becomes 6 and 0.02.
func llama3Budget(limbs int, tol float64) (int, float64) {
	if limbs < Llama3Limbs {
		limbs = Llama3Limbs
	}
	if tol > Llama3Tol {
		tol = Llama3Tol
	}
	return limbs, tol
}

func limbName(k string, l int) string { return k + "_limb" + strconv.Itoa(l) }

// limbData is one extra limb of a tensor.
type limbData struct {
	r int
	d []uint8 // digits 0..r-1, [R][C]
	s float64
}

// quantLimb is the best radix-r digit matrix for x: digits clip(round(x/step)) + c over half-octave
// steps from absmean/16 to absmean×16, then the least-squares scale for those digits.
// The grid is the body's. A singular entry is cut out before this fit (singularNeck); a step that
// runs up to that entry codes the spike and puts the body on the zero digit. err is ||x - s(d - c)||^2.
func quantLimb(x []float32, r int) (d []uint8, s, err float64) {
	c := (r - 1) / 2
	lo, hi := float64(-c), float64(r-1-c)
	am := float64(absmean(x))
	if am == 0 {
		d = make([]uint8, len(x))
		for i := range d {
			d[i] = uint8(c)
		}
		return d, 0, 0
	}
	best := math.Inf(1)
	var bs float64
	for k := -8; k <= 8; k++ {
		step := am * math.Pow(2, float64(k)/2)
		num, den, sum := limbFit(x, step, lo, hi)
		if den == 0 {
			continue
		}
		e := sum - num*num/den
		if e < best {
			best, bs = e, step
		}
	}
	d = make([]uint8, len(x))
	var num, den float64
	for i, v := range x {
		q := math.Max(lo, math.Min(hi, math.RoundToEven(float64(v)/bs)))
		d[i] = uint8(int(q) + c)
		num += float64(v) * q
		den += q * q
	}
	if den > 0 {
		s = num / den
	}
	for i, v := range x {
		e := float64(v) - s*float64(int(d[i])-c)
		err += e * e
	}
	return d, s, err
}

// limbFit is the least-squares fit of clip(round(x/step)) to x: x·q, ||q||^2, ||x||^2.
func limbFit(x []float32, step, lo, hi float64) (num, den, sum float64) {
	fold := func(a, b int) (num, den, sum float64) {
		for _, v := range x[a:b] {
			fv := float64(v)
			sum += fv * fv
			q := math.Max(lo, math.Min(hi, math.RoundToEven(fv/step)))
			num += fv * q
			den += q * q
		}
		return num, den, sum
	}
	if len(x) < 1<<16 {
		return fold(0, len(x))
	}
	const P = 8
	type part struct{ num, den, sum float64 }
	parts := make([]part, P)
	var wg sync.WaitGroup
	for p := 0; p < P; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			n, d, s := fold(p*len(x)/P, (p+1)*len(x)/P)
			parts[p] = part{n, d, s}
		}(p)
	}
	wg.Wait()
	for _, p := range parts {
		num += p.num
		den += p.den
		sum += p.sum
	}
	return num, den, sum
}

// t1Tau is the T1 extreme-value threshold of singularity surgery §5: for n i.i.d. Gaussian
// coordinates, max |x| / median|x| stays at most this with probability 1 - 2^-8. The 2^-8 is the
// BF16 resolution of paper_v2 App. A. τ_2560 = 7.67, τ_6912 = 7.95.
func t1Tau(n int) float64 {
	if n < 2 {
		return math.Inf(1)
	}
	// Φ^{-1}(3/4) = √2 erfinv(1/2): median |x| of a Gaussian is this many σ.
	z := math.Sqrt2 * math.Erfinv(0.5)
	return math.Sqrt(2*math.Log(float64(n)/math.Ldexp(1, -8))) / z
}

// singularNeck is the T1 neck of x: entries with |x| beyond τ·median|x|. Nil when there is none.
// Surgery cuts this set out of the field the later limbs are fit on and out of the lock.
func singularNeck(x []float32) []bool {
	n := len(x)
	if n < 2 {
		return nil
	}
	abs := make([]float64, n)
	for i, v := range x {
		abs[i] = math.Abs(float64(v))
	}
	med := median(abs)
	if med == 0 {
		return nil
	}
	thr := t1Tau(n) * med
	var mask []bool
	for i, a := range abs {
		if a > thr {
			if mask == nil {
				mask = make([]bool, n)
			}
			mask[i] = true
		}
	}
	return mask
}

// residualLimbs are the limbs after limb 0 (ternary codes q0, scale s0) of W, up to max-1 of them,
// while the residual is above tol of ||W||. R, C is the channel shape (R*C = len(W)). rel is the
// relative residual after each limb (rel[0] after limb 0).
//
// The singular neck is cut out of the residual before any body limb is fit, so a spike does not
// choose the body step (Alg. S steps 5–6). Each body limb takes the Kuramoto lock against limb 0;
// a harmonic bin that does not lock stays with limb 0. The neck is then coded on NeckLimbs of its
// own: center on the body, its own scale, no lock. Locking it would put the neck back at center.
func residualLimbs(W []float32, q0 []int8, s0 float64, max int, tol float64, R, C int) (limbs []limbData, rel []float64) {
	gap := make([]float32, len(W))
	ref := make([]float32, len(W))
	var wn float64
	for i, v := range W {
		ref[i] = float32(s0 * float64(q0[i]))
		gap[i] = v - ref[i]
		wn += float64(v) * float64(v)
	}
	wn = math.Sqrt(wn)
	norm := func(g []float32) float64 {
		var s float64
		for _, v := range g {
			s += float64(v) * float64(v)
		}
		return math.Sqrt(s) / math.Max(wn, 1e-30)
	}
	sing := singularNeck(gap)
	work := append([]float32(nil), gap...)
	if sing != nil {
		for i, s := range sing {
			if s {
				work[i] = 0
			}
		}
	}
	rel = append(rel, norm(gap))
	// part is the residual of the neck (onNeck) or of the body, as a fraction of ||W||.
	// The two are fit separately, so a spike must not decide whether a body limb moved.
	part := func(g []float32, onNeck bool) float64 {
		var s float64
		for i, v := range g {
			neck := sing != nil && sing[i]
			if neck != onNeck {
				continue
			}
			s += float64(v) * float64(v)
		}
		return math.Sqrt(s) / math.Max(wn, 1e-30)
	}
	lock := R > 0 && C > 0 && R*C == len(W)
	for len(limbs)+1 < max && rel[len(rel)-1] > tol {
		var bd []uint8
		var br int
		var bsc float64
		be := math.Inf(1)
		for _, r := range []int{3, 4, 2} {
			d, s, e := quantLimb(work, r)
			if e < be {
				bd, br, bsc, be = d, r, s, e
			}
		}
		if bsc == 0 || len(bd) == 0 {
			break
		}
		lb := limbData{br, bd, bsc}
		if lock {
			lb = lockLimb(ref, lb, R, C)
		}
		c := (lb.r - 1) / 2
		if sing != nil {
			cen := uint8(c)
			for i, s := range sing {
				if s {
					lb.d[i] = cen
				}
			}
		}
		next := make([]float32, len(gap))
		var moved bool
		for i := range gap {
			d := float32(lb.s * float64(int(lb.d[i])-c))
			next[i] = gap[i] - d
			if d != 0 {
				moved = true
			}
		}
		if !moved {
			break
		}
		nr := norm(next)
		// A drop below the BF16 refine tolerance is not a limb. hodge-4 kept tails of
		// 1e-7 after the residual had already stopped, and the loader accumulates every limb.
		if part(gap, false)-part(next, false) < RefineTol {
			break
		}
		for i := range gap {
			gap[i] = next[i]
			work[i] = next[i]
			if sing != nil && sing[i] {
				work[i] = 0
			}
		}
		limbs = append(limbs, lb)
		rel = append(rel, nr)
	}
	// A T1 neck is a spike tail. The hodge-4 mix residual marked about half of every
	// weight, and the loader accumulated that limb at 0.25 of the base rms.
	if neckWide(sing) {
		sing = nil
	}
	for _, lb := range neckLimbs(gap, sing, NeckLimbs) {
		c := (lb.r - 1) / 2
		next := append([]float32(nil), gap...)
		for i := range next {
			next[i] -= float32(lb.s * float64(int(lb.d[i])-c))
		}
		if part(gap, true)-part(next, true) < RefineTol {
			break
		}
		gap = next
		limbs = append(limbs, lb)
		rel = append(rel, norm(gap))
	}
	return limbs, rel
}

// neckWide is a mask that is not a spike tail. One spike's ternarization shoulder
// marks about an eighth of a small grid. The hodge-4 mix residual marked about half
// of each tensor, and the loader accumulated that limb at a quarter of the base rms.
// A fifth of the tensor sits between those two.
func neckWide(sing []bool) bool {
	if len(sing) == 0 {
		return false
	}
	c := 0
	for _, s := range sing {
		if s {
			c++
		}
	}
	return c > len(sing)/5
}

// neckLimbs codes the singular neck of gap. Positions outside sing stay the center digit, so the
// scale is the neck's and the body is not quantized by it. No Kuramoto lock: the neck was cut out
// of the lock, and a lock against limb 0 would cap it again.
func neckLimbs(gap []float32, sing []bool, max int) []limbData {
	if sing == nil || max < 1 {
		return nil
	}
	left := make([]float32, len(gap))
	var energy float64
	for i, on := range sing {
		if !on {
			continue
		}
		left[i] = gap[i]
		energy += float64(gap[i]) * float64(gap[i])
	}
	if energy == 0 {
		return nil
	}
	var out []limbData
	for len(out) < max {
		var bd []uint8
		br := 3
		var bsc float64
		be := math.Inf(1)
		for _, r := range []int{3, 4, 2} {
			d, s, e := quantLimb(left, r)
			if e < be {
				bd, br, bsc, be = d, r, s, e
			}
		}
		if bsc == 0 || len(bd) != len(left) {
			break
		}
		cen := (br - 1) / 2
		for i, on := range sing {
			if !on {
				bd[i] = uint8(cen)
			}
		}
		next := make([]float32, len(left))
		var moved bool
		var n2 float64
		for i := range left {
			d := bsc * float64(int(bd[i])-cen)
			if d != 0 {
				moved = true
			}
			next[i] = left[i] - float32(d)
			n2 += float64(next[i]) * float64(next[i])
		}
		if !moved || n2 >= energy {
			break
		}
		out = append(out, limbData{br, bd, bsc})
		left = next
		energy = n2
	}
	return out
}

// lockLimb is the Hodge–Kuramoto lock of one limb against limb 0 (paper_v2 Eq.(14)–(20), Alg. S
// step 5). Oscillators are the harmonic bins of the channel grid; limb 0 is the reference.
// A bin that does not lock is not merged: it stays with limb 0. The neck has no support here,
// because the caller cut it out of the residual this limb was fit on.
func lockLimb(ref []float32, lb limbData, R, C int) limbData {
	n := R * C
	if n != len(ref) || n != len(lb.d) || lb.s == 0 {
		return lb
	}
	S := NewSpectrum([2]int{R, C}, 0)
	var modes []int
	for i, h := range S.Harm {
		if h {
			modes = append(modes, i)
		}
	}
	if len(modes) == 0 {
		return lb
	}
	c0 := (lb.r - 1) / 2
	base := make([]float64, n)
	cand := make([]float64, n)
	for i := range ref {
		base[i] = float64(ref[i])
		cand[i] = lb.s * float64(int(lb.d[i])-c0)
	}
	Fb := fftn(toC(base), S.dims, false)
	Fc := fftn(toC(cand), S.dims, false)
	nm := len(modes)
	no := 2 * nm
	theta := make([]float64, no)
	level := make([]int, no)
	for mi, m := range modes {
		level[mi] = S.Level[m]
		level[nm+mi] = S.Level[m]
		theta[nm+mi] = cmplx.Phase(Fc[m]) - cmplx.Phase(Fb[m])
	}
	weak := math.Ldexp(1, -(GOmegaShift - GThetaShift))
	Cm := make([]float64, no*no)
	for a := 0; a < no; a++ {
		for b := 0; b < no; b++ {
			if a == b {
				continue
			}
			cpl := weak
			if level[a] == level[b] {
				cpl = 1
			}
			Cm[a*no+b] = cpl
		}
	}
	Kb := couplingMatrix(Cm, no, math.Ldexp(1, -GThetaShift))
	refOf := make([]int, no)
	for mi := range modes {
		refOf[mi] = mi
		refOf[nm+mi] = mi
	}
	kp := KParams{
		GTheta: math.Ldexp(1, -GThetaShift),
		Weak:   weak,
		WLock:  WLock,
		NLost:  NLost,
		RTol:   RefineTol,
		MaxWin: S.NLevel + 2,
		Anneal: true,
	}
	kr := kuramotoTick(theta, Kb, level, refOf, kp, "limb")
	drop := map[int]bool{}
	for mi, m := range modes {
		if kr.Locked[nm+mi] {
			continue
		}
		drop[m] = true
		drop[conjIdx(m, S.dims)] = true
	}
	if len(drop) == 0 {
		return lb
	}
	for m := range drop {
		Fc[m] = 0
	}
	g := realPart(fftn(Fc, S.dims, true))
	out := limbData{r: lb.r, s: lb.s, d: make([]uint8, n)}
	lo, hi := float64(-c0), float64(lb.r-1-c0)
	for i, v := range g {
		q := math.Max(lo, math.Min(hi, math.RoundToEven(v/lb.s)))
		out.d[i] = uint8(int(q) + c0)
	}
	return out
}

// packDigits packs raw digits (0..3) [R][C] four rows per byte, the layout of the ternary codes.
func packDigits(d []uint8, R, C int) []byte {
	R4 := R / 4
	o := make([]byte, R4*C)
	for i := 0; i < R4*C; i++ {
		var b byte
		for s := 0; s < 4; s++ {
			b |= (d[s*R4*C+i] & 3) << (2 * s)
		}
		o[i] = b
	}
	return o
}

func unpackDigits(b []byte, R, C int) []uint8 {
	R4 := R / 4
	d := make([]uint8, R*C)
	for i := 0; i < R4*C; i++ {
		for s := 0; s < 4; s++ {
			d[s*R4*C+i] = (b[i] >> (2 * s)) & 3
		}
	}
	return d
}

// writeLimbs adds the extra limbs of tensor k (shape R x C) to W, numbered from 1.
func writeLimbs(W *STW, k string, R, C int, limbs []limbData) {
	writeLimbsFrom(W, k, R, C, 1, limbs)
}

// writeLimbsFrom numbers the first limb as `first` (1-based).
func writeLimbsFrom(W *STW, k string, R, C int, first int, limbs []limbData) {
	for i, lb := range limbs {
		n := limbName(k, first+i)
		W.Add(n, "U8", []int{R / 4, C}, packDigits(lb.d, R, C))
		W.AddBF16(n+"_scale", []int{1}, []float32{float32(lb.s)})
		rb := make([]byte, 4)
		binary.LittleEndian.PutUint32(rb, uint32(lb.r))
		W.Add(n+"_radix", "I32", []int{1}, rb)
	}
}

// limbsOf reads tensor k's extra limbs from s (radix, digits, scale).
func (s *ST) limbsOf(k string) []limbData {
	var out []limbData
	if !s.Has(k) {
		return nil
	}
	sh := s.LogicalShape(k)
	if len(sh) != 2 {
		return nil
	}
	for l := 1; ; l++ {
		n := limbName(k, l)
		if !s.Has(n) || !s.Has(n+"_scale") || !s.Has(n+"_radix") {
			return out
		}
		b, err1 := s.Raw(n)
		sc, err2 := s.F32(n + "_scale")
		rb, err3 := s.Raw(n + "_radix")
		if err1 != nil || err2 != nil || err3 != nil || len(sc) != 1 || len(rb) != 4 {
			return out
		}
		out = append(out, limbData{int(binary.LittleEndian.Uint32(rb)), unpackDigits(b, sh[0], sh[1]), float64(sc[0])})
	}
}

// isLimbTensor reports a limb tensor (or its scale / radix) name.
func isLimbTensor(n string) bool {
	for _, suf := range []string{"_scale", "_radix", ""} {
		m := n[:len(n)-len(suf)]
		if len(n) < len(suf) || n[len(n)-len(suf):] != suf {
			continue
		}
		i := len(m) - 1
		for i >= 0 && m[i] >= '0' && m[i] <= '9' {
			i--
		}
		if i < len(m)-1 && i >= 11 && m[i-11:i+1] == ".weight_limb" {
			return true
		}
	}
	return false
}

// rewriteLimb overwrites tensor k's extra limb i (1-based) of out in place: digits, scale, radix.
// A limb the file does not have is an error. Indexing its header panics (Off is empty).
func rewriteLimb(f *os.File, out *ST, k string, i int, lb limbData) error {
	n := limbName(k, i)
	sh := out.LogicalShape(k)
	dig, ok1 := out.H[n]
	sc, ok2 := out.H[n+"_scale"]
	rd, ok3 := out.H[n+"_radix"]
	if len(sh) != 2 || !ok1 || !ok2 || !ok3 || len(dig.Off) == 0 || len(sc.Off) == 0 || len(rd.Off) == 0 {
		return fmt.Errorf("%s: limb %d is not in the file", k, i)
	}
	R, C := sh[0], sh[1]
	if len(lb.d) != R*C {
		return fmt.Errorf("%s: limb %d has %d digits for %d x %d", k, i, len(lb.d), R, C)
	}
	if _, err := f.WriteAt(packDigits(lb.d, R, C), out.base+dig.Off[0]); err != nil {
		return err
	}
	u := bf16(float32(lb.s))
	if _, err := f.WriteAt([]byte{byte(u), byte(u >> 8)}, out.base+sc.Off[0]); err != nil {
		return err
	}
	rb := make([]byte, 4)
	binary.LittleEndian.PutUint32(rb, uint32(lb.r))
	_, err := f.WriteAt(rb, out.base+rd.Off[0])
	return err
}

// zeroLimbs makes every extra limb of k zero (digits at their centre, scale 0).
func zeroLimbs(f *os.File, out *ST, k string) error {
	for i, lb := range out.limbsOf(k) {
		c := uint8((lb.r - 1) / 2)
		for j := range lb.d {
			lb.d[j] = c
		}
		lb.s = 0
		if err := rewriteLimb(f, out, k, i+1, lb); err != nil {
			return err
		}
	}
	return nil
}

// refillLimbs requantizes k's extra limbs from W given its new limb 0 (q0, s0), in the same slots.
// residualLimbs appends neck limbs past the body budget. The file has no slot for those, and
// writing one panics. The refill keeps only as many limbs as the file already stores.
func refillLimbs(f *os.File, out *ST, k string, W []float32, q0 []int8, s0 float64) error {
	have := out.limbsOf(k)
	n := len(have)
	if n == 0 {
		return nil
	}
	sh := out.LogicalShape(k)
	R, C := 0, 0
	if len(sh) == 2 {
		R, C = sh[0], sh[1]
	}
	limbs, _ := residualLimbs(W, q0, s0, n+1, 0, R, C)
	if len(limbs) > n {
		limbs = limbs[:n]
	}
	for i, lb := range limbs {
		if err := rewriteLimb(f, out, k, i+1, lb); err != nil {
			return err
		}
	}
	for i := len(limbs); i < n; i++ {
		lb := have[i]
		c := uint8((lb.r - 1) / 2)
		for j := range lb.d {
			lb.d[j] = c
		}
		lb.s = 0
		if err := rewriteLimb(f, out, k, i+1, lb); err != nil {
			return err
		}
	}
	return nil
}

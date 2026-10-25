package tower

import (
	"math"
	"math/cmplx"
	"sort"
	"sync"
)

const (
	// VisSide is the image the tower resizes to (D17). Patch is the ViT-B/16 cell (D18).
	VisSide = 224
	Patch   = 16
)

// Patches is the number of non-overlapping Patch cells in an image.
func Patches(w, h int) int {
	if w < Patch || h < Patch {
		return 0
	}
	return (w / Patch) * (h / Patch)
}

// Vision resizes rgb (interleaved, row-major) to 224×224 and returns one
// 46-vector per patch, in raster order, plus the RoPE phase of each patch.
func Vision(rgb []float32, w, h int) ([][]float32, []float64, error) {
	return vision(rgb, w, h, true)
}

// VisionRaw encodes an image that is already a multiple of the patch.
// Probes use it on a single 16×16 cell.
func VisionRaw(rgb []float32, w, h int) ([][]float32, []float64, error) {
	return vision(rgb, w, h, false)
}

func vision(rgb []float32, w, h int, resize bool) ([][]float32, []float64, error) {
	if !Ready() {
		return nil, nil, errf("L-A tower is not ready")
	}
	if w <= 0 || h <= 0 || len(rgb) != w*h*3 {
		return nil, nil, errf("rgb length %d, want %d×%d×3", len(rgb), w, h)
	}
	if resize && (w != VisSide || h != VisSide) {
		rgb = resizeRGB(rgb, w, h, VisSide, VisSide)
		w, h = VisSide, VisSide
	}
	if w%Patch != 0 || h%Patch != 0 {
		return nil, nil, errf("image %d×%d is not a multiple of the %d patch", w, h, Patch)
	}
	n := w * h
	L := make([]float64, n)
	w1 := make([]complex128, n)
	w2 := make([]complex128, n)
	abs1 := make([]float64, n)
	abs2 := make([]float64, n)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			r := float64(rgb[3*i])
			g := float64(rgb[3*i+1])
			bl := float64(rgb[3*i+2])
			L[i] = 0.299*r + 0.587*g + 0.114*bl
			w2[i] = complex(r-g, 0.5*(r+g)-bl)
			abs2[i] = cmplx.Abs(w2[i])
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			dx, dy := deriv(L, w, h, x, y)
			w1[i] = complex(dx, dy)
			abs1[i] = cmplx.Abs(w1[i])
		}
	}
	R1, err := embedField(w1, median(abs1))
	if err != nil {
		return nil, nil, err
	}
	R2, err := embedField(w2, median(abs2))
	if err != nil {
		return nil, nil, err
	}
	nw, nh := w/Patch, h/Patch
	feats := make([][]float32, nw*nh)
	phis := make([]float64, nw*nh)
	for py := 0; py < nh; py++ {
		for px := 0; px < nw; px++ {
			f := patchFeature(R1, R2, L, w, px*Patch, py*Patch)
			if len(f) != VisFeat {
				return nil, nil, errf("patch feature width %d, want %d", len(f), VisFeat)
			}
			k := py*nw + px
			feats[k] = f
			phis[k] = Phi(f)
		}
	}
	return feats, phis, nil
}

func embedField(w []complex128, s float64) ([]complex128, error) {
	R := make([]complex128, len(w))
	var mu sync.Mutex
	bad := 0
	parallel(len(w), func(i int) {
		z, ok := Embed(Seed(Squash(w[i], s)))
		if !ok || !finite(z) {
			mu.Lock()
			bad++
			mu.Unlock()
			return
		}
		R[i] = z
	})
	if bad == len(w) && len(w) > 0 {
		return nil, errf("every pixel fell outside the L-A disc")
	}
	return R, nil
}

func deriv(L []float64, w, h, x, y int) (dx, dy float64) {
	xm, xp := x-1, x+1
	if xm < 0 {
		xm = x
	}
	if xp >= w {
		xp = x
	}
	ym, yp := y-1, y+1
	if ym < 0 {
		ym = y
	}
	if yp >= h {
		yp = y
	}
	if d := xp - xm; d != 0 {
		dx = (L[y*w+xp] - L[y*w+xm]) / float64(d)
	}
	if d := yp - ym; d != 0 {
		dy = (L[yp*w+x] - L[ym*w+x]) / float64(d)
	}
	return dx, dy
}

// patchFeature is the D21 vision vector: per field, mean R (2), eight low
// DFT coefficients (16), winding counts (2), log curvature (1), mean log|R|
// (1); then the patch luma mean and standard deviation. 22×2+2 = 46.
func patchFeature(R1, R2 []complex128, L []float64, stride, x0, y0 int) []float32 {
	f := make([]float32, 0, VisFeat)
	f = append(f, fieldStats(fieldOf(R1, stride, x0, y0))...)
	f = append(f, fieldStats(fieldOf(R2, stride, x0, y0))...)
	var sum, ss float64
	for y := 0; y < Patch; y++ {
		for x := 0; x < Patch; x++ {
			v := L[(y0+y)*stride+(x0+x)]
			sum += v
			ss += v * v
		}
	}
	n := float64(Patch * Patch)
	mean := sum / n
	f = append(f, float32(mean), float32(math.Sqrt(math.Max(ss/n-mean*mean, 0))))
	return f
}

func fieldStats(R []complex128) []float32 {
	var sum complex128
	var logA float64
	for _, z := range R {
		sum += z
		a := cmplx.Abs(z)
		if a < 1e-30 {
			a = 1e-30
		}
		logA += math.Log(a)
	}
	n := float64(len(R))
	mean := sum / complex(n, 0)
	o := []float32{float32(real(mean)), float32(imag(mean))}
	for _, c := range lowDFT(R, Patch, 8) {
		o = append(o, float32(real(c)), float32(imag(c)))
	}
	wp, wm := windings(R, Patch)
	o = append(o, float32(wp), float32(wm), float32(math.Log(1+meanGrad2(R, Patch))), float32(logA/n))
	return o
}

type freq struct{ u, v, d int }

func lowDFT(R []complex128, n, k int) []complex128 {
	var bins []freq
	for u := 0; u < n; u++ {
		fu := u
		if u > n/2 {
			fu = u - n
		}
		for v := 0; v < n; v++ {
			if u == 0 && v == 0 {
				continue
			}
			fv := v
			if v > n/2 {
				fv = v - n
			}
			bins = append(bins, freq{u, v, fu*fu + fv*fv})
		}
	}
	sort.Slice(bins, func(i, j int) bool {
		if bins[i].d != bins[j].d {
			return bins[i].d < bins[j].d
		}
		if bins[i].u != bins[j].u {
			return bins[i].u < bins[j].u
		}
		return bins[i].v < bins[j].v
	})
	if k > len(bins) {
		k = len(bins)
	}
	out := make([]complex128, k)
	for i := 0; i < k; i++ {
		u, v := bins[i].u, bins[i].v
		var s complex128
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				ang := -2 * math.Pi * float64(u*x+v*y) / float64(n)
				s += R[y*n+x] * cmplx.Exp(complex(0, ang))
			}
		}
		out[i] = s / complex(float64(n*n), 0)
	}
	return out
}

func windings(R []complex128, n int) (plus, minus int) {
	for y := 0; y < n-1; y++ {
		for x := 0; x < n-1; x++ {
			q := plaquette(R[y*n+x], R[y*n+x+1], R[(y+1)*n+x+1], R[(y+1)*n+x])
			switch q {
			case 1:
				plus++
			case -1:
				minus++
			}
		}
	}
	return plus, minus
}

func plaquette(a, b, c, d complex128) int {
	p := []complex128{a, b, c, d}
	var s float64
	for i := 0; i < 4; i++ {
		s += phaseStep(p[i], p[(i+1)%4])
	}
	return int(math.Round(s / (2 * math.Pi)))
}

func phaseStep(a, b complex128) float64 {
	if a == 0 || b == 0 {
		return 0
	}
	d := cmplx.Phase(b) - cmplx.Phase(a)
	for d > math.Pi {
		d -= 2 * math.Pi
	}
	for d < -math.Pi {
		d += 2 * math.Pi
	}
	return d
}

func meanGrad2(R []complex128, n int) float64 {
	var s float64
	var c int
	for y := 0; y < n; y++ {
		for x := 0; x < n-1; x++ {
			d := phaseStep(R[y*n+x], R[y*n+x+1])
			s += d * d
			c++
		}
	}
	for y := 0; y < n-1; y++ {
		for x := 0; x < n; x++ {
			d := phaseStep(R[y*n+x], R[(y+1)*n+x])
			s += d * d
			c++
		}
	}
	if c == 0 {
		return 0
	}
	return s / float64(c)
}

func resizeRGB(src []float32, sw, sh, dw, dh int) []float32 {
	dst := make([]float32, dw*dh*3)
	for y := 0; y < dh; y++ {
		fy := (float64(y)+0.5)*float64(sh)/float64(dh) - 0.5
		y0 := int(math.Floor(fy))
		ty := fy - float64(y0)
		if y0 < 0 {
			y0, ty = 0, 0
		}
		y1 := y0 + 1
		if y1 >= sh {
			y1 = sh - 1
		}
		if y0 >= sh {
			y0 = sh - 1
		}
		for x := 0; x < dw; x++ {
			fx := (float64(x)+0.5)*float64(sw)/float64(dw) - 0.5
			x0 := int(math.Floor(fx))
			tx := fx - float64(x0)
			if x0 < 0 {
				x0, tx = 0, 0
			}
			x1 := x0 + 1
			if x1 >= sw {
				x1 = sw - 1
			}
			if x0 >= sw {
				x0 = sw - 1
			}
			for c := 0; c < 3; c++ {
				v00 := float64(src[(y0*sw+x0)*3+c])
				v10 := float64(src[(y0*sw+x1)*3+c])
				v01 := float64(src[(y1*sw+x0)*3+c])
				v11 := float64(src[(y1*sw+x1)*3+c])
				v := v00*(1-tx)*(1-ty) + v10*tx*(1-ty) + v01*(1-tx)*ty + v11*tx*ty
				dst[(y*dw+x)*3+c] = float32(v)
			}
		}
	}
	return dst
}

func fieldOf(R []complex128, stride, x0, y0 int) []complex128 {
	o := make([]complex128, Patch*Patch)
	for y := 0; y < Patch; y++ {
		for x := 0; x < Patch; x++ {
			o[y*Patch+x] = R[(y0+y)*stride+(x0+x)]
		}
	}
	return o
}

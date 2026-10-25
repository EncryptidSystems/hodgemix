package tower

import (
	"math"
	"math/cmplx"
	"sync"
)

const (
	// SampleRate, Win and Hop are the Whisper front end (D19).
	SampleRate = 16000
	Win        = 400
	Hop        = 160
	MelBands   = 80
	Bins       = Win/2 + 1 // 201
)

// Tokens is how many audio tokens a clip of n samples yields (D19).
// A 30 s clip (480000 samples) yields 1500.
func Tokens(n int) int {
	if n < Hop {
		return 0
	}
	frames := n/Hop + 1
	pairs := frames - 1
	if pairs < 2 {
		return 0
	}
	return pairs / 2
}

// Audio encodes mono 16 kHz pcm. Clips longer than 30 s are taken in 30 s chunks.
func Audio(pcm []float32) ([][]float32, []float64, error) {
	if !Ready() {
		return nil, nil, errf("L-A tower is not ready")
	}
	if Tokens(len(pcm)) == 0 {
		return nil, nil, errf("audio has %d samples, a token needs 320", len(pcm))
	}
	const chunk = SampleRate * 30
	var feats [][]float32
	var phis []float64
	for off := 0; off < len(pcm); off += chunk {
		end := off + chunk
		if end > len(pcm) {
			end = len(pcm)
		}
		if Tokens(end-off) == 0 {
			break
		}
		f, p, err := audioChunk(pcm[off:end])
		if err != nil {
			return nil, nil, err
		}
		feats = append(feats, f...)
		phis = append(phis, p...)
	}
	if len(feats) == 0 {
		return nil, nil, errf("audio produced no tokens")
	}
	return feats, phis, nil
}

func audioChunk(pcm []float32) ([][]float32, []float64, error) {
	frames := stft(pcm)
	pairs := len(frames) - 1
	w := make([][]complex128, pairs)
	for t := 0; t < pairs; t++ {
		w[t] = make([]complex128, Bins)
		for k := 0; k < Bins; k++ {
			het := cmplx.Exp(complex(0, -2*math.Pi*float64(k)*float64(Hop)/float64(Win)))
			w[t][k] = frames[t+1][k] * cmplx.Conj(frames[t][k]) * het
		}
	}
	s := make([]float64, Bins)
	col := make([]float64, pairs)
	for k := 0; k < Bins; k++ {
		for t := 0; t < pairs; t++ {
			col[t] = cmplx.Abs(w[t][k])
		}
		s[k] = median(col)
	}
	R := make([][]complex128, pairs)
	for t := 0; t < pairs; t++ {
		R[t] = make([]complex128, Bins)
	}
	var mu sync.Mutex
	bad := 0
	parallel(pairs*Bins, func(i int) {
		t, k := i/Bins, i%Bins
		z, ok := Embed(Seed(Squash(w[t][k], s[k])))
		if !ok || !finite(z) {
			mu.Lock()
			bad++
			mu.Unlock()
			return
		}
		R[t][k] = z
	})
	if bad == pairs*Bins {
		return nil, nil, errf("every audio bin fell outside the L-A disc")
	}
	mel := melBank()
	nt := pairs / 2
	feats := make([][]float32, nt)
	phis := make([]float64, nt)
	for i := 0; i < nt; i++ {
		f0 := frameFeat(R[2*i], frames[2*i], mel)
		f1 := frameFeat(R[2*i+1], frames[2*i+1], mel)
		f := make([]float32, AudFeat)
		for j := range f {
			f[j] = (f0[j] + f1[j]) / 2
		}
		feats[i] = f
		phis[i] = AudioPhi(f)
	}
	return feats, phis, nil
}

func frameFeat(R, X []complex128, mel [][]float64) []float32 {
	f := make([]float32, AudFeat)
	for b := 0; b < MelBands; b++ {
		var num complex128
		var den, e float64
		for k, w := range mel[b] {
			if w == 0 {
				continue
			}
			num += complex(w, 0) * R[k]
			den += w
			a := cmplx.Abs(X[k])
			e += w * a * a
		}
		if den > 0 {
			num /= complex(den, 0)
		}
		if e < 1e-10 {
			e = 1e-10
		}
		f[2*b] = float32(real(num))
		f[2*b+1] = float32(imag(num))
		f[160+b] = float32(math.Log(e))
	}
	return f
}

// stft returns one spectrum per frame. Frame t is the Hann window centered at t*Hop.
func stft(pcm []float32) [][]complex128 {
	n := len(pcm)/Hop + 1
	out := make([][]complex128, n)
	hann := make([]float64, Win)
	for i := 0; i < Win; i++ {
		hann[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(Win)))
	}
	for t := 0; t < n; t++ {
		spec := make([]complex128, Bins)
		start := t*Hop - Win/2
		for k := 0; k < Bins; k++ {
			var s complex128
			for n := 0; n < Win; n++ {
				idx := start + n
				var x float64
				if idx >= 0 && idx < len(pcm) {
					x = float64(pcm[idx])
				}
				ang := -2 * math.Pi * float64(k*n) / float64(Win)
				s += complex(x*hann[n], 0) * cmplx.Exp(complex(0, ang))
			}
			spec[k] = s
		}
		out[t] = spec
	}
	return out
}

var (
	melOnce sync.Once
	melW    [][]float64
)

func melBank() [][]float64 {
	melOnce.Do(func() { melW = slaneyMel(SampleRate, Win, MelBands, 0, float64(SampleRate)/2) })
	return melW
}

// slaneyMel is the triangular mel matrix Whisper's front end is built on:
// 80 bands, 201 bins, Slaney scale and normalisation.
func slaneyMel(sr, nfft, nmel int, fmin, fmax float64) [][]float64 {
	nbin := nfft/2 + 1
	melF := make([]float64, nmel+2)
	m0, m1 := hzToMel(fmin), hzToMel(fmax)
	for i := range melF {
		melF[i] = melToHz(m0 + (m1-m0)*float64(i)/float64(nmel+1))
	}
	ff := make([]float64, nbin)
	for k := range ff {
		ff[k] = float64(k) * float64(sr) / float64(nfft)
	}
	w := make([][]float64, nmel)
	for i := 0; i < nmel; i++ {
		w[i] = make([]float64, nbin)
		for k, f := range ff {
			lower := (f - melF[i]) / (melF[i+1] - melF[i])
			upper := (melF[i+2] - f) / (melF[i+2] - melF[i+1])
			v := lower
			if upper < v {
				v = upper
			}
			if v > 0 {
				w[i][k] = v
			}
		}
		enorm := 2 / (melF[i+2] - melF[i])
		for k := range w[i] {
			w[i][k] *= enorm
		}
	}
	return w
}

func hzToMel(f float64) float64 {
	const fsp = 200.0 / 3
	const minLog = 1000.0
	mel := f / fsp
	minLogMel := minLog / fsp
	if f >= minLog {
		mel = minLogMel + math.Log(f/minLog)/math.Log(6.4)*27
	}
	return mel
}

func melToHz(m float64) float64 {
	const fsp = 200.0 / 3
	const minLog = 1000.0
	minLogMel := minLog / fsp
	if m < minLogMel {
		return fsp * m
	}
	return minLog * math.Exp((m-minLogMel)*math.Log(6.4)/27)
}

//go:build js && wasm

package main

// The audio encoder's front end (hkruntime model/encoder.go, decision E7): log-mel frames
// of mono 16 kHz pcm. HTK pre-emphasis 0.97, periodic Hann window of 512 samples (32 ms),
// hop 160 (10 ms), 1024-point FFT magnitude, 128 HTK mel bands over 125-7600 Hz, log with
// a floor of 1e-5.

import (
	"math"
	"math/cmplx"
)

const (
	melRate  = 16000
	melWin   = 512
	melHop   = 160
	melFFT   = 1024
	melBands = 128
	melLo    = 125.0
	melHi    = 7600.0
	melFloor = 1e-5
	melPre   = 0.97
)

func htkMel(f float64) float64   { return 2595 * math.Log10(1+f/700) }
func htkHz(m float64) float64    { return 700 * (math.Pow(10, m/2595) - 1) }

// melBank is [melBands][melFFT/2+1] triangular HTK filters, unnormalised.
func melBank() [][]float64 {
	bins := melFFT/2 + 1
	lo, hi := htkMel(melLo), htkMel(melHi)
	pts := make([]float64, melBands+2)
	for i := range pts {
		pts[i] = htkHz(lo + (hi-lo)*float64(i)/float64(melBands+1))
	}
	bank := make([][]float64, melBands)
	for b := 0; b < melBands; b++ {
		row := make([]float64, bins)
		l, c, r := pts[b], pts[b+1], pts[b+2]
		for k := 0; k < bins; k++ {
			f := float64(k) * melRate / melFFT
			switch {
			case f > l && f <= c:
				row[k] = (f - l) / (c - l)
			case f > c && f < r:
				row[k] = (r - f) / (r - c)
			}
		}
		bank[b] = row
	}
	return bank
}

// fft is an in-place radix-2 transform; len(a) is a power of two.
func fft(a []complex128) {
	n := len(a)
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	for size := 2; size <= n; size <<= 1 {
		w := cmplx.Exp(complex(0, -2*math.Pi/float64(size)))
		for start := 0; start < n; start += size {
			z := complex(1, 0)
			for k := 0; k < size/2; k++ {
				u, v := a[start+k], a[start+k+size/2]*z
				a[start+k], a[start+k+size/2] = u+v, u-v
				z *= w
			}
		}
	}
}

// logMel is frames x melBands, row-major.
func logMel(pcm []float32) ([]float32, int) {
	if len(pcm) < melWin {
		return nil, 0
	}
	x := make([]float64, len(pcm))
	x[0] = float64(pcm[0]) * (1 - melPre)
	for i := 1; i < len(pcm); i++ {
		x[i] = float64(pcm[i]) - melPre*float64(pcm[i-1])
	}
	frames := 1 + (len(x)-melWin)/melHop
	win := make([]float64, melWin)
	for i := range win {
		win[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/melWin)
	}
	bank := melBank()
	out := make([]float32, frames*melBands)
	buf := make([]complex128, melFFT)
	mag := make([]float64, melFFT/2+1)
	for f := 0; f < frames; f++ {
		for i := range buf {
			buf[i] = 0
		}
		for i := 0; i < melWin; i++ {
			buf[i] = complex(x[f*melHop+i]*win[i], 0)
		}
		fft(buf)
		for k := range mag {
			mag[k] = cmplx.Abs(buf[k])
		}
		for b := 0; b < melBands; b++ {
			var s float64
			for k, w := range bank[b] {
				if w != 0 {
					s += w * mag[k]
				}
			}
			out[f*melBands+b] = float32(math.Log(math.Max(s, melFloor)))
		}
	}
	return out, frames
}

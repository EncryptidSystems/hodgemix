package tower

import "math"

// ProbeVision runs n single-patch images through the vision readout.
// The rows are the closed-form fit's feature matrix (D6).
func ProbeVision(n int) ([]float64, error) {
	if n <= 0 {
		return nil, errf("vision probe count %d", n)
	}
	out := make([]float64, n*VisFeat)
	errc := make(chan error, n)
	parallel(n, func(i int) {
		f, _, err := VisionRaw(probeRGB(i), Patch, Patch)
		if err != nil {
			errc <- err
			return
		}
		if len(f) != 1 || len(f[0]) != VisFeat {
			errc <- errf("vision probe %d produced %d vectors", i, len(f))
			return
		}
		for j, v := range f[0] {
			if !finite64(float64(v)) {
				errc <- errf("vision probe %d feature %d is not finite", i, j)
				return
			}
			out[i*VisFeat+j] = float64(v)
		}
	})
	close(errc)
	for err := range errc {
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ProbeAudio runs n one-token tones through the audio readout.
func ProbeAudio(n int) ([]float64, error) {
	if n <= 0 {
		return nil, errf("audio probe count %d", n)
	}
	out := make([]float64, n*AudFeat)
	errc := make(chan error, n)
	parallel(n, func(i int) {
		f, _, err := Audio(probePCM(i))
		if err != nil {
			errc <- err
			return
		}
		if len(f) < 1 || len(f[0]) != AudFeat {
			errc <- errf("audio probe %d produced %d tokens", i, len(f))
			return
		}
		for j, v := range f[0] {
			if !finite64(float64(v)) {
				errc <- errf("audio probe %d feature %d is not finite", i, j)
				return
			}
			out[i*AudFeat+j] = float64(v)
		}
	})
	close(errc)
	for err := range errc {
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func probeRGB(i int) []float32 {
	o := make([]float32, Patch*Patch*3)
	f1 := float64(i%7 + 1)
	f2 := float64(i%5 + 1)
	for y := 0; y < Patch; y++ {
		for x := 0; x < Patch; x++ {
			xf, yf := float64(x)/Patch, float64(y)/Patch
			r := 0.5 + 0.5*math.Sin(2*math.Pi*f1*xf)*math.Cos(2*math.Pi*f2*yf)
			g := 0.5 + 0.5*math.Sin(2*math.Pi*f2*xf+float64(i)*0.17)
			b := 0.5 + 0.5*math.Cos(2*math.Pi*f1*yf)
			p := (y*Patch + x) * 3
			o[p], o[p+1], o[p+2] = float32(r), float32(g), float32(b)
		}
	}
	return o
}

func probePCM(i int) []float32 {
	const n = 320
	o := make([]float32, n)
	hz := 80 + float64(i)*30
	for t := 0; t < n; t++ {
		o[t] = float32(0.3 * math.Sin(2*math.Pi*hz*float64(t)/SampleRate))
	}
	return o
}

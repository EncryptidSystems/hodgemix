//go:build js && wasm

// towerwasm runs the L-A vision and audio towers in the page. It is the tower
// package unchanged, so the features are the ones the embed_vision and
// embed_audio projectors were fitted to. Build:
//
//	GOOS=js GOARCH=wasm go build -o tower.wasm ./towerwasm
//
// The page gets globalThis.hodgeTower with
//
//	vision(rgb Uint8Array, w, h)  rgb is w*h*3 little-endian f32 in [0, 1], row-major
//	audio(pcm Uint8Array)         pcm is mono 16 kHz little-endian f32
//	mel(pcm Uint8Array)           the audio encoder's log-mel frames (mel.go): {frames, bins, feat}
//
// Each returns {n, width, feat, phi}: n tokens of width features (feat, f32)
// and each token's RoPE phase (phi, f32), or {error}.
package main

import (
	"encoding/binary"
	"math"
	"syscall/js"

	"hodgemix/tower"
)

func f32s(v js.Value) []float32 {
	b := make([]byte, v.Get("length").Int())
	js.CopyBytesToGo(b, v)
	o := make([]float32, len(b)/4)
	for i := range o {
		o[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return o
}

func bytesOf(v []float32) js.Value {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	u := js.Global().Get("Uint8Array").New(len(b))
	js.CopyBytesToJS(u, b)
	return u
}

func result(feats [][]float32, phi []float64, width int, err error) any {
	if err != nil {
		return map[string]any{"error": err.Error()}
	}
	flat := make([]float32, 0, len(feats)*width)
	for _, f := range feats {
		flat = append(flat, f...)
	}
	ph := make([]float32, len(phi))
	for i, p := range phi {
		ph[i] = float32(p)
	}
	return map[string]any{"n": len(feats), "width": width, "feat": bytesOf(flat), "phi": bytesOf(ph)}
}

func main() {
	api := map[string]any{
		"vision": js.FuncOf(func(_ js.Value, a []js.Value) any {
			if len(a) != 3 {
				return map[string]any{"error": "vision(rgb, w, h)"}
			}
			feats, phi, err := tower.Vision(f32s(a[0]), a[1].Int(), a[2].Int())
			return result(feats, phi, tower.VisFeat, err)
		}),
		"audio": js.FuncOf(func(_ js.Value, a []js.Value) any {
			if len(a) != 1 {
				return map[string]any{"error": "audio(pcm)"}
			}
			feats, phi, err := tower.Audio(f32s(a[0]))
			return result(feats, phi, tower.AudFeat, err)
		}),
		"mel": js.FuncOf(func(_ js.Value, a []js.Value) any {
			if len(a) != 1 {
				return map[string]any{"error": "mel(pcm)"}
			}
			feat, frames := logMel(f32s(a[0]))
			if frames == 0 {
				return map[string]any{"error": "the clip is shorter than one frame"}
			}
			return map[string]any{"frames": frames, "bins": melBands, "feat": bytesOf(feat)}
		}),
		"ready": tower.Ready(),
	}
	js.Global().Set("hodgeTower", js.ValueOf(api))
	select {}
}

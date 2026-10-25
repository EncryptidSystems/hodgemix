package eval

import (
	"encoding/binary"
	"fmt"
	"math"

	"hodgemix/tower"
)

// towerTab is one modality's embedding layer: the projector into the residual
// stream, the feature mean it was centred on, and the embedding rms it matches.
type towerTab struct {
	D, df          int
	W              []float32
	meanF, meanRow []float32
	rms            float64
}

type towerSlot struct {
	kind string
	x    []float32
	phi  float64
}

func (m *model) loadTowers(s *shards) error {
	vis, err := s.towerEmbed(tower.VisEmbed, tower.VisFeat)
	if err != nil {
		return err
	}
	aud, err := s.towerEmbed(tower.AudEmbed, tower.AudFeat)
	if err != nil {
		return err
	}
	if vis == nil && aud == nil {
		return nil
	}
	if vis == nil || aud == nil {
		return fmt.Errorf("towers: the file has one embedding layer and not the other")
	}
	if vis.D != m.D || aud.D != m.D {
		return fmt.Errorf("tower embedding width vision %d audio %d, model hidden %d", vis.D, aud.D, m.D)
	}
	if err := s.checkTower(tower.VisTower); err != nil {
		return err
	}
	if err := s.checkTower(tower.AudTower); err != nil {
		return err
	}
	m.vis, m.aud = vis, aud
	return nil
}

func (s *shards) towerEmbed(prefix string, feat int) (*towerTab, error) {
	wt, ok := s.hdr[prefix+".weight"]
	if !ok {
		return nil, nil
	}
	var W []float32
	var D, df int
	switch wt.Dtype {
	case "U8":
		if len(wt.Shape) != 2 {
			return nil, fmt.Errorf("%s.weight shape %v", prefix, wt.Shape)
		}
		D, df = wt.Shape[0]*4, wt.Shape[1]
		b, err := s.raw(prefix + ".weight")
		if err != nil {
			return nil, err
		}
		sc, err := s.vec(prefix+".weight_scale", 1)
		if err != nil {
			return nil, err
		}
		limbs, err := s.towerLimbs(prefix+".weight", D, df)
		if err != nil {
			return nil, err
		}
		W = tower.Dense(b, D, df, sc[0], limbs)
	default:
		if !isFloat(wt.Dtype) || len(wt.Shape) != 2 {
			return nil, fmt.Errorf("%s.weight is %s %v", prefix, wt.Dtype, wt.Shape)
		}
		D, df = wt.Shape[0], wt.Shape[1]
		var err error
		if W, err = s.vec(prefix+".weight", D*df); err != nil {
			return nil, err
		}
	}
	if df != feat || D < 1 {
		return nil, fmt.Errorf("%s.weight is [%d,%d], want [*,%d]", prefix, D, df, feat)
	}
	mf, err := s.vec(prefix+".mean_feature", df)
	if err != nil {
		return nil, err
	}
	my, err := s.vec(prefix+".mean_row", D)
	if err != nil {
		return nil, err
	}
	rms, err := s.vec(prefix+".rms", 1)
	if err != nil {
		return nil, err
	}
	return &towerTab{D: D, df: df, W: W, meanF: mf, meanRow: my, rms: float64(rms[0])}, nil
}

func (s *shards) towerLimbs(weight string, D, df int) ([]tower.Limb, error) {
	var out []tower.Limb
	R4 := D / 4
	for i := 1; ; i++ {
		n := fmt.Sprintf("%s_limb%d", weight, i)
		t, ok := s.hdr[n]
		if !ok {
			return out, nil
		}
		if t.Dtype != "U8" || len(t.Shape) != 2 || t.Shape[0] != R4 || t.Shape[1] != df {
			return nil, fmt.Errorf("%s is %s %v, want U8 [%d %d]", n, t.Dtype, t.Shape, R4, df)
		}
		b, err1 := s.raw(n)
		sc, err2 := s.vec(n+"_scale", 1)
		rb, err3 := s.raw(n + "_radix")
		if err1 != nil || err2 != nil || err3 != nil || len(rb) != 4 {
			return nil, fmt.Errorf("%s: incomplete limb", n)
		}
		out = append(out, tower.Limb{Radix: int(binary.LittleEndian.Uint32(rb)), Digits: b, Scale: sc[0]})
	}
}

func (s *shards) checkTower(prefix string) error {
	sig, err := s.complexConst(prefix + ".sigma")
	if err != nil {
		return err
	}
	if cmplxAbs(sig-tower.Sigma()) > 2e-5 {
		return fmt.Errorf("%s.sigma is %v, the L-A bias is %v", prefix, sig, tower.Sigma())
	}
	zs, err := s.complexConst(prefix + ".zstar")
	if err != nil {
		return err
	}
	if cmplxAbs(zs-tower.ZStar) > 2e-5 {
		return fmt.Errorf("%s.zstar is %v, want %v", prefix, zs, tower.ZStar)
	}
	b, err := s.raw(prefix + ".depth")
	if err != nil {
		return err
	}
	if len(b) < 4 || int32(binary.LittleEndian.Uint32(b)) != tower.Depth {
		return fmt.Errorf("%s.depth is not %d", prefix, tower.Depth)
	}
	return nil
}

func (s *shards) complexConst(name string) (complex128, error) {
	v, err := s.vec(name, 2)
	if err != nil {
		return 0, err
	}
	return complex(float64(v[0]), float64(v[1])), nil
}

func cmplxAbs(z complex128) float64 {
	return math.Hypot(real(z), imag(z))
}

func (s *seq) phaseAt(pos int) float64 {
	if s == nil || pos < 0 || pos >= len(s.phi) {
		return 0
	}
	return s.phi[pos]
}

func (m *model) install(tab *towerTab, kind string, base int, feats [][]float32, phi []float64) ([]int, error) {
	if tab == nil {
		return nil, fmt.Errorf("this model has no %s embedding layer", kind)
	}
	if m.D != 0 && tab.D != m.D {
		return nil, fmt.Errorf("%s projector width %d, model hidden %d", kind, tab.D, m.D)
	}
	if m.slots == nil {
		m.slots = map[int]towerSlot{}
	}
	for id, sl := range m.slots {
		if sl.kind == kind {
			delete(m.slots, id)
		}
	}
	ids := make([]int, len(feats))
	for i, f := range feats {
		x, err := tower.Project(tab.W, tab.D, tab.df, tab.meanF, tab.meanRow, tab.rms, f)
		if err != nil {
			return nil, err
		}
		p := 0.0
		if i < len(phi) {
			p = phi[i]
		}
		id := base + i
		m.slots[id] = towerSlot{kind: kind, x: x, phi: p}
		ids[i] = id
	}
	return ids, nil
}

// Vision encodes an image into reserved token ids, one past the text embedding.
// The ids are layer-0 states: the vision embedding layer replaces the token row (D3).
func (s *Session) Vision(rgb []float32, w, h int) ([]int, error) {
	if s.m == nil || s.m.vis == nil {
		return nil, fmt.Errorf("this model has no vision embedding layer")
	}
	feats, phi, err := tower.Vision(rgb, w, h)
	if err != nil {
		return nil, err
	}
	return s.m.install(s.m.vis, "vision", s.m.nEmb, feats, phi)
}

// Audio encodes mono 16 kHz pcm into reserved token ids after the vision range.
func (s *Session) Audio(pcm []float32) ([]int, error) {
	if s.m == nil || s.m.aud == nil {
		return nil, fmt.Errorf("this model has no audio embedding layer")
	}
	feats, phi, err := tower.Audio(pcm)
	if err != nil {
		return nil, err
	}
	return s.m.install(s.m.aud, "audio", s.m.nEmb+tower.VisSlots, feats, phi)
}

package main

// OV chart (--ov chart, default): merge each attention head's OV circuit to first order, keeping the
// base's value map as the head's chart.
//
// A head writes O_h V_g x (V_g its kv head's value rows, O_h its columns of o_proj): the circuit is the
// product O V, and V is the chart, the addresses the head reads. Mixing both factors gives
//     (O_b + dO)(V_b + dV) = O_b V_b + dO V_b + O_b dV + dO dV,
// and the cross term dO dV is the source's own circuit read through the base's addresses: each of v or
// o mixed alone is fine, both together broke every 334-token chat (2B4T + Qwen3) while 14-token chats
// answered. The chart keeps V_b and moves the circuit in its coordinates:
//     V_t = V_b,   O_t,h = O_h,mix + O_b,h (dV_g V_b,g^+),   V^+ = V^T (V V^T)^-1,
// so O_t V_b = O_b V_b + dO V_b + O_b dV P_g (P_g the projection on V_b,g's row space): exactly the first
// order change of the circuit, no cross term, rank and shapes unchanged. Per kv group the solve is
// hd x hd. Applied to the written ternary file (dO, dV from its codes), before surgery.
// No paper equation.

import (
	"fmt"
	"math"
	"os"
)

// OVMode is --ov: chart | mix.
var OVMode = "chart"

// solveSPD returns K G^-1 for G symmetric positive definite (n x n), K (n x n), row major.
func solveSPD(K, G []float64, n int) []float64 {
	A := append([]float64(nil), G...)
	var tr float64
	for i := 0; i < n; i++ {
		tr += A[i*n+i]
	}
	for i := 0; i < n; i++ {
		A[i*n+i] += 1e-9 * tr / float64(n)
	}
	// X G = K  <=>  G X^T = K^T (G symmetric).
	Kt := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Kt[j*n+i] = K[i*n+j]
		}
	}
	Xt := cholSolve(A, Kt, n, n)
	X := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			X[i*n+j] = Xt[j*n+i]
		}
	}
	return X
}

// ternarize is the pipeline's absmean code choice and least-squares weight_scale.
func ternarize(W []float32) ([]int8, float64) {
	s := absmean(W)
	q := make([]int8, len(W))
	var nz float64
	n := 0
	for i, v := range W {
		q[i] = int8(ternQ(float64(v), s))
		if q[i] != 0 {
			nz += math.Abs(float64(v))
			n++
		}
	}
	if n == 0 {
		return q, s
	}
	return q, nz / float64(n)
}

type ovReport struct {
	Layers int     `json:"layers"`
	Cross  float64 `json:"cross_term"` // mean ||dO dV|| / ||O_b V_b|| removed, per head
}

// ovChart rewrites v_proj (to the base's) and o_proj (the first-order circuit) of outTern in place.
func ovChart(basePath, outTern string, heads, kv, hd int) (*ovReport, error) {
	base, err := OpenST(basePath)
	if err != nil {
		return nil, err
	}
	defer base.Close()
	out, err := OpenST(outTern)
	if err != nil {
		return nil, err
	}
	defer out.Close()
	f, err := os.OpenFile(outTern, os.O_RDWR, 0)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	rep := &ovReport{}
	G := heads / kv
	var cross float64
	var ncross int
	for l := 0; ; l++ {
		p := fmt.Sprintf("model.layers.%d.self_attn.", l)
		vn, on := p+"v_proj.weight", p+"o_proj.weight"
		if !out.Has(vn) || !out.Has(on) {
			break
		}
		if !base.Has(vn) || !base.Has(on) || !out.isPacked(vn) || !out.isPacked(on) {
			continue
		}
		Vb, vsh, err1 := base.LoadDense(vn)
		Vm, _, err2 := out.LoadDense(vn)
		Ob, osh, err3 := base.LoadDense(on)
		Om, _, err4 := out.LoadDense(on)
		if err1 != nil || err2 != nil || err3 != nil || err4 != nil {
			return nil, fmt.Errorf("layer %d: %v %v %v %v", l, err1, err2, err3, err4)
		}
		D := vsh[1]
		if vsh[0] != kv*hd || osh[0] != D || osh[1] != heads*hd {
			return nil, fmt.Errorf("layer %d: v %v / o %v are not [%d,%d] / [%d,%d]", l, vsh, osh, kv*hd, D, D, heads*hd)
		}
		Ot := make([]float32, len(Om))
		copy(Ot, Om)
		for g := 0; g < kv; g++ {
			Gm, K := make([]float64, hd*hd), make([]float64, hd*hd)
			parFor(hd, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					vi := Vb[(g*hd+i)*D : (g*hd+i+1)*D]
					di := Vm[(g*hd+i)*D : (g*hd+i+1)*D]
					for j := 0; j < hd; j++ {
						vj := Vb[(g*hd+j)*D : (g*hd+j+1)*D]
						var a, b float64
						for c := 0; c < D; c++ {
							a += float64(vi[c]) * float64(vj[c])
							b += (float64(di[c]) - float64(vi[c])) * float64(vj[c])
						}
						Gm[i*hd+j], K[i*hd+j] = a, b
					}
				}
			})
			X := solveSPD(K, Gm, hd) // dV_g V_b,g^+ (hd x hd)
			for j := 0; j < G; j++ {
				h := g*G + j
				parFor(D, func(lo, hi int) {
					for r := lo; r < hi; r++ {
						row := Ob[r*heads*hd+h*hd : r*heads*hd+(h+1)*hd]
						dst := Ot[r*heads*hd+h*hd : r*heads*hd+(h+1)*hd]
						for c := 0; c < hd; c++ {
							var s float64
							for k := 0; k < hd; k++ {
								s += float64(row[k]) * X[k*hd+c]
							}
							dst[c] += float32(s)
						}
					}
				})
			}
			// Size of the removed cross term for this group, ||dO dV||_F / ||O_b V_b||_F, sampled on the
			// first head (O^T O and V V^T Gram traces avoid forming the D x D products).
			h := g * G
			var num, den float64
			dOg, Obg := make([]float64, hd*hd), make([]float64, hd*hd)
			for r := 0; r < D; r++ {
				for a := 0; a < hd; a++ {
					oa := float64(Ob[r*heads*hd+h*hd+a])
					da := float64(Om[r*heads*hd+h*hd+a]) - oa
					for b := 0; b < hd; b++ {
						ob := float64(Ob[r*heads*hd+h*hd+b])
						dOg[a*hd+b] += da * (float64(Om[r*heads*hd+h*hd+b]) - ob)
						Obg[a*hd+b] += oa * ob
					}
				}
			}
			dVg := make([]float64, hd*hd)
			for a := 0; a < hd; a++ {
				for b := 0; b < hd; b++ {
					var s float64
					for c := 0; c < D; c++ {
						s += (float64(Vm[(g*hd+a)*D+c]) - float64(Vb[(g*hd+a)*D+c])) * (float64(Vm[(g*hd+b)*D+c]) - float64(Vb[(g*hd+b)*D+c]))
					}
					dVg[a*hd+b] = s
				}
			}
			for i := range dOg {
				num += dOg[i] * dVg[i]
				den += Obg[i] * Gm[i]
			}
			if den > 0 {
				cross += math.Sqrt(math.Max(num, 0) / den)
				ncross++
			}
		}
		// v_proj back to the base's codes and scale; o_proj ternarized from O_t.
		// A packed base gives its codes and scale as stored; a float base is ternarized the way the
		// pipeline ternarizes (its raw BF16 bytes do not fit the packed slot and would overrun it).
		var vb, vs []byte
		if base.isPacked(vn) {
			var e1, e2 error
			vb, e1 = base.Raw(vn)
			vs, e2 = base.Raw(scaleKey(vn))
			if e1 != nil || e2 != nil {
				return nil, fmt.Errorf("layer %d: base v_proj: %v %v", l, e1, e2)
			}
		} else {
			vq, vws := ternarize(Vb)
			vb = packTrits(vq, vsh[0], vsh[1])
			u := bf16(float32(vws))
			vs = []byte{byte(u), byte(u >> 8)}
		}
		ov, os2 := out.H[vn], out.H[scaleKey(vn)]
		if int64(len(vb)) != ov.Off[1]-ov.Off[0] {
			return nil, fmt.Errorf("layer %d: base v_proj is %d bytes, the output's slot %d", l, len(vb), ov.Off[1]-ov.Off[0])
		}
		if _, err := f.WriteAt(vb, out.base+ov.Off[0]); err != nil {
			return nil, err
		}
		if int64(len(vs)) != os2.Off[1]-os2.Off[0] {
			return nil, fmt.Errorf("layer %d: base v_proj weight_scale is %d bytes, the output's slot %d", l, len(vs), os2.Off[1]-os2.Off[0])
		}
		if _, err := f.WriteAt(vs, out.base+os2.Off[0]); err != nil {
			return nil, err
		}
		q, ws := ternarize(Ot)
		oo, oso := out.H[on], out.H[scaleKey(on)]
		if _, err := f.WriteAt(packTrits(q, osh[0], osh[1]), out.base+oo.Off[0]); err != nil {
			return nil, err
		}
		b := make([]byte, 2)
		u := bf16(float32(ws))
		b[0], b[1] = byte(u), byte(u>>8)
		if _, err := f.WriteAt(b, out.base+oso.Off[0]); err != nil {
			return nil, err
		}
		// Dynamic MRH limbs (limbs.go): v is the base's single ternary limb again; o's extra limbs
		// carry what its new limb 0 leaves of O_t.
		if err := zeroLimbs(f, out, vn); err != nil {
			return nil, err
		}
		if err := refillLimbs(f, out, on, Ot, q, ws); err != nil {
			return nil, err
		}
		rep.Layers++
	}
	if ncross > 0 {
		rep.Cross = cross / float64(ncross)
	}
	return rep, nil
}

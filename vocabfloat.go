package main

// Vocabulary on the instrumented residual cochain.
//
// An instrumented model is one whose volume.safetensors the page at
// ai.encryptid.systems reads. cell.D is the residual axis: channel j sits in
// cell cell.D[j]. The page's node D, and eval.cellSums, are the sums of the
// channels in each cell. That sum is the 0-cochain of one vocabulary row.
//
// The floating point is the channel vector whose cell sums are that cochain.
// Each channel in a cell receives an equal share of the cell sum, so a row that
// was already constant on each cell comes back unchanged.

import (
	"encoding/binary"
	"fmt"
	"math"
)

// vocabCochain maps each embedding row to the residual cell-sum cochain.
func vocabCochain(emb []float32, D int, cell []int32, n int) []float32 {
	if D == 0 || n == 0 || len(emb)%D != 0 || len(cell) != D {
		return nil
	}
	V := len(emb) / D
	o := make([]float32, V*n)
	acc := make([]float64, n)
	for t := 0; t < V; t++ {
		for i := range acc {
			acc[i] = 0
		}
		row := emb[t*D : (t+1)*D]
		for j, v := range row {
			g := int(cell[j])
			if g < 0 || g >= n {
				return nil
			}
			acc[g] += float64(v)
		}
		for i, v := range acc {
			o[t*n+i] = float32(v)
		}
	}
	return o
}

// cochainFloat maps a residual cell-sum cochain back to one float per channel.
func cochainFloat(co []float32, D int, cell []int32, n int) []float32 {
	if D == 0 || n == 0 || len(cell) != D || len(co)%n != 0 {
		return nil
	}
	count := make([]int, n)
	for _, c := range cell {
		g := int(c)
		if g < 0 || g >= n {
			return nil
		}
		count[g]++
	}
	V := len(co) / n
	o := make([]float32, V*D)
	for t := 0; t < V; t++ {
		c := co[t*n : (t+1)*n]
		row := o[t*D : (t+1)*D]
		for j := 0; j < D; j++ {
			k := count[cell[j]]
			if k == 0 {
				return nil
			}
			row[j] = c[cell[j]] / float32(k)
		}
	}
	return o
}

// vocabFloat reads the served embedding and the volume's cell.D, and writes
// cochain [V, cells] and float [V, D] beside the model. The checkpoint
// embedding is not changed.
func vocabFloat(modelPath, volumePath, outPath string) error {
	st, err := OpenST(modelPath)
	if err != nil {
		return err
	}
	ek := embedKey(st)
	if ek == "" {
		return fmt.Errorf("%s: no token embedding", modelPath)
	}
	sh := st.LogicalShape(ek)
	if len(sh) != 2 {
		return fmt.Errorf("%s: %s shape %v", modelPath, ek, sh)
	}
	V, D := sh[0], sh[1]
	vol, err := OpenST(volumePath)
	if err != nil {
		return err
	}
	cell, err := readI32(vol, "cell.D")
	if err != nil {
		return err
	}
	if len(cell) != D {
		return fmt.Errorf("cell.D has %d channels, the embedding has %d", len(cell), D)
	}
	n := 0
	for _, c := range cell {
		if int(c) < 0 {
			return fmt.Errorf("cell.D has a negative cell")
		}
		if int(c)+1 > n {
			n = int(c) + 1
		}
	}
	emb, err := st.F32(ek)
	if err != nil {
		return err
	}
	if len(emb) != V*D {
		return fmt.Errorf("%s: embedding %d, shape %v", ek, len(emb), sh)
	}
	count := make([]int, n)
	for _, c := range cell {
		g := int(c)
		if g >= n {
			return fmt.Errorf("cell.D names cell %d of %d", g, n)
		}
		count[g]++
	}
	for g, k := range count {
		if k == 0 {
			return fmt.Errorf("residual cell %d holds no channel", g)
		}
	}
	co := make([]float32, V*n)
	acc := make([]float64, n)
	W, err := NewSTW(outPath, [][2]string{{"format", "pt"}, {"vocab_of", modelPath}, {"volume", volumePath}})
	if err != nil {
		return err
	}
	tw := W.Append("float", "F32", []int{V, D})
	buf := make([]byte, D*4)
	for t := 0; t < V; t++ {
		for i := range acc {
			acc[i] = 0
		}
		row := emb[t*D : (t+1)*D]
		for j, v := range row {
			acc[cell[j]] += float64(v)
		}
		c := co[t*n : (t+1)*n]
		for i, v := range acc {
			c[i] = float32(v)
		}
		for j := 0; j < D; j++ {
			v := c[cell[j]] / float32(count[cell[j]])
			binary.LittleEndian.PutUint32(buf[4*j:], math.Float32bits(v))
		}
		if _, err := tw.Write(buf); err != nil {
			W.Abort()
			return err
		}
	}
	W.AddF32("cochain", []int{V, n}, co)
	logf("  vocab %s: %d tokens, %d residual cells -> %s", ek, V, n, outPath)
	if err := W.Close(nil); err != nil {
		return err
	}
	return nil
}

func readI32(st *ST, k string) ([]int32, error) {
	b, err := st.Raw(k)
	if err != nil {
		return nil, err
	}
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("%s: %d bytes is not i32", k, len(b))
	}
	o := make([]int32, len(b)/4)
	for i := range o {
		o[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return o, nil
}

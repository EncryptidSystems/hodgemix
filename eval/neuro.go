package eval

// Graded spikes and the non-adjacent form of an int8 BitLinear code.
//
// quant already made every BitLinear input an integer in [-128, 127]. A zero
// code adds nothing, so it is not an event. A nonzero code is one graded
// spike. The non-adjacent form writes that same integer as at most four
// digits in {-1, 0, +1} across eight places (Reitwiesner). Both forms produce
// the integer accumulator applyIn reduces, so the scaled output matches it
// bit for bit. kernels/neuro.wgsl runs the graded form on WebGPU.

// NAFDigits is the unique non-adjacent form of q, digits d[t] in {-1, 0, +1}
// with q = sum_t d[t] * 2^t and no two nonzero digits adjacent. q is an int8
// code. Eight places are enough: the value fits in [-128, 127].
func NAFDigits(q int) [8]int8 {
	var d [8]int8
	k := q
	for t := 0; t < 8; t++ {
		if k&1 != 0 {
			r := k % 4
			if r < 0 {
				r += 4
			}
			dig := int8(1)
			if r == 3 {
				dig = -1
			}
			d[t] = dig
			k -= int(dig)
		}
		k /= 2
	}
	return d
}

// NAFWeight is the number of nonzero digits in NAFDigits(q).
func NAFWeight(q int) int {
	d := NAFDigits(q)
	n := 0
	for _, v := range d {
		if v != 0 {
			n++
		}
	}
	return n
}

// EventAcc is the graded-spike accumulator. One event per nonzero q[j]:
// acc[i] = sum_j (digits[i, j] - center) * q[j]. A zero code is skipped.
// digits is row-major [n][k], each entry a radix digit 0..r-1, and center is
// (r-1)/2, the same centre a limb uses.
func EventAcc(digits []uint8, n, k, center int, q []int8) []int32 {
	ev := make([]int, 0, k)
	for j := 0; j < k; j++ {
		if q[j] != 0 {
			ev = append(ev, j)
		}
	}
	out := make([]int32, n)
	for i := 0; i < n; i++ {
		row := digits[i*k:]
		var acc int32
		for _, j := range ev {
			acc += int32(int(row[j])-center) * int32(q[j])
		}
		out[i] = acc
	}
	return out
}

// nafAcc is the same accumulator with each code expanded into its non-adjacent
// digits. acc = sum_t 2^t * D * s_t, and that equals EventAcc.
func nafAcc(digits []uint8, n, k, center int, q []int8) []int32 {
	out := make([]int32, n)
	for j := 0; j < k; j++ {
		d := NAFDigits(int(q[j]))
		for t := 0; t < 8; t++ {
			if d[t] == 0 {
				continue
			}
			step := int32(d[t]) << uint(t)
			for i := 0; i < n; i++ {
				out[i] += int32(int(digits[i*k+j])-center) * step
			}
		}
	}
	return out
}

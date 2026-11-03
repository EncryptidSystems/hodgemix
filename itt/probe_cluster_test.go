//go:build probe

package main

import (
	"math"
	"math/rand"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"unicode"

	"hodgemix/eval"
)

const probeDir = `C:\Users\grego\models\dev\hodge-3.5-instruct`

const potter = `Hagrid raised a gigantic fist and knocked three times on the castle door. The door swung open at once. A tall, black-haired witch in emerald-green robes stood there. She had a very stern face and Harry's first thought was that this was not someone to cross. The first-years, Professor McGonagall, said Hagrid. Thank you, Hagrid. I will take them from here. She pulled the door wide. The Entrance Hall was so big you could have fitted the whole of the Dursleys' house in it.`

const neutral = `The committee met on Monday to review the budget and the schedule. People asked questions about the cost and the time. A report will be sent to the office next week. They agreed to wait until the numbers are ready before they decide. The city will publish the notes after the meeting so that residents can read them.`

const pronouns = `She left the chamber. Harry swallowed. How exactly do they sort us into houses he asked Ron. Some sort of test I think Fred said it hurts a lot but I think he was joking. Harry's heart gave a horrible jolt. A test in front of the whole school. But he didn't know any magic yet. What on earth would he have to do. He hadn't expected something like this the moment they arrived. He looked around anxiously.`

type probeWord struct {
	text string
	v    []float64
}

func TestPotterCluster(t *testing.T) {
	co, V, C, sess := openProbe(t)
	hp := takeWords(t, sess, co, V, C, potter, 50, 3)
	pr := takeWords(t, sess, co, V, C, pronouns, 50, 3)
	neu := takeWords(t, sess, co, V, C, neutral, 50, 3)
	prShort := takeWords(t, sess, co, V, C, pronouns, 50, 2)
	reportSet(t, "harry potter opening", hp)
	reportSet(t, "harry potter pronouns", pr)
	reportSet(t, "neutral prose", neu)

	mu := vocabMean(co, V, C)
	rng := rand.New(rand.NewSource(1))
	var randCos, randPair, cenCos, cenPair []float64
	var example []string
	var firstRand []probeWord
	for trial := 0; trial < 20; trial++ {
		xs := sampleWords(rng, sess, co, V, C, 50)
		cos, pair, _ := tightness(vecsOf(xs))
		randCos = append(randCos, cos)
		randPair = append(randPair, pair)
		cc, cp := centeredStats(xs, mu)
		cenCos = append(cenCos, cc)
		cenPair = append(cenPair, cp)
		if trial == 0 {
			firstRand = xs
			for _, w := range xs {
				example = append(example, w.text)
			}
			t.Logf("one random 50: %s", strings.Join(example, ", "))
			t.Logf("mean length  potter %.0f  pronouns %.0f  neutral %.0f  random %.0f",
				meanLen(hp), meanLen(pr), meanLen(neu), meanLen(xs))
		}
	}
	t.Logf("random vocab words x20  cosine to centroid mean %.3f (min %.3f max %.3f)  pairwise mean %.3f (min %.3f max %.3f)",
		meanOf(randCos), minOf(randCos), maxOf(randCos), meanOf(randPair), minOf(randPair), maxOf(randPair))
	t.Logf("random vocab words x20 centered  cosine to centroid mean %.3f (min %.3f max %.3f)  pairwise mean %.3f (min %.3f max %.3f)",
		meanOf(cenCos), minOf(cenCos), maxOf(cenCos), meanOf(cenPair), minOf(cenPair), maxOf(cenPair))

	hi := indexWord(hp, "Harry", "Harry's")
	if hi < 0 {
		t.Fatal("Harry not in the 50")
	}
	t.Logf("center check: Potter word closest to its own centroid is above; Harry's row is %q", hp[hi].text)
	var around []string
	for _, n := range nearest(hp[hi].v, vecsOf(hp), 12) {
		if n == hi {
			continue
		}
		around = append(around, hp[n].text)
	}
	t.Logf("nearest of the 50 Potter words to Harry: %s", strings.Join(around, ", "))
	t.Logf("nearest vocab words to Harry: %s", strings.Join(topVocab(hp[hi].v, co, V, C, 25, sess), ", "))
	reportCentered(t, "potter opening", hp, mu)
	reportCentered(t, "potter pronouns", pr, mu)
	reportCentered(t, "neutral prose", neu, mu)
	reportGraph(t, "potter opening", hp, mu)
	reportGraph(t, "potter pronouns", pr, mu)
	reportGraph(t, "neutral prose", neu, mu)
	reportGraph(t, "random vocab", firstRand, mu)
	reportCentered(t, "pronouns with he/him kept", prShort, mu)
	reportGraph(t, "pronouns with he/him kept", prShort, mu)
	if i := indexWord(prShort, "Harry", "Harry's"); i >= 0 {
		var around []string
		for _, n := range nearestAt(prShort[i].v, vecsOf(prShort), mu, 12) {
			if n == i {
				continue
			}
			around = append(around, prShort[n].text)
		}
		t.Logf("centered nearest of the pronoun passage to Harry: %s", strings.Join(around, ", "))
	}
	t.Logf("Harry's vocab neighbors, shared direction removed: %s", strings.Join(topVocabCentered(hp[hi].v, co, V, C, mu, 20, sess), ", "))

	cOpen := centeredCentroid(hp, mu)
	cPro := centeredCentroid(pr, mu)
	cNeu := centeredCentroid(neu, mu)
	cRnd := centeredCentroid(firstRand, mu)
	t.Logf("watermark centroid cosine  opening·pronouns %.3f  opening·neutral %.3f  pronouns·neutral %.3f  opening·random %.3f",
		cosine(cOpen, cPro), cosine(cOpen, cNeu), cosine(cPro, cNeu), cosine(cOpen, cRnd))
	shared, against := signSplit(cOpen, cPro, cNeu)
	t.Logf("watermark cells  both Potter passages same sign %d/64  and neutral has the other sign %d/64", shared, against)
	reportRipple(t, "potter opening", hp, mu, rng)
	reportRipple(t, "potter pronouns", pr, mu, rng)
	reportRipple(t, "neutral prose", neu, mu, rng)
	reportRipple(t, "random vocab", firstRand, mu, rng)
	reportLattice(t, "potter opening", hp, mu, rng)
	reportLattice(t, "potter pronouns", pr, mu, rng)
	reportLattice(t, "neutral prose", neu, mu, rng)
	reportLattice(t, "random vocab", firstRand, mu, rng)
}

func rowFreq(xs []probeWord, mu []float64) (steady, ripple float64) {
	R := len(xs)
	var total, kr0, kr1 float64
	col := make([]float64, R)
	for c := range mu {
		for r := 0; r < R; r++ {
			col[r] = xs[r].v[c] - mu[c]
		}
		for kr := 0; kr < R; kr++ {
			var re, im float64
			for r := 0; r < R; r++ {
				ang := -2 * math.Pi * float64(kr*r) / float64(R)
				re += col[r] * math.Cos(ang)
				im += col[r] * math.Sin(ang)
			}
			e := re*re + im*im
			total += e
			if kr == 0 {
				kr0 += e
			}
			if kr == 1 || kr == R-1 {
				kr1 += e
			}
		}
	}
	return kr0 / total, kr1 / total
}

func reportLattice(t *testing.T, name string, xs []probeWord, mu []float64, rng *rand.Rand) {
	t.Helper()
	steady, wave := rowFreq(xs, mu)
	var sh []float64
	for trial := 0; trial < 20; trial++ {
		cp := append([]probeWord(nil), xs...)
		rng.Shuffle(len(cp), func(i, j int) { cp[i], cp[j] = cp[j], cp[i] })
		_, w := rowFreq(cp, mu)
		sh = append(sh, w)
	}
	t.Logf("lattice %s  steady share %.3f  ripple share %.3f  shuffle ripple %.3f (min %.3f max %.3f)",
		name, steady, wave, meanOf(sh), minOf(sh), maxOf(sh))
}

func centeredCentroid(xs []probeWord, mu []float64) []float64 {
	c := centroid(vecsOf(xs))
	for i := range c {
		c[i] -= mu[i]
	}
	return c
}

func signSplit(a, b, other []float64) (shared, against int) {
	for i := range a {
		if (a[i] >= 0) != (b[i] >= 0) {
			continue
		}
		shared++
		if (a[i] >= 0) != (other[i] >= 0) {
			against++
		}
	}
	return shared, against
}

func pathSmooth(xs []probeWord, mu []float64) float64 {
	var sum float64
	for i := 0; i < len(xs)-1; i++ {
		sum += cosineAt(xs[i].v, xs[i+1].v, mu)
	}
	return sum / float64(len(xs)-1)
}

func pathRipple(xs []probeWord) float64 {
	var sum float64
	n := 0
	prev := diffVec(xs[0].v, xs[1].v)
	for i := 1; i < len(xs)-1; i++ {
		next := diffVec(xs[i].v, xs[i+1].v)
		sum += cosine(prev, next)
		n++
		prev = next
	}
	return sum / float64(n)
}

func diffVec(a, b []float64) []float64 {
	o := make([]float64, len(a))
	for i := range a {
		o[i] = b[i] - a[i]
	}
	return o
}

func reportRipple(t *testing.T, name string, xs []probeWord, mu []float64, rng *rand.Rand) {
	t.Helper()
	smooth := pathSmooth(xs, mu)
	wave := pathRipple(xs)
	var shSmooth, shWave []float64
	for trial := 0; trial < 20; trial++ {
		sh := append([]probeWord(nil), xs...)
		rng.Shuffle(len(sh), func(i, j int) { sh[i], sh[j] = sh[j], sh[i] })
		shSmooth = append(shSmooth, pathSmooth(sh, mu))
		shWave = append(shWave, pathRipple(sh))
	}
	t.Logf("ripple %s  step cosine %.3f  shuffle %.3f (min %.3f max %.3f)  turn cosine %.3f  shuffle %.3f (min %.3f max %.3f)",
		name, smooth, meanOf(shSmooth), minOf(shSmooth), maxOf(shSmooth),
		wave, meanOf(shWave), minOf(shWave), maxOf(shWave))
}

func vocabMean(co []int32, V, C int) []float64 {
	o := make([]float64, C)
	for id := 0; id < V; id++ {
		base := id * C
		for g := 0; g < C; g++ {
			o[g] += float64(co[base+g])
		}
	}
	for g := range o {
		o[g] /= float64(V)
	}
	return o
}

func centeredStats(xs []probeWord, mu []float64) (meanCos, meanPair float64) {
	vs := vecsOf(xs)
	c := centroid(vs)
	var sum float64
	for _, v := range vs {
		sum += cosineAt(v, c, mu)
	}
	var pairs, n float64
	for i := 0; i < len(vs); i++ {
		for j := i + 1; j < len(vs); j++ {
			pairs += cosineAt(vs[i], vs[j], mu)
			n++
		}
	}
	return sum / float64(len(vs)), pairs / n
}

func reportCentered(t *testing.T, name string, xs []probeWord, mu []float64) {
	t.Helper()
	vs := vecsOf(xs)
	c := centroid(vs)
	var sum float64
	best, center := -2.0, 0
	for i, v := range vs {
		s := cosineAt(v, c, mu)
		sum += s
		if s > best {
			best, center = s, i
		}
	}
	var pairs, n float64
	for i := 0; i < len(vs); i++ {
		for j := i + 1; j < len(vs); j++ {
			pairs += cosineAt(vs[i], vs[j], mu)
			n++
		}
	}
	t.Logf("centered %s  cosine to centroid %.3f  pairwise %.3f  closest %q",
		name, sum/float64(len(vs)), pairs/n, xs[center].text)
}

func reportGraph(t *testing.T, name string, xs []probeWord, mu []float64) {
	t.Helper()
	vs := vecsOf(xs)
	const k = 3
	nbr := make([][]int, len(vs))
	for i := range vs {
		nbr[i] = knn(i, vs, mu, k)
	}
	mutual := 0
	var edgeCos float64
	deg := make([]int, len(vs))
	var lines []string
	for i := range vs {
		for _, j := range nbr[i] {
			back := false
			for _, h := range nbr[j] {
				if h == i {
					back = true
				}
			}
			if back && i < j {
				mutual++
				edgeCos += cosineAt(vs[i], vs[j], mu)
				deg[i]++
				deg[j]++
				lines = append(lines, xs[i].text+"—"+xs[j].text)
			}
		}
	}
	hub, best := 0, -1
	for i, d := range deg {
		if d > best {
			best, hub = d, i
		}
	}
	meanEdge := 0.0
	if mutual > 0 {
		meanEdge = edgeCos / float64(mutual)
	}
	t.Logf("topo %s  mutual 3-neighbor edges %d  mean edge cosine %.3f  hub %q degree %d", name, mutual, meanEdge, xs[hub].text, best)
	if len(lines) > 18 {
		lines = lines[:18]
	}
	t.Logf("topo %s  edges %s", name, strings.Join(lines, ", "))
}

func knn(i int, vs [][]float64, mu []float64, k int) []int {
	type pair struct {
		j int
		s float64
	}
	all := make([]pair, 0, len(vs)-1)
	for j := range vs {
		if j == i {
			continue
		}
		all = append(all, pair{j, cosineAt(vs[i], vs[j], mu)})
	}
	sort.Slice(all, func(a, b int) bool { return all[a].s > all[b].s })
	out := make([]int, k)
	for n := 0; n < k; n++ {
		out[n] = all[n].j
	}
	return out
}

func cosineAt(a, b, mu []float64) float64 {
	var d, na, nb float64
	for i := range a {
		x, y := a[i]-mu[i], b[i]-mu[i]
		d += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return d / math.Sqrt(na*nb)
}

func topVocabCentered(target []float64, co []int32, V, C int, mu []float64, k int, sess *eval.Session) []string {
	type pair struct {
		id int
		s  float64
	}
	best := make([]pair, 0, k)
	push := func(p pair) {
		if len(best) < k {
			best = append(best, p)
			return
		}
		min := 0
		for i := 1; i < len(best); i++ {
			if best[i].s < best[min].s {
				min = i
			}
		}
		if p.s > best[min].s {
			best[min] = p
		}
	}
	row := make([]float64, C)
	for id := 0; id < V; id++ {
		base := id * C
		var na float64
		for g := 0; g < C; g++ {
			row[g] = float64(co[base+g])
			na += row[g] * row[g]
		}
		if na == 0 {
			continue
		}
		w := strings.TrimSpace(sess.Decode([]int{id}))
		if !isProbeWord(w) {
			continue
		}
		push(pair{id, cosineAt(target, row, mu)})
	}
	sort.Slice(best, func(i, j int) bool { return best[i].s > best[j].s })
	out := make([]string, len(best))
	for i, p := range best {
		out[i] = strings.TrimSpace(sess.Decode([]int{p.id}))
	}
	return out
}

func wordID(sess *eval.Session, w string) int {
	ids := sess.Encode(w, false, false)
	if len(ids) != 1 {
		return -1
	}
	return ids[0]
}

func meanLen(xs []probeWord) float64 {
	var s float64
	for _, x := range xs {
		s += math.Sqrt(dot(x.v, x.v))
	}
	return s / float64(len(xs))
}

func dot(a, b []float64) float64 {
	var d float64
	for i := range a {
		d += a[i] * b[i]
	}
	return d
}

func openProbe(t *testing.T) ([]int32, int, int, *eval.Session) {
	t.Helper()
	co, V, C, err := loadCochain(filepath.Join(probeDir, "model.cochain.safetensors"))
	if err != nil {
		t.Skip(err)
	}
	sess, err := eval.OpenTok(filepath.Join(probeDir, "tokenizer.json"), filepath.Join(probeDir, "tokenizer_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	return co, V, C, sess
}

func takeWords(t *testing.T, sess *eval.Session, co []int32, V, C int, text string, n, minLen int) []probeWord {
	t.Helper()
	var out []probeWord
	for _, ids := range sess.Segments(text) {
		w := strings.TrimSpace(sess.Decode(ids))
		if !isWord(w, minLen) {
			continue
		}
		out = append(out, probeWord{w, sumRows(co, V, C, ids)})
		if len(out) == n {
			break
		}
	}
	if len(out) < n {
		t.Fatalf("only %d words in %q", len(out), text[:40])
	}
	return out
}

func sampleWords(rng *rand.Rand, sess *eval.Session, co []int32, V, C, n int) []probeWord {
	var out []probeWord
	seen := map[int]bool{}
	for len(out) < n {
		id := rng.Intn(V)
		if seen[id] {
			continue
		}
		seen[id] = true
		w := strings.TrimSpace(sess.Decode([]int{id}))
		if !isProbeWord(w) {
			continue
		}
		out = append(out, probeWord{w, sumRows(co, V, C, []int{id})})
	}
	return out
}

func sumRows(co []int32, V, C int, ids []int) []float64 {
	o := make([]float64, C)
	for _, id := range ids {
		if id < 0 || id >= V {
			continue
		}
		base := id * C
		for g := 0; g < C; g++ {
			o[g] += float64(co[base+g])
		}
	}
	return o
}

func isProbeWord(s string) bool {
	return isWord(s, 3)
}

func isWord(s string, minLen int) bool {
	rs := []rune(s)
	if len(rs) < minLen || len(rs) > 24 {
		return false
	}
	for _, r := range rs {
		if r != '\'' && !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}

func nearestAt(target []float64, vs [][]float64, mu []float64, k int) []int {
	type pair struct {
		i int
		s float64
	}
	all := make([]pair, len(vs))
	for i, v := range vs {
		all[i] = pair{i, cosineAt(target, v, mu)}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].s > all[j].s })
	if k > len(all) {
		k = len(all)
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = all[i].i
	}
	return out
}

func reportSet(t *testing.T, name string, xs []probeWord) {
	t.Helper()
	cos, pair, center := tightness(vecsOf(xs))
	t.Logf("%s  n=%d  mean cosine to centroid %.3f  mean pairwise %.3f  word closest to centroid %q",
		name, len(xs), cos, pair, xs[center].text)
}

func vecsOf(xs []probeWord) [][]float64 {
	o := make([][]float64, len(xs))
	for i := range xs {
		o[i] = xs[i].v
	}
	return o
}

func tightness(vs [][]float64) (meanCos, meanPair float64, center int) {
	c := centroid(vs)
	var sum float64
	best := -2.0
	for i, v := range vs {
		s := cosine(v, c)
		sum += s
		if s > best {
			best = s
			center = i
		}
	}
	meanCos = sum / float64(len(vs))
	var pairs, n float64
	for i := 0; i < len(vs); i++ {
		for j := i + 1; j < len(vs); j++ {
			pairs += cosine(vs[i], vs[j])
			n++
		}
	}
	return meanCos, pairs / n, center
}

func centroid(vs [][]float64) []float64 {
	o := make([]float64, len(vs[0]))
	for _, v := range vs {
		for i, x := range v {
			o[i] += x
		}
	}
	k := float64(len(vs))
	for i := range o {
		o[i] /= k
	}
	return o
}

func cosine(a, b []float64) float64 {
	var d, na, nb float64
	for i := range a {
		d += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return d / math.Sqrt(na*nb)
}

func nearest(target []float64, vs [][]float64, k int) []int {
	type pair struct {
		i int
		s float64
	}
	all := make([]pair, len(vs))
	for i, v := range vs {
		all[i] = pair{i, cosine(target, v)}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].s > all[j].s })
	if k > len(all) {
		k = len(all)
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = all[i].i
	}
	return out
}

func topVocab(target []float64, co []int32, V, C, k int, sess *eval.Session) []string {
	type pair struct {
		id int
		s  float64
	}
	best := make([]pair, 0, k)
	push := func(p pair) {
		if len(best) < k {
			best = append(best, p)
			return
		}
		min := 0
		for i := 1; i < len(best); i++ {
			if best[i].s < best[min].s {
				min = i
			}
		}
		if p.s > best[min].s {
			best[min] = p
		}
	}
	row := make([]float64, C)
	for id := 0; id < V; id++ {
		base := id * C
		var na float64
		for g := 0; g < C; g++ {
			row[g] = float64(co[base+g])
			na += row[g] * row[g]
		}
		if na == 0 {
			continue
		}
		w := strings.TrimSpace(sess.Decode([]int{id}))
		if !isProbeWord(w) {
			continue
		}
		push(pair{id, cosine(target, row)})
	}
	sort.Slice(best, func(i, j int) bool { return best[i].s > best[j].s })
	out := make([]string, len(best))
	for i, p := range best {
		out[i] = strings.TrimSpace(sess.Decode([]int{p.id}))
	}
	return out
}

func indexWord(xs []probeWord, names ...string) int {
	for i, x := range xs {
		for _, n := range names {
			if strings.EqualFold(x.text, n) {
				return i
			}
		}
	}
	return -1
}

func meanOf(x []float64) float64 {
	var s float64
	for _, v := range x {
		s += v
	}
	return s / float64(len(x))
}

func minOf(x []float64) float64 {
	m := x[0]
	for _, v := range x[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func maxOf(x []float64) float64 {
	m := x[0]
	for _, v := range x[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

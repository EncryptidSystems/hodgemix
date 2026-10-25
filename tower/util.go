package tower

import (
	"fmt"
	"math"
	"runtime"
	"sort"
	"sync"
)

func errf(f string, a ...any) error { return fmt.Errorf(f, a...) }

func parallel(n int, fn func(i int)) {
	if n <= 0 {
		return
	}
	if n == 1 {
		fn(0)
		return
	}
	w := runtime.NumCPU()
	if w > n {
		w = n
	}
	ch := make(chan int, n)
	for i := 0; i < n; i++ {
		ch <- i
	}
	close(ch)
	var wg sync.WaitGroup
	for k := 0; k < w; k++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				fn(i)
			}
		}()
	}
	wg.Wait()
}

func median(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	b := append([]float64(nil), a...)
	sort.Float64s(b)
	n := len(b)
	if n%2 == 1 {
		return b[n/2]
	}
	return 0.5 * (b[n/2-1] + b[n/2])
}

func finite64(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

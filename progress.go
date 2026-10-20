package main

// Live stderr status: log lines plus one in-place (\r) status line made of the overall bar and the inner bar.
// ASCII only, works in Windows PowerShell / cmd.

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

var T0 = time.Now()

type bar struct {
	label     string
	done, tot float64
	start     time.Time
	unit      string
}

var (
	pmu      sync.Mutex
	outer    *bar
	inner    *bar
	lastDraw time.Time
	lineLen  int
)

// Paper: no equation (helper: I/O, progress or utility)
func fmtDur(s float64) string {
	if s < 0 || s > 1e9 {
		return "--"
	}
	h := int(s) / 3600
	m := (int(s) % 3600) / 60
	if h > 0 {
		return fmt.Sprintf("%dh%02dm", h, m)
	}
	return fmt.Sprintf("%dm%02ds", m, int(s)%60)
}

// Paper: no equation (helper: I/O, progress or utility)
func (b *bar) str(w int) string {
	f := 0.0
	if b.tot > 0 {
		f = b.done / b.tot
	}
	if f > 1 {
		f = 1
	}
	n := int(f * float64(w))
	el := time.Since(b.start).Seconds()
	eta := -1.0
	if f > 0 {
		eta = el/f - el
	}
	cnt := fmt.Sprintf("%.0f/%.0f", b.done, b.tot)
	if b.unit == "B" {
		cnt = fmt.Sprintf("%.1f/%.1fMB", b.done/1e6, b.tot/1e6)
	}
	return fmt.Sprintf("%s [%s%s] %5.1f%% %s el %s eta %s", b.label, strings.Repeat("#", n), strings.Repeat(".", w-n), 100*f, cnt, fmtDur(el), fmtDur(eta))
}

// Paper: no equation (helper: I/O, progress or utility)
func drawLocked() {
	var s string
	if outer != nil {
		s = outer.str(20)
	}
	if inner != nil {
		if s != "" {
			s += " | "
		}
		s += inner.str(16)
	}
	pad := ""
	if len(s) < lineLen {
		pad = strings.Repeat(" ", lineLen-len(s))
	}
	fmt.Fprint(os.Stderr, "\r"+s+pad)
	lineLen = len(s)
	lastDraw = time.Now()
}

// Paper: no equation (helper: I/O, progress or utility)
func clearLocked() {
	if lineLen > 0 {
		fmt.Fprint(os.Stderr, "\r"+strings.Repeat(" ", lineLen)+"\r")
		lineLen = 0
	}
}

// logf prints a timestamped line above the status line.
// Paper: no equation (helper: I/O, progress or utility)
var quiet bool

func logf(format string, a ...any) {
	if quiet {
		return
	}
	pmu.Lock()
	defer pmu.Unlock()
	clearLocked()
	fmt.Fprintf(os.Stderr, "[%8.1fs] %s\n", time.Since(T0).Seconds(), fmt.Sprintf(format, a...))
	if outer != nil || inner != nil {
		drawLocked()
	}
}

// Paper: no equation (helper: I/O, progress or utility)
func setOuter(label string, tot float64) {
	pmu.Lock()
	outer = &bar{label: label, tot: tot, start: time.Now()}
	drawLocked()
	pmu.Unlock()
}

// Paper: no equation (helper: I/O, progress or utility)
func outerSet(done float64) {
	pmu.Lock()
	if outer != nil {
		outer.done = done
		drawLocked()
	}
	pmu.Unlock()
}

// Paper: no equation (helper: I/O, progress or utility)
func endOuter() { pmu.Lock(); outer = nil; clearLocked(); pmu.Unlock() }

// Paper: no equation (helper: I/O, progress or utility)
func setInner(label string, tot float64, unit string) {
	pmu.Lock()
	inner = &bar{label: label, tot: tot, start: time.Now(), unit: unit}
	drawLocked()
	pmu.Unlock()
}

// innerSet throttles redraws to 10/s.
// Paper: no equation (helper: I/O, progress or utility)
func innerSet(done, tot float64) {
	pmu.Lock()
	if inner != nil {
		inner.done = done
		if tot > 0 {
			inner.tot = tot
		}
		if time.Since(lastDraw) > 100*time.Millisecond || done >= inner.tot {
			drawLocked()
		}
	}
	pmu.Unlock()
}

// Paper: no equation (helper: I/O, progress or utility)
func endInner() {
	pmu.Lock()
	inner = nil
	clearLocked()
	if outer != nil {
		drawLocked()
	}
	pmu.Unlock()
}

package main

// Derived constants of paper_v2 (Appendix A "Parameter derivations"). Every value cites its source.
// None of these is a free tuning knob; flags may override them for experiments only.

import "math"

const (
	// Eq.(15): KTE phase coupling gain 2^-g_theta, g_theta = 4 (jev_kuramoto R-KTE-1/R-KTE-8, lccsac_asic_spec.tex:620)
	GThetaShift = 4
	// Eq.(15): weak/strong ratio = 2^-(g_omega-g_theta), g_omega = 8 (lccsac_asic_spec.tex:621)
	GOmegaShift = 8
	// Eq.(16): N_lost = 255 ticks (lccsac_asic_spec.tex, Tab. ktepar)
	NLost = 255
	// Eq.(7): 12-bit saturating key (DMT chip arrival width, saturates at 4095)
	KeyBits = 12
	// Eq.(9): 16 buckets per MRH level (DMT three-level 16-bucket ordering unit) -> top 4-bit digit
	BucketBits = 4
	// Eq.(6): radix-4 digits (radix-2^2 FFT), at most LMax digits per axis
	Radix = 4
	LMax  = 4
)

// Eq.(16): lock window W_lock = 1/16 turn (lccsac_asic_spec.tex:624)
var WLock = 2 * math.Pi / 16

// Eq.(20): refine tolerance 2^-8 = BF16 relative resolution (8-bit significand) and KTE frequency gain
var RefineTol = math.Ldexp(1, -8)

// dtft_lock App.A: PSR >= 10, uniqueness >= 1.4 (author's phase-correlation engine gate, dtft-chip E11)
const (
	PSRMin  = 10.0
	UniqMin = 1.4
)

// dtft_lock (L2): pair lock bound |d omega| <= 2 K W_lock = 2 * 2^-4 * 1/16 = 1/128 turn/tick
var PairBound = 2 * math.Ldexp(1, -GThetaShift) * (1.0 / 16)

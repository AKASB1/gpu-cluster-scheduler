// Package clock holds the time conventions of the project: virtual time is a
// time.Duration since the start of the trace that is always a whole number of
// milliseconds, and trace or report times are decimal seconds with at most
// three decimals.
package clock

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// Clock is an injected source of virtual time. Policies never read the wall
// clock; the simulator advances a Virtual clock.
type Clock interface {
	Now() time.Duration
}

// Virtual is a manually advanced clock.
type Virtual struct{ t time.Duration }

// Now returns the current virtual time.
func (v *Virtual) Now() time.Duration { return v.t }

// Set moves the clock. Time is monotonic: moving backwards panics, because it
// would be a simulator bug.
func (v *Virtual) Set(t time.Duration) {
	if t < v.t {
		panic(fmt.Sprintf("clock: time moved backwards from %v to %v", v.t, t))
	}
	v.t = t
}

// Tolerance absorbs float error when a duration is computed from a rate: a
// value within 1e-6 ms above a whole millisecond is not rounded up.
const toleranceMs = 1e-6

// ErrDecimal is returned for values that are not plain decimals.
var ErrDecimal = errors.New("not a non-negative decimal with at most three decimals")

// ParseSeconds parses a non-negative decimal number of seconds with at most
// three decimals ("12", "12.5", "0.001") into an exact millisecond duration.
// Signs, exponents, and more than three decimals are rejected.
func ParseSeconds(s string) (time.Duration, error) {
	if s == "" {
		return 0, ErrDecimal
	}
	intPart, frac, hasDot := strings.Cut(s, ".")
	if intPart == "" || (hasDot && (frac == "" || len(frac) > 3)) {
		return 0, ErrDecimal
	}
	for _, r := range intPart + frac {
		if r < '0' || r > '9' {
			return 0, ErrDecimal
		}
	}
	whole, err := strconv.ParseInt(intPart, 10, 64)
	if err != nil || whole > math.MaxInt64/int64(time.Second) {
		return 0, ErrDecimal
	}
	ms := whole * 1000
	if hasDot {
		f, _ := strconv.Atoi(frac + strings.Repeat("0", 3-len(frac)))
		ms += int64(f)
	}
	return time.Duration(ms) * time.Millisecond, nil
}

// FormatSeconds writes a whole-millisecond duration as the shortest decimal
// number of seconds ("12", "12.5", "0.001"). Negative values keep their sign.
func FormatSeconds(d time.Duration) string {
	ms := d.Milliseconds()
	sign := ""
	if ms < 0 {
		sign, ms = "-", -ms
	}
	whole, frac := ms/1000, ms%1000
	if frac == 0 {
		return sign + strconv.FormatInt(whole, 10)
	}
	f := strings.TrimRight(fmt.Sprintf("%03d", frac), "0")
	return sign + strconv.FormatInt(whole, 10) + "." + f
}

// Sec converts a duration to float seconds (exact division of whole ms).
func Sec(d time.Duration) float64 { return float64(d.Milliseconds()) / 1000 }

// CeilMs rounds a non-negative float number of milliseconds up to a whole
// millisecond duration, ignoring float error below the tolerance.
func CeilMs(ms float64) time.Duration {
	if ms <= 0 {
		return 0
	}
	return time.Duration(math.Ceil(ms-toleranceMs)) * time.Millisecond
}

// FromSecondsCeil converts float seconds to a whole-ms duration, rounding up.
func FromSecondsCeil(s float64) time.Duration { return CeilMs(s * 1000) }

// WorkToDuration is the time a job needs to do work reference-seconds of work
// at rate work-seconds per second: ceil(work*1000/rate - 1e-6) ms. The same
// formula is used by the simulator for true run times and by policies for
// estimates, and by the Python harness, so that both sides agree bit for bit.
func WorkToDuration(work, rate float64) time.Duration {
	if work <= 0 {
		return 0
	}
	return CeilMs(work * 1000 / rate)
}

// RoundMs rounds float seconds to the nearest whole millisecond (used by
// generators before writing a trace).
func RoundMs(s float64) time.Duration {
	return time.Duration(math.Round(s*1000)) * time.Millisecond
}

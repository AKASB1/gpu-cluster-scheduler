package clock

import (
	"testing"
	"time"
)

func TestParseSeconds(t *testing.T) {
	ok := map[string]time.Duration{
		"0": 0, "12": 12 * time.Second, "12.5": 12500 * time.Millisecond,
		"0.001": time.Millisecond, "3.140": 3140 * time.Millisecond, "007.25": 7250 * time.Millisecond,
	}
	for in, want := range ok {
		got, err := ParseSeconds(in)
		if err != nil || got != want {
			t.Errorf("ParseSeconds(%q) = %v, %v; want %v", in, got, err, want)
		}
	}
	for _, in := range []string{"", "-1", "+1", "1.", ".5", "1.0001", "1e3", "NaN", "inf", "1,5", " 1", "1.2.3", "99999999999999999999"} {
		if _, err := ParseSeconds(in); err == nil {
			t.Errorf("ParseSeconds(%q) accepted", in)
		}
	}
}

func TestFormatSecondsRoundTrip(t *testing.T) {
	cases := map[time.Duration]string{0: "0", time.Millisecond: "0.001", 1500 * time.Millisecond: "1.5",
		12 * time.Second: "12", 3140 * time.Millisecond: "3.14", -2500 * time.Millisecond: "-2.5"}
	for d, want := range cases {
		if got := FormatSeconds(d); got != want {
			t.Errorf("FormatSeconds(%v) = %q, want %q", d, got, want)
		}
		if d >= 0 {
			back, err := ParseSeconds(want)
			if err != nil || back != d {
				t.Errorf("round trip %q -> %v, %v", want, back, err)
			}
		}
	}
}

func TestWorkToDurationRoundsUpWithTolerance(t *testing.T) {
	// 1000 s of work at rate 1/1.15: exactly 1150 s in real arithmetic; float
	// error must not add a millisecond.
	if got := WorkToDuration(1000, 1/1.15); got != 1150*time.Second {
		t.Fatalf("got %v", got)
	}
	if got := WorkToDuration(1.0005, 1); got != 1001*time.Millisecond {
		t.Fatalf("got %v want 1.001s", got)
	}
	if got := WorkToDuration(0, 0.5); got != 0 {
		t.Fatalf("got %v", got)
	}
	if got := WorkToDuration(10, 0.4); got != 25*time.Second {
		t.Fatalf("got %v", got)
	}
}

func TestVirtualClockMonotonic(t *testing.T) {
	var c Virtual
	c.Set(time.Second)
	if c.Now() != time.Second {
		t.Fatal("Set")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("moving backwards must panic")
		}
	}()
	c.Set(0)
}

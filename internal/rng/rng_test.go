package rng

import "testing"

func TestFNV1a64KnownValues(t *testing.T) {
	// Reference values of 64-bit FNV-1a.
	if got := FNV1a64(""); got != 0xcbf29ce484222325 {
		t.Fatalf("empty: %x", got)
	}
	if got := FNV1a64("a"); got != 0xaf63dc4c8601ec8c {
		t.Fatalf("a: %x", got)
	}
}

func TestSplitMix64KnownValue(t *testing.T) {
	// First output of SplitMix64 seeded with 0.
	if got := SplitMix64(0); got != 0xe220a8397b1dcdaf {
		t.Fatalf("got %x", got)
	}
}

func TestStreamsAreReproducibleAndIndependent(t *testing.T) {
	a1, a2, b := New(7, "arrivals"), New(7, "arrivals"), New(7, "sizes")
	c := New(8, "arrivals")
	same, diffName, diffSeed := true, false, false
	for i := 0; i < 100; i++ {
		x, y, z, w := a1.Uint64(), a2.Uint64(), b.Uint64(), c.Uint64()
		same = same && x == y
		diffName = diffName || x != z
		diffSeed = diffSeed || x != w
	}
	if !same || !diffName || !diffSeed {
		t.Fatalf("same=%v diffName=%v diffSeed=%v", same, diffName, diffSeed)
	}
}

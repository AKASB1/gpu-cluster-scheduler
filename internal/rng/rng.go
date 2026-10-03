// Package rng derives one independent random stream per component from the
// run seed and the component name (contracts §1.2):
//
//	x  = seed XOR fnv1a64(name)
//	s1 = splitmix64(x); s2 = splitmix64(s1)
//	stream = math/rand/v2 PCG(s1, s2)
package rng

import "math/rand/v2"

const (
	fnvOffset = 0xcbf29ce484222325
	fnvPrime  = 0x100000001b3
)

// FNV1a64 is the 64-bit FNV-1a hash of the UTF-8 bytes of s.
func FNV1a64(s string) uint64 {
	h := uint64(fnvOffset)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= fnvPrime
	}
	return h
}

// SplitMix64 is one step of the SplitMix64 output function.
func SplitMix64(z uint64) uint64 {
	z += 0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// Seeds returns the two PCG seed words for (seed, name).
func Seeds(seed uint64, name string) (uint64, uint64) {
	s1 := SplitMix64(seed ^ FNV1a64(name))
	return s1, SplitMix64(s1)
}

// New returns the stream of component name under the run seed.
func New(seed uint64, name string) *rand.Rand {
	s1, s2 := Seeds(seed, name)
	return rand.New(rand.NewPCG(s1, s2))
}

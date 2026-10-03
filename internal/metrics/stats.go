package metrics

import "math"

// CI is a mean with a two-sided 95% Student-t confidence interval.
type CI struct {
	N          int
	Mean, Half float64 // interval is Mean +- Half (NaN when N < 2)
}

// MeanCI computes the mean and the t-based 95% interval of x, ignoring NaN.
func MeanCI(x []float64) CI {
	var v []float64
	for _, a := range x {
		if !math.IsNaN(a) {
			v = append(v, a)
		}
	}
	c := CI{N: len(v), Mean: Mean(v), Half: math.NaN()}
	if len(v) < 2 {
		return c
	}
	ss := 0.0
	for _, a := range v {
		ss += (a - c.Mean) * (a - c.Mean)
	}
	sd := math.Sqrt(ss / float64(len(v)-1))
	c.Half = TQuantile(0.975, float64(len(v)-1)) * sd / math.Sqrt(float64(len(v)))
	return c
}

// TQuantile is the p-quantile of Student's t with df degrees of freedom,
// by bisection on the CDF.
func TQuantile(p, df float64) float64 {
	lo, hi := -1e3, 1e3
	for i := 0; i < 200; i++ {
		mid := (lo + hi) / 2
		if TCDF(mid, df) < p {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// TCDF is the CDF of Student's t with df degrees of freedom.
func TCDF(t, df float64) float64 {
	x := df / (df + t*t)
	ib := RegIncBeta(df/2, 0.5, x)
	if t >= 0 {
		return 1 - ib/2
	}
	return ib / 2
}

// RegIncBeta is the regularized incomplete beta function I_x(a, b)
// (continued fraction, Lentz's method).
func RegIncBeta(a, b, x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	lga, _ := math.Lgamma(a)
	lgb, _ := math.Lgamma(b)
	lgab, _ := math.Lgamma(a + b)
	front := math.Exp(lgab - lga - lgb + a*math.Log(x) + b*math.Log(1-x))
	if x > (a+1)/(a+b+2) {
		return 1 - RegIncBeta(b, a, 1-x)
	}
	const tiny = 1e-300
	f, c, d := 1.0, 1.0, 0.0
	for i := 0; i <= 400; i++ {
		m := float64(i / 2)
		var num float64
		switch {
		case i == 0:
			num = 1
		case i%2 == 0:
			num = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
		default:
			num = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
		}
		d = 1 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		d = 1 / d
		c = 1 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		cd := c * d
		f *= cd
		if math.Abs(1-cd) < 1e-14 {
			break
		}
	}
	return front * (f - 1) / a
}

// WTL counts wins, ties, and losses of a against b per paired seed. lower
// says whether lower values are better. A tie is a relative difference of at
// most 1e-9 (identical schedules give exact ties).
func WTL(a, b []float64, lower bool) (win, tie, loss int) {
	for i := range a {
		if math.IsNaN(a[i]) || math.IsNaN(b[i]) {
			continue
		}
		d := a[i] - b[i]
		if math.Abs(d) <= 1e-9*math.Max(math.Max(math.Abs(a[i]), math.Abs(b[i])), 1e-12) {
			tie++
			continue
		}
		if (d < 0) == lower {
			win++
		} else {
			loss++
		}
	}
	return
}

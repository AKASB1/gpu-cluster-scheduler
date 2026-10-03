package policy

import (
	"math"
	"slices"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// ratioModel is the empirical distribution of runtime / estimate in the
// completion history, per user and over all users (risk backfilling). The
// slices are kept sorted, so a quantile is one index lookup.
type ratioModel struct {
	q          float64
	minHistory int
	user       map[string][]float64
	all        []float64
	seen       int
}

func insertSorted(s []float64, x float64) []float64 {
	i, _ := slices.BinarySearch(s, x)
	return slices.Insert(s, i, x)
}

func (m *ratioModel) update(h []api.CompletedJob) {
	for ; m.seen < len(h); m.seen++ {
		c := h[m.seen]
		r := clock.Sec(c.Runtime) / clock.Sec(c.Estimate)
		m.user[c.User] = insertSorted(m.user[c.User], r)
		m.all = insertSorted(m.all, r)
	}
}

// ratio is the q-quantile (nearest rank) of the user's ratios when the user
// has at least minHistory completed jobs, else of all users' ratios, else 1.
func (m *ratioModel) ratio(user string) float64 {
	s := m.user[user]
	if len(s) < m.minHistory {
		s = m.all
	}
	if len(s) == 0 {
		return 1
	}
	k := int(math.Ceil(m.q*float64(len(s)))) - 1
	return s[min(max(k, 0), len(s)-1)]
}

// work is the q-quantile prediction of a job's run time (reference seconds).
func (m *ratioModel) work(user string, estimate float64) float64 {
	return estimate * m.ratio(user)
}

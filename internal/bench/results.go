// Package bench runs scenarios (generate traces, run policies in parallel,
// compute metrics), tunes parameterized policies, and aggregates results.
package bench

import (
	"bufio"
	"cmp"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/metrics"
)

// Row is the result of one run: deterministic fields and wall-clock fields.
type Row struct {
	Scenario string
	Policy   string
	Seed     uint64
	Fields   []metrics.Field // deterministic
	Wall     []metrics.Field // wall_ columns, written to a separate file
	Extra    []metrics.Field // scenario-specific deterministic fields (e.g. J)
}

// Get returns a deterministic field (NaN if absent).
func (r *Row) Get(name string) float64 {
	for _, f := range r.Extra {
		if f.Name == name {
			return f.Value
		}
	}
	for _, f := range r.Fields {
		if f.Name == name {
			return f.Value
		}
	}
	return math.NaN()
}

func sortRows(rows []Row) {
	slices.SortStableFunc(rows, func(a, b Row) int {
		if c := cmp.Compare(a.Scenario, b.Scenario); c != 0 {
			return c
		}
		if c := cmp.Compare(a.Policy, b.Policy); c != 0 {
			return c
		}
		return cmp.Compare(a.Seed, b.Seed)
	})
}

// columns returns the union of field names in order of first appearance.
func columns(rows []Row, pick func(*Row) []metrics.Field) []string {
	seen := map[string]bool{}
	var cols []string
	for i := range rows {
		for _, f := range pick(&rows[i]) {
			if !seen[f.Name] {
				seen[f.Name] = true
				cols = append(cols, f.Name)
			}
		}
	}
	return cols
}

// WriteRows writes runs.csv (deterministic columns) and wall.csv (wall_
// columns) into dir. Rows are sorted by (scenario, policy, seed).
func WriteRows(dir string, rows []Row) error {
	sortRows(rows)
	det := func(r *Row) []metrics.Field { return append(slices.Clone(r.Extra), r.Fields...) }
	if err := writeTable(filepath.Join(dir, "runs.csv"), rows, columns(rows, det), det); err != nil {
		return err
	}
	wall := func(r *Row) []metrics.Field { return r.Wall }
	return writeTable(filepath.Join(dir, "wall.csv"), rows, columns(rows, wall), wall)
}

func writeTable(path string, rows []Row, cols []string, pick func(*Row) []metrics.Field) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	bw := bufio.NewWriter(f)
	w := csv.NewWriter(bw)
	_ = w.Write(append([]string{"scenario", "policy", "seed"}, cols...))
	rec := make([]string, 3+len(cols))
	for i := range rows {
		r := &rows[i]
		vals := map[string]float64{}
		for _, fl := range pick(r) {
			vals[fl.Name] = fl.Value
		}
		rec[0], rec[1], rec[2] = r.Scenario, r.Policy, strconv.FormatUint(r.Seed, 10)
		for k, c := range cols {
			v, ok := vals[c]
			if !ok {
				v = math.NaN()
			}
			rec[3+k] = metrics.Format(v)
		}
		_ = w.Write(rec)
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return err
	}
	return bw.Flush()
}

// ReadRows reads a runs.csv written by WriteRows (deterministic fields only).
func ReadRows(path string) ([]Row, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	cr := csv.NewReader(f)
	head, err := cr.Read()
	if err != nil {
		return nil, err
	}
	var rows []Row
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		seed, _ := strconv.ParseUint(rec[2], 10, 64)
		r := Row{Scenario: rec[0], Policy: rec[1], Seed: seed}
		for k := 3; k < len(rec); k++ {
			v := math.NaN()
			if rec[k] != "" {
				v, _ = strconv.ParseFloat(rec[k], 64)
			}
			r.Fields = append(r.Fields, metrics.Field{Name: head[k], Value: v})
		}
		rows = append(rows, r)
	}
	return rows, nil
}

// HigherIsBetter lists the metrics where larger values are better; every
// other metric is lower-is-better for win/tie/loss counts.
var HigherIsBetter = map[string]bool{"utilization": true, "goodput_ratio": true, "jain_bsld": true,
	"slo_attainment": true}

// Aggregate is one (scenario, policy, metric) summary.
type Aggregate struct {
	Scenario, Policy, Metric string
	CI                       metrics.CI
}

// Paired is the paired comparison of a policy against the baseline.
type Paired struct {
	Scenario, Policy, Baseline, Metric string
	Diff                               metrics.CI
	Win, Tie, Loss                     int
}

// Summarize computes per-(scenario, policy, metric) means with 95% t
// intervals, and paired differences against baseline on common seeds.
func Summarize(rows []Row, baseline string, only []string) ([]Aggregate, []Paired) {
	sortRows(rows)
	type key struct{ s, p string }
	groups := map[key][]*Row{}
	var keys []key
	for i := range rows {
		k := key{rows[i].Scenario, rows[i].Policy}
		if groups[k] == nil {
			keys = append(keys, k)
		}
		groups[k] = append(groups[k], &rows[i])
	}
	cols := columns(rows, func(r *Row) []metrics.Field { return append(slices.Clone(r.Extra), r.Fields...) })
	if only != nil {
		cols = slices.DeleteFunc(slices.Clone(cols), func(c string) bool { return !slices.Contains(only, c) })
	}
	var aggs []Aggregate
	var pairs []Paired
	for _, k := range keys {
		g := groups[k]
		base := map[uint64]*Row{}
		for _, r := range groups[key{k.s, baseline}] {
			base[r.Seed] = r
		}
		for _, c := range cols {
			x := make([]float64, len(g))
			for i, r := range g {
				x[i] = r.Get(c)
			}
			ci := metrics.MeanCI(x)
			if ci.N == 0 {
				continue // the metric is not defined in this group
			}
			aggs = append(aggs, Aggregate{k.s, k.p, c, ci})
			if k.p == baseline || len(base) == 0 {
				continue
			}
			var a, b, d []float64
			for _, r := range g {
				if br, ok := base[r.Seed]; ok {
					a, b = append(a, r.Get(c)), append(b, br.Get(c))
					d = append(d, r.Get(c)-br.Get(c))
				}
			}
			w, t, l := metrics.WTL(a, b, !HigherIsBetter[c])
			pairs = append(pairs, Paired{k.s, k.p, baseline, c, metrics.MeanCI(d), w, t, l})
		}
	}
	return aggs, pairs
}

// WriteSummaries writes aggregate.csv and paired.csv.
func WriteSummaries(dir string, aggs []Aggregate, pairs []Paired) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f := metrics.Format
	lo := func(c metrics.CI) string { return f(c.Mean - c.Half) }
	hi := func(c metrics.CI) string { return f(c.Mean + c.Half) }
	var a [][]string
	a = append(a, []string{"scenario", "policy", "metric", "n", "mean", "ci95_low", "ci95_high"})
	for _, g := range aggs {
		a = append(a, []string{g.Scenario, g.Policy, g.Metric, strconv.Itoa(g.CI.N), f(g.CI.Mean), lo(g.CI), hi(g.CI)})
	}
	if err := writeCSV(filepath.Join(dir, "aggregate.csv"), a); err != nil {
		return err
	}
	var p [][]string
	p = append(p, []string{"scenario", "policy", "baseline", "metric", "n", "diff_mean", "diff_ci95_low", "diff_ci95_high", "win", "tie", "loss"})
	for _, g := range pairs {
		p = append(p, []string{g.Scenario, g.Policy, g.Baseline, g.Metric, strconv.Itoa(g.Diff.N), f(g.Diff.Mean), lo(g.Diff), hi(g.Diff),
			strconv.Itoa(g.Win), strconv.Itoa(g.Tie), strconv.Itoa(g.Loss)})
	}
	return writeCSV(filepath.Join(dir, "paired.csv"), p)
}

func writeCSV(path string, recs [][]string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	fh, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fh.Close()
	w := csv.NewWriter(fh)
	if err := w.WriteAll(recs); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

// WriteCSV writes records to a CSV file (LF line ends), creating its folder.
func WriteCSV(path string, recs [][]string) error { return writeCSV(path, recs) }

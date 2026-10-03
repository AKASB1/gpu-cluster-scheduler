package trace

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const row1 = "j1,0,t0,u1,4,2,1,,any,16,64,100.5,120,1,600,900"
const row2 = "j2,1.25,t1,,0,8,2,a100,rack,96,512.5,3600,3000,0,0,"

func doc(rows ...string) string { return Header + "\n" + strings.Join(rows, "\n") + "\n" }

func TestLoadValid(t *testing.T) {
	jobs, err := Parse([]byte(doc(row1, row2)))
	if err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 2 {
		t.Fatalf("got %d jobs", len(jobs))
	}
	a, b := jobs[0], jobs[1]
	if a.ID != "j1" || a.GPUs != 2 || a.Runtime != 100500*time.Millisecond || !a.Preemptible || !a.HasMaxWait || a.MaxWait != 900*time.Second || a.CheckpointInterval != 600*time.Second {
		t.Fatalf("row 1 parsed wrongly: %+v", a)
	}
	if b.User != "t1" || b.GPUClass != "a100" || b.Topology != "rack" || b.MemGB != 512.5 || b.HasMaxWait || b.Submit != 1250*time.Millisecond {
		t.Fatalf("row 2 parsed wrongly: %+v", b)
	}
}

func TestLoadCRLFAndEmptyTrace(t *testing.T) {
	jobs, err := Parse([]byte(strings.ReplaceAll(doc(row1, row2), "\n", "\r\n")))
	if err != nil || len(jobs) != 2 {
		t.Fatalf("CRLF: %v %d", err, len(jobs))
	}
	jobs, err = Parse([]byte(Header + "\n"))
	if err != nil || len(jobs) != 0 {
		t.Fatalf("header-only trace must be valid and empty: %v", err)
	}
	if _, err := Parse(nil); err == nil {
		t.Fatal("zero-byte file must be invalid")
	}
}

func TestLoadRejectsWithLineNumbers(t *testing.T) {
	cases := []struct {
		name, in string
		line     int
	}{
		{"wrong header", strings.Replace(doc(row1), "job_id", "id", 1), 1},
		{"missing header", row1 + "\n", 1},
		{"field count", doc(row1, "j2,1,t0"), 3},
		{"unsorted", doc(row2, strings.Replace(row1, "j1,0,", "j1,1.2,", 1)), 3},
		{"duplicate", doc(row1, strings.Replace(row2, "j2,", "j1,", 1)), 3},
		{"bad priority", doc(strings.Replace(row1, ",4,2,1,", ",10,2,1,", 1)), 2},
		{"bad gpus", doc(strings.Replace(row1, ",4,2,1,", ",4,0,1,", 1)), 2},
		{"bad topology", doc(strings.Replace(row1, ",any,", ",pod,", 1)), 2},
		{"zero runtime", doc(strings.Replace(row1, ",100.5,", ",0,", 1)), 2},
		{"four decimals", doc(strings.Replace(row1, ",100.5,", ",100.5001,", 1)), 2},
		{"negative submit", doc(strings.Replace(row1, "j1,0,", "j1,-1,", 1)), 2},
		{"empty tenant", doc(strings.Replace(row1, ",t0,", ",,", 1)), 2},
		{"bad preemptible", doc(strings.Replace(row1, ",1,600,", ",2,600,", 1)), 2},
		{"nan mem", doc(strings.Replace(row1, ",64,", ",NaN,", 1)), 2},
		{"empty id", doc(strings.Replace(row1, "j1,", ",", 1)), 2},
	}
	for _, c := range cases {
		_, err := Parse([]byte(c.in))
		var le *LineError
		if !errors.As(err, &le) {
			t.Errorf("%s: want LineError, got %v", c.name, err)
			continue
		}
		if le.Line != c.line {
			t.Errorf("%s: line %d, want %d (%v)", c.name, le.Line, c.line, err)
		}
	}
}

func TestWriteParseRoundTrip(t *testing.T) {
	jobs, err := Parse([]byte(doc(row1, row2)))
	if err != nil {
		t.Fatal(err)
	}
	out := Encode(jobs)
	back, err := Parse(out)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(Encode(back), out) {
		t.Fatal("encode is not stable")
	}
	// user defaults to the tenant and is written explicitly
	if !strings.Contains(string(out), "j2,1.25,t1,t1,") {
		t.Fatalf("unexpected encoding:\n%s", out)
	}
}

func TestManifestVerification(t *testing.T) {
	dir := t.TempDir()
	jobs, _ := Parse([]byte(doc(row1, row2)))
	p := filepath.Join(dir, "tr.csv")
	m, err := SaveWithManifest(p, jobs, GeneratorInfo{Name: "test", Version: 1, Params: []byte(`{}`)}, 9)
	if err != nil {
		t.Fatal(err)
	}
	if m.Jobs != 2 || m.DurationS != 1.25 || m.Seed != 9 {
		t.Fatalf("manifest %+v", m)
	}
	if _, _, err := LoadVerified(p); err != nil {
		t.Fatal(err)
	}
	// Tamper with the CSV: the hash check must fail.
	data, _ := os.ReadFile(p)
	if err := os.WriteFile(p, bytes.Replace(data, []byte("100.5"), []byte("100.6"), 1), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := LoadVerified(p); err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("tampered trace accepted: %v", err)
	}
}

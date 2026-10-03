// Package trace reads, writes, validates, and generates job traces of schema
// v1 (docs/contracts.md §2) and their manifests.
package trace

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
	"github.com/AKASB1/gpu-cluster-scheduler/internal/clock"
)

// SchemaVersion is the trace schema version this package reads and writes.
const SchemaVersion = 1

// Header is the exact header line of schema v1.
const Header = "job_id,submit_s,tenant,user,priority,gpus,workers,gpu_class,topology,cpus,mem_gb,runtime_s,estimate_s,preemptible,checkpoint_interval_s,max_wait_s"

var columns = strings.Split(Header, ",")

// LineError is a validation error that names the 1-based line number.
type LineError struct {
	Line int
	Msg  string
}

func (e *LineError) Error() string { return fmt.Sprintf("line %d: %s", e.Line, e.Msg) }

// Load parses and validates a schema-v1 trace.
func Load(r io.Reader) ([]api.Job, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return Parse(data)
}

// LoadFile parses and validates a trace file (without checking a manifest).
func LoadFile(path string) ([]api.Job, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Parse(data)
}

// Parse validates the bytes of a schema-v1 trace.
func Parse(data []byte) ([]api.Job, error) {
	if len(data) == 0 {
		return nil, &LineError{1, "empty file (a valid empty trace still has the header line)"}
	}
	cr := csv.NewReader(bytes.NewReader(data))
	cr.FieldsPerRecord = -1
	cr.ReuseRecord = true
	head, err := cr.Read()
	if err != nil {
		return nil, &LineError{1, "unreadable header: " + err.Error()}
	}
	if strings.Join(head, ",") != Header {
		return nil, &LineError{1, "header must be exactly: " + Header}
	}
	var jobs []api.Job
	seen := map[string]bool{}
	var prev time.Duration
	for {
		rec, err := cr.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			var pe *csv.ParseError
			if errors.As(err, &pe) {
				return nil, &LineError{pe.Line, pe.Err.Error()}
			}
			return nil, err
		}
		line, _ := cr.FieldPos(0)
		if len(rec) != len(columns) {
			return nil, &LineError{line, fmt.Sprintf("expected %d fields, got %d", len(columns), len(rec))}
		}
		j, msg := parseRow(rec)
		if msg != "" {
			return nil, &LineError{line, msg}
		}
		if seen[j.ID] {
			return nil, &LineError{line, "duplicate job_id " + j.ID}
		}
		seen[j.ID] = true
		if j.Submit < prev {
			return nil, &LineError{line, "submit_s is smaller than the previous row's"}
		}
		prev = j.Submit
		jobs = append(jobs, j)
	}
	return jobs, nil
}

func parseRow(f []string) (api.Job, string) {
	var j api.Job
	var err error
	bad := func(col, why string) (api.Job, string) { return j, col + ": " + why }
	if j.ID = f[0]; j.ID == "" || strings.Contains(j.ID, ",") {
		return bad("job_id", "must be non-empty and contain no comma")
	}
	if j.Submit, err = clock.ParseSeconds(f[1]); err != nil {
		return bad("submit_s", err.Error())
	}
	if j.Tenant = f[2]; j.Tenant == "" {
		return bad("tenant", "required")
	}
	if j.User = f[3]; j.User == "" {
		j.User = j.Tenant
	}
	if j.Priority, err = strconv.Atoi(f[4]); err != nil || j.Priority < 0 || j.Priority > 9 {
		return bad("priority", "integer 0 to 9 required")
	}
	if j.GPUs, err = strconv.Atoi(f[5]); err != nil || j.GPUs < 1 {
		return bad("gpus", "integer >= 1 required")
	}
	if j.Workers, err = strconv.Atoi(f[6]); err != nil || j.Workers < 1 {
		return bad("workers", "integer >= 1 required")
	}
	j.GPUClass = f[7]
	switch api.Topology(f[8]) {
	case api.TopoAny, api.TopoRack, api.TopoNode:
		j.Topology = api.Topology(f[8])
	default:
		return bad("topology", "must be any, rack, or node")
	}
	if j.CPUs, err = strconv.Atoi(f[9]); err != nil || j.CPUs < 0 {
		return bad("cpus", "integer >= 0 required")
	}
	if j.MemGB, err = strconv.ParseFloat(f[10], 64); err != nil || math.IsNaN(j.MemGB) || math.IsInf(j.MemGB, 0) || j.MemGB < 0 {
		return bad("mem_gb", "finite decimal >= 0 required")
	}
	if j.Runtime, err = clock.ParseSeconds(f[11]); err != nil || j.Runtime <= 0 {
		return bad("runtime_s", "decimal > 0 with at most three decimals required")
	}
	if j.Estimate, err = clock.ParseSeconds(f[12]); err != nil || j.Estimate <= 0 {
		return bad("estimate_s", "decimal > 0 with at most three decimals required")
	}
	switch f[13] {
	case "0":
	case "1":
		j.Preemptible = true
	default:
		return bad("preemptible", "must be 0 or 1")
	}
	if j.CheckpointInterval, err = clock.ParseSeconds(f[14]); err != nil {
		return bad("checkpoint_interval_s", err.Error())
	}
	if f[15] != "" {
		if j.MaxWait, err = clock.ParseSeconds(f[15]); err != nil {
			return bad("max_wait_s", err.Error())
		}
		j.HasMaxWait = true
	}
	return j, ""
}

// Write writes jobs as a schema-v1 CSV with LF line endings.
func Write(w io.Writer, jobs []api.Job) error {
	bw := bufio.NewWriter(w)
	cw := csv.NewWriter(bw)
	if err := cw.Write(columns); err != nil {
		return err
	}
	rec := make([]string, len(columns))
	for i := range jobs {
		j := &jobs[i]
		rec[0] = j.ID
		rec[1] = clock.FormatSeconds(j.Submit)
		rec[2] = j.Tenant
		rec[3] = j.User
		rec[4] = strconv.Itoa(j.Priority)
		rec[5] = strconv.Itoa(j.GPUs)
		rec[6] = strconv.Itoa(j.Workers)
		rec[7] = j.GPUClass
		rec[8] = string(j.Topology)
		rec[9] = strconv.Itoa(j.CPUs)
		rec[10] = strconv.FormatFloat(j.MemGB, 'f', -1, 64)
		rec[11] = clock.FormatSeconds(j.Runtime)
		rec[12] = clock.FormatSeconds(j.Estimate)
		rec[13] = "0"
		if j.Preemptible {
			rec[13] = "1"
		}
		rec[14] = clock.FormatSeconds(j.CheckpointInterval)
		rec[15] = ""
		if j.HasMaxWait {
			rec[15] = clock.FormatSeconds(j.MaxWait)
		}
		if err := cw.Write(rec); err != nil {
			return err
		}
	}
	cw.Flush()
	if err := cw.Error(); err != nil {
		return err
	}
	return bw.Flush()
}

// Encode returns the CSV bytes of jobs.
func Encode(jobs []api.Job) []byte {
	var b bytes.Buffer
	_ = Write(&b, jobs) // writing to a bytes.Buffer cannot fail
	return b.Bytes()
}

// GeneratorInfo names the program that produced a trace.
type GeneratorInfo struct {
	Name    string          `json:"name"`
	Version int             `json:"version"`
	Params  json.RawMessage `json:"params"`
}

// Manifest describes a trace file (<name>.manifest.json beside <name>.csv).
type Manifest struct {
	SchemaVersion int           `json:"schema_version"`
	Generator     GeneratorInfo `json:"generator"`
	Seed          uint64        `json:"seed"`
	Jobs          int           `json:"jobs"`
	DurationS     float64       `json:"duration_s"`
	ContentSHA256 string        `json:"content_sha256"`
}

// ManifestPath returns the manifest path for a trace CSV path.
func ManifestPath(csvPath string) string {
	return strings.TrimSuffix(csvPath, filepath.Ext(csvPath)) + ".manifest.json"
}

// SHA256Hex is the hex SHA-256 of b.
func SHA256Hex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// NewManifest builds the manifest of encoded trace bytes.
func NewManifest(data []byte, jobs []api.Job, gen GeneratorInfo, seed uint64) Manifest {
	var dur time.Duration
	if n := len(jobs); n > 0 {
		dur = jobs[n-1].Submit
	}
	return Manifest{SchemaVersion: SchemaVersion, Generator: gen, Seed: seed, Jobs: len(jobs),
		DurationS: clock.Sec(dur), ContentSHA256: SHA256Hex(data)}
}

// SaveWithManifest writes <csvPath> and its manifest.
func SaveWithManifest(csvPath string, jobs []api.Job, gen GeneratorInfo, seed uint64) (Manifest, error) {
	data := Encode(jobs)
	m := NewManifest(data, jobs, gen, seed)
	if err := os.MkdirAll(filepath.Dir(csvPath), 0o755); err != nil {
		return m, err
	}
	if err := os.WriteFile(csvPath, data, 0o644); err != nil {
		return m, err
	}
	mb, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return m, err
	}
	return m, os.WriteFile(ManifestPath(csvPath), append(mb, '\n'), 0o644)
}

// LoadVerified loads a trace and checks its manifest's schema version, hash,
// and job count before use.
func LoadVerified(csvPath string) ([]api.Job, Manifest, error) {
	var m Manifest
	data, err := os.ReadFile(csvPath)
	if err != nil {
		return nil, m, err
	}
	mb, err := os.ReadFile(ManifestPath(csvPath))
	if err != nil {
		return nil, m, fmt.Errorf("manifest: %w", err)
	}
	if err := json.Unmarshal(mb, &m); err != nil {
		return nil, m, fmt.Errorf("manifest: %w", err)
	}
	if m.SchemaVersion != SchemaVersion {
		return nil, m, fmt.Errorf("manifest: unknown schema_version %d", m.SchemaVersion)
	}
	if got := SHA256Hex(data); got != m.ContentSHA256 {
		return nil, m, fmt.Errorf("manifest: content_sha256 mismatch (file %s, manifest %s)", got, m.ContentSHA256)
	}
	jobs, err := Parse(data)
	if err != nil {
		return nil, m, err
	}
	if len(jobs) != m.Jobs {
		return nil, m, fmt.Errorf("manifest: %d jobs declared, %d in file", m.Jobs, len(jobs))
	}
	return jobs, m, nil
}

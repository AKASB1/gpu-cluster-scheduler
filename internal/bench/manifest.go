package bench

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/extpolicy"
)

// Hardware is the hardware class recorded next to every result.
type Hardware struct {
	CPU         string  `json:"cpu"`
	LogicalCPUs int     `json:"logical_cpus"`
	RAMGB       float64 `json:"ram_gb"`
	OS          string  `json:"os"`
}

// Versions of the toolchain.
type Versions struct {
	Go     string `json:"go"`
	Python string `json:"python"`
	SciPy  string `json:"scipy"`
	HiGHS  string `json:"highs"`
}

// Manifest is written beside the results of a benchmark command.
type Manifest struct {
	Experiment   string   `json:"experiment"`
	Mode         string   `json:"mode"`
	Command      string   `json:"command"`
	Commit       string   `json:"commit"`
	Dirty        bool     `json:"dirty"`
	ConfigSHA256 string   `json:"config_sha256"`
	ConfigFiles  []string `json:"config_files"`
	TunedSHA256  string   `json:"tuned_sha256,omitempty"`
	TunedCommit  string   `json:"tuned_commit,omitempty"`
	TuningSeeds  []uint64 `json:"tuning_seeds"`
	EvalSeeds    []uint64 `json:"eval_seeds"`
	Workers      int      `json:"workers"`
	Versions     Versions `json:"versions"`
	Hardware     Hardware `json:"hardware"`
	LoadNote     string   `json:"load_note"`
	WallStart    string   `json:"wall_started"`
	WallSeconds  float64  `json:"wall_seconds"`
	Label        string   `json:"label"`
	Skipped      []string `json:"skipped_scenarios,omitempty"`
}

// ConfigHash hashes the experiment's configuration files (path and content).
func ConfigHash(root string, files []string) (string, error) {
	h := sha256.New()
	for _, f := range files {
		data, err := os.ReadFile(filepath.Join(root, f))
		if err != nil {
			return "", err
		}
		s := sha256.Sum256(data)
		h.Write([]byte(f + "\x00" + hex.EncodeToString(s[:]) + "\n"))
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// FileSHA256 is the hex SHA-256 of a file ("" if unreadable).
func FileSHA256(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	s := sha256.Sum256(data)
	return hex.EncodeToString(s[:])
}

func run(ctx context.Context, dir string, name string, args ...string) string {
	ctx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "PYTHONUTF8=1")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// GitState returns the HEAD commit and whether the working tree is dirty.
func GitState(ctx context.Context, root string) (string, bool) {
	commit := run(ctx, root, "git", "rev-parse", "HEAD")
	if commit == "" {
		return "unknown", false
	}
	return commit, run(ctx, root, "git", "status", "--porcelain") != ""
}

// LastCommitOf returns the last commit that changed path.
func LastCommitOf(ctx context.Context, root, path string) string {
	return run(ctx, root, "git", "log", "-1", "--format=%H", "--", path)
}

// ProbeVersions asks the Python harness for its versions.
func ProbeVersions(ctx context.Context, root string) Versions {
	v := Versions{Go: runtime.Version(), Python: "unavailable", SciPy: "unavailable", HiGHS: "unavailable"}
	out := run(ctx, root, extpolicy.Python(), "-m", "harness.milp.solve", "--versions")
	var m map[string]string
	if json.Unmarshal([]byte(out), &m) == nil {
		v.Python, v.SciPy, v.HiGHS = m["python"], m["scipy"], m["highs"]
	}
	return v
}

// ProbeHardware detects the CPU model, logical CPUs, RAM, and OS.
func ProbeHardware(ctx context.Context) Hardware {
	hw := Hardware{LogicalCPUs: runtime.NumCPU(), OS: runtime.GOOS + "/" + runtime.GOARCH, CPU: "unknown"}
	switch runtime.GOOS {
	case "linux":
		if f, err := os.Open("/proc/cpuinfo"); err == nil {
			sc := bufio.NewScanner(f)
			for sc.Scan() {
				if k, v, ok := strings.Cut(sc.Text(), ":"); ok && strings.TrimSpace(k) == "model name" {
					hw.CPU = strings.TrimSpace(v)
					break
				}
			}
			f.Close()
		}
		if data, err := os.ReadFile("/proc/meminfo"); err == nil {
			if m := regexp.MustCompile(`MemTotal:\s+(\d+) kB`).FindSubmatch(data); m != nil {
				kb, _ := strconv.ParseFloat(string(m[1]), 64)
				hw.RAMGB = round1(kb / 1024 / 1024)
			}
		}
	case "windows":
		out := run(ctx, "", "reg", "query", `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0`, "/v", "ProcessorNameString")
		if i := strings.Index(out, "REG_SZ"); i >= 0 {
			hw.CPU = strings.TrimSpace(out[i+len("REG_SZ"):])
		}
		mem := run(ctx, "", "powershell", "-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
		if b, err := strconv.ParseFloat(mem, 64); err == nil {
			hw.RAMGB = round1(b / (1 << 30))
		}
	case "darwin":
		hw.CPU = run(ctx, "", "sysctl", "-n", "machdep.cpu.brand_string")
		if b, err := strconv.ParseFloat(run(ctx, "", "sysctl", "-n", "hw.memsize"), 64); err == nil {
			hw.RAMGB = round1(b / (1 << 30))
		}
	}
	return hw
}

func round1(x float64) float64 { return float64(int(x*10+0.5)) / 10 }

// Save writes the manifest as indented JSON.
func (m *Manifest) Save(path string) error {
	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

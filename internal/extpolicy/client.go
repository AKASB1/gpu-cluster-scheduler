// Package extpolicy is the Go side of the external-policy protocol v1
// (docs/contracts.md §6): it starts a policy as a child process (no shell),
// exchanges one JSON object per line over stdin/stdout, enforces wall-clock
// timeouts, copies the child's stderr to a log, and makes sure the child is
// gone on every exit path. There is no silent fallback: a timeout, a crash,
// a malformed line, or an invalid action aborts the run.
package extpolicy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

// DefaultTimeout is the default wall-clock limit per call.
const DefaultTimeout = 60 * time.Second

// Config describes how to start the policy process.
type Config struct {
	// Command is the program and its arguments (no shell). Default:
	// Python() -m harness.policy_server.
	Command []string
	// Dir is the working directory (the repository root, so that the harness
	// package is importable).
	Dir string
	// Timeout per call (default 60 s).
	Timeout time.Duration
	// Stderr receives the child's stderr (nil: discarded, but the last 4 KB
	// are kept for error messages).
	Stderr io.Writer
	// Seed is sent in the handshake.
	Seed uint64
}

// Python returns the interpreter: $PYTHON, else "python".
func Python() string {
	if p := os.Getenv("PYTHON"); p != "" {
		return p
	}
	return "python"
}

// RepoRoot walks up from the working directory to the directory with go.mod.
func RepoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", errors.New("go.mod not found above the working directory")
		}
		dir = parent
	}
}

// Client is a running external policy. It implements policy.Policy.
type Client struct {
	name    string
	spec    wirePolicy
	cfg     Config
	cmd     *exec.Cmd
	cancel  context.CancelFunc
	stdin   io.WriteCloser
	lines   chan []byte
	readErr error
	done    chan struct{}
	waitErr error
	tail    *tailBuffer
	seq     int
	hist    int
	started bool
	dead    bool
	mu      sync.Mutex
}

// tailBuffer keeps the last 4 KB written to it and forwards everything.
type tailBuffer struct {
	mu  sync.Mutex
	buf []byte
	fwd io.Writer
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	if len(t.buf) > 4096 {
		t.buf = t.buf[len(t.buf)-4096:]
	}
	if t.fwd != nil {
		_, _ = t.fwd.Write(p)
	}
	return len(p), nil
}

func (t *tailBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(bytes.TrimSpace(t.buf))
}

// New prepares a client for the policy named name (without the "py:"
// prefix) with its JSON parameters. The process starts at the first call,
// because the handshake carries the cluster description from the first view.
func New(name string, params json.RawMessage, cfg Config) *Client {
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}
	if len(cfg.Command) == 0 {
		cfg.Command = []string{Python(), "-m", "harness.policy_server"}
	}
	return &Client{name: "py:" + name, spec: wirePolicy{Name: name, Params: params}, cfg: cfg,
		tail: &tailBuffer{fwd: cfg.Stderr}}
}

// Name implements policy.Policy.
func (c *Client) Name() string { return c.name }

func (c *Client) start(v *api.View) error {
	ctx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(ctx, c.cfg.Command[0], c.cfg.Command[1:]...)
	cmd.Dir = c.cfg.Dir
	cmd.Env = append(os.Environ(), "PYTHONUTF8=1", "PYTHONUNBUFFERED=1", "PYTHONIOENCODING=utf-8")
	cmd.WaitDelay = 2 * time.Second
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return err
	}
	pr, pw := io.Pipe()
	cmd.Stdout = pw
	cmd.Stderr = c.tail
	if err := cmd.Start(); err != nil {
		cancel()
		return fmt.Errorf("policy %q: cannot start %v: %w", c.name, c.cfg.Command, err)
	}
	c.cmd, c.cancel, c.stdin, c.started = cmd, cancel, stdin, true
	c.lines, c.done = make(chan []byte, 1), make(chan struct{})
	go func() {
		c.waitErr = cmd.Wait()
		pw.Close()
		close(c.done)
	}()
	go func() {
		r := bufio.NewReaderSize(pr, 1<<16)
		for {
			line, err := r.ReadBytes('\n')
			if len(line) > 0 && err == nil {
				c.lines <- line
				continue
			}
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			if len(line) > 0 {
				err = fmt.Errorf("unterminated line %q", trunc(line))
			}
			c.readErr = err
			close(c.lines)
			return
		}
	}()
	hello := helloMsg{Type: "hello", ProtocolVersion: ProtocolVersion, Policy: c.spec, Seed: c.cfg.Seed, Cluster: clusterOf(v)}
	var rep helloReply
	if err := c.call(hello, &rep); err != nil {
		return err
	}
	if rep.Type != "hello" {
		return c.fail(fmt.Errorf("handshake: expected a hello reply, got type %q %s", rep.Type, rep.Message))
	}
	return nil
}

func trunc(b []byte) string {
	if len(b) > 200 {
		return string(b[:200]) + "..."
	}
	return string(b)
}

// call sends msg and decodes the one-line reply into out, with the timeout.
func (c *Client) call(msg any, out any) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return c.fail(err)
	}
	data = append(data, '\n')
	timer := time.NewTimer(c.cfg.Timeout)
	defer timer.Stop()
	wrote := make(chan error, 1)
	go func() {
		_, err := c.stdin.Write(data)
		wrote <- err
	}()
	select {
	case err := <-wrote:
		if err != nil {
			return c.fail(fmt.Errorf("write: %w", err))
		}
	case <-timer.C:
		return c.fail(fmt.Errorf("no answer within %v (write blocked)", c.cfg.Timeout))
	}
	select {
	case line, ok := <-c.lines:
		if !ok {
			return c.fail(fmt.Errorf("the policy process exited or closed stdout (%v)", c.readErr))
		}
		dec := json.NewDecoder(bytes.NewReader(line))
		dec.DisallowUnknownFields()
		if err := dec.Decode(out); err != nil {
			return c.fail(fmt.Errorf("malformed reply line %q: %w", trunc(bytes.TrimSpace(line)), err))
		}
		return nil
	case <-timer.C:
		return c.fail(fmt.Errorf("no answer within %v", c.cfg.Timeout))
	}
}

// fail kills the process and returns err with the policy name and the tail
// of its stderr.
func (c *Client) fail(err error) error {
	c.kill()
	msg := fmt.Sprintf("external policy %q: %v", c.name, err)
	if t := c.tail.String(); t != "" {
		msg += "\n--- policy stderr (tail) ---\n" + t
	}
	return errors.New(msg)
}

func (c *Client) kill() {
	if !c.started || c.dead {
		return
	}
	c.dead = true
	c.cancel() // kills the process (exec.CommandContext)
	_ = c.stdin.Close()
	<-c.done
}

// Schedule implements policy.Policy.
func (c *Client) Schedule(v *api.View) (api.Decision, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dead {
		return api.Decision{}, fmt.Errorf("external policy %q: process is not running", c.name)
	}
	if !c.started {
		if err := c.start(v); err != nil {
			return api.Decision{}, err
		}
	}
	c.seq++
	msg := scheduleMsg{Type: "schedule", Seq: c.seq, View: viewOf(v, c.hist)}
	c.hist = len(v.History)
	var rep decisionReply
	if err := c.call(msg, &rep); err != nil {
		return api.Decision{}, err
	}
	switch {
	case rep.Type == "error":
		return api.Decision{}, c.fail(fmt.Errorf("policy reported an error: %s", rep.Message))
	case rep.Type != "decision":
		return api.Decision{}, c.fail(fmt.Errorf("expected a decision, got type %q", rep.Type))
	case rep.Seq != c.seq:
		return api.Decision{}, c.fail(fmt.Errorf("reply seq %d, expected %d", rep.Seq, c.seq))
	}
	d, err := decisionOf(&rep, v)
	if err != nil {
		return api.Decision{}, c.fail(err)
	}
	return d, nil
}

// Close ends the session (bye), waits for the process to exit, and kills it
// if it does not. It is safe to call more than once and after errors.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.started || c.dead {
		return nil
	}
	old := c.cfg.Timeout
	c.cfg.Timeout = min(old, 10*time.Second)
	var rep helloReply
	err := c.call(map[string]string{"type": "bye"}, &rep)
	c.cfg.Timeout = old
	if err != nil {
		return err
	}
	_ = c.stdin.Close()
	select {
	case <-c.done:
	case <-time.After(5 * time.Second):
		c.kill()
		return fmt.Errorf("external policy %q: did not exit after bye; killed", c.name)
	}
	c.dead = true
	c.cancel()
	return nil
}

// Exited reports whether the child process has exited and been reaped (for
// tests: no child is left behind).
func (c *Client) Exited() bool {
	if !c.started {
		return true
	}
	select {
	case <-c.done:
		return c.cmd.ProcessState != nil
	default:
		return false
	}
}

// PID returns the child's process ID (0 before start).
func (c *Client) PID() int {
	if c.cmd == nil || c.cmd.Process == nil {
		return 0
	}
	return c.cmd.Process.Pid
}

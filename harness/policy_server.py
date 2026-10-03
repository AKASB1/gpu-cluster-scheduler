"""External-policy server: `python -m harness.policy_server`.

Speaks protocol v1 (docs/contracts.md section 6) on stdin/stdout: one JSON
object per line, UTF-8, LF only, binary pipes (no newline translation).
stdout carries protocol lines only; logs go to stderr.
"""
from __future__ import annotations

import json
import sys
import time
import traceback

from .model import Cluster, fmt_s, parse_view
from .policies import make_policy

PROTOCOL_VERSION = 1
VERSION = "gcs-harness 1"


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def send(out, obj) -> None:
    out.write(json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
    out.flush()


def wire_decision(seq: int, d: dict) -> dict:
    """Normalise a policy's decision into the wire reply (times in seconds)."""
    rep = {"type": "decision", "seq": seq, "actions": d.get("actions", [])}
    wake = d.get("wake_at_ms")
    rep["wake_at_s"] = None if wake is None else fmt_s(wake)
    rep["reservations"] = [{"job_id": r["job_id"], "shadow_s": fmt_s(r["shadow_ms"])} for r in d.get("reservations", [])]
    rep["solver"] = d.get("solver")
    return rep


def serve(inp, out) -> int:
    policy, cluster, seq = None, None, 0
    while True:
        line = inp.readline()
        if not line:
            log("stdin closed without bye")
            return 1
        msg = json.loads(line.decode("utf-8"))
        kind = msg.get("type")
        if kind == "hello":
            if msg.get("protocol_version") != PROTOCOL_VERSION:
                send(out, {"type": "error", "message": f"unsupported protocol version {msg.get('protocol_version')}"})
                return 1
            cluster = Cluster.parse(msg["cluster"])
            spec = msg["policy"]
            policy = make_policy(spec["name"], spec.get("params"), int(msg["seed"]), cluster)
            log(f"policy {spec['name']} ready (seed {msg['seed']}, {len(cluster.nodes)} nodes)")
            send(out, {"type": "hello", "name": spec["name"], "version": VERSION})
        elif kind == "schedule":
            seq = msg["seq"]
            view = parse_view(msg["view"], cluster)
            t0 = time.perf_counter()
            d = policy.schedule(view)
            if d is None and hasattr(policy, "raw_reply"):
                out.write(policy.raw_reply)
                out.flush()
                continue
            rep = wire_decision(seq, d)
            if rep["solver"] is not None:
                rep["solver"].setdefault("solve_s", time.perf_counter() - t0)
            send(out, rep)
        elif kind == "bye":
            if hasattr(policy, "worker"):
                policy.worker.close()  # the solver worker process of a MILP policy
            send(out, {"type": "bye"})
            return 0
        else:
            send(out, {"type": "error", "message": f"unknown message type {kind!r}"})
            return 1


def main() -> int:
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    try:
        return serve(inp, out)
    except Exception as e:  # report, then exit non-zero: the Go side aborts the run
        log(traceback.format_exc())
        try:
            send(out, {"type": "error", "message": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Deployment

There is nothing to deploy yet. This repository contains a simulator, scheduling policies evaluated in simulation, and a benchmark; all of it runs locally with Go and Python and needs no Docker, no network, and no GPU.

| Item | State |
|---|---|
| Local simulator and benchmark (`cmd/scheduler`, `cmd/benchmark`) | available |
| Python harness as a child process (`python -m harness.policy_server`) | available (stdin/stdout, no ports) |
| Kubernetes scheduling-framework plugin or extender | planned (Tier 3) |
| Prometheus text-format `/metrics` of a replayed simulation (`cmd/scheduler -replay-speedup N`, 127.0.0.1 only) | available (simulated values; no Prometheus server or dashboards are shipped) |
| gRPC transport for the external-policy protocol | planned |

This project does not replace the Kubernetes scheduler, Kueue, or Volcano. A future adapter would feed the same policy objects from the cluster state; until it exists, no claim about production behaviour is made.

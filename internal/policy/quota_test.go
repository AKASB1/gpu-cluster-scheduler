package policy

import (
	"encoding/json"
	"testing"

	"github.com/AKASB1/gpu-cluster-scheduler/internal/api"
)

const halfHalf = `{"quota_shares": {"A": 0.5, "B": 0.5}}`

func tenantJob(id, tenant string, submit float64, gpus int) api.PendingJob {
	p := pend(id, submit, 1, gpus, 1, 100)
	p.Tenant = tenant
	return p
}

// Two nodes of 8 GPUs, nominal quota 8 GPUs per tenant. A runs 8 GPUs on n0
// (at its quota); a4 (A, 4 GPUs) comes before b4 (B, 4 GPUs) in FIFO order.
func quotaView() *api.View {
	r := running("RA", 1, 8, 1, api.Placement{{Node: 0, Workers: 1}}, 1000)
	r.Tenant, r.Preemptible = "A", true
	v := view([]api.NodeState{node(0, "r", 8, 0), node(1, "r", 8, 8)}, []api.RunningJob{r},
		tenantJob("a4", "A", 1, 4), tenantJob("b4", "B", 2, 4))
	v.Tenants = []api.TenantUsage{{Tenant: "A", RunningGPUs: 8}, {Tenant: "B"}}
	return v
}

func TestQuotaAdmission(t *testing.T) {
	// strict: A is at its quota, so a4 is not admitted; b4 starts.
	d, err := mustNew(t, "fifo+first_fit+none+quota_strict", halfHalf).Schedule(quotaView())
	if err != nil || acts(d) != "start:b4[{1 1}] " {
		t.Fatalf("strict: %s %v", acts(d), err)
	}
	// borrowing: b4 (within quota) first, then a4 borrows B's unused quota.
	d, _ = mustNew(t, "fifo+first_fit+none+quota", halfHalf).Schedule(quotaView())
	if acts(d) != "start:b4[{1 1}] start:a4[{1 1}] " {
		t.Fatalf("borrow: %s", acts(d))
	}
	// without quotas FIFO starts a4 first
	d, _ = mustNew(t, "fifo+first_fit+none", "").Schedule(quotaView())
	if acts(d) != "start:a4[{1 1}] start:b4[{1 1}] " {
		t.Fatalf("no quota: %s", acts(d))
	}
}

// Reclaim: A borrowed the whole cluster (two 8-GPU jobs, 16 > quota 8); B's
// 8-GPU job is within B's quota and does not fit: it preempts the A job that
// loses less work (RA2 did 100 s without checkpoint, RA1 500 s).
func TestQuotaReclaim(t *testing.T) {
	r1 := running("RA1", 1, 8, 1, api.Placement{{Node: 0, Workers: 1}}, 1000)
	r2 := running("RA2", 1, 8, 1, api.Placement{{Node: 1, Workers: 1}}, 1000)
	r1.Tenant, r1.Preemptible, r1.WorkDone = "A", true, 500
	r2.Tenant, r2.Preemptible, r2.WorkDone = "A", true, 100
	v := view([]api.NodeState{node(0, "r", 8, 0), node(1, "r", 8, 0)}, []api.RunningJob{r1, r2}, tenantJob("b8", "B", 1, 8))
	v.Tenants = []api.TenantUsage{{Tenant: "A", RunningGPUs: 16}, {Tenant: "B"}}
	d, _ := mustNew(t, "fifo+first_fit+easy+quota+reclaim", halfHalf).Schedule(v)
	if acts(d) != "preempt:RA2[] start:b8[{1 1}] " {
		t.Fatalf("reclaim: %s", acts(d))
	}
	// without reclaim B waits
	d, _ = mustNew(t, "fifo+first_fit+easy+quota", halfHalf).Schedule(v)
	if len(d.Actions) != 0 {
		t.Fatalf("no reclaim: %s", acts(d))
	}
	// A within its quota cannot be reclaimed: give A a share of 1
	d, _ = mustNew(t, "fifo+first_fit+easy+quota+reclaim", `{"quota_shares": {"A": 1, "B": 0}}`).Schedule(v)
	if len(d.Actions) != 0 {
		t.Fatalf("victim within quota: %s", acts(d))
	}
}

func TestQuotaNames(t *testing.T) {
	for _, bad := range []Spec{{Name: "fifo+first_fit+none+reclaim", Params: json.RawMessage(halfHalf)},
		{Name: "fifo+first_fit+none+quota+quota_strict", Params: json.RawMessage(halfHalf)},
		{Name: "fifo+first_fit+none+quota+quota", Params: json.RawMessage(halfHalf)},
		{Name: "fifo+first_fit+none+quota"},
		{Name: "fifo+first_fit+none+quota", Params: json.RawMessage(`{"quota_shares": {"A": 0.7, "B": 0.7}}`)}} {
		if _, err := New(bad, Env{}); err == nil {
			t.Errorf("%+v accepted", bad)
		}
	}
	if p := mustNew(t, "priority+best_fit+easy+preempt+quota+reclaim", halfHalf); p.Name() != "priority+best_fit+easy+preempt+quota+reclaim" {
		t.Fatal("name")
	}
}

package cluster

import (
	"strings"
	"testing"
	"time"
)

const valid = `{"schema_version":1,"name":"t","classes":[{"name":"a100","speed":1},{"name":"v100","speed":0.4}],
"node_groups":[{"count":2,"prefix":"r1-n","rack":"r1","class":"a100","gpus":8,"cpus":128,"mem_gb":1024}],
"nodes":[{"name":"x","rack":"r0","class":"v100","gpus":4,"cpus":64,"mem_gb":512}],
"cross_node_factor":1.1,"cross_rack_factor":1.25,"restart_overhead_s":120,"preempt_grace_s":0}`

func TestLoadValidAndSorted(t *testing.T) {
	c, err := Parse([]byte(valid))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, n := range c.Nodes {
		names = append(names, n.Name)
	}
	if strings.Join(names, ",") != "x,r1-n00,r1-n01" {
		t.Fatalf("nodes not sorted by (rack, name): %v", names)
	}
	if c.Info.TotalGPUs != 20 || c.WorkCapacity() != 17.6 || c.Info.RestartOverhead != 120*time.Second {
		t.Fatalf("totals %+v work %v", c.Info, c.WorkCapacity())
	}
	if len(c.Info.Racks) != 2 || c.FastestSpeed("") != 1 || c.FastestSpeed("v100") != 0.4 {
		t.Fatalf("racks/speeds %+v", c.Info.Racks)
	}
	s := c.WithFactors(2, 0.5)
	if s.Info.CrossNode != 1.2 || s.Info.CrossRack != 1.5 || s.Info.RestartOverhead != 60*time.Second {
		t.Fatalf("scaled %+v", s.Info)
	}
}

func TestInvalidClusterFiles(t *testing.T) {
	bad := map[string]string{
		"version":       strings.Replace(valid, `"schema_version":1`, `"schema_version":2`, 1),
		"unknown class": strings.Replace(valid, `"class":"v100"`, `"class":"h100"`, 1),
		"dup node":      strings.Replace(valid, `"name":"x"`, `"name":"r1-n00"`, 1),
		"zero gpus":     strings.Replace(valid, `"gpus":4`, `"gpus":0`, 1),
		"zero mem":      strings.Replace(valid, `"mem_gb":512`, `"mem_gb":0`, 1),
		"bad factors":   strings.Replace(valid, `"cross_rack_factor":1.25`, `"cross_rack_factor":1.05`, 1),
		"speed":         strings.Replace(valid, `"speed":0.4`, `"speed":0`, 1),
		"unknown field": strings.Replace(valid, `"name":"t"`, `"name":"t","foo":1`, 1),
		"no rack":       strings.Replace(valid, `"rack":"r0"`, `"rack":""`, 1),
		"dup class":     strings.Replace(valid, `"name":"v100"`, `"name":"a100"`, 1),
		"overhead":      strings.Replace(valid, `"restart_overhead_s":120`, `"restart_overhead_s":-1`, 1),
		"not json":      `{"schema_version":1,`,
	}
	for name, in := range bad {
		if _, err := Parse([]byte(in)); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

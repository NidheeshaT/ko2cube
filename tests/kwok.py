"""
Integration tests for KWOKAdapter.

These tests interact with real local kwokctl clusters — no mocking.
Requires kwokctl to be installed (checked in /opt/homebrew/bin).

Test Coverage:
  - Cluster Lifecycle: initialization and cleanup
  - Resource Creation: K8sNode and K8sPod using both Pydantic models and raw dicts
  - Resource Deletion: Node and Pod deletion with verification
  - Resource Injection: CPU/Memory auto-injection from infrastructure.json
  - Error Handling:
      - NodeValidationError (missing instance-type label)
      - InstanceTypeError (unknown instance type)
      - PodValidationError (invalid nodeName, missing nodeName)
      - KWOKError (completely wrong input type)
  - Cluster Discovery
"""

import unittest
import sys
import os
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Ensure kwokctl binary is reachable
os.environ["PATH"] = f"/opt/homebrew/bin:/usr/local/bin:{os.environ.get('PATH', '')}"

from ko2cube.server.kwok.kwok import KWOKAdapter
from ko2cube.models import K8sNode, K8sPod, K8sMetadata, K8sPodSpec, DeleteNode, DeletePod
from ko2cube.server.kwok.error import (
    KWOKError, NodeValidationError, PodValidationError, InstanceTypeError
)

REGION = "us-east-1"


class TestKWOKIntegration(unittest.TestCase):
    adapter: KWOKAdapter = None

    @classmethod
    def setUpClass(cls):
        """Boot real clusters for integration testing."""
        print("\n[SETUP] Initializing KWOKAdapter (real clusters)...")
        cls.adapter = KWOKAdapter()
        time.sleep(5)  # allow clusters to fully start

    @classmethod
    def tearDownClass(cls):
        """Delete all clusters created during the test suite."""
        if cls.adapter:
            print("\n[TEARDOWN] Cleaning up all clusters...")
            cls.adapter.cleanup()

    # ------------------------------------------------------------------
    # Cluster discovery
    # ------------------------------------------------------------------

    def test_cluster_discovery(self):
        """Adapter must report the expected regional clusters after init."""
        clusters = self.adapter.get_clusters()
        print("\n[DEBUG] Discovered clusters:")
        print(json.dumps(clusters, indent=2))
        names = [c["name"] for c in clusters]
        self.assertIn("us-east-1", names)
        self.assertIn("us-west-2", names)
        self.assertIn("eu-west-1", names)

    # ------------------------------------------------------------------
    # Node lifecycle: create, verify injection, delete
    # ------------------------------------------------------------------

    def test_node_create_and_resource_injection(self):
        """Node is created and CPU/Memory are auto-injected from infrastructure.json."""
        node = K8sNode(
            apiVersion="v1",
            metadata=K8sMetadata(
                name="inject-node",
                labels={"node.kubernetes.io/instance-type": "m5.large"}
            )
        )
        self.adapter.create_from_dict([node], REGION)
        
        nodes = self.adapter.get_nodes()
        print(f"\n[DEBUG] Nodes after creation in {REGION}:")
        print(json.dumps(nodes, indent=2))

        created = next((n for n in nodes if n["name"] == "inject-node" and n["cluster"] == REGION), None)
        self.assertIsNotNone(created, "inject-node was not found after creation")
        # m5.large: 2 vCPU, 8 GiB RAM
        self.assertEqual(created["cpu"], "2")
        self.assertEqual(created["memory"], "8Gi")

        # Cleanup
        ok = self.adapter.delete_node("inject-node", REGION)
        self.assertTrue(ok, "delete_node should return True")

        # Verify it's gone
        nodes_after = self.adapter.get_nodes()
        print(f"\n[DEBUG] Nodes after deletion in {REGION}:")
        print(json.dumps(nodes_after, indent=2))
        remains = any(n["name"] == "inject-node" and n["cluster"] == REGION for n in nodes_after)
        self.assertFalse(remains, "inject-node should have been deleted")

    def test_node_resource_injection_for_all_instance_types(self):
        """Each instance type gets its correct CPU and memory injected."""
        expected = {
            "m5.large":   ("2", "8Gi"),
            "m5.xlarge":  ("4", "16Gi"),
            "m5.2xlarge": ("8", "32Gi"),
            "c5.large":   ("2", "4Gi"),
            "c5.xlarge":  ("4", "8Gi"),
            "r5.large":   ("2", "16Gi"),
        }

        for idx, (instance_type, (exp_cpu, exp_mem)) in enumerate(expected.items()):
            node_name = f"type-test-node-{idx}"
            node = K8sNode(
                apiVersion="v1",
                metadata=K8sMetadata(
                    name=node_name,
                    labels={"node.kubernetes.io/instance-type": instance_type}
                )
            )
            self.adapter.create_from_dict([node], REGION)

            nodes = self.adapter.get_nodes()
            print(f"\n[DEBUG] Nodes after creation in {REGION}:")
            print(json.dumps(nodes, indent=2))
            
            created = next((n for n in nodes if n["name"] == node_name and n["cluster"] == REGION), None)
            self.assertIsNotNone(created, f"{node_name} not found for type {instance_type}")
            self.assertEqual(created["cpu"], exp_cpu, f"Wrong CPU for {instance_type}")
            self.assertEqual(created["memory"], exp_mem, f"Wrong memory for {instance_type}")

            self.adapter.delete_node(node_name, REGION)
            
            # Verify and debug print after deletion
            nodes_after = self.adapter.get_nodes()
            print(f"[DEBUG] Nodes after deleting {node_name} in {REGION}:")
            print(json.dumps(nodes_after, indent=2))

    # ------------------------------------------------------------------
    # Pod lifecycle: create, verify, delete
    # ------------------------------------------------------------------

    def test_pod_create_and_delete(self):
        """Pod is created on a valid node and deleted cleanly."""
        # First create the backing node
        node = K8sNode(
            apiVersion="v1",
            metadata=K8sMetadata(
                name="pod-host-node",
                labels={"node.kubernetes.io/instance-type": "m5.xlarge"}
            )
        )
        pod = K8sPod(
            apiVersion="v1",
            metadata=K8sMetadata(name="lifecycle-pod"),
            spec=K8sPodSpec(
                nodeName="pod-host-node",
                containers=[{
                    "name": "app",
                    "image": "nginx",
                    "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}}
                }]
            )
        )

        self.adapter.create_from_dict([node, pod], REGION)

        # Verify pod exists
        pods = self.adapter.get_pods()
        print(f"\n[DEBUG] Pods after creation in {REGION}:")
        print(json.dumps(pods, indent=2))

        created = next((p for p in pods if p["name"] == "lifecycle-pod" and p["cluster"] == REGION), None)
        self.assertIsNotNone(created, "lifecycle-pod was not found after creation")
        self.assertEqual(created["node"], "pod-host-node")

        # Delete pod, verify gone
        self.assertTrue(self.adapter.delete_pod("lifecycle-pod", REGION))
        pods_after = self.adapter.get_pods()
        print(f"\n[DEBUG] Pods after deletion in {REGION}:")
        print(json.dumps(pods_after, indent=2))
        self.assertFalse(any(p["name"] == "lifecycle-pod" and p["cluster"] == REGION for p in pods_after))


        # Cleanup node
        self.adapter.delete_node("pod-host-node", REGION)

    # ------------------------------------------------------------------
    # Raw dict input (backward compatibility)
    # ------------------------------------------------------------------

    def test_raw_dict_input_raises_kwok_error(self):
        """Raw dicts raise KWOKError — callers must provide typed K8sNode/K8sPod models."""
        resources = [
            {
                "apiVersion": "v1",
                "kind": "Node",
                "metadata": {"name": "dict-node", "labels": {"node.kubernetes.io/instance-type": "c5.large"}}
            },
        ]
        with self.assertRaises(KWOKError):
            self.adapter.create_from_dict(resources, REGION)

    # ------------------------------------------------------------------
    # Error handling: NodeValidationError
    # ------------------------------------------------------------------

    def test_error_node_missing_instance_type_label(self):
        """NodeValidationError raised when instance-type label is absent."""
        node = K8sNode(
            apiVersion="v1",
            metadata=K8sMetadata(name="missing-label-node")  # no labels
        )
        with self.assertRaises(NodeValidationError):
            self.adapter.create_from_dict([node], REGION)

    # ------------------------------------------------------------------
    # Error handling: InstanceTypeError
    # ------------------------------------------------------------------

    def test_error_unknown_instance_type(self):
        """InstanceTypeError raised when label has an unknown instance type."""
        node = K8sNode(
            apiVersion="v1",
            metadata=K8sMetadata(
                name="bad-type-node",
                labels={"node.kubernetes.io/instance-type": "t9.nano-invalid"}
            )
        )
        with self.assertRaises(InstanceTypeError):
            self.adapter.create_from_dict([node], REGION)

    # ------------------------------------------------------------------
    # Error handling: PodValidationError
    # ------------------------------------------------------------------

    def test_error_pod_assigned_to_nonexistent_node(self):
        """PodValidationError raised when pod's nodeName doesn't exist in the cluster."""
        pod = K8sPod(
            apiVersion="v1",
            metadata=K8sMetadata(name="orphan-pod"),
            spec=K8sPodSpec(
                nodeName="ghost-node",
                containers=[{
                    "name": "app",
                    "image": "nginx",
                    "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}}
                }]
            )
        )
        with self.assertRaises(PodValidationError):
            self.adapter.create_from_dict([pod], REGION)

    # ------------------------------------------------------------------
    # Error handling: KWOKError (unsupported input type)
    # ------------------------------------------------------------------

    def test_error_unsupported_input_type(self):
        """KWOKError raised when input list contains an unrecognised object type."""
        with self.assertRaises(KWOKError):
            self.adapter.create_from_dict(["this-is-a-string"], REGION)

        with self.assertRaises(KWOKError):
            self.adapter.create_from_dict([42], REGION)

    def test_error_non_list_input(self):
        """ValueError raised when data is not a list at all."""
        with self.assertRaises(ValueError):
            self.adapter.create_from_dict({"kind": "Node"}, REGION)

    def test_delete_from_dict(self):
        """Verify deleting resources using DeleteNode and DeletePod models."""
        node_name = "del-test-node"
        pod_name = "del-test-pod"

        # 1. Setup: Create node and pod
        node = K8sNode(
            apiVersion="v1",
            metadata=K8sMetadata(name=node_name, labels={"node.kubernetes.io/instance-type": "m5.large"})
        )
        pod = K8sPod(
            apiVersion="v1",
            metadata=K8sMetadata(name=pod_name),
            spec=K8sPodSpec(
                nodeName=node_name,
                containers=[{
                    "name": "test",
                    "image": "nginx",
                    "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}}
                }]
            )
        )
        self.adapter.create_from_dict([node, pod], REGION)
        
        # Verify creation before deletion
        nodes_before = self.adapter.get_nodes()
        print(f"\n[DEBUG] Nodes after creation in {REGION}:")
        print(json.dumps(nodes_before, indent=2))
        pods_before = self.adapter.get_pods()
        print(f"[DEBUG] Pods after creation in {REGION}:")
        print(json.dumps(pods_before, indent=2))

        self.assertTrue(any(n["name"] == node_name and n["cluster"] == REGION for n in nodes_before))
        self.assertTrue(any(p["name"] == pod_name and p["cluster"] == REGION for p in pods_before))
        print(f"\n[DEBUG] Resources successfully created in {REGION} before delete_from_dict")

        # 2. Delete using delete_from_dict
        delete_list = [
            DeletePod(name=pod_name),
            DeleteNode(name=node_name)
        ]
        self.adapter.delete_from_dict(delete_list, REGION)

        # 3. Verify deletion
        nodes = self.adapter.get_nodes()
        pods = self.adapter.get_pods()
        
        print(f"\n[DEBUG] Nodes after delete_from_dict in {REGION}:")
        print(json.dumps(nodes, indent=2))
        print(f"[DEBUG] Pods after delete_from_dict in {REGION}:")
        print(json.dumps(pods, indent=2))

        self.assertFalse(any(n["name"] == node_name and n["cluster"] == REGION for n in nodes))
        self.assertFalse(any(p["name"] == pod_name and p["cluster"] == REGION for p in pods))

    def test_pod_scheduling_capacity_exhaustion(self):
        """Verify that pods remain Pending when capacity is exhausted and schedule when freed."""
        # 1. Create a node with 2 CPU (c5.large)
        node_name = "capped-node"
        node = K8sNode(
            metadata=K8sMetadata(
                name=node_name,
                labels={"node.kubernetes.io/instance-type": "c5.large"}
            )
        )
        self.adapter.create_from_dict([node], REGION)
        
        # Verify node creation
        nodes = self.adapter.get_nodes()
        print(f"\n[DEBUG] Nodes after creation in {REGION}:")
        print(json.dumps(nodes, indent=2))
        
        # 2. Create 3 pods, each requesting 1 CPU, no nodeName
        pods = []
        for i in range(3):
            pods.append(K8sPod(
                metadata=K8sMetadata(name=f"load-pod-{i}"),
                spec=K8sPodSpec(
                    containers=[{
                        "name": "app",
                        "image": "nginx",
                        "resources": {"requests": {"cpu": "1"}}
                    }]
                )
            ))
        
        self.adapter.create_from_dict(pods, REGION)
        
        # 3. Wait and check status
        # We expect 2 to be Running/Scheduled and 1 to be Pending
        print("\n[DEBUG] Waiting for scheduler to process pods...")
        time.sleep(15) # wait for scheduler
        
        all_pods = self.adapter.get_pods()
        print(f"\n[DEBUG] All Pods after initial scheduling in {REGION}:")
        print(json.dumps(all_pods, indent=2))
        
        my_pods = [p for p in all_pods if p["name"].startswith("load-pod-") and p["cluster"] == REGION]
        
        running = [p for p in my_pods if p["status"] == "Running"]
        pending = [p for p in my_pods if p["status"] == "Pending"]
        
        print(f"\n[DEBUG] Pod Summary after initial scheduling in {REGION}:")
        for p in my_pods:
            print(f"  {p['name']}: {p['status']} on {p['node']}")
            
        self.assertEqual(len(running), 2, f"Should have 2 running pods, found {len(running)}")
        self.assertEqual(len(pending), 1, f"Should have 1 pending pod, found {len(pending)}")
        
        # 4. Delete one running pod
        to_delete = running[0]["name"]
        print(f"\n[DEBUG] Deleting running pod {to_delete} to free capacity...")
        self.adapter.delete_pod(to_delete, REGION)
        
        # 5. Wait and see if pending pod gets scheduled
        print("[DEBUG] Waiting for pending pod to be scheduled...")
        time.sleep(15)
        
        all_pods_after = self.adapter.get_pods()
        print(f"\n[DEBUG] All Pods after freeing capacity in {REGION}:")
        print(json.dumps(all_pods_after, indent=2))
        
        my_pods_after = [p for p in all_pods_after if p["name"].startswith("load-pod-") and p["cluster"] == REGION]
        
        running_after = [p for p in my_pods_after if p["status"] == "Running"]
        print(f"\n[DEBUG] Pod Summary after freeing capacity in {REGION}:")
        for p in my_pods_after:
            print(f"  {p['name']}: {p['status']} on {p['node']}")
            
        self.assertEqual(len(running_after), 2, f"Should now have 2 running pods, found {len(running_after)}")
        
        # Cleanup
        for p in my_pods_after:
            self.adapter.delete_pod(p["name"], REGION)
        self.adapter.delete_node(node_name, REGION)

    def test_error_pod_missing_resources(self):
        """Pod creation should fail if resources are missing in containers."""
        # 1. Test Pydantic validation (missing field)
        with self.assertRaises(Exception): # Pydantic ValidationError
            K8sPod(
                metadata=K8sMetadata(name="no-resource-pod"),
                spec=K8sPodSpec(
                    containers=[{"name": "app", "image": "nginx"}] # missing resources
                )
            )

        # 2. Test Adapter/Config validation (empty dict)
        pod = K8sPod(
            metadata=K8sMetadata(name="empty-resource-pod"),
            spec=K8sPodSpec(
                containers=[{"name": "app", "image": "nginx", "resources": {}}]
            )
        )
        with self.assertRaises(PodValidationError):
            self.adapter.create_from_dict([pod], REGION)


if __name__ == "__main__":
    unittest.main(verbosity=2)

from kubernetes import client, utils
from .config import load_kwok_kubeconfig
from .constants import EC2_INSTANCE_TYPES


class NodeDict(dict):
    """Generates the Kubernetes dictionary representation of a Node."""
    def __init__(self, name: str, instance_type: str, region: str, capacity_type: str = "ON_DEMAND"):
        spec = EC2_INSTANCE_TYPES.get(instance_type)
        if spec is None:
            raise ValueError(f"Unknown instance type '{instance_type}'. See constants.EC2_INSTANCE_TYPES for valid options.")

        super().__init__({
            "apiVersion": "v1",
            "kind": "Node",
            "metadata": {
                "name": name,
                "annotations": {
                    "node.kubernetes.io/ttl": "0",        # Fixed: removed deprecated alpha prefix
                    "kwok.x-k8s.io/node": "fake",
                },
                "labels": {
                    # Removed: deprecated beta.kubernetes.io/arch and beta.kubernetes.io/os
                "kubernetes.io/arch": spec["arch"],
                "kubernetes.io/hostname": name,
                "kubernetes.io/os": "linux",
                "kubernetes.io/role": "agent",
                "node-role.kubernetes.io/agent": "",
                "node.kubernetes.io/instance-type": instance_type,
                "topology.kubernetes.io/region": region,
                "eks.amazonaws.com/capacityType": capacity_type.upper(),
                "type": "kwok",
            },
        },
        "spec": {
            "taints": [
                {
                    "effect": "NoSchedule",
                    "key": "kwok.x-k8s.io/node",
                    "value": "fake"
                }
            ]
        },
        "status": {
            "capacity": {
                "cpu": spec["cpu"],
                "memory": spec["memory"],
                "pods": str(spec["pods"]),
            },
            "allocatable": {
                "cpu": spec["cpu"],
                "memory": spec["memory"],
                "pods": str(spec["pods"]),
            },
            "nodeInfo": {
                "architecture": spec["arch"],
                "bootID": "",
                "containerRuntimeVersion": "",
                "kernelVersion": "",
                "kubeProxyVersion": "fake",
                "kubeletVersion": "fake",
                "machineID": "",
                "operatingSystem": "linux",
                "osImage": "",
                "systemUUID": "",
            },
            "phase": "Running",
        },
    })


class Node:
    """
    Represents a simulated kwok Kubernetes node.

    Example:
        node = Node(name="kwok-node-0", instance_type="m5.xlarge", region="us-east", capacity_type="SPOT")
        node.create()
    """

    def __init__(
        self,
        name: str = "kwok-node-0",
        instance_type: str = "m5.large",
        region: str = "us-east",
        cluster_name: str = "kwok-cluster",
        capacity_type: str = "ON_DEMAND",
        on_delete_callback = None,
    ):
        self.name = name
        self.instance_type = instance_type
        self.region = region
        self.cluster_name = cluster_name
        self.capacity_type = capacity_type
        self.on_delete_callback = on_delete_callback

        self._dict = NodeDict(name, instance_type, region, capacity_type)

    @property
    def spec(self) -> dict:
        """Return the node spec dict."""
        return self._dict

    def create(self):
        """Apply this node to the kwok cluster."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        k8s_client = client.ApiClient()
        print(f"Creating node '{self.name}' ({self.instance_type}, {self.region})...")
        try:
            print(self._dict)
            utils.create_from_dict(k8s_client, self._dict)
            print(f"Node '{self.name}' created successfully!")
        except Exception as e:
            print(f"Failed to create node: {e}")
            raise

    def delete(self):
        """Delete this node from the kwok cluster."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        v1 = client.CoreV1Api()
        print(f"Deleting node '{self.name}'...")
        try:
            v1.delete_node(self.name)
            print(f"Node '{self.name}' deleted.")
        except Exception as e:
            print(f"Failed to delete node: {e}")
            raise

    def simulate_spot_interruption(self, delay_seconds: int = 0, callback=None):
        """
        Simulate an AWS Spot Interruption via a lightweight background thread.
        Waits for `delay_seconds`, forcefully deletes this node from the kwok cluster,
        and optionally invokes a custom callback function.
        """
        import threading
        import time

        def interruption_routine():
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            
            print(f"\n[Spot Interruption] Terminating node '{self.name}'...")
            try:
                self.delete()
            except Exception as e:
                print(f"[Spot Interruption] Deletion failed: {e}")
            
            if callback:
                try:
                    callback()
                except Exception as e:
                    print(f"[Spot Interruption] Callback failed: {e}")

        t = threading.Thread(target=interruption_routine, daemon=True)
        t.start()

    def __repr__(self):
        return f"Node(name={self.name!r}, instance_type={self.instance_type!r}, region={self.region!r})"
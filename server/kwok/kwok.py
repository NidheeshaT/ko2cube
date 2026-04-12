import atexit
from typing import List, Union
import subprocess
import json
import os
from pathlib import Path
from kubernetes import client, config, utils
from ko2cube.server.kwok.config import (
    INFRA_PATH, get_valid_instance_types, validate_node_resource, 
    validate_pod_resource, get_instance_resources
)

from ko2cube.models import K8sResource, K8sNode, K8sPod, DeleteResource, DeleteNode, DeletePod
from ko2cube.server.kwok.error import KWOKError

class KWOKAdapter:
    """
    KWOKAdapter provides a high-level API for managing simulated Kubernetes clusters using KWOK.
    
    This adapter simplifies the interaction with kwokctl and the Kubernetes API for development
    and testing environments. It manages multi-region cluster initialization, resource 
    provisioning with automatic resource injection, and lifecycle management.

    Key Features:
    - **Multi-Region Support**: Automatically initializes clusters for every region defined 
      in the infrastructure configuration.
    - **Strict Validation**: Enforces resource naming conventions and validates pod-to-node 
      assignments before applying changes to the cluster.
    - **Automatic Resource Injection**: Nodes created via the adapter automatically receive 
      status.capacity and status.allocatable fields based on their AWS/Cloud instance types.
    - **Automated Lifecycle**: Uses `atexit` to ensure all simulated clusters are torn down 
      when the application process exits, preventing resource leakage.

    Usage:
        adapter = KWOKAdapter()
        # Resources are provisioned as lists of dictionaries (Pydantic-compatible)
        adapter.create_from_dict([node_data, pod_data], cluster_name="us-east-1")
    """

    @staticmethod
    def cleanup_all_clusters():
        """
        Global cleanup function registered with atexit.
        Queries kwokctl for all active clusters and deletes them to ensure
        no simulated resources are left behind when the process terminates.
        """
        try:
            # Query kwokctl for all currently running clusters
            result = subprocess.run(["kwokctl", "get", "clusters"], capture_output=True, text=True, check=True)
            clusters = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if clusters:
                print(f"\n[KWOK] Detected {len(clusters)} active clusters. Starting cleanup...")
                for cluster in clusters:
                    print(f"[KWOK] Deleting cluster: {cluster}")
                    subprocess.run(["kwokctl", "delete", "cluster", "--name", cluster], check=False)
                print("[KWOK] Cleanup complete.")
        except Exception as e:
            # Use simple print as logging might already be shut down during exit
            print(f"[KWOK] Automatic cleanup failed: {e}")

    def __init__(self, cluster_prefix: str = None):
        # Register global cleanup to run on process exit
        atexit.register(self.cleanup_all_clusters)

        self.cluster_prefix = cluster_prefix

        # Load infrastructure regions
        try:
            with open(INFRA_PATH, "r") as f:
                self.infra = json.load(f)
        except Exception as e:
            print(f"Failed to load infrastructure.json: {e}")
            self.infra = {"regions": []}
        
        self.regions = self.infra.get("regions", [])
        self.active_clusters = []

        # Start clusters for each region
        for region in self.regions:
            full_name = self._full_name(region)
            print(f"Starting kwok cluster: {full_name}")
            try:
                # kwokctl create cluster --name <full_name>
                # Use check=False to avoid crashing if cluster already exists
                subprocess.run(["kwokctl", "create", "cluster", "--name", full_name], check=False)
                self.active_clusters.append(full_name)
            except Exception as e:
                print(f"Failed to start cluster {region}: {e}")

    def _full_name(self, region: str) -> str:
        """Helper to map a region name to its prefixed cluster name."""
        if self.cluster_prefix:
            return f"{self.cluster_prefix}-{region}"
        return region

    def cleanup(self):
        """Manually trigger the deletion of all clusters tracked by this adapter."""
        for cluster in self.active_clusters:
             subprocess.run(["kwokctl", "delete", "cluster", "--name", cluster], check=False)
        self.active_clusters = []

    def delete_cluster(self, region: str):
        """Delete a specific cluster by region name."""
        cluster_name = self._full_name(region)
        try:
            print(f"Deleting kwok cluster: {cluster_name}")
            subprocess.run(["kwokctl", "delete", "cluster", "--name", cluster_name], check=True)
            if cluster_name in self.active_clusters:
                self.active_clusters.remove(cluster_name)
        except Exception as e:
            print(f"Failed to delete cluster {cluster_name}: {e}")

    def _load_config(self, cluster_name: str):
        """Load kubeconfig context for the specified region cluster."""
        try:
            # kwokctl typically creates contexts named 'kwok-<name>'
            config.load_kube_config(context=f"kwok-{cluster_name}")
        except Exception as e:
            print(f"Failed to load kubeconfig context for cluster {cluster_name}: {e}")

    def get_pods(self):
        """Aggregate pods from all active regional clusters."""
        all_pods = []
        for cluster in self.active_clusters:
            self._load_config(cluster)
            v1 = client.CoreV1Api()
            try:
                # List pods across all namespaces
                pod_list = v1.list_pod_for_all_namespaces()
                for p in pod_list.items:
                    all_pods.append({
                        "name": p.metadata.name,
                        "namespace": p.metadata.namespace,
                        "cluster": cluster,
                        "status": p.status.phase,
                        "node": p.spec.node_name
                    })
            except Exception as e:
                print(f"Error listing pods in cluster {cluster}: {e}")
        return all_pods

    def get_nodes(self):
        """Aggregate nodes from all active regional clusters."""
        all_nodes = []
        for cluster in self.active_clusters:
            self._load_config(cluster)
            v1 = client.CoreV1Api()
            try:
                node_list = v1.list_node()
                for n in node_list.items:
                    # Check Ready condition
                    ready_status = "NotReady"
                    if n.status.conditions:
                        for condition in n.status.conditions:
                            if condition.type == "Ready" and condition.status == "True":
                                ready_status = "Ready"
                                break
                    
                    all_nodes.append({
                        "name": n.metadata.name,
                        "cluster": cluster,
                        "status": ready_status,
                        "cpu": n.status.capacity.get("cpu"),
                        "memory": n.status.capacity.get("memory")
                    })
            except Exception as e:
                print(f"Error listing nodes in cluster {cluster}: {e}")
        return all_nodes

    def delete_pod(self, pod_name: str, region: str) -> bool:
        """Delete a pod from the specified regional cluster."""
        cluster_name = self._full_name(region)
        self._load_config(cluster_name)
        v1 = client.CoreV1Api()
        try:
            v1.delete_namespaced_pod(name=pod_name, namespace="default")
            print(f"Deleted pod {pod_name} from cluster {cluster_name}")
            return True
        except Exception as e:
            print(f"Failed to delete pod {pod_name} in cluster {cluster_name}: {e}")
            return False

    def delete_node(self, node_name: str, region: str) -> bool:
        """Delete a node from the specified regional cluster."""
        cluster_name = self._full_name(region)
        self._load_config(cluster_name)
        v1 = client.CoreV1Api()
        try:
            v1.delete_node(name=node_name)
            print(f"Deleted node {node_name} from cluster {cluster_name}")
            return True
        except Exception as e:
            print(f"Failed to delete node {node_name} in cluster {cluster_name}: {e}")
            return False

    def _create_nodes(self, nodes: List[K8sNode], valid_types: list, current_active_nodes: set, k8s_client: client.ApiClient) -> set:
        """Validate K8sNode models, inject resource constraints, create them, and return updated active node names."""
        for node in nodes:
            res = node.model_dump()
            validate_node_resource(res, valid_types)

            # Inject resource constraints based on the instance-type label
            instance_type = node.metadata.labels.get("node.kubernetes.io/instance-type")
            resources_constraints = get_instance_resources(instance_type)
            if resources_constraints:
                res["status"] = {
                    "capacity": {
                        "cpu": resources_constraints["cpu"],
                        "memory": resources_constraints["memory"],
                        "pods": "110"
                    },
                    "allocatable": {
                        "cpu": resources_constraints["cpu"],
                        "memory": resources_constraints["memory"],
                        "pods": "110"
                    },
                    "nodeInfo": {
                        "kubeletVersion": "fake",
                        "architecture": "amd64",
                        "operatingSystem": "linux"
                    }
                }

            utils.create_from_dict(k8s_client, res)
            current_active_nodes.add(node.metadata.name)
        return current_active_nodes

    def _create_pods(self, pods: List[K8sPod], active_node_names: set, k8s_client: client.ApiClient):
        """Validate K8sPod models for node assignment and create them in the cluster."""
        for pod in pods:
            res = pod.model_dump()
            validate_pod_resource(res, active_node_names)
            utils.create_from_dict(k8s_client, res)

    def create_from_dict(self, data: List[K8sResource], region: str):
        """
        Create resources from a list in the specified regional cluster.

        Accepts a mixed list of raw dicts or Pydantic K8sNode/K8sPod models.
        Resources are automatically segregated by type, then Nodes are validated,
        injected, and created before Pods are validated against the resulting node set.
        """
        if not isinstance(data, list):
            raise ValueError("Input data must be a list of dicts or K8sNode/K8sPod models")

        cluster_name = self._full_name(region)
        self._load_config(cluster_name)
        k8s_client = client.ApiClient()

        # Segregate resources using Pydantic models
        nodes_to_create: List[K8sNode] = []
        pods_to_create: List[K8sPod] = []

        for item in data:
            if isinstance(item, K8sNode):
                nodes_to_create.append(item)
            elif isinstance(item, K8sPod):
                pods_to_create.append(item)
            else:
                raise KWOKError(
                    f"Unsupported resource type: {type(item).__name__}. "
                    f"Expected K8sNode or K8sPod, got {type(item)}."
                )

        valid_types = get_valid_instance_types()

        # Get currently active nodes in this specific cluster
        current_nodes = self.get_nodes()
        active_node_names = {n["name"] for n in current_nodes if n["cluster"] == cluster_name}

        print(f"[KWOK] Creating {len(nodes_to_create)} node(s) and {len(pods_to_create)} pod(s) in cluster '{cluster_name}'...")
        try:
            # Handle Nodes first (Validation + Injection + Creation)
            active_node_names = self._create_nodes(nodes_to_create, valid_types, active_node_names, k8s_client)

            # Handle Pods second (Validation + Creation)
            self._create_pods(pods_to_create, active_node_names, k8s_client)

            return True
        except Exception as e:
            print(f"[KWOK] Failed to create resources in cluster '{cluster_name}': {e}")
            raise

    def delete_from_dict(self, data: List[DeleteResource], region: str):
        """
        Delete resources from a regional cluster using a list of dicts or DeleteResource models.
        """
        if not isinstance(data, list):
            raise ValueError("Input data must be a list of dicts or DeleteResource models")

        cluster_name = self._full_name(region)
        self._load_config(cluster_name)

        print(f"[KWOK] Deleting resources from cluster '{cluster_name}'...")
        for item in data:
            try:
                if isinstance(item, DeleteNode):
                    self.delete_node(item.name, region)
                elif isinstance(item, DeletePod):
                    self.delete_pod(item.name, region)
                else:
                    raise KWOKError(f"Unsupported delete type: {type(item).__name__}")
            except Exception as e:
                print(f"[KWOK] Failed to delete resource: {e}")

    def get_clusters(self):
        """Return a list of all active regional clusters using kwokctl."""
        try:
            # kwokctl get clusters returns one name per line by default
            result = subprocess.run(["kwokctl", "get", "clusters"], capture_output=True, text=True, check=True)
            clusters = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return [{"name": c, "region": c} for c in clusters]
        except Exception as e:
            print(f"Failed to get clusters from kwokctl: {e}")
            # Fallback to tracked clusters if CLI fails
            return [{"name": c, "region": c} for c in self.active_clusters]


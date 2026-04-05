import subprocess
import atexit
from kubernetes import client

from .config import CLUSTER_CONFIG, get_cluster_config, load_kwok_kubeconfig
from .node import Node
from .pod import Pod
from .job import Job
from .deployment import Deployment
from .stage import Stage


# Maps config dict keys → kwokctl CLI flag names
_FLAG_MAP = {
    "kubeApiserverPort": "--kube-apiserver-port",
}


def _config_to_flags(config: dict) -> list[str]:
    """
    Translate a cluster config dict into kwokctl CLI flag pairs.
    Only keys present in _FLAG_MAP are emitted; 'region' and 'labels'
    are metadata and do not map to flags.
    """
    flags = []
    for key, flag in _FLAG_MAP.items():
        if key in config:
            flags.extend([flag, str(config[key])])
    return flags


class Cluster:
    """
    Manages a kwok-simulated Kubernetes cluster lifecycle.
    Configuration is sourced from CLUSTER_CONFIG in config.py — no YAML files.

    Example:
        cluster = Cluster(name="my-cluster", region="us-east")
        cluster.create()
        cluster.stop()
        cluster.delete()
    """

    def __init__(self, region: str = "default", name: str | None = None):
        self.region = region
        self.name = name or region
        self.config = get_cluster_config(region)
        
        # Attach user-requested metadata to the cluster object
        self.metadata = self.config.get("labels", {}).copy()
        self.metadata["region"] = self.region
        
        # Internal state tracking mapped by name -> Python Object Reference
        self.kwok_nodes = {}
        self.kwok_pods = {}
        self.kwok_jobs = {}
        self.kwok_deployments = {}
        
        # Automatically stop the cluster when the Python process exits
        atexit.register(self._atexit_cleanup)

    def _atexit_cleanup(self):
        """Called automatically at process exit — stops the cluster gracefully."""
        print(f"[atexit] Stopping cluster '{self.name}'...")
        try:
            subprocess.run(
                ["kwokctl", "stop", "cluster", "--name", self.name],
                check=False,  # don't raise — we're already exiting
                capture_output=True,
            )
        except Exception:
            pass  # best-effort cleanup

    def exists(self) -> bool:
        """Return True if a cluster with this name already exists in kwokctl."""
        result = subprocess.run(
            ["kwokctl", "get", "clusters"],
            capture_output=True,
            text=True,
            check=False,
        )
        return self.name in result.stdout.splitlines()

    def create(self):
        """Create the kwok cluster using the region config dict."""
        if self.exists():
            raise RuntimeError(
                f"Cluster '{self.name}' is already running. "
                "Call stop() or delete() before creating a new one."
            )
        extra_flags = _config_to_flags(self.config)
        cmd = ["kwokctl", "create", "cluster", "--name", self.name] + extra_flags

        print(f"Creating cluster '{self.name}' (region={self.config['region']}, "
              f"port={self.config.get('kubeApiserverPort', 'random')})...")
        print(f"Executing: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print(f"Cluster '{self.name}' created successfully!")

        except subprocess.CalledProcessError as e:
            print(f"Failed to create cluster: {e}")
            raise

    def stop(self):
        """Stop the kwok cluster without deleting it."""
        cmd = ["kwokctl", "stop", "cluster", "--name", self.name]
        print(f"Stopping cluster '{self.name}'...")
        try:
            subprocess.run(cmd, check=True)
            print(f"Cluster '{self.name}' stopped.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to stop cluster: {e}")
            raise

    def delete(self):
        """Delete the kwok cluster."""
        cmd = ["kwokctl", "delete", "cluster", "--name", self.name]
        print(f"Deleting cluster '{self.name}'...")
        try:
            subprocess.run(cmd, check=True)
            print(f"Cluster '{self.name}' deleted.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to delete cluster: {e}")
            raise

    def add_node(self, name: str, instance_type: str = "m5.large", capacity_type: str = "ON_DEMAND"):
        """Create and add a node to the cluster."""
        node = Node(
            name=name,
            instance_type=instance_type,
            region=self.region,
            cluster_name=self.name,
            capacity_type=capacity_type,
            on_delete_callback=lambda n: self.kwok_nodes.pop(n, None)
        )
        node.create()
        self.kwok_nodes[name] = node
        return node

    def add_pod(
        self,
        name: str,
        node_name: str,
        namespace: str = "default",
        cpu: str = "100m",
        memory: str = "128Mi",
        annotations: dict | None = None,
        restart_policy: str = "Always",
    ):
        """Create and add a pod to the cluster."""
        pod = Pod(
            name=name,
            namespace=namespace,
            node_name=node_name,
            cpu=cpu,
            memory=memory,
            cluster_name=self.name,
            annotations=annotations,
            restart_policy=restart_policy,
            on_delete_callback=lambda n: self.kwok_pods.pop(n, None)
        )
        pod.create()
        self.kwok_pods[name] = pod
        return pod

    def add_job(
        self,
        name: str,
        node_name: str,
        *,
        active_deadline_seconds: int | None = None,  # None = no hard deadline
        namespace: str = "default",
        cpu: str = "100m",
        memory: str = "128Mi",
        completions: int = 1,
        parallelism: int = 1,
        ttl_seconds_after_finished: int | None = 86400,
        backoff_limit: int = 6,
        annotations: dict | None = None,
        duration: str | None = None,
        jitter: str | None = None,
    ):
        """Create and add a job to the cluster."""
        job = Job(
            name=name,
            namespace=namespace,
            node_name=node_name,
            cpu=cpu,
            memory=memory,
            completions=completions,
            parallelism=parallelism,
            cluster_name=self.name,
            active_deadline_seconds=active_deadline_seconds,
            ttl_seconds_after_finished=ttl_seconds_after_finished,
            backoff_limit=backoff_limit,
            annotations=annotations,
            on_delete_callback=lambda n: self.kwok_jobs.pop(n, None),
            duration=duration,
            jitter=jitter,
        )
        job.create()
        self.kwok_jobs[name] = job
        return job

    def add_deployment(
        self,
        name: str,
        namespace: str = "default",
        replicas: int = 1,
        cpu: str = "100m",
        memory: str = "128Mi",
        image: str = "fake-image:v1",
    ):
        """Create and add an apps/v1 Deployment to natively manage identical replicas."""
        deploy = Deployment(
            name=name,
            namespace=namespace,
            replicas=replicas,
            cpu=cpu,
            memory=memory,
            image=image,
            cluster_name=self.name,
            on_delete_callback=lambda n: self.kwok_deployments.pop(n, None)
        )
        deploy.create()
        self.kwok_deployments[name] = deploy
        return deploy

    def get_nodes(self) -> list[dict]:
        """
        Return a list of nodes running in this cluster.

        Each entry contains:
            name, status (Ready/NotReady), instance_type, region, cpu, memory, pods_capacity
        """
        load_kwok_kubeconfig(self.name)
        v1 = client.CoreV1Api()
        nodes = []
        for n in v1.list_node().items:
            ready = any(
                c.type == "Ready" and c.status == "True"
                for c in (n.status.conditions or [])
            )
            nodes.append({
                "name":          n.metadata.name,
                "status":        "Ready" if ready else "NotReady",
                "instance_type": n.metadata.labels.get("node.kubernetes.io/instance-type"),
                "region":        n.metadata.labels.get("topology.kubernetes.io/region"),
                "cpu":           n.status.capacity.get("cpu") if n.status.capacity else None,
                "memory":        n.status.capacity.get("memory") if n.status.capacity else None,
                "pods_capacity": n.status.capacity.get("pods") if n.status.capacity else None,
            })
        return nodes

    def get_pods(self, namespace: str = "") -> list[dict]:
        """
        Return a list of pods in this cluster.

        Args:
            namespace: filter by namespace; empty string returns pods from all namespaces.

        Each entry contains:
            name, namespace, phase, node_name, containers
        """
        load_kwok_kubeconfig(self.name)
        v1 = client.CoreV1Api()
        pod_list = (
            v1.list_namespaced_pod(namespace)
            if namespace
            else v1.list_pod_for_all_namespaces()
        )
        pods = []
        for p in pod_list.items:
            pods.append({
                "name":       p.metadata.name,
                "namespace":  p.metadata.namespace,
                "phase":      p.status.phase,
                "node_name":  p.spec.node_name,
                "containers": [c.name for c in (p.spec.containers or [])],
            })
        return pods

    def get_jobs(self, namespace: str = "") -> list[dict]:
        """
        Return a list of batch jobs in this cluster.

        Args:
            namespace: filter by namespace; empty string returns jobs from all namespaces.

        Each entry contains:
            name, namespace, completions, parallelism, succeeded, failed, active, status
        """
        load_kwok_kubeconfig(self.name)
        batch = client.BatchV1Api()
        job_list = (
            batch.list_namespaced_job(namespace)
            if namespace
            else batch.list_job_for_all_namespaces()
        )
        jobs = []
        for j in job_list.items:
            conditions = j.status.conditions or []
            status = next(
                (c.type for c in conditions if c.status == "True"),
                "Running" if (j.status.active or 0) > 0 else "Pending",
            )
            jobs.append({
                "name":        j.metadata.name,
                "namespace":   j.metadata.namespace,
                "completions": j.spec.completions,
                "parallelism": j.spec.parallelism,
                "succeeded":   j.status.succeeded or 0,
                "failed":      j.status.failed or 0,
                "active":      j.status.active or 0,
                "status":      status,
            })
        return jobs

    def wait_for_job(
        self,
        name: str,
        namespace: str = "default",
        timeout: int = 120,
        poll_interval: float = 1.0,
    ) -> dict:
        """
        Block until the named job reaches 'Complete' or 'Failed', then return
        the final job status dict.  Raises TimeoutError if *timeout* seconds
        elapse first.

        Args:
            name:          Job name to watch.
            namespace:     Namespace the job lives in.
            timeout:       Max seconds to wait (default 120 s).
            poll_interval: Seconds between polls (default 1 s).

        Returns:
            The job status dict as returned by get_jobs().
        """
        import time as _time
        deadline = _time.monotonic() + timeout
        terminal = {"Complete", "Failed"}
        while _time.monotonic() < deadline:
            for j in self.get_jobs(namespace):
                if j["name"] == name and j["status"] in terminal:
                    return j
            _time.sleep(poll_interval)
        raise TimeoutError(
            f"Job '{name}' did not reach Complete/Failed within {timeout}s"
        )

    def __repr__(self):
        return f"Cluster(name={self.name!r}, region={self.region!r})"


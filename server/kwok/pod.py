from kubernetes import client, utils
from .config import load_kwok_kubeconfig

COMPLETE_TRIGGER_ANNOTATION = "kwok.x-k8s.io/trigger-complete"

class PodDict(dict):
    """Generates the Kubernetes dictionary representation of a Pod."""
    def __init__(self, name: str, namespace: str, node_name: str, cpu: str, memory: str, annotations: dict | None = None, restart_policy: str = "Always"):
        super().__init__({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "labels": {
                    "app": name,
                },
                **(
                    {"annotations": annotations}
                    if annotations else {}
                ),
            },
            "spec": {
                "nodeName": node_name,
                "restartPolicy": restart_policy,
                "automountServiceAccountToken": False,
                "containers": [
                    {
                        "name": "fake-container",
                        "image": "fake-image",
                        "resources": {
                            "requests": {"cpu": cpu, "memory": memory},
                            "limits":   {"cpu": cpu, "memory": memory},
                        },
                    }
                ],
            },
        })


class Pod:
    """
    Represents a simulated kwok Kubernetes pod.

    Example:
        pod = Pod(name="my-pod", node_name="kwok-node-0", cpu="100m", memory="128Mi")
        pod.create()
    """

    def __init__(
        self,
        name: str = "kwok-pod-0",
        namespace: str = "default",
        node_name: str = "kwok-node-0",
        cpu: str = "100m",
        memory: str = "128Mi",
        cluster_name: str = "kwok-cluster",
        annotations: dict | None = None,
        restart_policy: str = "Always",
        on_delete_callback = None,
    ):
        self.name = name
        self.namespace = namespace
        self.node_name = node_name
        self.cluster_name = cluster_name
        self.annotations = annotations
        self.restart_policy = restart_policy
        self.on_delete_callback = on_delete_callback

        self._dict = PodDict(name, namespace, node_name, cpu, memory, annotations, restart_policy)

    @property
    def spec(self) -> dict:
        """Return the pod spec dict."""
        return self._dict

    def create(self):
        """Apply this pod to the kwok cluster."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        k8s_client = client.ApiClient()
        print(f"Creating pod '{self.name}' on node '{self.node_name}'...")
        try:
            utils.create_from_dict(k8s_client, self._dict)
            print(f"Pod '{self.name}' created successfully!")
        except Exception as e:
            print(f"Failed to create pod: {e}")
            raise
    

    def complete(self):
        """Forces the pod into a Succeeded (Complete) state immediately."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        v1 = client.CoreV1Api()
        try:
            v1.patch_namespaced_pod_status(
                self.name,
                self.namespace,
                {"status": {"phase": "Succeeded", "reason": "Completed"}}
            )
            print(f"Pod '{self.name}' status changed to Succeeded")
        except Exception as e:
            print(f"Failed to change pod '{self.name}' status to Succeeded: {e}")
            raise

    def delete(self):
        """Delete this pod from the kwok cluster."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        v1 = client.CoreV1Api()
        print(f"Deleting pod '{self.name}' in namespace '{self.namespace}'...")
        try:
            v1.delete_namespaced_pod(self.name, self.namespace)
            print(f"Pod '{self.name}' deleted.")
            if self.on_delete_callback:
                self.on_delete_callback(self.name)
        except Exception as e:
            print(f"Failed to delete pod: {e}")
            raise

    def simulate_failure(self):
        """Forces the pod into a Failed state immediately."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return
            
        v1 = client.CoreV1Api()
        try:
            v1.patch_namespaced_pod_status(
                self.name, 
                self.namespace, 
                {"status": {"phase": "Failed", "reason": "SimulatedFailure"}}
            )
            print(f"Simulated failure for pod '{self.name}'")
        except Exception as e:
            print(f"Failed to simulate failure for pod '{self.name}': {e}")

    def __repr__(self):
        return f"Pod(name={self.name!r}, namespace={self.namespace!r}, node_name={self.node_name!r})"
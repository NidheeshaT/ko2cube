"""
KWOK Adapter — Pod YAML template builder.

Builds Kubernetes Pod manifests from Ko2cube Job + JobAssignment objects.
Pods are scheduled to matching fake KWOK nodes via nodeSelector.
"""
from typing import Optional


def build_pod_manifest(
    job_id: str,
    cpu_cores: float,
    memory_gb: float,
    region: str,
    instance_type: str,
    machine_type: str,
    eta_minutes: Optional[int],
    namespace: str = "ko2cube",
) -> dict:
    """
    Build a minimal K8s Pod manifest for a Ko2cube job.

    The pod runs a pause container (zero real compute).
    KWOK intercepts it and simulates the node binding without any kubelet.
    """
    # K8s resource strings
    cpu_str = f"{int(cpu_cores * 1000)}m" if cpu_cores > 0 else "100m"
    mem_str = f"{int(memory_gb * 1024)}Mi"

    # Sanitize pod name (k8s names: lowercase alphanumeric + hyphens)
    pod_name = job_id.lower().replace("_", "-")

    manifest: dict = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": namespace,
            "labels": {
                "app": "ko2cube-job",
                "ko2cube/job-id": job_id,
                "ko2cube/region": region,
                "ko2cube/instance-type": instance_type,
                "ko2cube/machine-type": machine_type,
            },
        },
        "spec": {
            # Pin pod to the matching fake KWOK node
            "nodeSelector": {
                "ko2cube/region": region,
                "ko2cube/instance-type": instance_type,
            },
            # Tolerate the KWOK taint so pod binds to virtual nodes
            "tolerations": [
                {
                    "key": "kwok.x-k8s.io/node",
                    "operator": "Exists",
                    "effect": "NoSchedule",
                }
            ],
            # Kill pod automatically when eta expires
            "activeDeadlineSeconds": (eta_minutes * 60) if eta_minutes else None,
            "restartPolicy": "Never",
            "containers": [
                {
                    "name": "job",
                    # pause image — does nothing, uses near-zero resources
                    "image": "registry.k8s.io/pause:3.9",
                    "resources": {
                        "requests": {"cpu": cpu_str, "memory": mem_str},
                        "limits":   {"cpu": cpu_str, "memory": mem_str},
                    },
                }
            ],
        },
    }

    # Remove None values (activeDeadlineSeconds must be omitted if null)
    if manifest["spec"]["activeDeadlineSeconds"] is None:
        del manifest["spec"]["activeDeadlineSeconds"]

    return manifest

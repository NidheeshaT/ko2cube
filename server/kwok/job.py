from kubernetes import client, utils
from .config import load_kwok_kubeconfig


class JobDict(dict):
    """Generates the Kubernetes dictionary representation of a Job."""
    def __init__(
        self,
        name: str,
        namespace: str,
        node_name: str,
        cpu: str,
        memory: str,
        completions: int,
        parallelism: int,
        active_deadline_seconds: int,
        ttl_seconds_after_finished: int | None = 86400,  # Default to 24 hours
        backoff_limit: int = 6,
        annotations: dict | None = None,
        pod_annotations: dict | None = None,
    ):
        # pod_annotations are placed on spec.template.metadata so KWOK Stages
        # can read them from child Pods (Job-level annotations are NOT inherited).
        super().__init__({
            "apiVersion": "batch/v1",
            "kind": "Job",
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
                "completions": completions,
                "parallelism": parallelism,
                "backoffLimit": backoff_limit,
                **(  # max wall-clock time before the job is killed
                    {"activeDeadlineSeconds": active_deadline_seconds}
                    if active_deadline_seconds is not None else {}
                ),
                **(  # seconds after completion before auto-cleanup
                    {"ttlSecondsAfterFinished": ttl_seconds_after_finished}
                    if ttl_seconds_after_finished is not None else {}
                ),
                "template": {
                    "metadata": {
                        "labels": {"app": name},
                        # Annotations here ARE visible on child Pods — required
                        # so the KWOK pod-complete Stage can read ko2cube.io/duration.
                        **(
                            {"annotations": pod_annotations}
                            if pod_annotations else {}
                        ),
                    },
                    "spec": {
                        "nodeName": node_name,
                        "restartPolicy": "Never",
                        "automountServiceAccountToken": False,
                        "containers": [
                            {
                                "name": "fake-job-container",
                                "image": "fake-image",
                                "resources": {
                                    "requests": {"cpu": cpu, "memory": memory},
                                    "limits":   {"cpu": cpu, "memory": memory},
                                },
                            }
                        ],
                    },
                },
            },
        })


class Job:
    """
    Represents a simulated Kubernetes batch Job on a kwok cluster.

    Example:
        job = Job(name="train-job", node_name="kwok-node-0", cpu="500m", memory="512Mi")
        job.create()
        job.delete()
    """

    def __init__(
        self,
        name: str = "kwok-job-0",
        namespace: str = "default",
        node_name: str = "kwok-node-0",
        cpu: str = "100m",
        memory: str = "128Mi",
        completions: int = 1,
        parallelism: int = 1,
        cluster_name: str = "kwok-cluster",
        *,
        active_deadline_seconds: int | None = None,  # None = no hard deadline
        ttl_seconds_after_finished: int | None = 86400,  # Default to 24 hours
        backoff_limit: int = 6,
        annotations: dict | None = None,
        on_delete_callback = None,
        duration: str | None = None,
        jitter: str | None = None,
    ):
        self.name = name
        self.namespace = namespace
        self.node_name = node_name
        self.cluster_name = cluster_name
        self.backoff_limit = backoff_limit
        self.on_delete_callback = on_delete_callback

        # Build job-level annotations (for human-readable metadata)
        self.annotations = dict(annotations or {})
        if duration:
            self.annotations["pod-complete.stage.kwok.x-k8s.io/delay"] = duration
        if jitter:
            self.annotations["pod-complete.stage.kwok.x-k8s.io/jitter-delay"] = jitter

        # Pod template annotations must carry the KWOK-native delay/jitter so that the
        # KWOK pod-complete Stage can read the annotation from the child Pod.
        pod_annotations = dict(annotations or {})
        if duration:
            pod_annotations["pod-complete.stage.kwok.x-k8s.io/delay"] = duration
        if jitter:
            pod_annotations["pod-complete.stage.kwok.x-k8s.io/jitter-delay"] = jitter

        self._dict = JobDict(
            name,
            namespace,
            node_name,
            cpu,
            memory,
            completions,
            parallelism,
            active_deadline_seconds,
            ttl_seconds_after_finished,
            backoff_limit,
            self.annotations if self.annotations else None,
            pod_annotations if pod_annotations else None,
        )

    @property
    def spec(self) -> dict:
        """Return the job spec dict."""
        return self._dict

    def create(self):
        """Apply this job to the kwok cluster."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        k8s_client = client.ApiClient()
        print(f"Creating job '{self.name}' on node '{self.node_name}'...")
        try:
            utils.create_from_dict(k8s_client, self._dict)
            print(f"Job '{self.name}' created successfully!")
        except Exception as e:
            print(f"Failed to create job: {e}")
            raise

    def delete(self):
        """Delete this job from the kwok cluster."""
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        batch_v1 = client.BatchV1Api()
        print(f"Deleting job '{self.name}' in namespace '{self.namespace}'...")
        try:
            batch_v1.delete_namespaced_job(
                self.name,
                self.namespace,
                propagation_policy="Foreground",  # also cleans up child pods
            )
            print(f"Job '{self.name}' deleted.")
            if self.on_delete_callback:
                self.on_delete_callback(self.name)
        except Exception as e:
            print(f"Failed to delete job: {e}")
            raise

    def simulate_failure(self):
        """Forces the Job into a Failed condition immediately."""
        from datetime import datetime, timezone
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            return

        batch_v1 = client.BatchV1Api()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            batch_v1.patch_namespaced_job_status(self.name, self.namespace, {
                "status": {
                    "conditions": [
                        {
                            "type": "Failed", 
                            "status": "True", 
                            "reason": "SimulatedFailure", 
                            "lastTransitionTime": now
                        }
                    ]
                }
            })
            print(f"Simulated failure for job '{self.name}'")
        except Exception as e:
            print(f"Failed to simulate failure for job '{self.name}': {e}")

    def __repr__(self):
        return f"Job(name={self.name!r}, namespace={self.namespace!r}, node_name={self.node_name!r})"

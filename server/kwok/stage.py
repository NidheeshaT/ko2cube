from kubernetes import client
from .config import load_kwok_kubeconfig


# Well-known KWOK stage names from kustomize/stage/pod/general
STAGE_POD_READY = "pod-ready"
STAGE_POD_COMPLETE = "pod-complete"
STAGE_POD_DELETE = "pod-delete"

KWOK_API_GROUP = "kwok.x-k8s.io"
KWOK_API_VERSION = "v1alpha1"
KWOK_STAGE_PLURAL = "stages"


class StageDict(dict):
    """
    Generates the Kubernetes dictionary representation of a KWOK Stage CRD.

    A Stage defines a lifecycle transition for a simulated pod/node.
    KWOK watches for the selector conditions and applies the statusTemplate
    when they are met.
    """

    def __init__(
        self,
        name: str,
        resource_kind: str = "Pod",
        resource_api_group: str = "v1",
        selector_match_labels: dict | None = None,
        selector_match_annotations: dict | None = None,
        selector_expressions: list[dict] | None = None,
        delay_ms: int = 0,
        jitter_ms: int | None = None,
        status_template: str | None = None,
        event_type: str | None = None,
        event_reason: str | None = None,
        event_message: str | None = None,
        delete: bool = False,
        immediate_next_stage: bool = False,
        weight: int | None = None,
    ):
        selector = {}
        if selector_match_labels:
            selector["matchLabels"] = selector_match_labels
        if selector_match_annotations:
            selector["matchAnnotations"] = selector_match_annotations
        if selector_expressions:
            selector["matchExpressions"] = selector_expressions

        delay = {"durationMilliseconds": delay_ms}
        if jitter_ms is not None:
            delay["jitterDurationMilliseconds"] = jitter_ms

        next_stage: dict = {}
        if status_template:
            next_stage["statusTemplate"] = status_template
        if delete:
            next_stage["delete"] = True
        if event_type and event_reason and event_message:
            next_stage["event"] = {
                "type": event_type,
                "reason": event_reason,
                "message": event_message,
            }

        spec: dict = {
            "resourceRef": {
                "apiGroup": resource_api_group,
                "kind": resource_kind,
            },
            "selector": selector,
            "delay": delay,
            "next": next_stage,
        }
        if immediate_next_stage:
            spec["immediateNextStage"] = True
        if weight is not None:
            spec["weight"] = weight

        super().__init__({
            "apiVersion": f"{KWOK_API_GROUP}/{KWOK_API_VERSION}",
            "kind": "Stage",
            "metadata": {"name": name},
            "spec": spec,
        })


class Stage:
    """
    Represents a KWOK Stage custom resource that defines a pod/node lifecycle transition.

    Example:
        stage = Stage.pod_complete_on_annotation(cluster_name="kwok-cluster")
        stage.create()
    """

    def __init__(
        self,
        name: str,
        cluster_name: str = "kwok-cluster",
        resource_kind: str = "Pod",
        resource_api_group: str = "v1",
        selector_match_labels: dict | None = None,
        selector_match_annotations: dict | None = None,
        selector_expressions: list[dict] | None = None,
        delay_ms: int = 0,
        jitter_ms: int | None = None,
        status_template: str | None = None,
        event_type: str | None = None,
        event_reason: str | None = None,
        event_message: str | None = None,
        delete: bool = False,
        immediate_next_stage: bool = False,
        weight: int | None = None,
    ):
        self.name = name
        self.cluster_name = cluster_name

        self._dict = StageDict(
            name=name,
            resource_kind=resource_kind,
            resource_api_group=resource_api_group,
            selector_match_labels=selector_match_labels,
            selector_match_annotations=selector_match_annotations,
            selector_expressions=selector_expressions,
            delay_ms=delay_ms,
            jitter_ms=jitter_ms,
            status_template=status_template,
            event_type=event_type,
            event_reason=event_reason,
            event_message=event_message,
            delete=delete,
            immediate_next_stage=immediate_next_stage,
            weight=weight,
        )

    # ------------------------------------------------------------------
    # Factory methods — pre-built stages matching pod/general behaviour
    # ------------------------------------------------------------------

    @classmethod
    def pod_complete_on_annotation(
        cls,
        cluster_name: str = "kwok-cluster",
        annotation_key: str = "kwok.x-k8s.io/trigger-complete",
        delay_ms: int = 0,
    ) -> "Stage":
        """
        Stage that moves a Running pod to Succeeded when the trigger
        annotation is patched onto it.

        Pair with pod.complete() which patches the annotation.
        """
        return cls(
            name="pod-complete-on-demand",
            cluster_name=cluster_name,
            selector_match_annotations={annotation_key: "true"},
            selector_expressions=[
                {
                    "key": ".status.phase",
                    "operator": "In",
                    "values": ["Running"],
                }
            ],
            delay_ms=delay_ms,
            event_type="Normal",
            event_reason="Completed",
            event_message="Pod completed via on-demand annotation trigger",
            status_template="""
phase: Succeeded
conditions:
  - type: Ready
    status: "False"
    reason: PodCompleted
containerStatuses:
  {{ range .spec.containers }}
  - name: {{ .name }}
    ready: false
    state:
      terminated:
        exitCode: 0
        reason: Completed
        finishedAt: {{ now | format "2006-01-02T15:04:05.000000000Z" }}
  {{ end }}
""".strip(),
        )

    @classmethod
    def pod_ready(
        cls,
        cluster_name: str = "kwok-cluster",
        match_labels: dict | None = None,
        delay_ms: int = 1000,
        jitter_ms: int = 3000,
    ) -> "Stage":
        """
        Stage that moves a Pending pod to Running/Ready after a delay.
        Mirrors the pod/general pod-ready stage.
        """
        return cls(
            name="pod-ready-general",
            cluster_name=cluster_name,
            selector_match_labels=match_labels,
            selector_expressions=[
                {
                    "key": ".status.phase",
                    "operator": "NotIn",
                    "values": ["Running", "Succeeded", "Failed", "Unknown"],
                }
            ],
            delay_ms=delay_ms,
            jitter_ms=jitter_ms,
            event_type="Normal",
            event_reason="Started",
            event_message="Pod is now Running",
            immediate_next_stage=True,
            status_template="""
phase: Running
conditions:
  - type: Ready
    status: "True"
  - type: ContainersReady
    status: "True"
  - type: Initialized
    status: "True"
  - type: PodScheduled
    status: "True"
containerStatuses:
  {{ range .spec.containers }}
  - name: {{ .name }}
    ready: true
    started: true
    state:
      running:
        startedAt: {{ now | format "2006-01-02T15:04:05.000000000Z" }}
  {{ end }}
""".strip(),
        )

    @classmethod
    def pod_delete(
        cls,
        cluster_name: str = "kwok-cluster",
        match_labels: dict | None = None,
    ) -> "Stage":
        """
        Stage that handles graceful pod deletion when deletionTimestamp is set.
        Mirrors the pod/general pod-delete stage.
        """
        return cls(
            name="pod-delete-general",
            cluster_name=cluster_name,
            selector_match_labels=match_labels,
            selector_expressions=[
                {
                    "key": ".metadata.deletionTimestamp",
                    "operator": "Exists",
                }
            ],
            delete=True,
            event_type="Normal",
            event_reason="Killing",
            event_message="Pod deleted",
        )

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    @property
    def spec(self) -> dict:
        """Return the Stage spec dict."""
        return self._dict

    def _get_custom_api(self) -> client.CustomObjectsApi:
        try:
            load_kwok_kubeconfig(self.cluster_name)
        except Exception as e:
            print(e)
            raise
        return client.CustomObjectsApi()

    def create(self):
        """Apply this Stage to the kwok cluster."""
        api = self._get_custom_api()
        print(f"Creating stage '{self.name}'...")
        try:
            api.create_cluster_custom_object(
                group=KWOK_API_GROUP,
                version=KWOK_API_VERSION,
                plural=KWOK_STAGE_PLURAL,
                body=self._dict,
            )
            print(f"Stage '{self.name}' created successfully!")
        except Exception as e:
            print(f"Failed to create stage '{self.name}': {e}")
            raise

    def delete(self):
        """Delete this Stage from the kwok cluster."""
        api = self._get_custom_api()
        print(f"Deleting stage '{self.name}'...")
        try:
            api.delete_cluster_custom_object(
                group=KWOK_API_GROUP,
                version=KWOK_API_VERSION,
                plural=KWOK_STAGE_PLURAL,
                name=self.name,
            )
            print(f"Stage '{self.name}' deleted.")
        except Exception as e:
            print(f"Failed to delete stage '{self.name}': {e}")
            raise

    def apply(self):
        """
        Create or replace the Stage (idempotent).
        Tries to create first; if it already exists, replaces it.
        """
        api = self._get_custom_api()
        try:
            existing = api.get_cluster_custom_object(
                group=KWOK_API_GROUP,
                version=KWOK_API_VERSION,
                plural=KWOK_STAGE_PLURAL,
                name=self.name,
            )
            # Preserve resourceVersion for the replace call
            self._dict["metadata"]["resourceVersion"] = (
                existing["metadata"]["resourceVersion"]
            )
            api.replace_cluster_custom_object(
                group=KWOK_API_GROUP,
                version=KWOK_API_VERSION,
                plural=KWOK_STAGE_PLURAL,
                name=self.name,
                body=self._dict,
            )
            print(f"Stage '{self.name}' updated.")
        except client.exceptions.ApiException as e:
            if e.status == 404:
                self.create()
            else:
                print(f"Failed to apply stage '{self.name}': {e}")
                raise

    def get(self) -> dict:
        """Fetch the current state of this Stage from the cluster."""
        api = self._get_custom_api()
        try:
            result = api.get_cluster_custom_object(
                group=KWOK_API_GROUP,
                version=KWOK_API_VERSION,
                plural=KWOK_STAGE_PLURAL,
                name=self.name,
            )
            return result
        except Exception as e:
            print(f"Failed to get stage '{self.name}': {e}")
            raise

    @staticmethod
    def list_all(cluster_name: str = "kwok-cluster") -> list[dict]:
        """List all Stages currently applied to the cluster."""
        try:
            load_kwok_kubeconfig(cluster_name)
        except Exception as e:
            print(e)
            raise
        api = client.CustomObjectsApi()
        try:
            result = api.list_cluster_custom_object(
                group=KWOK_API_GROUP,
                version=KWOK_API_VERSION,
                plural=KWOK_STAGE_PLURAL,
            )
            stages = result.get("items", [])
            print(f"Found {len(stages)} stage(s).")
            return stages
        except Exception as e:
            print(f"Failed to list stages: {e}")
            raise

    def __repr__(self):
        return f"Stage(name={self.name!r}, cluster_name={self.cluster_name!r})"
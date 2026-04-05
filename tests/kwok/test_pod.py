"""
tests/kwok/test_pod.py

Integration tests for Pod scheduling, phase transitions, and failure/restart.
"""

import time
import pytest


pytestmark = pytest.mark.integration


def _wait_for_pod_phase(cluster, name: str, namespace: str, phase: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pods = cluster.get_pods(namespace)
        pod = next((p for p in pods if p["name"] == name), None)
        if pod and pod["phase"] == phase:
            return True
        time.sleep(1)
    return False


class TestPodIntegration:

    def test_pod_reaches_running_phase(self, kwok_cluster):
        from server.kwok.pod import Pod
        ns = kwok_cluster["namespace"]
        pod = Pod(name="int-pod-running", node_name=kwok_cluster["node_name"],
                  cpu="100m", memory="128Mi", cluster_name=kwok_cluster["cluster_name"])
        pod.create()
        cluster = kwok_cluster["cluster"]
        reached = _wait_for_pod_phase(cluster, "int-pod-running", ns, "Running", timeout=30)
        assert reached, "Pod did not reach Running within 30s"
        pod.delete()

    def test_pod_delete_removes_from_cluster(self, kwok_cluster):
        from server.kwok.pod import Pod
        ns = kwok_cluster["namespace"]
        pod = Pod(name="int-pod-delete", node_name=kwok_cluster["node_name"],
                  cpu="50m", memory="64Mi", cluster_name=kwok_cluster["cluster_name"])
        pod.create()
        pod.delete()

        cluster = kwok_cluster["cluster"]
        names = [p["name"] for p in cluster.get_pods(ns)]
        assert "int-pod-delete" not in names


"""
tests/kwok/conftest.py

Shared fixtures for kwok integration tests.
"""

import shutil
import pytest

CLUSTER_NAME = "us-east"
NODE_NAME    = "int-node-0"
NAMESPACE    = "default"


@pytest.fixture(scope="session", autouse=True)
def kwok_cluster():
    """
    Session-scoped fixture: creates a kwok cluster + one worker node
    before all integration tests, then cleans up after.
    """
    if not shutil.which("kwokctl"):
        pytest.skip("kwokctl not found on PATH — skipping integration tests")

    from server.kwok.cluster import Cluster
    from server.kwok.node import Node
    from server.kwok.config import load_kwok_kubeconfig

    cluster = Cluster(region="us-east")

    if cluster.exists():
        cluster.delete()

    cluster.create()

    node = Node(name=NODE_NAME, instance_type="m5.large",
                region="us-east", cluster_name=CLUSTER_NAME)
    node.create()

    load_kwok_kubeconfig(CLUSTER_NAME)

    # Kubernetes controller manager takes a moment to create the default service account.
    # Pod creation will fail with 'serviceaccount default/default not found' if we don't wait.
    import time
    import subprocess
    for _ in range(30):
        res = subprocess.run(
            ["kubectl", "get", "sa", "default", "-n", NAMESPACE, "--context", f"kwok-{CLUSTER_NAME}"],
            capture_output=True
        )
        if res.returncode == 0:
            break
        time.sleep(1)

    yield {"cluster": cluster, "node": node,
           "cluster_name": CLUSTER_NAME, "node_name": NODE_NAME,
           "namespace": NAMESPACE}

    try:
        node.delete()
    except Exception:
        pass
    cluster.delete()

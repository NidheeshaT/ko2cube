"""
tests/kwok/test_deployment.py

Integration tests for Deployment rollout and replica management.
"""

import pytest

pytestmark = pytest.mark.integration


class TestDeploymentIntegration:
    def test_deployment_scaling_and_rollout(self, kwok_cluster):
        cluster = kwok_cluster["cluster"]
        ns = kwok_cluster["namespace"]
        
        # 1. Create a deployment with 3 replicas
        deploy = cluster.add_deployment(
            name="test-deploy",
            namespace=ns,
            replicas=3,
            image="fake-image:v1"
        )
        
        # Wait for deployment to spawn 3 pods
        import time
        reached_3 = False
        for _ in range(15):
            pods = [p for p in cluster.get_pods(ns) if p["name"].startswith("test-deploy")]
            if len(pods) == 3:
                reached_3 = True
                break
            time.sleep(1)
        
        assert reached_3, "Deployment failed to scale up to 3 replicas"
        
        # 2. Patch the deployment with a new image (rollout)
        deploy.apply_new_rollout("fake-image:v2")
        
        # 3. Clean up
        deploy.delete()
        
        # 4. Verify tracking
        assert "test-deploy" not in cluster.kwok_deployments

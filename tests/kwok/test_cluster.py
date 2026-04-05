"""
tests/kwok/test_cluster.py

Integration tests for Cluster lifecycle.
"""

import pytest


pytestmark = pytest.mark.integration


class TestClusterIntegration:

    def test_cluster_exists_after_create(self, kwok_cluster):
        cluster = kwok_cluster["cluster"]
        assert cluster.exists(), "Cluster should exist after creation"

    def test_cluster_raises_on_duplicate_create(self, kwok_cluster):
        from server.kwok.cluster import Cluster
        dupe = Cluster(name=kwok_cluster["cluster_name"], region="us-east")
        with pytest.raises(RuntimeError, match="already running"):
            dupe.create()

    def test_api_server_reachable(self, kwok_cluster):
        """Our cluster helper should be able to reach the API server."""
        cluster = kwok_cluster["cluster"]
        cluster.get_nodes()  # raises if unreachable

    def test_multiple_clusters_cross_region(self):
        """Verify we can spin up multiple clusters with different regions smoothly."""
        from server.kwok.cluster import Cluster

        cluster_west = Cluster(region="us-west")
        cluster_eu = Cluster(region="eu-west")

        # Cleanup in case previous interrupted tests left them running
        try: cluster_west.delete()
        except: pass
        try: cluster_eu.delete()
        except: pass

        cluster_west.create()
        cluster_eu.create()

        assert cluster_west.exists()
        assert cluster_eu.exists()

        assert cluster_west.metadata["region"] == "us-west"
        assert cluster_eu.metadata["region"] == "eu-west"
        assert cluster_west.config["kubeApiserverPort"] != cluster_eu.config["kubeApiserverPort"]

        cluster_west.delete()
        cluster_eu.delete()

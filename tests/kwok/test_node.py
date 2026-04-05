"""
tests/kwok/test_node.py

Integration tests for Node creation and metadata.
"""

import pytest
from server.kwok.constants import EC2_INSTANCE_TYPES

pytestmark = pytest.mark.integration


class TestNodeIntegration:

    def test_node_appears_in_cluster_with_correct_metadata(self, kwok_cluster):
        cluster = kwok_cluster["cluster"]
        nodes = cluster.get_nodes()
        node_names = [n["name"] for n in nodes]
        
        assert kwok_cluster["node_name"] in node_names
        
        # Find the node dict for the test node
        node_dict = next(n for n in nodes if n["name"] == kwok_cluster["node_name"])
        
        # Verify labels and metadata
        assert node_dict["instance_type"] == "m5.large"
        assert node_dict["region"] == "us-east"
        
        # Verify capacity reflects instance type
        expected = EC2_INSTANCE_TYPES["m5.large"]
        assert node_dict["cpu"] == expected["cpu"]
        assert node_dict["memory"] == expected["memory"]

"""
KWOK Node Setup — Creates fake virtual nodes in the KWOK cluster.

Run this once before starting the environment server when KWOK_ENABLED=true.

Each Ko2cube region gets a set of virtual nodes (one per instance type).
Pods are pinned to nodes via nodeSelector on ko2cube/region and ko2cube/instance-type labels.

Usage:
    python -m server.kwok.node_setup          # sets up all nodes
    python -m server.kwok.node_setup --teardown  # deletes all ko2cube nodes
"""

import argparse
import json
import logging
import os
import subprocess
import sys

import yaml

log = logging.getLogger("kwok_node_setup")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Matches infrastructure.json exactly
REGIONS = ["us-east-1", "us-west-2", "eu-west-1"]
INSTANCE_TYPES = [
    {"name": "m5.large",   "cpu": "2",  "memory": "8Gi"},
    {"name": "m5.xlarge",  "cpu": "4",  "memory": "16Gi"},
    {"name": "m5.2xlarge", "cpu": "8",  "memory": "32Gi"},
    {"name": "m5.4xlarge", "cpu": "16", "memory": "64Gi"},
    {"name": "c5.large",   "cpu": "2",  "memory": "4Gi"},
    {"name": "c5.xlarge",  "cpu": "4",  "memory": "8Gi"},
    {"name": "r5.large",   "cpu": "2",  "memory": "16Gi"},
]

# Number of fake nodes per (region, instance_type) pair
NODES_PER_TYPE = int(os.environ.get("KWOK_NODES_PER_TYPE", "3"))


def build_node_manifest(region: str, instance: dict, index: int) -> dict:
    """Build a KWOK virtual Node manifest."""
    # e.g. ko2cube-us-east-1-m5-xlarge-0
    node_name = f"ko2cube-{region}-{instance['name'].replace('.', '-')}-{index}"

    return {
        "apiVersion": "v1",
        "kind": "Node",
        "metadata": {
            "name": node_name,
            "labels": {
                "type": "kwok",
                "ko2cube/region": region,
                "ko2cube/instance-type": instance["name"],
                # Standard K8s topology labels (good practice)
                "topology.kubernetes.io/region": region,
                "node.kubernetes.io/instance-type": instance["name"],
            },
            "annotations": {
                # Tells KWOK to manage this node
                "kwok.x-k8s.io/node": "fake",
            },
        },
        "spec": {
            # KWOK taint — prevents real pods from landing here accidentally
            "taints": [
                {
                    "key": "kwok.x-k8s.io/node",
                    "value": "fake",
                    "effect": "NoSchedule",
                }
            ],
        },
        "status": {
            "allocatable": {
                "cpu": instance["cpu"],
                "memory": instance["memory"],
                "pods": "110",
            },
            "capacity": {
                "cpu": instance["cpu"],
                "memory": instance["memory"],
                "pods": "110",
            },
            "conditions": [
                {
                    "type": "Ready",
                    "status": "True",
                    "reason": "KWOKNodeReady",
                    "message": "KWOK virtual node is always ready",
                }
            ],
        },
    }


def default_kubectl_bin() -> str:
    return os.environ.get("KWOK_KUBECTL", "kubectl")

def kubectl(args: list) -> bool:
    kubeconfig = os.environ.get("KWOK_KUBECONFIG", "")
    cmd = [default_kubectl_bin()]
    if kubeconfig:
        cmd += ["--kubeconfig", kubeconfig]
    cmd += args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("kubectl %s: %s", " ".join(args[:2]), result.stderr.strip())
    return result.returncode == 0


def kubectl_apply(manifest: dict) -> bool:
    kubeconfig = os.environ.get("KWOK_KUBECONFIG", "")
    cmd = [default_kubectl_bin()]
    if kubeconfig:
        cmd += ["--kubeconfig", kubeconfig]
    cmd += ["apply", "-f", "-"]
    yaml_str = yaml.dump(manifest, default_flow_style=False)
    result = subprocess.run(cmd, input=yaml_str, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("apply failed: %s", result.stderr.strip())
        return False
    log.info("  ✓ %s", result.stdout.strip())
    return True


def setup_nodes():
    log.info("Setting up KWOK virtual nodes for Ko2cube...")
    total = 0
    for region in REGIONS:
        for instance in INSTANCE_TYPES:
            for i in range(NODES_PER_TYPE):
                manifest = build_node_manifest(region, instance, i)
                if kubectl_apply(manifest):
                    total += 1
    log.info("Done — %d virtual nodes created across %d regions.", total, len(REGIONS))


def teardown_nodes():
    log.info("Tearing down all Ko2cube KWOK nodes...")
    kubectl([
        "delete", "nodes",
        "-l", "ko2cube/region",
        "--ignore-not-found=true",
    ])
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ko2cube KWOK node setup")
    parser.add_argument("--teardown", action="store_true", help="Delete all ko2cube nodes")
    args = parser.parse_args()

    if args.teardown:
        teardown_nodes()
    else:
        setup_nodes()

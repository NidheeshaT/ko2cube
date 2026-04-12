import json
from pathlib import Path

from ko2cube.server.kwok.error import NodeValidationError, PodValidationError, InstanceTypeError

# Path to the infrastructure configuration file
INFRA_PATH = Path(__file__).parent.parent / "data" / "infrastructure.json"

def get_infra_data():
    """Load and return the infrastructure configuration."""
    try:
        with open(INFRA_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading infrastructure.json: {e}")
        return {"regions": [], "instances": []}

def get_valid_instance_types():
    """Return a list of valid instance type names from infrastructure.json."""
    infra = get_infra_data()
    return [inst["name"] for inst in infra.get("instances", [])]

def validate_node_resource(resource: dict, valid_types: list):
    """
    Validate a Node resource dictionary.
    Ensures the node has a valid 'node.kubernetes.io/instance-type' label.
    Node name can be anything.
    """
    metadata = resource.get("metadata", {})
    name = metadata.get("name", "")
    
    if not name:
        raise NodeValidationError("Node resource must have a name in metadata")

    # 1. Validate labels
    labels = metadata.get("labels", {})
    instance_type_tag = labels.get("node.kubernetes.io/instance-type")
    if not instance_type_tag:
        raise NodeValidationError(f"Node '{name}' is missing required label 'node.kubernetes.io/instance-type'")
    
    # 2. Validate instance type against infrastructure.json
    if instance_type_tag not in valid_types:
        raise InstanceTypeError(f"Unknown instance type '{instance_type_tag}' in node '{name}' label")

def validate_pod_resource(resource: dict, active_node_names: set):
    """
    Validate a Pod resource dictionary.
    Checks for spec.nodeName and verifies it exists in the targeted cluster.
    """
    metadata = resource.get("metadata", {})
    name = metadata.get("name", "")
    spec = resource.get("spec", {})
    node_name = spec.get("nodeName")
    
    # Validate each container has resources
    containers = spec.get("containers", [])
    if not containers:
        raise PodValidationError(f"Pod '{name}' must have at least one container")
    
    for container in containers:
        if not container.get("resources"):
            raise PodValidationError(f"Container '{container.get('name')}' in pod '{name}' is missing mandatory 'resources' constraints")
    
    if not node_name:
        # If nodeName is not specified, we allow it (the K8s scheduler will handle assignment)
        return
    
    if node_name not in active_node_names:
        raise PodValidationError(f"Pod '{name}' assigned to non-existent node '{node_name}' in this cluster")

def get_instance_resources(instance_type: str) -> dict | None:
    """Return CPU and Memory strings for the specified instance type."""
    infra = get_infra_data()
    for inst in infra.get("instances", []):
        if inst["name"] == instance_type:
            return {
                "cpu": str(inst["cpu"]),
                "memory": f"{inst['mem']}Gi"
            }
    return None

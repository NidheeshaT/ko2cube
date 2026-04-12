from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union

class K8sMetadata(BaseModel):
    name: str
    namespace: Optional[str] = "default"
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)

class K8sNodeSpec(BaseModel):
    # Standard Node spec is complex; we use a basic version or Dict
    taints: Optional[List[dict]] = None
    unschedulable: Optional[bool] = None

class K8sContainer(BaseModel):
    name: str
    image: str
    resources: dict
    # Optional fields
    command: Optional[List[str]] = None
    args: Optional[List[str]] = None
    env: Optional[List[dict]] = None
    ports: Optional[List[dict]] = None

class K8sPodSpec(BaseModel):
    nodeName: Optional[str] = None
    containers: List[K8sContainer] = Field(..., description="List of containers in the pod")
    volumes: Optional[List[dict]] = None

class K8sNode(BaseModel):
    apiVersion: str = "v1"
    kind: str = "Node"
    metadata: K8sMetadata
    spec: Optional[K8sNodeSpec] = None
    # status is usually injected by the KWOK adapter, but included here for completeness
    status: Optional[dict] = None

class K8sPod(BaseModel):
    apiVersion: str = "v1"
    kind: str = "Pod"
    metadata: K8sMetadata
    spec: K8sPodSpec
    status: Optional[dict] = None

# Union type for the list items expected by create_from_dict
K8sResource = Union[K8sNode, K8sPod]

class DeleteNode(BaseModel):
    kind: str = "Node"
    name: str

class DeletePod(BaseModel):
    kind: str = "Pod"
    name: str

# Union type for the list items expected by delete_from_dict
DeleteResource = Union[DeleteNode, DeletePod]

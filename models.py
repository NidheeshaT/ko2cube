from typing import Union
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field

from openenv.core.env_server.types import State, Action, Observation

# Sub-models for Observation / State
class CarbonData(BaseModel):
    current_intensity: float = Field(..., description="Current carbon intensity (gCO2/kWh)")
    forecast: List[float] = Field(
        ...,
        description="Carbon intensity forecast for next N steps. Length matches scenario lookahead config."
    )

class InstanceType(BaseModel):
    name: str = Field(..., description="Instance type name, e.g. m5.xlarge")
    cpu_cores: float = Field(..., description="Available vCPUs on this instance")
    memory_gb: float = Field(..., description="Available memory in GB")
    spot_price: float = Field(..., description="Current hourly spot price for this type")
    on_demand_price: float = Field(..., description="Fixed hourly on-demand price for this type")
    available_count: int = Field(..., description="Number of free instances of this type right now")

class RegionInfo(BaseModel):
    region_name: str = Field(..., description="Region identifier")
    carbon: CarbonData
    available_instances: List[InstanceType] = Field(..., description="List of specific machine types available in this region")

ALWAYS_ON: int = -1  # sentinel for never-expiring jobs

class Job(BaseModel):
    job_id: str = Field(..., description="Unique identifier for the job")
    arrival_step: int = Field(default=0, description="Step at which this job appears in the queue")
    eta_minutes: Optional[int] = Field(None, description="Expected runtime in minutes. Null if always-on.")
    cpu_cores: float = Field(..., description="CPU cores required")
    memory_gb: float = Field(..., description="Memory required in GB")
    sla_start: int = Field(..., description="Earliest simulation step the job can begin")
    sla_end: int = Field(..., description="Latest step the job must finish. -1 (ALWAYS_ON) means never expires.")
    delay_tolerant: bool = Field(..., description="Whether the job can wait for a better carbon window without failing")
    instance_preference: Literal["spot", "on-demand"] = Field(..., description="Preferred instance lifecycle")
    
    # Status fields tracked by the simulator
    status: Literal["queued", "running", "completed", "dropped", "sla_missed", "deferred"] = Field(
        default="queued", description="Current status of the job in the system."
    )
    start_step: Optional[int] = Field(None, description="The step at which the job started running, if applicable.")
    completion_step: Optional[int] = Field(None, description="The step at which the job finished running, if applicable.")
    region: Optional[str] = Field(None, description="Selected region for this job.")
    machine_type: Optional[Literal["spot", "on-demand"]] = Field(None, description="Selected instance type for this job.")
    theoretical_min_carbon: Optional[float] = Field(0, description="Theoretical minimum carbon for this job.")
    theoretical_min_cost: Optional[float] = Field(0, description="Theoretical minimum cost for this job.")
    
    # Baseline metrics (computed over SLA window)
    baseline_carbon_intensity: float = Field(0.0, description="Average carbon intensity across all regions over the SLA window.")
    baseline_spot_price: float = Field(0.0, description="Average spot price across all regions over the SLA window.")

class RunningJob(BaseModel):
    job_id: str = Field(..., description="ID of the running job")
    region: str = Field(..., description="Region where the job is currently running")
    steps_remaining: int = Field(..., description="How many steps until this job completes")
    machine_type: str = Field(..., description="Machine type the job is using")

# Main Request / Response Models
class ClusterState(BaseModel):
    nodes: List[dict] = Field(default_factory=list)
    pods: List[dict] = Field(default_factory=list)

class Ko2cubeObservation(Observation):
    """
    The observation returned to the agent at every step.
    Contains the queue of jobs, and the state of the world to make carbon-aware decisions.
    """
    current_step: int = Field(..., description="Current simulation step")
    job_queue: List[Job] = Field(..., description="List of jobs currently waiting to be scheduled")
    active_jobs: List[RunningJob] = Field(..., description="List of jobs actively running in the simulated cluster")
    regions: Dict[str, RegionInfo] = Field(..., description="Data about carbon and pricing per region")
    infra_clusters: Dict[str, ClusterState] = Field(
        default_factory=dict, 
        description="Current K8s nodes and pods grouped by cluster name"
    )
    last_action_result: str = Field(
        default="",
        description="Human readable result of last action."
    )


class JobAssignment(BaseModel):
    job_id: str = Field(..., description="ID of the job you are taking action on")
    decision: Literal["schedule", "defer", "drop"] = Field(
        ..., 
        description="Whether to schedule the job now, defer it to a later step, or drop it entirely (violating SLA)."
    )
    region: Optional[str] = Field(
        None, 
        description="Target region if scheduling. Null if deferring or dropping."
    )
    machine_type: Optional[Literal["spot", "on-demand"]] = Field(
        None, 
        description="Target machine lifecycle (spot/on-demand). Note your choice impacts both price and reliability."
    )
    instance_type: Optional[str] = Field(
        None,
        description="Target instance type name (e.g. m5.xlarge) from the region's available_instances list."
    )
    defer_to_step: Optional[int] = Field(
        None,
        description="Step to retry scheduling. Required if decision=defer. Must be within job sla_end."
    )

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


class Ko2cubeAction(Action):
    """
    The action the agent provides. 
    It must contain a decision for EVERY job currently in the `job_queue`.
    """
    assignments: List[JobAssignment] = Field(
        ..., 
        description="List of decisions for the jobs in the queue."
    )
    resources_to_delete: Dict[str, List[DeleteResource]] = Field(
        default_factory=dict,
        description="List of resources(nodes/pods) to delete per region."
    )
    resources_to_create: Dict[str, List[K8sResource]] = Field(
        default_factory=dict,
        description="List of resources(nodes/pods) to create per region."
    )
        

# Internal State Model
class Ko2cubeState(State):
    """
    Internal State tracked by the simulator, kept hidden from the agent.
    """
    task_id: str = Field(default="")
    scenario_name: str = Field(default="")
    difficulty: str = Field(default="easy")
    current_step: int = Field(default=0)
    all_jobs: Dict[str, Job] = Field(default_factory=dict, description="Registry of all jobs over time")
    total_carbon_gco2: float = Field(default=0.0, description="Total operational carbon emitted")
    total_cost_usd: float = Field(default=0.0, description="Total dollars spent")
    baseline_carbon_gco2: float = Field(default=0.0, description="Expected carbon at regional-average intensity (reward baseline)")
    baseline_cost_usd: float = Field(default=0.0, description="Expected cost at regional-average spot price (reward baseline)")
    min_carbon_gco2: float = Field(default=0.0, description="Theoretical minimum possible carbon for scheduled jobs (for normalization)")
    min_cost_usd: float = Field(default=0.0, description="Theoretical minimum possible cost for scheduled jobs (for normalization)")
    sla_violations: int = Field(default=0, description="Total number of SLA violations")
    jobs_completed: int = Field(default=0, description="Number of successfully completed jobs")
    jobs_dropped: int = Field(default=0, description="Number of dropped jobs")
    # tracks how many times KWOK cluster failed to delete a pod when a job finished
    kwok_errors: int = Field(default=0, description="Cumulative count of KWOK infrastructure errors (failed pod/node operations)")
    step_duration_minutes: int = Field(default=60, description="Simulation time per step")
    is_done: bool = Field(default=False)

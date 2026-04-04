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

class PriceData(BaseModel):
    spot_price: float = Field(..., description="Current spot price per hour")
    on_demand_price: float = Field(..., description="Current on-demand price per hour")

class RegionInfo(BaseModel):
    region_name: str = Field(..., description="Region identifier")
    carbon: CarbonData
    pricing: PriceData
    available_capacity_units: int = Field(
        ..., description="Free compute slots in this region right now"
    )

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
    status: Literal["queued", "running", "completed", "dropped", "sla_missed"] = Field(
        default="queued", description="Current status of the job in the system."
    )
    start_step: Optional[int] = Field(None, description="The step at which the job started running, if applicable.")
    completion_step: Optional[int] = Field(None, description="The step at which the job finished running, if applicable.")
    region: Optional[str] = Field(None, description="Selected region for this job.")
    machine_type: Optional[Literal["spot", "on-demand"]] = Field(None, description="Selected instance type for this job.")

class RunningJob(BaseModel):
    job_id: str = Field(..., description="ID of the running job")
    region: str = Field(..., description="Region where the job is currently running")
    steps_remaining: int = Field(..., description="How many steps until this job completes")
    machine_type: str = Field(..., description="Machine type the job is using")

# Main Request / Response Models
class Ko2cubeObservation(Observation):
    """
    The observation returned to the agent at every step.
    Contains the queue of jobs, and the state of the world to make carbon-aware decisions.
    """
    current_step: int = Field(..., description="Current simulation step")
    job_queue: List[Job] = Field(..., description="List of jobs currently waiting to be scheduled")
    active_jobs: List[RunningJob] = Field(..., description="List of jobs actively running in the simulated cluster")
    regions: Dict[str, RegionInfo] = Field(..., description="Data about carbon and pricing per region")
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
        description="Target machine type if scheduling. Note your choice impacts both price and reliability."
    )
    defer_to_step: Optional[int] = Field(
        None,
        description="Step to retry scheduling. Required if decision=defer. Must be within job sla_end."
    )
    reasoning: str = Field(
        default="", 
        description="Step-by-step reasoning for this specific decision. Explain why instances/regions were picked or deferred."
    )

class Ko2cubeAction(Action):
    """
    The action the agent provides. 
    It must contain a decision for EVERY job currently in the `job_queue`.
    """
    assignments: List[JobAssignment] = Field(
        ..., 
        description="List of decisions for the jobs in the queue."
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
    sla_violations: int = Field(default=0, description="Total number of SLA violations")
    jobs_completed: int = Field(default=0, description="Number of successfully completed jobs")
    jobs_dropped: int = Field(default=0, description="Number of dropped jobs")
    step_duration_minutes: int = Field(default=60, description="Simulation time per step")
    is_done: bool = Field(default=False)
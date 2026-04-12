import os
import csv
import json
import math
import copy
from uuid import uuid4
from typing import Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ko2cube.models import (
    Ko2cubeAction, Ko2cubeObservation, Ko2cubeState,
    Job, RunningJob, RegionInfo, CarbonData, InstanceType,
    JobAssignment, ALWAYS_ON,
)
from ko2cube.server.rewards import (
    compute_step_reward, compute_terminal_reward,
    compute_grader_score, _potential,
    expected_carbon_baseline, expected_cost_baseline,
    actual_carbon, actual_cost,
)
from ko2cube.server.data.scenarios import get_scenario, Scenario
from ko2cube.server.rewards import KWOK_ERROR_PENALTY
from ko2cube.server.kwok.kwok import KWOKAdapter
from ko2cube.server.kwok.error import KWOKError

class Ko2cubeEnvironment(Environment):
    """
    Ko2cube: A carbon-aware cloud job scheduler environment.
    Uses synthetic timeseries for carbon and pricing.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        # Configuration
        self._root_dir = os.path.dirname(os.path.abspath(__file__))
        self._data_dir = os.path.join(self._root_dir, "data")
        
        # Load infrastructure
        infra_path = os.path.join(self._data_dir, "infrastructure.json")
        with open(infra_path, "r") as f:
            self._infra_config = json.load(f)
            
        # Load timeseries
        self._timeseries = []
        csv_path = os.path.join(self._data_dir, "cleaned_timeseries_data.csv")
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._timeseries.append(row)
        
        # Scenario management
        self._scenario: Optional[Scenario] = None
        self._reset_count = 0
        self._job_queue: List[Job] = []
        self._active_jobs: List[RunningJob] = []
        self._deferred: Dict[str, int] = {} # job_id -> step to retry
        self._state = self._init_state()
        self._kwok = KWOKAdapter()

    def _init_state(self) -> Ko2cubeState:
        return Ko2cubeState(
            episode_id=str(uuid4()),
            step_count=0,
            current_step=0,
            step_duration_minutes=60, # 1 hour per step
        )

    def reset(self, task_id: Optional[str] = None) -> Ko2cubeObservation:
        """Resets the environment for a new episode."""
        task = task_id or os.environ.get("KO2CUBE_TASK", "easy")
        
        # Deep copy the scenario to avoid shared state between episodes
        self._scenario = copy.deepcopy(get_scenario(task))
        scenario = self._scenario

        self._state = self._init_state()
        self._state.task_id = task
        self._state.scenario_name = scenario.name
        self._state.difficulty = scenario.difficulty
        self._state.step_duration_minutes = scenario.step_duration_minutes
        
        self._reset_count += 1
        self._job_queue = []
        self._active_jobs = []
        self._deferred = {}

        # Reset job registry and precalculate fair baselines
        # Precalculate SLA-window averages so rewards aren't 0.0
        self._calculate_baselines(scenario)

        for job in scenario.job_pool:
            job.status = "queued"
            self._state.all_jobs[job.job_id] = job

        # Surface jobs arriving at step 0
        self._surface_arriving_jobs(scenario, 0)
        
        regions = self._build_regions(scenario, 0)

        return Ko2cubeObservation(
            current_step=0,
            job_queue=list(self._job_queue),
            active_jobs=list(self._active_jobs),
            regions=regions,
            last_action_result="Episode started.",
            done=False,
            reward=0.01,
            metadata={"scenario": scenario.name}
        )

    def step(self, action: Ko2cubeAction) -> Ko2cubeObservation: # type: ignore[override]
        """Executes one step in the environment."""
        current_step = self._state.current_step
        scenario = self._scenario
        if not scenario:
            raise RuntimeError("Environment must be reset before calling step().")
        
        regions = self._build_regions(scenario, current_step)

        # Snapshot before mutation
        # Capture phi_before BEFORE any state changes
        phi_before = _potential(self._state)
        queue_snapshot = list(self._job_queue)
        assignment_map = {a.job_id: a for a in action.assignments}
        result_parts = []

        # 1. Process agent decisions
        for job in list(self._job_queue):
            assignment = assignment_map.get(job.job_id)
            if not assignment:
                continue

            if assignment.decision == "schedule":
                # Check feasibility (CPU, Memory, and Region availability)
                ok, msg = self._try_schedule(job, assignment, regions)
                if ok:
                    result_parts.append(f"{job.job_id}: scheduled in {assignment.region} [{assignment.instance_type}]")
                    self._job_queue.remove(job)
                else:
                    # Treat failed schedule attempt as an implicit defer penalty
                    result_parts.append(f"{job.job_id}: schedule failed ({msg})")
            
            elif assignment.decision == "defer":
                if job.sla_end == ALWAYS_ON:
                    # Heavily penalized in rewards, treated as a 'drop' attempt on always-on
                    job.status = "dropped"
                    self._state.jobs_dropped += 1
                    result_parts.append(f"{job.job_id}: always-on defer blocked (dropped)")
                elif assignment.defer_to_step is not None:
                    job.status = "deferred"
                    self._deferred[job.job_id] = assignment.defer_to_step
                    result_parts.append(f"{job.job_id}: deferred to step {assignment.defer_to_step}")
                self._job_queue.remove(job)

            elif assignment.decision == "drop":
                job.status = "dropped"
                self._state.jobs_dropped += 1
                result_parts.append(f"{job.job_id}: dropped")
                self._job_queue.remove(job)

        # 2. Tick active jobs
        still_running = []
        kwok_errors_this_step = 0  # count pod deletions that failed this step
        for rj in self._active_jobs:
            rj.steps_remaining -= 1
            if rj.steps_remaining <= 0:
                original = self._state.all_jobs.get(rj.job_id)
                if original:
                    original.status = "completed"
                    original.completion_step = current_step
                self._state.jobs_completed += 1

                # command KWOK to delete the finished pod from the simulated cluster
                try:
                    self._kwok.delete_pod(rj.job_id, rj.region)
                    result_parts.append(f"{rj.job_id}: pod deleted from KWOK cluster '{rj.region}'")
                except KWOKError as e:
                    kwok_errors_this_step += 1
                    self._state.kwok_errors += 1
                    result_parts.append(f"{rj.job_id}: KWOK pod deletion failed [{type(e).__name__}]: {e}")
                except Exception as e:
                    kwok_errors_this_step += 1
                    self._state.kwok_errors += 1
                    result_parts.append(f"{rj.job_id}: KWOK pod deletion error: {e}")
            else:
                still_running.append(rj)
        self._active_jobs = still_running

        # 3. SLA expiry check (using pre-increment current_step)
        for job in list(self._state.all_jobs.values()):
            if job.status in ["queued", "deferred"] and job.sla_end != ALWAYS_ON and current_step > job.sla_end:
                job.status = "sla_missed"
                self._state.sla_violations += 1
                self._job_queue = [j for j in self._job_queue if j.job_id != job.job_id]
                result_parts.append(f"{job.job_id}: SLA breached")

        # 4. Reward & Potential Computation
        phi_after = _potential(self._state)
        # We pass both potentials to ensure accurate marginal reward computation
        # Pass current_step explicitly — state.current_step hasn't been incremented yet
        # so it equals the step we just processed. This prevents an off-by-one in SLA checks.
        rb = compute_step_reward(
            assignments=action.assignments,
            queue=queue_snapshot,
            regions=regions,
            state=self._state,
            phi_before=phi_before,
            phi_after=phi_after,
            current_step=current_step,
        )

        # apply a penalty for each KWOK pod deletion that failed this step
        if kwok_errors_this_step > 0:
            kwok_penalty = kwok_errors_this_step * KWOK_ERROR_PENALTY
            rb.sla += kwok_penalty
            rb.components[f"kwok_errors_x{kwok_errors_this_step}"] = round(kwok_penalty, 4)

        # 5. Transition to next step
        self._state.current_step += 1
        self._state.step_count = self._state.current_step
        next_step = self._state.current_step
        
        self._surface_arriving_jobs(scenario, next_step)
        self._surface_deferred_jobs(next_step)
        
        done = next_step >= scenario.total_steps
        self._state.is_done = done

        if done:
            trb = compute_terminal_reward(self._state)
            rb.terminal = trb.terminal
            rb.components.update(trb.components)

        # Clamp step reward strictly within (0, 1)
        clamped_reward = max(0.01, min(0.99, round(rb.total, 4)))

        return Ko2cubeObservation(
            current_step=next_step,
            job_queue=list(self._job_queue),
            active_jobs=list(self._active_jobs),
            regions=self._build_regions(scenario, next_step),
            last_action_result=" | ".join(result_parts) or "Clock ticked.",
            done=done,
            reward=clamped_reward,
            metadata=rb.to_dict()
        )

    def _build_regions(self, scenario: Scenario, step: int) -> Dict[str, RegionInfo]:
        """Constructs the regional observation data for a specific step."""
        data_idx = step % len(self._timeseries)
        data = self._timeseries[data_idx]
        lookahead = scenario.lookahead_steps
        
        regions = {}
        # Resource capacity limits based on difficulty
        capacity_count = {"easy": 10, "medium": 6, "hard": 3}.get(scenario.difficulty, 10)

        for rname in self._infra_config["regions"]:
            # Extract carbon forecast
            hist_vals = []
            for i in range(lookahead):
                idx = (step + i) % len(self._timeseries)
                v = self._timeseries[idx].get(f"carbon_{rname}", 0.1)
                hist_vals.append(float(v))
                
            # Create instance list with current spot prices
            spot_mult = float(data.get(f"spot_mult_{rname}", 1.0))
            instances = []
            for inst in self._infra_config["instances"]:
                instances.append(InstanceType(
                    name=inst["name"],
                    cpu_cores=float(inst["cpu"]),
                    memory_gb=float(inst["mem"]),
                    spot_price=round(inst["on_demand"] * spot_mult, 4),
                    on_demand_price=inst["on_demand"],
                    available_count=capacity_count
                ))
            
            regions[rname] = RegionInfo(
                region_name=rname,
                carbon=CarbonData(current_intensity=hist_vals[0], forecast=hist_vals),
                available_instances=instances
            )
        return regions

    def _try_schedule(self, job: Job, assignment: JobAssignment, regions: Dict[str, RegionInfo]) -> tuple[bool, str]:
        """Validate requirements and update state in case of success."""
        region_info = regions.get(assignment.region or "")
        if not region_info:
            return False, "invalid region"
            
        instance = next((i for i in region_info.available_instances if i.name == assignment.instance_type), None)
        if not instance:
            return False, "invalid instance type"
            
        if instance.cpu_cores < job.cpu_cores or instance.memory_gb < job.memory_gb:
            return False, "insufficient instance resources"

        # Success - update internal tracking for the job
        job.status = "running"
        job.start_step = self._state.current_step
        job.region = assignment.region
        job.machine_type = assignment.machine_type
        
        runtime_h = (job.eta_minutes or 60) / 60.0
        agent_c = actual_carbon(job, assignment, regions)
        agent_cost = actual_cost(job, assignment, regions)
        
        # Update running totals in state
        self._state.total_carbon_gco2 += agent_c
        self._state.total_cost_usd += agent_cost
        
        # Baselines (window averages)
        self._state.baseline_carbon_gco2 += expected_carbon_baseline(job)
        self._state.baseline_cost_usd += expected_cost_baseline(job)
        
        # Min cases for grader normalization (using pre-calculated theoretical minimums)
        # Note: baseline_* metrics are already on the Job object
        self._state.min_cost_usd += getattr(job, "theoretical_min_cost", 0.0)
        self._state.min_carbon_gco2 += getattr(job, "theoretical_min_carbon", 0.0)

        self._active_jobs.append(RunningJob(
            job_id=job.job_id,
            region=assignment.region,
            steps_remaining=math.ceil((job.eta_minutes or 60) / self._state.step_duration_minutes),
            machine_type=assignment.machine_type or "on-demand"
        ))
        
        return True, ""

    def _calculate_baselines(self, scenario: Scenario):
        """Precalculates the fair market average (baseline) and theoretical minimum for each job."""
        regions_list = self._infra_config["regions"]

        for job in scenario.job_pool:
            if job.sla_end == ALWAYS_ON or job.eta_minutes is None:
                job.baseline_carbon_intensity = 0.0
                job.baseline_spot_price = 0.0
                job.theoretical_min_carbon = 0.0
                job.theoretical_min_cost = 0.0
                continue
                
            # Define window
            if job.eta_minutes is None:
                job.baseline_carbon_intensity = 0.0
                job.baseline_spot_price = 0.0
                job.theoretical_min_carbon = 0.0
                job.theoretical_min_cost = 0.0
                continue

            runtime_h = job.eta_minutes / 60.0
            cpu_factor = max(job.cpu_cores, 1.0)
            window_steps = range(job.sla_start, job.sla_end + 1)
            
            # 1. Average Baseline (mean across all regions and all steps in window)
            total_intensity = 0.0
            total_multiplier = 0.0
            divisor = len(window_steps) * len(regions_list)
            
            # 2. Theoretical Min (Absolute best region/step in the window)
            abs_min_carbon_intensity = float('inf')
            abs_min_price = float('inf')

            # Find cheapest fitting instance base price
            fitting_prices = [
                i["on_demand"] for i in self._infra_config["instances"] 
                if i["cpu"] >= job.cpu_cores and i["mem"] >= job.memory_gb
            ]
            ideal_base_price = min(fitting_prices) if fitting_prices else 0.0
            
            for t in window_steps:
                idx = t % len(self._timeseries)
                row = self._timeseries[idx]
                for rname in regions_list:
                    c = float(row.get(f"carbon_{rname}", 0.0))
                    m = float(row.get(f"spot_mult_{rname}", 1.0))
                    
                    total_intensity += c
                    total_multiplier += m
                    
                    # Track absolute minimums
                    if c < abs_min_carbon_intensity:
                        abs_min_carbon_intensity = c
                    
                    current_spot_p = ideal_base_price * m
                    if current_spot_p < abs_min_price:
                        abs_min_price = current_spot_p
            
            # Save baselines
            if divisor > 0:
                job.baseline_carbon_intensity = total_intensity / divisor
                job.baseline_spot_price = ideal_base_price * (total_multiplier / divisor)
            else:
                job.baseline_carbon_intensity = 0.0
                job.baseline_spot_price = 0.0
            
            # Save theoretical minimums for the Grader normalization
            job.theoretical_min_carbon = abs_min_carbon_intensity * runtime_h * cpu_factor
            job.theoretical_min_cost = abs_min_price * runtime_h


    def _surface_arriving_jobs(self, scenario: Scenario, step: int):
        existing_ids = {j.job_id for j in self._job_queue}
        for job in scenario.job_pool:
            if job.arrival_step == step and job.job_id not in existing_ids:
                self._job_queue.append(job)

    def _surface_deferred_jobs(self, step: int):
        to_move = [jid for jid, target in self._deferred.items() if target <= step]
        for jid in to_move:
            job = self._state.all_jobs[jid]
            job.status = "queued" 
            self._job_queue.append(job)
            del self._deferred[jid]

    @property
    def state(self) -> State:
        return self._state

    def get_observation(self) -> Ko2cubeObservation:
        """Returns the current observation without advancing the simulation."""
        scenario = self._scenario
        if not scenario:
            raise RuntimeError("Environment must be reset before calling get_observation().")
            
        return Ko2cubeObservation(
            current_step=self._state.current_step,
            job_queue=list(self._job_queue),
            active_jobs=list(self._active_jobs),
            regions=self._build_regions(scenario, self._state.current_step),
            last_action_result="Observation retrieved.",
            done=self._state.is_done,
            reward=0.0,
            metadata={}
        )

    def grader_score(self) -> float:
        return compute_grader_score(self._state)

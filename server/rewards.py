"""
Ko2cube Reward System

Priority order:
  1. SLA compliance     - dominant signal, largest magnitudes
  2. Carbon efficiency  - measured vs regional average intensity in the SLA window
  3. Cost efficiency    - actual cost vs expected cost at average SLA-window prices

Baseline design:
  Cost baseline  = avg(spot_price across available regions)
                   x (eta_minutes / 60)
                   x cpu_cores
  If agent_cost < baseline_cost → proportional reward
  If agent_cost > baseline_cost → proportional penalty
  Normalized to [-1, 1].

  Carbon baseline = avg(carbon_intensity across available regions)
                    x (eta_minutes / 60)
                    x cpu_cores
  If agent_carbon < baseline_carbon → proportional reward
  Normalized to [0, 2].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import Job, JobAssignment, Ko2cubeState, RegionInfo, ALWAYS_ON

# Constants
# SLA penalties (dominant - largest magnitudes)
SLA_BREACH_PENALTY       = -5.0   # job missed its deadline
DROP_PENALTY             = -4.0   # agent explicitly dropped a job
DEFER_PAST_END_PENALTY   = -4.5   # deferred to a step beyond sla_end
ALWAYS_ON_PENALTY        = -6.0   # deferred / dropped an always-on job
ALWAYS_ON_SPOT_PENALTY   = -3.0   # scheduled always-on job on spot (eviction risk)

# SLA bonuses
SCHEDULE_ON_TIME_BONUS   = +1.0   # scheduled within SLA window
URGENT_JOB_BONUS         = +0.5   # non-deferrable job scheduled in the same step it arrived

# Carbon signal
CARBON_REWARD_SCALE      = 2.0    # multiplier on fractional saving below regional average
DEFER_TO_CLEAN_BONUS     = +0.8   # smart deferral: carbon is currently above regional avg
DEFER_CLEAN_ALREADY_PENALTY = -0.3  # deferred when carbon is already below average

# Cost signal
COST_REWARD_SCALE        = 1.0    # multiplier on fractional saving below regional average
RIGHT_SIZING_WASTE_SCALE = 1.5    # multiplier on the dollar amount wasted (vs cheapest valid instance)

# Terminal multipliers
TERMINAL_COMPLETION_BONUS  = +3.0
TERMINAL_CARBON_BONUS_MAX  = +2.0
TERMINAL_COST_BONUS_MAX    = +1.0

# Baseline helpers
def _current_avg_intensity(regions: Dict[str, RegionInfo]) -> float:
    """Current average carbon intensity across all regions."""
    if not regions:
        return 0.0
    return sum(r.carbon.current_intensity for r in regions.values()) / len(regions)


def expected_cost_baseline(job: Job) -> float:
    """
    Fair cost baseline based on the average spot price in the job's SLA window.
    Expected cost = window_avg_spot_price x runtime_hours x cpu_cores
    """
    if job.eta_minutes is None:
        return 0.0  # always-on jobs excluded from cost baseline
    
    runtime_hours = job.eta_minutes / 60.0
    cpu = max(job.cpu_cores, 1.0)
    return job.baseline_spot_price * runtime_hours * cpu


def expected_carbon_baseline(job: Job) -> float:
    """
    Fair carbon baseline based on the average intensity in the job's SLA window.
    Expected carbon = window_avg_intensity x runtime_hours x cpu_cores (gCO2)
    """
    if job.eta_minutes is None:
        return 0.0  # always-on jobs excluded from carbon baseline
    
    runtime_hours = job.eta_minutes / 60.0
    cpu = max(job.cpu_cores, 1.0)
    return job.baseline_carbon_intensity * runtime_hours * cpu


def actual_cost(job: Job, assignment: JobAssignment, regions: Dict[str, RegionInfo]) -> float:
    """Actual cost for the agent's scheduling choice looking up the specific instance type."""
    if assignment.decision != "schedule" or not assignment.region or not assignment.instance_type or job.eta_minutes is None:
        return 0.0
    region_info = regions.get(assignment.region)
    if not region_info:
        return 0.0
    
    # Find the specific machine type requested
    instance = next((i for i in region_info.available_instances if i.name == assignment.instance_type), None)
    if not instance:
        return 0.0  # Or apply a penalty for invalid instance type in a real simulator
    
    runtime_hours = job.eta_minutes / 60.0
    price = (instance.spot_price if assignment.machine_type == "spot"
             else instance.on_demand_price)
    # We use the instance's total price, or we can scale by job requirement. 
    # Usually, you pay for the whole instance.
    return price * runtime_hours


def actual_carbon(job: Job, assignment: JobAssignment, regions: Dict[str, RegionInfo]) -> float:
    """Actual carbon for the agent's scheduling choice (gCO2)."""
    if assignment.decision != "schedule" or not assignment.region or job.eta_minutes is None:
        return 0.0
    region_info = regions.get(assignment.region)
    if not region_info:
        return 0.0
    runtime_hours = job.eta_minutes / 60.0
    cpu = max(job.cpu_cores, 1.0)
    return region_info.carbon.current_intensity * runtime_hours * cpu


# Potential-based shaping
def _potential(state: Ko2cubeState) -> float:
    """
    φ(s) - Progress potential. Fraction of jobs that have been completed.
    Increases toward 1.0 as jobs finish within SLA.
    Returns 0.0 if no jobs exist.
    """
    total = len(state.all_jobs)
    if total == 0:
        return 0.0
    return state.jobs_completed / total


# Reward breakdown dataclass
@dataclass
class RewardBreakdown:
    sla: float = 0.0
    carbon: float = 0.0
    cost: float = 0.0
    waste: float = 0.0
    shaping: float = 0.0
    terminal: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (self.sla + self.carbon + self.cost
                + self.waste + self.shaping + self.terminal)

    def to_dict(self) -> Dict[str, float]:
        return {
            "sla":       round(self.sla, 4),
            "carbon":    round(self.carbon, 4),
            "cost":      round(self.cost, 4),
            "waste":     round(self.waste, 4),
            "shaping":   round(self.shaping, 4),
            "terminal":  round(self.terminal, 4),
            "total":     round(self.total, 4),
            **{k: round(v, 4) for k, v in self.components.items()},
        }


# Step reward
def compute_step_reward(
    assignments: List[JobAssignment],
    queue: List[Job],
    regions: Dict[str, RegionInfo],
    state: Ko2cubeState,
    phi_after: float,
) -> RewardBreakdown:
    """
    Compute the reward for one step's batch of assignments.

    Parameters
    ----------
    assignments : Agent's decisions for this step
    queue       : Jobs that were in the queue at the start of this step
    regions     : Current region data (carbon + pricing)
    state       : Current simulator state (before mutation)
    phi_after   : Potential φ(s') after state has been mutated - used for shaping
    """
    rb = RewardBreakdown()
    job_map: Dict[str, Job] = {j.job_id: j for j in queue}
    current_step = state.current_step

    avg_intensity = _current_avg_intensity(regions)

    # Per-assignment SLA + Carbon + Cost-
    for assignment in assignments:
        job = job_map.get(assignment.job_id)
        if not job:
            continue

        is_always_on = (job.sla_end == ALWAYS_ON)

        # SLA
        if assignment.decision == "drop":
            penalty = ALWAYS_ON_PENALTY if is_always_on else DROP_PENALTY
            rb.sla += penalty
            rb.components[f"drop_{job.job_id}"] = penalty

        elif assignment.decision == "defer":
            if is_always_on:
                rb.sla += ALWAYS_ON_PENALTY
                rb.components[f"defer_alwayson_{job.job_id}"] = ALWAYS_ON_PENALTY

            elif not job.delay_tolerant:
                rb.sla += SLA_BREACH_PENALTY
                rb.components[f"defer_urgent_{job.job_id}"] = SLA_BREACH_PENALTY

            elif assignment.defer_to_step is not None:
                if assignment.defer_to_step > job.sla_end:
                    rb.sla += DEFER_PAST_END_PENALTY
                    rb.components[f"defer_past_sla_{job.job_id}"] = DEFER_PAST_END_PENALTY
                else:
                    # Consistency Fix: Defer smart bonus now uses the job's personal SLA-window baseline
                    # instead of the current regional average.
                    region_for_job = (
                        regions.get(assignment.region) or next(iter(regions.values()), None)
                    )
                    current_intensity = region_for_job.carbon.current_intensity if region_for_job else job.baseline_carbon_intensity
                    
                    if current_intensity > job.baseline_carbon_intensity:
                        rb.carbon += DEFER_TO_CLEAN_BONUS
                        rb.components[f"defer_smart_{job.job_id}"] = DEFER_TO_CLEAN_BONUS
                    else:
                        rb.carbon += DEFER_CLEAN_ALREADY_PENALTY
                        rb.components[f"defer_unnecessary_{job.job_id}"] = DEFER_CLEAN_ALREADY_PENALTY
            else:
                # No defer_to_step specified
                rb.sla += -0.5
                rb.components[f"defer_no_target_{job.job_id}"] = -0.5

        elif assignment.decision == "schedule":
            if current_step < job.sla_start:
                rb.sla += -1.0
                rb.components[f"early_schedule_{job.job_id}"] = -1.0

            elif not is_always_on and current_step > job.sla_end:
                rb.sla += SLA_BREACH_PENALTY
                rb.components[f"late_schedule_{job.job_id}"] = SLA_BREACH_PENALTY

            else:
                # Valid schedule
                rb.sla += SCHEDULE_ON_TIME_BONUS
                rb.components[f"sched_ok_{job.job_id}"] = SCHEDULE_ON_TIME_BONUS

                # Bonus: non-deferrable job scheduled immediately
                if not job.delay_tolerant and current_step == job.sla_start:
                    rb.sla += URGENT_JOB_BONUS
                    rb.components[f"urgent_bonus_{job.job_id}"] = URGENT_JOB_BONUS

                # Carbon (comparing actual to the job's personal SLA-window baseline)
                baseline_c = expected_carbon_baseline(job)
                agent_c = actual_carbon(job, assignment, regions)
                if baseline_c > 0:
                    carbon_saving_frac = (baseline_c - agent_c) / baseline_c
                    # Reward for efficiency, penalty for exceeding baseline
                    carbon_reward = carbon_saving_frac * CARBON_REWARD_SCALE
                    rb.carbon += carbon_reward
                    rb.components[f"carbon_frac_{job.job_id}"] = round(carbon_saving_frac, 3)
                    rb.components[f"carbon_reward_{job.job_id}"] = round(carbon_reward, 4)

                # Cost (comparing actual to the job's personal SLA-window baseline)
                baseline_cost = expected_cost_baseline(job)
                agent_cost = actual_cost(job, assignment, regions)
                if baseline_cost > 0:
                    cost_saving_frac = (baseline_cost - agent_cost) / baseline_cost
                    cost_signal = cost_saving_frac * COST_REWARD_SCALE
                    rb.cost += cost_signal
                    rb.components[f"cost_frac_{job.job_id}"] = round(cost_saving_frac, 3)
                    rb.components[f"cost_signal_{job.job_id}"] = round(cost_signal, 4)

                # Waste (Over-provisioning)
                # Calculate what the cheapest valid instance would have cost in this region
                if region_info:
                    valid_instances = [
                        inst for inst in region_info.available_instances
                        if inst.cpu_cores >= job.cpu_cores and inst.memory_gb >= job.memory_gb
                    ]
                    if valid_instances:
                        runtime_hours = job.eta_minutes / 60.0
                        # Find the cheapest price (using the same machine_type preference)
                        ideal_unit_price = min(
                            inst.spot_price if assignment.machine_type == "spot" else inst.on_demand_price
                            for inst in valid_instances
                        )
                        ideal_cost = ideal_unit_price * runtime_hours
                        
                        if agent_cost > ideal_cost:
                            waste_dollars = agent_cost - ideal_cost
                            waste_penalty = -waste_dollars * RIGHT_SIZING_WASTE_SCALE
                            rb.waste += waste_penalty
                            rb.components[f"waste_penalty_{job.job_id}"] = round(waste_penalty, 4)
                            rb.components[f"oversize_dollars_{job.job_id}"] = round(waste_dollars, 4)

                # Penalty: used spot for an always-on job (eviction risk)
                if is_always_on and assignment.machine_type == "spot":
                    rb.sla += ALWAYS_ON_SPOT_PENALTY
                    rb.components[f"always_on_spot_{job.job_id}"] = ALWAYS_ON_SPOT_PENALTY

    # Shaping
    phi_before = _potential(state)
    rb.shaping = phi_after - phi_before

    return rb


# Terminal reward
def compute_terminal_reward(state: Ko2cubeState) -> RewardBreakdown:
    """
    Compute the end-of-episode bonus/penalty.

    Carbon and cost efficiency are measured against the regional-average
    baseline accumulated over the episode (stored in state.baseline_*).
    """
    rb = RewardBreakdown()
    total = max(len(state.all_jobs), 1)

    # Completion rate
    completion_rate = state.jobs_completed / total
    rb.terminal += TERMINAL_COMPLETION_BONUS * completion_rate
    rb.components["completion_rate"] = round(completion_rate, 3)

    # SLA violation penalty
    violation_rate = state.sla_violations / total
    sla_term_penalty = -violation_rate * TERMINAL_COMPLETION_BONUS
    rb.terminal += sla_term_penalty
    rb.components["violation_rate"] = round(violation_rate, 3)
    rb.components["sla_terminal_penalty"] = round(sla_term_penalty, 4)

    # Carbon efficiency vs accumulated baseline
    if state.baseline_carbon_gco2 > 0:
        carbon_improvement = max(
            0.0,
            (state.baseline_carbon_gco2 - state.total_carbon_gco2) / state.baseline_carbon_gco2
        )
        carbon_bonus = TERMINAL_CARBON_BONUS_MAX * carbon_improvement
        rb.terminal += carbon_bonus
        rb.components["carbon_improvement"] = round(carbon_improvement, 3)
        rb.components["carbon_terminal_bonus"] = round(carbon_bonus, 4)

    # Cost efficiency vs accumulated baseline
    # Reward: actual cost below expected average cost
    if state.baseline_cost_usd > 0:
        cost_improvement = max(
            0.0,
            (state.baseline_cost_usd - state.total_cost_usd) / state.baseline_cost_usd
        )
        cost_bonus = TERMINAL_COST_BONUS_MAX * cost_improvement
        rb.terminal += cost_bonus
        rb.components["cost_improvement"] = round(cost_improvement, 3)
        rb.components["cost_terminal_bonus"] = round(cost_bonus, 4)

    return rb


# Grader score (0.0 – 1.0)
def compute_grader_score(state: Ko2cubeState) -> float:
    """
    Normalized [0, 1] grader score.

    Weights:
      SLA compliance  50%
      Carbon savings  35%
      Cost savings    15%

    Each component is independently normalized to [0, 1] before weighting.
    """
    total = max(len(state.all_jobs), 1)

    # SLA: success rate (completed / total jobs)
    sla_score = state.jobs_completed / total

    # Carbon: Rescale improvement relative to the best possible theoretical emissions.
    # Score = 1.0 if agent is equal to best-case. Score = 0.0 if agent is equal to the baseline.
    if state.baseline_carbon_gco2 > state.min_carbon_gco2:
        carbon_score = max(0.0, min(1.0, 
            (state.baseline_carbon_gco2 - state.total_carbon_gco2) / 
            (state.baseline_carbon_gco2 - state.min_carbon_gco2)
        ))
    else:
        # If baseline == min (e.g. single region, no optimization possible), 
        # use a soft score relative to the baseline.
        carbon_score = max(0.0, 1.0 - state.total_carbon_gco2 / max(state.baseline_carbon_gco2, 1.0))

    # Cost: Rescale improvement relative to the best possible theoretical cost.
    if state.baseline_cost_usd > state.min_cost_usd:
        cost_score = max(0.0, min(1.0, 
            (state.baseline_cost_usd - state.total_cost_usd) / 
            (state.baseline_cost_usd - state.min_cost_usd)
        ))
    else:
        cost_score = max(0.0, 1.0 - state.total_cost_usd / max(state.baseline_cost_usd, 0.01))

    final = 0.50 * sla_score + 0.35 * carbon_score + 0.15 * cost_score
    return round(min(1.0, max(0.0, final)), 4)

"""
Baseline agents for evaluating the Ko2cube environment.

These agents implement different scheduling strategies to establish performance bounds:
- RandomAgent: Random scheduling decisions (lower bound)
- GreedyAgent: Always schedules immediately in cheapest region
- CarbonAwareGreedy: Always schedules in lowest carbon region
- OracleAgent: Uses full forecast knowledge for optimal decisions (upper bound)
"""

import random
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from models import (
    Ko2cubeAction, Ko2cubeObservation, Job, JobAssignment,
    RegionInfo, ALWAYS_ON,
)


@dataclass
class AgentMetrics:
    """Metrics collected during an episode."""
    total_reward: float = 0.0
    total_carbon_gco2: float = 0.0
    total_cost_usd: float = 0.0
    jobs_scheduled: int = 0
    jobs_deferred: int = 0
    jobs_dropped: int = 0
    sla_violations: int = 0
    steps: int = 0


class BaselineAgent(ABC):
    """Abstract base class for baseline agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = AgentMetrics()
    
    @abstractmethod
    def act(self, obs: Ko2cubeObservation) -> Ko2cubeAction:
        """Generate an action given the current observation."""
        pass
    
    def reset(self):
        """Reset agent state for a new episode."""
        self.metrics = AgentMetrics()
    
    def update_metrics(self, reward: float):
        """Update metrics after receiving a reward."""
        self.metrics.total_reward += reward
        self.metrics.steps += 1
    
    def _get_fitting_instance(
        self, job: Job, region: RegionInfo
    ) -> Optional[str]:
        """Find an instance type that fits the job requirements."""
        for inst in region.available_instances:
            if (inst.cpu_cores >= job.cpu_cores and 
                inst.memory_gb >= job.memory_gb and
                inst.available_count > 0):
                return inst.name
        return None
    
    def _can_schedule_in_region(
        self, job: Job, region: RegionInfo
    ) -> bool:
        """Check if a job can be scheduled in a region."""
        return self._get_fitting_instance(job, region) is not None


class RandomAgent(BaselineAgent):
    """
    Random baseline: makes random scheduling decisions.
    
    For each job, randomly chooses to:
    - Schedule in a random region (if instance available)
    - Defer to a random future step within SLA
    - Drop the job (with low probability)
    """
    
    def __init__(self, drop_prob: float = 0.05, defer_prob: float = 0.3):
        super().__init__("RandomAgent")
        self.drop_prob = drop_prob
        self.defer_prob = defer_prob
    
    def act(self, obs: Ko2cubeObservation) -> Ko2cubeAction:
        assignments = []
        
        for job in obs.job_queue:
            decision = self._make_decision(job, obs)
            assignments.append(decision)
        
        return Ko2cubeAction(assignments=assignments)
    
    def _make_decision(
        self, job: Job, obs: Ko2cubeObservation
    ) -> JobAssignment:
        roll = random.random()
        
        # Small chance to drop
        if roll < self.drop_prob:
            self.metrics.jobs_dropped += 1
            return JobAssignment(job_id=job.job_id, decision="drop")
        
        # Chance to defer if delay tolerant
        if roll < self.drop_prob + self.defer_prob and job.delay_tolerant:
            if job.sla_end != ALWAYS_ON and job.sla_end > obs.current_step + 1:
                defer_to = random.randint(obs.current_step + 1, job.sla_end)
                self.metrics.jobs_deferred += 1
                return JobAssignment(
                    job_id=job.job_id,
                    decision="defer",
                    defer_to_step=defer_to
                )
        
        # Otherwise schedule in random region
        schedulable_regions = [
            (name, region) for name, region in obs.regions.items()
            if self._can_schedule_in_region(job, region)
        ]
        
        if schedulable_regions:
            region_name, region = random.choice(schedulable_regions)
            instance = self._get_fitting_instance(job, region)
            self.metrics.jobs_scheduled += 1
            return JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=region_name,
                machine_type=job.instance_preference,
                instance_type=instance
            )
        
        # No capacity - defer if possible
        if job.delay_tolerant and job.sla_end != ALWAYS_ON:
            defer_to = min(obs.current_step + 1, job.sla_end)
            self.metrics.jobs_deferred += 1
            return JobAssignment(
                job_id=job.job_id,
                decision="defer",
                defer_to_step=defer_to
            )
        
        # Must drop
        self.metrics.jobs_dropped += 1
        return JobAssignment(job_id=job.job_id, decision="drop")


class GreedyCostAgent(BaselineAgent):
    """
    Greedy cost baseline: always schedules immediately in the cheapest region.
    
    Prioritizes minimizing cost without considering carbon.
    """
    
    def __init__(self):
        super().__init__("GreedyCostAgent")
    
    def act(self, obs: Ko2cubeObservation) -> Ko2cubeAction:
        assignments = []
        
        for job in obs.job_queue:
            decision = self._make_decision(job, obs)
            assignments.append(decision)
        
        return Ko2cubeAction(assignments=assignments)
    
    def _get_cheapest_region(
        self, job: Job, obs: Ko2cubeObservation
    ) -> Optional[Tuple[str, RegionInfo, str]]:
        """Find the cheapest region where the job can run."""
        best = None
        best_price = float('inf')
        
        for region_name, region in obs.regions.items():
            instance_name = self._get_fitting_instance(job, region)
            if instance_name:
                for inst in region.available_instances:
                    if inst.name == instance_name:
                        price = (inst.spot_price if job.instance_preference == "spot" 
                                else inst.on_demand_price)
                        if price < best_price:
                            best_price = price
                            best = (region_name, region, instance_name)
                        break
        
        return best
    
    def _make_decision(
        self, job: Job, obs: Ko2cubeObservation
    ) -> JobAssignment:
        result = self._get_cheapest_region(job, obs)
        
        if result:
            region_name, _, instance_name = result
            self.metrics.jobs_scheduled += 1
            return JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=region_name,
                machine_type=job.instance_preference,
                instance_type=instance_name
            )
        
        # No capacity - defer if possible
        if job.delay_tolerant and job.sla_end != ALWAYS_ON:
            defer_to = min(obs.current_step + 1, job.sla_end)
            self.metrics.jobs_deferred += 1
            return JobAssignment(
                job_id=job.job_id,
                decision="defer",
                defer_to_step=defer_to
            )
        
        self.metrics.jobs_dropped += 1
        return JobAssignment(job_id=job.job_id, decision="drop")


class CarbonAwareGreedyAgent(BaselineAgent):
    """
    Carbon-aware greedy baseline: schedules in the lowest carbon region.
    
    Always schedules immediately in the region with lowest current carbon intensity.
    Does not consider future forecasts.
    """
    
    def __init__(self):
        super().__init__("CarbonAwareGreedyAgent")
    
    def act(self, obs: Ko2cubeObservation) -> Ko2cubeAction:
        assignments = []
        
        for job in obs.job_queue:
            decision = self._make_decision(job, obs)
            assignments.append(decision)
        
        return Ko2cubeAction(assignments=assignments)
    
    def _get_greenest_region(
        self, job: Job, obs: Ko2cubeObservation
    ) -> Optional[Tuple[str, RegionInfo, str]]:
        """Find the region with lowest carbon where job can run."""
        best = None
        best_carbon = float('inf')
        
        for region_name, region in obs.regions.items():
            instance_name = self._get_fitting_instance(job, region)
            if instance_name:
                carbon = region.carbon.current_intensity
                if carbon < best_carbon:
                    best_carbon = carbon
                    best = (region_name, region, instance_name)
        
        return best
    
    def _make_decision(
        self, job: Job, obs: Ko2cubeObservation
    ) -> JobAssignment:
        result = self._get_greenest_region(job, obs)
        
        if result:
            region_name, _, instance_name = result
            self.metrics.jobs_scheduled += 1
            return JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=region_name,
                machine_type=job.instance_preference,
                instance_type=instance_name
            )
        
        # No capacity - defer if possible
        if job.delay_tolerant and job.sla_end != ALWAYS_ON:
            defer_to = min(obs.current_step + 1, job.sla_end)
            self.metrics.jobs_deferred += 1
            return JobAssignment(
                job_id=job.job_id,
                decision="defer",
                defer_to_step=defer_to
            )
        
        self.metrics.jobs_dropped += 1
        return JobAssignment(job_id=job.job_id, decision="drop")


class OracleAgent(BaselineAgent):
    """
    Oracle baseline: uses full forecast knowledge for optimal decisions.
    
    For each delay-tolerant job, looks ahead to find the optimal time and region
    based on the carbon forecast. This represents an upper bound on performance
    (minus the cost of additional compute cycles from deferral).
    """
    
    def __init__(self, carbon_weight: float = 0.7, cost_weight: float = 0.3):
        super().__init__("OracleAgent")
        self.carbon_weight = carbon_weight
        self.cost_weight = cost_weight
    
    def act(self, obs: Ko2cubeObservation) -> Ko2cubeAction:
        assignments = []
        
        for job in obs.job_queue:
            decision = self._make_decision(job, obs)
            assignments.append(decision)
        
        return Ko2cubeAction(assignments=assignments)
    
    def _find_optimal_schedule(
        self, job: Job, obs: Ko2cubeObservation
    ) -> Tuple[int, str, str, float]:
        """
        Find the optimal (step, region, instance, score) for scheduling.
        
        Returns the step offset, region name, instance type, and score.
        """
        best_step = 0
        best_region = None
        best_instance = None
        best_score = float('inf')
        
        # Determine how far ahead we can look
        max_defer_steps = 0
        if job.delay_tolerant and job.sla_end != ALWAYS_ON:
            max_defer_steps = job.sla_end - obs.current_step
        
        # Evaluate each future step using forecast
        for step_offset in range(max_defer_steps + 1):
            for region_name, region in obs.regions.items():
                instance_name = self._get_fitting_instance(job, region)
                if not instance_name:
                    continue
                
                # Get carbon intensity for this step
                if step_offset == 0:
                    carbon = region.carbon.current_intensity
                elif step_offset < len(region.carbon.forecast):
                    carbon = region.carbon.forecast[step_offset]
                else:
                    # Beyond forecast horizon, use last known value
                    carbon = region.carbon.forecast[-1] if region.carbon.forecast else region.carbon.current_intensity
                
                # Get price for the instance
                for inst in region.available_instances:
                    if inst.name == instance_name:
                        price = (inst.spot_price if job.instance_preference == "spot"
                                else inst.on_demand_price)
                        break
                else:
                    price = 1.0  # Fallback
                
                # Normalize and combine scores
                carbon_norm = carbon / 500.0  # Rough normalization (typical range 50-500)
                cost_norm = price / 1.0  # Rough normalization (typical range $0.1-$1)
                
                score = (self.carbon_weight * carbon_norm + 
                        self.cost_weight * cost_norm)
                
                # Add small penalty for deferral (opportunity cost)
                defer_penalty = step_offset * 0.01
                score += defer_penalty
                
                if score < best_score:
                    best_score = score
                    best_step = step_offset
                    best_region = region_name
                    best_instance = instance_name
        
        return best_step, best_region, best_instance, best_score
    
    def _make_decision(
        self, job: Job, obs: Ko2cubeObservation
    ) -> JobAssignment:
        step_offset, region, instance, _ = self._find_optimal_schedule(job, obs)
        
        if region is None:
            # No viable schedule found
            if job.delay_tolerant and job.sla_end != ALWAYS_ON:
                defer_to = min(obs.current_step + 1, job.sla_end)
                self.metrics.jobs_deferred += 1
                return JobAssignment(
                    job_id=job.job_id,
                    decision="defer",
                    defer_to_step=defer_to
                )
            self.metrics.jobs_dropped += 1
            return JobAssignment(job_id=job.job_id, decision="drop")
        
        if step_offset == 0:
            # Schedule now
            self.metrics.jobs_scheduled += 1
            return JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=region,
                machine_type=job.instance_preference,
                instance_type=instance
            )
        else:
            # Defer to optimal step
            defer_to = obs.current_step + step_offset
            self.metrics.jobs_deferred += 1
            return JobAssignment(
                job_id=job.job_id,
                decision="defer",
                defer_to_step=defer_to
            )


class HybridAgent(BaselineAgent):
    """
    Hybrid baseline: balances carbon, cost, and SLA constraints.
    
    Uses a weighted scoring system that considers:
    - Carbon intensity
    - Cost
    - SLA urgency (remaining time window)
    - Forecast trends
    """
    
    def __init__(
        self, 
        carbon_weight: float = 0.5,
        cost_weight: float = 0.3,
        urgency_weight: float = 0.2,
        defer_threshold: float = 0.3
    ):
        super().__init__("HybridAgent")
        self.carbon_weight = carbon_weight
        self.cost_weight = cost_weight
        self.urgency_weight = urgency_weight
        self.defer_threshold = defer_threshold
    
    def act(self, obs: Ko2cubeObservation) -> Ko2cubeAction:
        assignments = []
        
        for job in obs.job_queue:
            decision = self._make_decision(job, obs)
            assignments.append(decision)
        
        return Ko2cubeAction(assignments=assignments)
    
    def _compute_urgency(self, job: Job, current_step: int) -> float:
        """Compute urgency score (0-1, higher = more urgent)."""
        if job.sla_end == ALWAYS_ON:
            return 0.5  # Always-on jobs have medium urgency
        
        remaining = job.sla_end - current_step
        total_window = job.sla_end - job.sla_start
        
        if total_window <= 0:
            return 1.0  # Must schedule now
        
        # Urgency increases as deadline approaches
        progress = 1.0 - (remaining / total_window)
        return min(1.0, max(0.0, progress))
    
    def _forecast_improving(self, region: RegionInfo) -> bool:
        """Check if carbon forecast is improving."""
        if not region.carbon.forecast:
            return False
        
        current = region.carbon.current_intensity
        future_avg = sum(region.carbon.forecast[:3]) / min(3, len(region.carbon.forecast))
        
        return future_avg < current * 0.9  # 10% improvement threshold
    
    def _score_region(
        self, job: Job, region: RegionInfo, obs: Ko2cubeObservation
    ) -> Optional[Tuple[str, float]]:
        """Score a region for scheduling this job. Returns (instance, score) or None."""
        instance_name = self._get_fitting_instance(job, region)
        if not instance_name:
            return None
        
        # Get price
        price = 0.5  # Default
        for inst in region.available_instances:
            if inst.name == instance_name:
                price = (inst.spot_price if job.instance_preference == "spot"
                        else inst.on_demand_price)
                break
        
        carbon = region.carbon.current_intensity
        urgency = self._compute_urgency(job, obs.current_step)
        
        # Normalize scores
        carbon_score = carbon / 500.0
        cost_score = price / 1.0
        urgency_score = 1.0 - urgency  # Lower is more urgent
        
        total_score = (
            self.carbon_weight * carbon_score +
            self.cost_weight * cost_score +
            self.urgency_weight * urgency_score
        )
        
        return instance_name, total_score
    
    def _make_decision(
        self, job: Job, obs: Ko2cubeObservation
    ) -> JobAssignment:
        urgency = self._compute_urgency(job, obs.current_step)
        
        # Score all regions
        scored_regions = []
        for region_name, region in obs.regions.items():
            result = self._score_region(job, region, obs)
            if result:
                instance, score = result
                improving = self._forecast_improving(region)
                scored_regions.append((region_name, instance, score, improving))
        
        if not scored_regions:
            # No capacity
            if job.delay_tolerant and job.sla_end != ALWAYS_ON:
                defer_to = min(obs.current_step + 1, job.sla_end)
                self.metrics.jobs_deferred += 1
                return JobAssignment(
                    job_id=job.job_id,
                    decision="defer",
                    defer_to_step=defer_to
                )
            self.metrics.jobs_dropped += 1
            return JobAssignment(job_id=job.job_id, decision="drop")
        
        # Sort by score (lower is better)
        scored_regions.sort(key=lambda x: x[2])
        best_region, best_instance, best_score, best_improving = scored_regions[0]
        
        # Consider deferral if:
        # - Job is delay tolerant
        # - Not urgent
        # - Forecast is improving
        should_defer = (
            job.delay_tolerant and
            urgency < self.defer_threshold and
            best_improving and
            job.sla_end != ALWAYS_ON and
            job.sla_end > obs.current_step + 1
        )
        
        if should_defer:
            defer_to = min(obs.current_step + 2, job.sla_end)
            self.metrics.jobs_deferred += 1
            return JobAssignment(
                job_id=job.job_id,
                decision="defer",
                defer_to_step=defer_to
            )
        
        # Schedule now
        self.metrics.jobs_scheduled += 1
        return JobAssignment(
            job_id=job.job_id,
            decision="schedule",
            region=best_region,
            machine_type=job.instance_preference,
            instance_type=best_instance
        )


# Agent registry for easy access
BASELINE_AGENTS = {
    "random": RandomAgent,
    "greedy_cost": GreedyCostAgent,
    "carbon_aware": CarbonAwareGreedyAgent,
    "oracle": OracleAgent,
    "hybrid": HybridAgent,
}


def create_agent(name: str, **kwargs) -> BaselineAgent:
    """Create a baseline agent by name."""
    if name not in BASELINE_AGENTS:
        raise ValueError(f"Unknown agent: {name}. Available: {list(BASELINE_AGENTS.keys())}")
    return BASELINE_AGENTS[name](**kwargs)


def run_episode(
    env,
    agent: BaselineAgent,
    task_id: str = "easy",
    max_steps: int = 1000,
    verbose: bool = False
) -> AgentMetrics:
    """
    Run a single episode with the given agent.
    
    Args:
        env: Ko2cubeEnvironment instance
        agent: BaselineAgent instance
        task_id: Task/scenario to run
        max_steps: Maximum steps before forced termination
        verbose: Print progress
    
    Returns:
        AgentMetrics with episode statistics
    """
    agent.reset()
    obs = env.reset(task_id=task_id)
    
    for step in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        
        agent.update_metrics(result.reward)
        
        if verbose and step % 10 == 0:
            print(f"Step {step}: reward={result.reward:.2f}, "
                  f"total={agent.metrics.total_reward:.2f}")
        
        if result.done:
            break
        
        obs = result
    
    # Update final metrics from environment state
    state = env.state
    agent.metrics.total_carbon_gco2 = state.total_carbon_gco2
    agent.metrics.total_cost_usd = state.total_cost_usd
    agent.metrics.sla_violations = state.sla_violations
    
    return agent.metrics


def evaluate_baselines(
    env,
    task_id: str = "easy",
    num_episodes: int = 3,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Evaluate all baseline agents on a task.
    
    Returns a dictionary of agent_name -> aggregated metrics.
    """
    results = {}
    
    for agent_name, agent_class in BASELINE_AGENTS.items():
        agent = agent_class()
        episode_metrics = []
        
        for ep in range(num_episodes):
            metrics = run_episode(env, agent, task_id=task_id, verbose=False)
            episode_metrics.append(metrics)
            
            if verbose:
                print(f"{agent_name} episode {ep+1}: "
                      f"reward={metrics.total_reward:.2f}, "
                      f"carbon={metrics.total_carbon_gco2:.2f}")
        
        # Aggregate metrics
        results[agent_name] = {
            "mean_reward": sum(m.total_reward for m in episode_metrics) / num_episodes,
            "mean_carbon": sum(m.total_carbon_gco2 for m in episode_metrics) / num_episodes,
            "mean_cost": sum(m.total_cost_usd for m in episode_metrics) / num_episodes,
            "mean_sla_violations": sum(m.sla_violations for m in episode_metrics) / num_episodes,
            "episodes": num_episodes,
        }
    
    return results

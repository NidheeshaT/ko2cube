"""
Adversarial scenario generator for Ko2cube environment.

Dynamically generates challenging scenarios that target the agent's
weaknesses, promoting robust skill development.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

from models import Job, ALWAYS_ON
from server.data.scenarios import Scenario, TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD


@dataclass
class AdversarialConfig:
    """Configuration for adversarial scenario generation."""
    
    weakness_focus_ratio: float = 0.6
    noise_probability: float = 0.2
    carbon_variance: float = 0.3
    job_burst_probability: float = 0.15
    tight_sla_probability: float = 0.25
    
    regions: List[str] = None
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = ["us-east-1", "us-west-2", "eu-west-1"]


class AdversarialGenerator:
    """
    Generates adversarial scenarios targeting agent weaknesses.
    
    Uses mastery tracker data to create scenarios that stress-test
    the agent's weakest areas while maintaining playability.
    """
    
    JOB_TEMPLATES = {
        "etl": {
            "cpu_cores": 4,
            "memory_gb": 16,
            "eta_minutes": 45,
            "delay_tolerant": True,
            "instance_preference": "spot",
        },
        "cicd": {
            "cpu_cores": 8,
            "memory_gb": 16,
            "eta_minutes": 12,
            "delay_tolerant": False,
            "instance_preference": "spot",
        },
        "ml_training": {
            "cpu_cores": 0,
            "memory_gb": 64,
            "eta_minutes": 240,
            "delay_tolerant": True,
            "instance_preference": "spot",
        },
        "video_transcode": {
            "cpu_cores": 16,
            "memory_gb": 16,
            "eta_minutes": 35,
            "delay_tolerant": True,
            "instance_preference": "spot",
        },
        "db_backup": {
            "cpu_cores": 2,
            "memory_gb": 8,
            "eta_minutes": 20,
            "delay_tolerant": True,
            "instance_preference": "spot",
        },
        "api_serving": {
            "cpu_cores": 4,
            "memory_gb": 8,
            "eta_minutes": None,
            "delay_tolerant": False,
            "instance_preference": "on-demand",
        },
        "batch_report": {
            "cpu_cores": 16,
            "memory_gb": 32,
            "eta_minutes": 90,
            "delay_tolerant": True,
            "instance_preference": "spot",
        },
        "data_quality": {
            "cpu_cores": 4,
            "memory_gb": 16,
            "eta_minutes": 25,
            "delay_tolerant": True,
            "instance_preference": "spot",
        },
    }
    
    def __init__(self, config: Optional[AdversarialConfig] = None):
        """
        Initialize the adversarial generator.
        
        Args:
            config: Configuration for scenario generation
        """
        self.config = config or AdversarialConfig()
        self._job_counter = 0
    
    def _create_job(
        self,
        job_type: str,
        arrival_step: int,
        sla_window: Tuple[int, int],
        override: Optional[Dict] = None,
    ) -> Job:
        """Create a job from template with optional overrides."""
        self._job_counter += 1
        template = self.JOB_TEMPLATES.get(job_type, self.JOB_TEMPLATES["etl"])
        
        sla_start, sla_end = sla_window
        is_always_on = job_type == "api_serving"
        
        job_data = {
            "job_id": f"adv_{job_type}_{self._job_counter:04d}",
            "arrival_step": arrival_step,
            "sla_start": sla_start,
            "sla_end": ALWAYS_ON if is_always_on else sla_end,
            **template,
        }
        
        if override:
            job_data.update(override)
        
        return Job(**job_data)
    
    def generate_scenario(
        self,
        weak_job_types: List[str],
        weak_regions: List[str],
        base_difficulty: str = "medium",
        total_steps: int = 24,
    ) -> Scenario:
        """
        Generate an adversarial scenario targeting weaknesses.
        
        Args:
            weak_job_types: Job types the agent struggles with
            weak_regions: Regions where agent makes suboptimal choices
            base_difficulty: Base difficulty level
            total_steps: Episode length in steps
            
        Returns:
            A Scenario object with adversarial characteristics
        """
        self._job_counter = 0
        
        job_pool: List[Job] = []
        
        num_focus_jobs = int(12 * self.config.weakness_focus_ratio)
        for i in range(num_focus_jobs):
            if weak_job_types:
                job_type = random.choice(weak_job_types)
            else:
                job_type = random.choice(list(self.JOB_TEMPLATES.keys()))
            
            arrival = random.randint(0, total_steps // 2)
            sla_window = self._generate_sla_window(job_type, arrival, total_steps)
            
            if random.random() < self.config.tight_sla_probability:
                sla_start, sla_end = sla_window
                sla_end = min(sla_start + 2, sla_end)
                sla_window = (sla_start, sla_end)
            
            job = self._create_job(job_type, arrival, sla_window)
            job_pool.append(job)
        
        other_types = [t for t in self.JOB_TEMPLATES.keys() if t not in weak_job_types]
        for i in range(12 - num_focus_jobs):
            if other_types:
                job_type = random.choice(other_types)
            else:
                job_type = random.choice(list(self.JOB_TEMPLATES.keys()))
            
            arrival = random.randint(0, total_steps * 2 // 3)
            sla_window = self._generate_sla_window(job_type, arrival, total_steps)
            job = self._create_job(job_type, arrival, sla_window)
            job_pool.append(job)
        
        if random.random() < self.config.job_burst_probability:
            burst_step = random.randint(total_steps // 3, total_steps * 2 // 3)
            burst_size = random.randint(3, 5)
            
            for _ in range(burst_size):
                job = self._create_job(
                    "cicd",
                    burst_step,
                    (burst_step, burst_step + 1),
                )
                job_pool.append(job)
        
        if base_difficulty == "hard" or random.random() < 0.3:
            job_pool.append(self._create_job(
                "api_serving",
                0,
                (0, ALWAYS_ON),
            ))
        
        return Scenario(
            name=f"adversarial_{len(job_pool)}jobs",
            difficulty="adversarial",
            description=(
                f"Adversarial scenario targeting: {', '.join(weak_job_types[:3])}. "
                f"Generated with {len(job_pool)} jobs over {total_steps} steps."
            ),
            total_steps=total_steps,
            step_duration_minutes=60,
            lookahead_steps=min(24, total_steps),
            regions=self.config.regions,
            job_pool=job_pool,
        )
    
    def _generate_sla_window(
        self,
        job_type: str,
        arrival: int,
        total_steps: int,
    ) -> Tuple[int, int]:
        """Generate appropriate SLA window for a job type."""
        template = self.JOB_TEMPLATES.get(job_type, {})
        eta_minutes = template.get("eta_minutes")
        delay_tolerant = template.get("delay_tolerant", True)
        
        if eta_minutes is None:
            return (arrival, ALWAYS_ON)
        
        eta_steps = max(1, eta_minutes // 60)
        
        if not delay_tolerant:
            return (arrival, arrival + eta_steps)
        
        if job_type in ["etl", "ml_training"]:
            window_size = random.randint(4, 8)
        elif job_type in ["batch_report"]:
            window_size = random.randint(6, 10)
        else:
            window_size = random.randint(2, 4)
        
        sla_start = arrival
        sla_end = min(arrival + window_size, total_steps - 1)
        
        return (sla_start, sla_end)
    
    def mutate_scenario(self, base_scenario: Scenario, mutation_strength: float = 0.2) -> Scenario:
        """
        Create a mutated version of an existing scenario.
        
        Args:
            base_scenario: Scenario to mutate
            mutation_strength: How much to change (0.0 to 1.0)
            
        Returns:
            Mutated scenario
        """
        scenario = deepcopy(base_scenario)
        scenario.name = f"{scenario.name}_mutated"
        
        for job in scenario.job_pool:
            if random.random() < mutation_strength:
                if job.sla_end != ALWAYS_ON and job.delay_tolerant:
                    current_window = job.sla_end - job.sla_start
                    new_window = max(1, current_window - random.randint(1, 2))
                    job.sla_end = job.sla_start + new_window
            
            if random.random() < mutation_strength * 0.5:
                shift = random.randint(-2, 2)
                job.arrival_step = max(0, job.arrival_step + shift)
                if job.sla_end != ALWAYS_ON:
                    job.sla_start = max(job.arrival_step, job.sla_start + shift)
                    job.sla_end = max(job.sla_start + 1, job.sla_end + shift)
        
        return scenario
    
    def generate_from_mastery(
        self,
        mastery_stats: Dict,
        base_difficulty: str = "medium",
    ) -> Scenario:
        """
        Generate scenario directly from mastery tracker stats.
        
        Args:
            mastery_stats: Stats dict from MasteryTracker.get_stats()
            base_difficulty: Base difficulty level
            
        Returns:
            Adversarial scenario targeting weak areas
        """
        job_type_success = mastery_stats.get("job_type_success", {})
        
        if job_type_success:
            sorted_types = sorted(job_type_success.items(), key=lambda x: x[1])
            weak_types = [t[0] for t in sorted_types[:3]]
        else:
            weak_types = ["cicd", "api_serving"]
        
        region_efficiency = mastery_stats.get("region_efficiency", {})
        if region_efficiency:
            sorted_regions = sorted(region_efficiency.items(), key=lambda x: x[1])
            weak_regions = [r[0] for r in sorted_regions[:2]]
        else:
            weak_regions = []
        
        return self.generate_scenario(
            weak_job_types=weak_types,
            weak_regions=weak_regions,
            base_difficulty=base_difficulty,
        )

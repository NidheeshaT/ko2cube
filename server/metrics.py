"""
Prometheus metrics for Ko2cube environment observability.

Provides metrics for monitoring:
- Carbon savings and emissions
- Episode duration and completion
- Job scheduling patterns
- SLA compliance
- Training progress
"""

import time
from typing import Optional
from contextlib import contextmanager

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


if PROMETHEUS_AVAILABLE:
    carbon_savings_total = Counter(
        "ko2cube_carbon_savings_gco2_total",
        "Total carbon saved vs baseline (gCO2)",
        ["scenario", "difficulty"],
    )

    carbon_emitted_total = Counter(
        "ko2cube_carbon_emitted_gco2_total",
        "Total carbon emitted (gCO2)",
        ["scenario", "difficulty", "region"],
    )

    cost_total = Counter(
        "ko2cube_cost_usd_total",
        "Total cost incurred (USD)",
        ["scenario", "difficulty"],
    )

    episode_duration = Histogram(
        "ko2cube_episode_duration_seconds",
        "Episode completion time in seconds",
        ["scenario", "difficulty"],
        buckets=(1, 5, 10, 30, 60, 120, 300, 600),
    )

    episode_steps = Histogram(
        "ko2cube_episode_steps_total",
        "Number of steps per episode",
        ["scenario"],
        buckets=(10, 20, 30, 40, 50, 100),
    )

    episode_reward = Summary(
        "ko2cube_episode_reward_total",
        "Total reward per episode",
        ["scenario", "difficulty"],
    )

    grader_score = Gauge(
        "ko2cube_grader_score",
        "Current grader score (0-1)",
        ["scenario"],
    )

    current_carbon_intensity = Gauge(
        "ko2cube_carbon_intensity_gco2_kwh",
        "Current carbon intensity by region",
        ["region"],
    )

    jobs_scheduled = Counter(
        "ko2cube_jobs_scheduled_total",
        "Total jobs scheduled",
        ["job_type", "region", "machine_type"],
    )

    jobs_completed = Counter(
        "ko2cube_jobs_completed_total",
        "Total jobs completed successfully",
        ["job_type", "region"],
    )

    jobs_dropped = Counter(
        "ko2cube_jobs_dropped_total",
        "Total jobs dropped",
        ["job_type", "reason"],
    )

    jobs_deferred = Counter(
        "ko2cube_jobs_deferred_total",
        "Total jobs deferred",
        ["job_type"],
    )

    sla_violations = Counter(
        "ko2cube_sla_violations_total",
        "Total SLA violations",
        ["job_type", "reason"],
    )

    active_jobs_gauge = Gauge(
        "ko2cube_active_jobs",
        "Number of currently running jobs",
        ["region"],
    )

    job_queue_size = Gauge(
        "ko2cube_job_queue_size",
        "Number of jobs waiting in queue",
    )

    training_episodes = Counter(
        "ko2cube_training_episodes_total",
        "Total training episodes completed",
        ["model"],
    )

    curriculum_level = Gauge(
        "ko2cube_curriculum_level",
        "Current curriculum difficulty level (0-4)",
    )

    environment_info = Info(
        "ko2cube_environment",
        "Environment information",
    )


class MetricsCollector:
    """
    Collects and records metrics during environment execution.
    
    Works with or without prometheus_client installed.
    When prometheus is not available, metrics are tracked internally
    and can be retrieved via get_internal_metrics().
    """

    def __init__(self, scenario: str = "unknown", difficulty: str = "unknown"):
        """
        Initialize the metrics collector.
        
        Args:
            scenario: Current scenario name
            difficulty: Current difficulty level
        """
        self.scenario = scenario
        self.difficulty = difficulty
        self._episode_start: Optional[float] = None
        
        self._internal_metrics = {
            "carbon_savings_gco2": 0.0,
            "carbon_emitted_gco2": 0.0,
            "cost_usd": 0.0,
            "jobs_scheduled": 0,
            "jobs_completed": 0,
            "jobs_dropped": 0,
            "jobs_deferred": 0,
            "sla_violations": 0,
            "episode_count": 0,
            "total_reward": 0.0,
        }

    def set_context(self, scenario: str, difficulty: str) -> None:
        """Update the scenario context."""
        self.scenario = scenario
        self.difficulty = difficulty

    def start_episode(self) -> None:
        """Mark the start of a new episode."""
        self._episode_start = time.time()
        self._internal_metrics["episode_count"] += 1

    def end_episode(self, total_reward: float, steps: int, grader: float) -> None:
        """
        Record end-of-episode metrics.
        
        Args:
            total_reward: Total reward for the episode
            steps: Number of steps taken
            grader: Final grader score (0-1)
        """
        if self._episode_start is not None:
            duration = time.time() - self._episode_start
            
            if PROMETHEUS_AVAILABLE:
                episode_duration.labels(
                    scenario=self.scenario,
                    difficulty=self.difficulty,
                ).observe(duration)
                
                episode_steps.labels(scenario=self.scenario).observe(steps)
                episode_reward.labels(
                    scenario=self.scenario,
                    difficulty=self.difficulty,
                ).observe(total_reward)
                grader_score.labels(scenario=self.scenario).set(grader)
        
        self._internal_metrics["total_reward"] += total_reward
        self._episode_start = None

    def record_carbon_savings(self, savings_gco2: float) -> None:
        """Record carbon savings vs baseline."""
        if PROMETHEUS_AVAILABLE:
            carbon_savings_total.labels(
                scenario=self.scenario,
                difficulty=self.difficulty,
            ).inc(max(0, savings_gco2))
        
        self._internal_metrics["carbon_savings_gco2"] += max(0, savings_gco2)

    def record_carbon_emission(self, gco2: float, region: str) -> None:
        """Record carbon emitted for a job."""
        if PROMETHEUS_AVAILABLE:
            carbon_emitted_total.labels(
                scenario=self.scenario,
                difficulty=self.difficulty,
                region=region,
            ).inc(gco2)
        
        self._internal_metrics["carbon_emitted_gco2"] += gco2

    def record_cost(self, usd: float) -> None:
        """Record cost incurred."""
        if PROMETHEUS_AVAILABLE:
            cost_total.labels(
                scenario=self.scenario,
                difficulty=self.difficulty,
            ).inc(usd)
        
        self._internal_metrics["cost_usd"] += usd

    def record_job_scheduled(
        self,
        job_type: str,
        region: str,
        machine_type: str,
    ) -> None:
        """Record a job being scheduled."""
        if PROMETHEUS_AVAILABLE:
            jobs_scheduled.labels(
                job_type=job_type,
                region=region,
                machine_type=machine_type,
            ).inc()
        
        self._internal_metrics["jobs_scheduled"] += 1

    def record_job_completed(self, job_type: str, region: str) -> None:
        """Record a job completing successfully."""
        if PROMETHEUS_AVAILABLE:
            jobs_completed.labels(
                job_type=job_type,
                region=region,
            ).inc()
        
        self._internal_metrics["jobs_completed"] += 1

    def record_job_dropped(self, job_type: str, reason: str = "agent_decision") -> None:
        """Record a job being dropped."""
        if PROMETHEUS_AVAILABLE:
            jobs_dropped.labels(
                job_type=job_type,
                reason=reason,
            ).inc()
        
        self._internal_metrics["jobs_dropped"] += 1

    def record_job_deferred(self, job_type: str) -> None:
        """Record a job being deferred."""
        if PROMETHEUS_AVAILABLE:
            jobs_deferred.labels(job_type=job_type).inc()
        
        self._internal_metrics["jobs_deferred"] += 1

    def record_sla_violation(self, job_type: str, reason: str = "deadline_missed") -> None:
        """Record an SLA violation."""
        if PROMETHEUS_AVAILABLE:
            sla_violations.labels(
                job_type=job_type,
                reason=reason,
            ).inc()
        
        self._internal_metrics["sla_violations"] += 1

    def update_carbon_intensity(self, region: str, intensity: float) -> None:
        """Update current carbon intensity for a region."""
        if PROMETHEUS_AVAILABLE:
            current_carbon_intensity.labels(region=region).set(intensity)

    def update_active_jobs(self, region: str, count: int) -> None:
        """Update count of active jobs in a region."""
        if PROMETHEUS_AVAILABLE:
            active_jobs_gauge.labels(region=region).set(count)

    def update_queue_size(self, size: int) -> None:
        """Update the job queue size."""
        if PROMETHEUS_AVAILABLE:
            job_queue_size.set(size)

    def record_training_episode(self, model: str = "unknown") -> None:
        """Record completion of a training episode."""
        if PROMETHEUS_AVAILABLE:
            training_episodes.labels(model=model).inc()

    def update_curriculum_level(self, level: int) -> None:
        """Update the current curriculum level."""
        if PROMETHEUS_AVAILABLE:
            curriculum_level.set(level)

    def get_internal_metrics(self) -> dict:
        """Get internal metrics (works without prometheus)."""
        return dict(self._internal_metrics)

    def reset_internal_metrics(self) -> None:
        """Reset internal counters."""
        for key in self._internal_metrics:
            if isinstance(self._internal_metrics[key], (int, float)):
                self._internal_metrics[key] = 0.0 if isinstance(
                    self._internal_metrics[key], float
                ) else 0


@contextmanager
def episode_timer(collector: MetricsCollector):
    """Context manager to time an episode."""
    collector.start_episode()
    try:
        yield
    finally:
        pass


def get_job_type_from_id(job_id: str) -> str:
    """Extract job type from job ID."""
    prefixes = [
        "etl", "cicd", "ml_training", "video_transcode",
        "db_backup", "api_serving", "batch_report", "dq_scan", "adv"
    ]
    
    for prefix in prefixes:
        if job_id.startswith(prefix):
            return prefix
    
    return "unknown"


def setup_environment_info(version: str = "0.1.0", mode: str = "training") -> None:
    """Set environment info metric."""
    if PROMETHEUS_AVAILABLE:
        environment_info.info({
            "version": version,
            "mode": mode,
        })

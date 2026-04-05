"""
Scenario definitions for the Ko2cube - carbon-aware scheduling environment.

A Scenario bundles:
  - Metadata (name, difficulty, timing config)
  - A job_pool: the full set of job templates available in this scenario
  - Each job has an `arrival_step` which controls when it surfaces into
    the agent's job_queue during the episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from models import Job, ALWAYS_ON

# Scenario dataclass
@dataclass
class Scenario:
    """A reproducible (task, job_pool) pair for one episode."""
    name: str
    difficulty: str                    # "easy" | "medium" | "hard"
    description: str
    total_steps: int                   # how many simulation steps per episode
    step_duration_minutes: int         # how many real minutes each step represents
    lookahead_steps: int               # how many steps of carbon forecast to show
    regions: List[str]                 # regions available in this scenario
    job_pool: List[Job]                # full job list - surfaced by arrival_step

# Job template helpers
def _etl(arrival: int, sla_start: int, sla_end: int) -> Job:
    """ETL Pipeline - Workload 1. Delay tolerant, wide window, spot OK."""
    return Job(
        job_id=f"etl_sales_{arrival:03d}",
        arrival_step=arrival,
        eta_minutes=45,
        cpu_cores=4,
        memory_gb=16,
        sla_start=sla_start,
        sla_end=sla_end,
        delay_tolerant=True,
        instance_preference="spot",
    )


def _db_backup(arrival: int, sla_start: int, sla_end: int) -> Job:
    """DB Backup - Workload 5. Tight 2-hour maintenance window."""
    return Job(
        job_id=f"db_backup_{arrival:03d}",
        arrival_step=arrival,
        eta_minutes=20,
        cpu_cores=2,
        memory_gb=8,
        sla_start=sla_start,
        sla_end=sla_end,
        delay_tolerant=True,
        instance_preference="spot",
    )


def _cicd(arrival: int, sla_start: int, uid: str = "") -> Job:
    """CI/CD Build - Workload 3. Developer waiting. Must NOT be deferred."""
    suffix = f"_{uid}" if uid else ""
    return Job(
        job_id=f"cicd_build_{arrival:03d}{suffix}",
        arrival_step=arrival,
        eta_minutes=12,
        cpu_cores=8,
        memory_gb=16,
        sla_start=sla_start,
        sla_end=sla_start + 1,   # 1-step (1 hour) hard deadline
        delay_tolerant=False,
        instance_preference="spot",
    )


def _video_transcode(arrival: int, sla_start: int, sla_end: int) -> Job:
    """Video Transcode - Workload 6. 2-hour window, moderate flexibility."""
    return Job(
        job_id=f"video_transcode_{arrival:03d}",
        arrival_step=arrival,
        eta_minutes=35,
        cpu_cores=16,
        memory_gb=16,
        sla_start=sla_start,
        sla_end=sla_end,
        delay_tolerant=True,
        instance_preference="spot",
    )


def _ml_training(arrival: int, sla_start: int, sla_end: int) -> Job:
    """ML Training - Workload 2. 4-hour GPU job, delay tolerant, 12-hour window."""
    return Job(
        job_id=f"ml_training_{arrival:03d}",
        arrival_step=arrival,
        eta_minutes=240,
        cpu_cores=0,
        memory_gb=64,
        sla_start=sla_start,
        sla_end=sla_end,
        delay_tolerant=True,
        instance_preference="spot",
    )


def _batch_report(arrival: int, sla_start: int, sla_end: int) -> Job:
    """Batch Report - Workload 4. 90 min, 8.5 hr window, hard deadline."""
    return Job(
        job_id=f"batch_report_{arrival:03d}",
        arrival_step=arrival,
        eta_minutes=90,
        cpu_cores=16,
        memory_gb=32,
        sla_start=sla_start,
        sla_end=sla_end,
        delay_tolerant=True,
        instance_preference="spot",
    )


def _data_quality(arrival: int, sla_start: int, sla_end: int) -> Job:
    """Data Quality Scan - Workload 8. 25 min, tight morning window."""
    return Job(
        job_id=f"dq_scan_{arrival:03d}",
        arrival_step=arrival,
        eta_minutes=25,
        cpu_cores=4,
        memory_gb=16,
        sla_start=sla_start,
        sla_end=sla_end,
        delay_tolerant=True,
        instance_preference="spot",
    )


def _api_serving(arrival: int) -> Job:
    """API Serving - Workload 7. Always-on, never defer, on-demand only."""
    return Job(
        job_id=f"api_serving_{arrival:03d}",
        arrival_step=arrival,
        eta_minutes=None,           # runs forever
        cpu_cores=4,
        memory_gb=8,
        sla_start=arrival,
        sla_end=ALWAYS_ON,          # sentinel: never expires
        delay_tolerant=False,
        instance_preference="on-demand",
    )


# Task 1 - Easy
# Only delay-tolerant jobs (ETL + DB Backup). Single region (us-east).
# Wide SLA windows. Agent just needs to learn: defer during high-carbon hours.
# Episode = 24 steps (24 simulated hours). Jobs arrive in first 12 steps.
TASK_1_EASY = Scenario(
    name="task1_easy",
    difficulty="easy",
    description=(
        "Only delay-tolerant batch jobs. Single region. Clear daily carbon cycle. "
        "Agent must learn to defer jobs to clean windows and avoid high-carbon hours."
    ),
    total_steps=24,
    step_duration_minutes=60,
    lookahead_steps=24,
    regions=["us-east"],
    job_pool=[
        # ETL jobs arriving throughout the day
        _etl(arrival=0,  sla_start=0,  sla_end=6),
        _etl(arrival=2,  sla_start=2,  sla_end=10),
        _etl(arrival=6,  sla_start=6,  sla_end=14),
        _etl(arrival=10, sla_start=10, sla_end=18),
        # DB Backup jobs in expected maintenance windows
        _db_backup(arrival=1,  sla_start=2,  sla_end=4),
        _db_backup(arrival=8,  sla_start=9,  sla_end=11),
        _db_backup(arrival=14, sla_start=14, sla_end=16),
        # Data quality scans (early morning)
        _data_quality(arrival=4, sla_start=5, sla_end=8),
        _data_quality(arrival=12, sla_start=12, sla_end=15),
    ],
)


# Task 2 - Medium
# Mix of delay-tolerant + non-deferrable jobs. Three regions.
# Agent must correctly classify urgency. Misclassifying CI/CD = SLA breach.
# Episode = 24 steps.
TASK_2_MEDIUM = Scenario(
    name="task2_medium",
    difficulty="medium",
    description=(
        "Three regions. Mix of CI/CD builds (non-deferrable) + ETL + Video Transcode. "
        "Agent must classify job urgency correctly and pick regions for carbon savings."
    ),
    total_steps=24,
    step_duration_minutes=60,
    lookahead_steps=24,
    regions=["us-east", "us-west", "eu-west"],
    job_pool=[
        # Delay-tolerant batch
        _etl(arrival=0,  sla_start=0,  sla_end=8),
        _etl(arrival=4,  sla_start=4,  sla_end=12),
        _etl(arrival=10, sla_start=10, sla_end=20),
        _video_transcode(arrival=2,  sla_start=2,  sla_end=4),
        _video_transcode(arrival=8,  sla_start=8,  sla_end=10),
        _video_transcode(arrival=14, sla_start=14, sla_end=16),
        _data_quality(arrival=4, sla_start=5, sla_end=8),
        _db_backup(arrival=2, sla_start=2, sla_end=4),
        # Non-deferrable CI/CD - agent must run immediately
        _cicd(arrival=3,  sla_start=3),
        _cicd(arrival=7,  sla_start=7),
        _cicd(arrival=11, sla_start=11),
        _cicd(arrival=17, sla_start=17),
    ],
)


# Task 3 - Hard
# All 8 workload types simultaneously. Mid-episode CI/CD burst (steps 10-14).
# API Serving is always-on and must never be touched.
# ML Training competes with Video Transcode for GPU capacity.
# Agent must triage, protect urgent jobs, recover after burst.
# Episode = 48 steps (2 simulated days).
TASK_3_HARD = Scenario(
    name="task3_hard",
    difficulty="hard",
    description=(
        "All 8 workload types simultaneously. Mid-episode CI/CD burst at steps 10-14. "
        "API Serving always-on job must never be deferred. GPU capacity contention. "
        "Agent must triage across all job types and recover after the burst."
    ),
    total_steps=48,
    step_duration_minutes=60,
    lookahead_steps=24,
    regions=["us-east", "us-west", "eu-west"],
    job_pool=[
        # Always-on API (must protect throughout)
        _api_serving(arrival=0),

        # Batch / delay-tolerant baseline
        _etl(arrival=0,  sla_start=0,  sla_end=8),
        _etl(arrival=6,  sla_start=6,  sla_end=14),
        _etl(arrival=18, sla_start=18, sla_end=26),
        _etl(arrival=30, sla_start=30, sla_end=40),

        _db_backup(arrival=2,  sla_start=2,  sla_end=4),
        _db_backup(arrival=26, sla_start=26, sla_end=28),

        _batch_report(arrival=0, sla_start=0, sla_end=9),
        _batch_report(arrival=24, sla_start=24, sla_end=33),

        _data_quality(arrival=4, sla_start=5, sla_end=8),
        _data_quality(arrival=28, sla_start=29, sla_end=32),

        # GPU-heavy (competition for capacity)
        _ml_training(arrival=4,  sla_start=4,  sla_end=16),
        _ml_training(arrival=20, sla_start=20, sla_end=32),
        _video_transcode(arrival=5,  sla_start=5,  sla_end=7),
        _video_transcode(arrival=9,  sla_start=9,  sla_end=11),
        _video_transcode(arrival=21, sla_start=21, sla_end=23),

        # Normal CI/CD before burst
        _cicd(arrival=2, sla_start=2),
        _cicd(arrival=8, sla_start=8),

        # --- Mid-episode CI/CD burst (steps 10-14) ---
        _cicd(arrival=10, sla_start=10, uid="a"),
        _cicd(arrival=10, sla_start=10, uid="b"),
        _cicd(arrival=11, sla_start=11),
        _cicd(arrival=12, sla_start=12, uid="a"),
        _cicd(arrival=12, sla_start=12, uid="b"),
        _cicd(arrival=13, sla_start=13),
        _cicd(arrival=14, sla_start=14),

        # Recovery phase CI/CD
        _cicd(arrival=20, sla_start=20),
        _cicd(arrival=30, sla_start=30),
    ],
)


# Scenario registry - used by environment to look up by name or difficulty
SCENARIO_REGISTRY: dict[str, Scenario] = {
    "task1_easy":   TASK_1_EASY,
    "task2_medium": TASK_2_MEDIUM,
    "task3_hard":   TASK_3_HARD,
}

DIFFICULTY_MAP: dict[str, Scenario] = {
    "easy":   TASK_1_EASY,
    "medium": TASK_2_MEDIUM,
    "hard":   TASK_3_HARD,
}


def get_scenario(name: str) -> Scenario:
    """Look up a scenario by name or difficulty string."""
    if name in SCENARIO_REGISTRY:
        return SCENARIO_REGISTRY[name]
    if name in DIFFICULTY_MAP:
        return DIFFICULTY_MAP[name]
    raise ValueError(
        f"Unknown scenario '{name}'. "
        f"Available: {list(SCENARIO_REGISTRY.keys())}"
    )

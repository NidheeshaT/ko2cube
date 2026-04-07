"""
Shared pytest fixtures for ko2cube tests.
"""
import os
import sys
import pytest
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import Ko2cubeEnvironment
from server.data.scenarios import (
    Scenario, TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD,
    SCENARIO_REGISTRY, get_scenario,
)
from models import (
    Ko2cubeAction, Ko2cubeObservation, Ko2cubeState,
    Job, JobAssignment, RegionInfo, CarbonData, InstanceType,
    ALWAYS_ON,
)


@pytest.fixture
def environment() -> Ko2cubeEnvironment:
    """Fresh environment instance for each test."""
    return Ko2cubeEnvironment()


@pytest.fixture
def easy_environment(environment: Ko2cubeEnvironment) -> Ko2cubeEnvironment:
    """Environment reset to easy difficulty."""
    environment.reset(task_id="easy")
    return environment


@pytest.fixture
def medium_environment(environment: Ko2cubeEnvironment) -> Ko2cubeEnvironment:
    """Environment reset to medium difficulty."""
    environment.reset(task_id="medium")
    return environment


@pytest.fixture
def hard_environment(environment: Ko2cubeEnvironment) -> Ko2cubeEnvironment:
    """Environment reset to hard difficulty."""
    environment.reset(task_id="hard")
    return environment


@pytest.fixture
def easy_scenario() -> Scenario:
    """Returns the easy scenario definition."""
    return TASK_1_EASY


@pytest.fixture
def medium_scenario() -> Scenario:
    """Returns the medium scenario definition."""
    return TASK_2_MEDIUM


@pytest.fixture
def hard_scenario() -> Scenario:
    """Returns the hard scenario definition."""
    return TASK_3_HARD


@pytest.fixture
def all_scenarios() -> List[Scenario]:
    """Returns all scenarios."""
    return [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD]


@pytest.fixture
def sample_job() -> Job:
    """A simple delay-tolerant job for testing."""
    return Job(
        job_id="test_job_001",
        arrival_step=0,
        eta_minutes=60,
        cpu_cores=2,
        memory_gb=8,
        sla_start=0,
        sla_end=10,
        delay_tolerant=True,
        instance_preference="spot",
    )


@pytest.fixture
def urgent_job() -> Job:
    """A non-deferrable urgent job for testing."""
    return Job(
        job_id="urgent_job_001",
        arrival_step=0,
        eta_minutes=30,
        cpu_cores=4,
        memory_gb=16,
        sla_start=0,
        sla_end=1,
        delay_tolerant=False,
        instance_preference="spot",
    )


@pytest.fixture
def always_on_job() -> Job:
    """An always-on job that should never be deferred."""
    return Job(
        job_id="always_on_001",
        arrival_step=0,
        eta_minutes=None,
        cpu_cores=2,
        memory_gb=8,
        sla_start=0,
        sla_end=ALWAYS_ON,
        delay_tolerant=False,
        instance_preference="on-demand",
    )


@pytest.fixture
def sample_region_info() -> RegionInfo:
    """Sample region data for reward calculation tests."""
    return RegionInfo(
        region_name="us-east-1",
        carbon=CarbonData(
            current_intensity=250.0,
            forecast=[250.0, 230.0, 210.0, 200.0, 220.0, 240.0],
        ),
        available_instances=[
            InstanceType(
                name="m5.large",
                cpu_cores=2,
                memory_gb=8,
                spot_price=0.05,
                on_demand_price=0.096,
                available_count=10,
            ),
            InstanceType(
                name="m5.xlarge",
                cpu_cores=4,
                memory_gb=16,
                spot_price=0.10,
                on_demand_price=0.192,
                available_count=10,
            ),
        ],
    )


@pytest.fixture
def sample_regions() -> Dict[str, RegionInfo]:
    """Multiple regions with varying carbon intensities."""
    return {
        "us-east-1": RegionInfo(
            region_name="us-east-1",
            carbon=CarbonData(current_intensity=350.0, forecast=[350.0, 320.0, 300.0]),
            available_instances=[
                InstanceType(name="m5.large", cpu_cores=2, memory_gb=8,
                            spot_price=0.05, on_demand_price=0.096, available_count=10),
                InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                            spot_price=0.10, on_demand_price=0.192, available_count=10),
            ],
        ),
        "us-west-2": RegionInfo(
            region_name="us-west-2",
            carbon=CarbonData(current_intensity=150.0, forecast=[150.0, 140.0, 130.0]),
            available_instances=[
                InstanceType(name="m5.large", cpu_cores=2, memory_gb=8,
                            spot_price=0.06, on_demand_price=0.096, available_count=10),
                InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                            spot_price=0.12, on_demand_price=0.192, available_count=10),
            ],
        ),
        "eu-west-1": RegionInfo(
            region_name="eu-west-1",
            carbon=CarbonData(current_intensity=100.0, forecast=[100.0, 90.0, 85.0]),
            available_instances=[
                InstanceType(name="m5.large", cpu_cores=2, memory_gb=8,
                            spot_price=0.055, on_demand_price=0.096, available_count=10),
                InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                            spot_price=0.11, on_demand_price=0.192, available_count=10),
            ],
        ),
    }


def create_schedule_action(
    job: Job,
    region: str = "us-east-1",
    instance_type: str = "m5.large",
    machine_type: str = "spot",
) -> Ko2cubeAction:
    """Helper to create a schedule action for a single job."""
    return Ko2cubeAction(
        assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=region,
                instance_type=instance_type,
                machine_type=machine_type,
            )
        ]
    )


def create_defer_action(job: Job, defer_to_step: int) -> Ko2cubeAction:
    """Helper to create a defer action for a single job."""
    return Ko2cubeAction(
        assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="defer",
                defer_to_step=defer_to_step,
            )
        ]
    )


def create_drop_action(job: Job) -> Ko2cubeAction:
    """Helper to create a drop action for a single job."""
    return Ko2cubeAction(
        assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="drop",
            )
        ]
    )


def schedule_all_jobs_in_region(
    jobs: List[Job],
    region: str = "us-east-1",
    instance_type: str = "m5.xlarge",
) -> Ko2cubeAction:
    """Helper to schedule all jobs in a given region."""
    return Ko2cubeAction(
        assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=region,
                instance_type=instance_type,
                machine_type=job.instance_preference,
            )
            for job in jobs
        ]
    )

"""
Tests for Pydantic models in models.py.
Validates model construction, field constraints, and serialization.
"""
import pytest
from pydantic import ValidationError

from models import (
    Job, JobAssignment, RunningJob,
    Ko2cubeAction, Ko2cubeObservation, Ko2cubeState,
    CarbonData, InstanceType, RegionInfo,
    ALWAYS_ON,
)


class TestJobModel:
    """Tests for the Job model."""

    def test_job_creation_minimal(self):
        """Job can be created with required fields."""
        job = Job(
            job_id="test_001",
            cpu_cores=2,
            memory_gb=8,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        assert job.job_id == "test_001"
        assert job.status == "queued"
        assert job.arrival_step == 0

    def test_job_creation_full(self):
        """Job can be created with all fields."""
        job = Job(
            job_id="full_job",
            arrival_step=5,
            eta_minutes=120,
            cpu_cores=8,
            memory_gb=32,
            sla_start=5,
            sla_end=15,
            delay_tolerant=False,
            instance_preference="on-demand",
            status="running",
            start_step=6,
            region="us-west-2",
            machine_type="on-demand",
        )
        assert job.eta_minutes == 120
        assert job.status == "running"
        assert job.region == "us-west-2"

    def test_job_always_on_sentinel(self):
        """Always-on jobs use ALWAYS_ON sentinel value."""
        job = Job(
            job_id="always_on",
            cpu_cores=2,
            memory_gb=8,
            sla_start=0,
            sla_end=ALWAYS_ON,
            delay_tolerant=False,
            instance_preference="on-demand",
        )
        assert job.sla_end == -1
        assert job.sla_end == ALWAYS_ON

    def test_job_instance_preference_validation(self):
        """Instance preference must be 'spot' or 'on-demand'."""
        with pytest.raises(ValidationError):
            Job(
                job_id="bad_job",
                cpu_cores=2,
                memory_gb=8,
                sla_start=0,
                sla_end=10,
                delay_tolerant=True,
                instance_preference="invalid",
            )

    def test_job_status_values(self):
        """Job status must be one of the allowed values."""
        valid_statuses = ["queued", "running", "completed", "dropped", "sla_missed"]
        for status in valid_statuses:
            job = Job(
                job_id=f"job_{status}",
                cpu_cores=2,
                memory_gb=8,
                sla_start=0,
                sla_end=10,
                delay_tolerant=True,
                instance_preference="spot",
                status=status,
            )
            assert job.status == status


class TestJobAssignmentModel:
    """Tests for the JobAssignment model."""

    def test_schedule_assignment(self):
        """Schedule assignment with all required fields."""
        assignment = JobAssignment(
            job_id="job_001",
            decision="schedule",
            region="us-east-1",
            instance_type="m5.xlarge",
            machine_type="spot",
        )
        assert assignment.decision == "schedule"
        assert assignment.region == "us-east-1"

    def test_defer_assignment(self):
        """Defer assignment with defer_to_step."""
        assignment = JobAssignment(
            job_id="job_002",
            decision="defer",
            defer_to_step=5,
        )
        assert assignment.decision == "defer"
        assert assignment.defer_to_step == 5

    def test_drop_assignment(self):
        """Drop assignment requires minimal fields."""
        assignment = JobAssignment(
            job_id="job_003",
            decision="drop",
        )
        assert assignment.decision == "drop"
        assert assignment.region is None

    def test_invalid_decision(self):
        """Invalid decision value should fail validation."""
        with pytest.raises(ValidationError):
            JobAssignment(
                job_id="bad_job",
                decision="invalid_decision",
            )


class TestRunningJobModel:
    """Tests for the RunningJob model."""

    def test_running_job_creation(self):
        """RunningJob can be created with required fields."""
        rj = RunningJob(
            job_id="running_001",
            region="us-west-2",
            steps_remaining=3,
            machine_type="spot",
        )
        assert rj.steps_remaining == 3
        assert rj.machine_type == "spot"


class TestCarbonDataModel:
    """Tests for the CarbonData model."""

    def test_carbon_data_creation(self):
        """CarbonData can be created with intensity and forecast."""
        carbon = CarbonData(
            current_intensity=250.5,
            forecast=[250.5, 240.0, 230.0, 220.0],
        )
        assert carbon.current_intensity == 250.5
        assert len(carbon.forecast) == 4

    def test_carbon_data_empty_forecast(self):
        """CarbonData can have empty forecast."""
        carbon = CarbonData(
            current_intensity=100.0,
            forecast=[],
        )
        assert carbon.forecast == []


class TestInstanceTypeModel:
    """Tests for the InstanceType model."""

    def test_instance_type_creation(self):
        """InstanceType can be created with all fields."""
        instance = InstanceType(
            name="m5.xlarge",
            cpu_cores=4,
            memory_gb=16,
            spot_price=0.10,
            on_demand_price=0.192,
            available_count=5,
        )
        assert instance.name == "m5.xlarge"
        assert instance.spot_price < instance.on_demand_price


class TestRegionInfoModel:
    """Tests for the RegionInfo model."""

    def test_region_info_creation(self):
        """RegionInfo can be created with carbon and instances."""
        region = RegionInfo(
            region_name="eu-west-1",
            carbon=CarbonData(current_intensity=100.0, forecast=[100.0, 90.0]),
            available_instances=[
                InstanceType(
                    name="m5.large",
                    cpu_cores=2,
                    memory_gb=8,
                    spot_price=0.05,
                    on_demand_price=0.096,
                    available_count=10,
                ),
            ],
        )
        assert region.region_name == "eu-west-1"
        assert len(region.available_instances) == 1


class TestKo2cubeActionModel:
    """Tests for the Ko2cubeAction model."""

    def test_action_with_single_assignment(self):
        """Action can contain a single assignment."""
        action = Ko2cubeAction(
            assignments=[
                JobAssignment(job_id="job_001", decision="schedule",
                             region="us-east-1", instance_type="m5.large",
                             machine_type="spot"),
            ]
        )
        assert len(action.assignments) == 1

    def test_action_with_multiple_assignments(self):
        """Action can contain multiple assignments."""
        action = Ko2cubeAction(
            assignments=[
                JobAssignment(job_id="job_001", decision="schedule",
                             region="us-east-1", instance_type="m5.large",
                             machine_type="spot"),
                JobAssignment(job_id="job_002", decision="defer",
                             defer_to_step=5),
                JobAssignment(job_id="job_003", decision="drop"),
            ]
        )
        assert len(action.assignments) == 3

    def test_action_empty_assignments(self):
        """Action can have empty assignments list."""
        action = Ko2cubeAction(assignments=[])
        assert len(action.assignments) == 0


class TestKo2cubeObservationModel:
    """Tests for the Ko2cubeObservation model."""

    def test_observation_creation(self):
        """Observation can be created with all required fields."""
        obs = Ko2cubeObservation(
            current_step=5,
            job_queue=[],
            active_jobs=[],
            regions={},
            last_action_result="Test result",
            done=False,
            reward=0.0,
        )
        assert obs.current_step == 5
        assert obs.done is False


class TestKo2cubeStateModel:
    """Tests for the Ko2cubeState model."""

    def test_state_defaults(self):
        """State initializes with correct defaults."""
        state = Ko2cubeState()
        assert state.current_step == 0
        assert state.total_carbon_gco2 == 0.0
        assert state.total_cost_usd == 0.0
        assert state.sla_violations == 0
        assert state.jobs_completed == 0
        assert state.is_done is False

    def test_state_with_values(self):
        """State can be created with custom values."""
        state = Ko2cubeState(
            task_id="hard",
            scenario_name="task3_hard",
            difficulty="hard",
            current_step=10,
            total_carbon_gco2=500.0,
            sla_violations=2,
        )
        assert state.task_id == "hard"
        assert state.total_carbon_gco2 == 500.0
        assert state.sla_violations == 2


class TestModelSerialization:
    """Tests for model JSON serialization."""

    def test_job_to_dict(self):
        """Job can be serialized to dict."""
        job = Job(
            job_id="ser_001",
            cpu_cores=2,
            memory_gb=8,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        data = job.model_dump()
        assert data["job_id"] == "ser_001"
        assert "cpu_cores" in data

    def test_action_to_json(self):
        """Action can be serialized to JSON."""
        action = Ko2cubeAction(
            assignments=[
                JobAssignment(job_id="job_001", decision="schedule",
                             region="us-east-1", instance_type="m5.large",
                             machine_type="spot"),
            ]
        )
        json_str = action.model_dump_json()
        assert "job_001" in json_str
        assert "schedule" in json_str

    def test_observation_round_trip(self):
        """Observation can be serialized and deserialized."""
        obs = Ko2cubeObservation(
            current_step=3,
            job_queue=[],
            active_jobs=[],
            regions={},
            last_action_result="Test",
            done=False,
            reward=1.5,
        )
        data = obs.model_dump()
        restored = Ko2cubeObservation(**data)
        assert restored.current_step == obs.current_step
        assert restored.reward == obs.reward

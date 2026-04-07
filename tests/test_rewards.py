"""
Tests for reward calculations in server/rewards.py.
Validates reward components, baselines, and grader score.
"""
import pytest
from typing import Dict

from server.rewards import (
    compute_step_reward, compute_terminal_reward, compute_grader_score,
    expected_carbon_baseline, expected_cost_baseline,
    actual_carbon, actual_cost, _potential, _current_avg_intensity,
    RewardBreakdown,
    SLA_BREACH_PENALTY, DROP_PENALTY, ALWAYS_ON_PENALTY,
    SCHEDULE_ON_TIME_BONUS, DEFER_TO_CLEAN_BONUS,
    CARBON_REWARD_SCALE, COST_REWARD_SCALE,
)
from models import (
    Job, JobAssignment, Ko2cubeState, RegionInfo, CarbonData, InstanceType,
    ALWAYS_ON,
)


class TestBaselineCalculations:
    """Tests for baseline cost and carbon calculations."""

    def test_expected_carbon_baseline_normal_job(self):
        """Carbon baseline scales with intensity, runtime, and CPU."""
        job = Job(
            job_id="test_001",
            eta_minutes=60,
            cpu_cores=4,
            memory_gb=16,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
            baseline_carbon_intensity=200.0,
        )
        baseline = expected_carbon_baseline(job)
        expected = 200.0 * 1.0 * 4.0
        assert baseline == expected

    def test_expected_carbon_baseline_always_on(self):
        """Always-on jobs have zero carbon baseline."""
        job = Job(
            job_id="always_on",
            eta_minutes=None,
            cpu_cores=2,
            memory_gb=8,
            sla_start=0,
            sla_end=ALWAYS_ON,
            delay_tolerant=False,
            instance_preference="on-demand",
        )
        assert expected_carbon_baseline(job) == 0.0

    def test_expected_cost_baseline_normal_job(self):
        """Cost baseline scales with price and runtime."""
        job = Job(
            job_id="test_001",
            eta_minutes=120,
            cpu_cores=4,
            memory_gb=16,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
            baseline_spot_price=0.10,
        )
        baseline = expected_cost_baseline(job)
        expected = 0.10 * 2.0
        assert baseline == expected

    def test_expected_cost_baseline_always_on(self):
        """Always-on jobs have zero cost baseline."""
        job = Job(
            job_id="always_on",
            eta_minutes=None,
            cpu_cores=2,
            memory_gb=8,
            sla_start=0,
            sla_end=ALWAYS_ON,
            delay_tolerant=False,
            instance_preference="on-demand",
        )
        assert expected_cost_baseline(job) == 0.0


class TestActualCalculations:
    """Tests for actual cost and carbon calculations."""

    @pytest.fixture
    def sample_regions(self) -> Dict[str, RegionInfo]:
        return {
            "us-east-1": RegionInfo(
                region_name="us-east-1",
                carbon=CarbonData(current_intensity=300.0, forecast=[300.0, 280.0]),
                available_instances=[
                    InstanceType(name="m5.large", cpu_cores=2, memory_gb=8,
                                spot_price=0.05, on_demand_price=0.096, available_count=10),
                    InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                                spot_price=0.10, on_demand_price=0.192, available_count=10),
                ],
            ),
            "us-west-2": RegionInfo(
                region_name="us-west-2",
                carbon=CarbonData(current_intensity=150.0, forecast=[150.0, 140.0]),
                available_instances=[
                    InstanceType(name="m5.large", cpu_cores=2, memory_gb=8,
                                spot_price=0.06, on_demand_price=0.096, available_count=10),
                ],
            ),
        }

    def test_actual_carbon_schedule(self, sample_regions):
        """Actual carbon for scheduled job uses region intensity."""
        job = Job(
            job_id="test_001",
            eta_minutes=60,
            cpu_cores=4,
            memory_gb=16,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        assignment = JobAssignment(
            job_id="test_001",
            decision="schedule",
            region="us-east-1",
            instance_type="m5.xlarge",
            machine_type="spot",
        )
        carbon = actual_carbon(job, assignment, sample_regions)
        expected = 300.0 * 1.0 * 4.0
        assert carbon == expected

    def test_actual_carbon_low_region(self, sample_regions):
        """Scheduling in low-carbon region reduces emissions."""
        job = Job(
            job_id="test_001",
            eta_minutes=60,
            cpu_cores=2,
            memory_gb=8,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        assignment_high = JobAssignment(
            job_id="test_001", decision="schedule",
            region="us-east-1", instance_type="m5.large", machine_type="spot",
        )
        assignment_low = JobAssignment(
            job_id="test_001", decision="schedule",
            region="us-west-2", instance_type="m5.large", machine_type="spot",
        )
        carbon_high = actual_carbon(job, assignment_high, sample_regions)
        carbon_low = actual_carbon(job, assignment_low, sample_regions)
        
        assert carbon_low < carbon_high

    def test_actual_carbon_defer_is_zero(self, sample_regions):
        """Deferred job has zero immediate carbon."""
        job = Job(
            job_id="test_001",
            eta_minutes=60,
            cpu_cores=4,
            memory_gb=16,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        assignment = JobAssignment(
            job_id="test_001",
            decision="defer",
            defer_to_step=5,
        )
        assert actual_carbon(job, assignment, sample_regions) == 0.0

    def test_actual_cost_spot_vs_on_demand(self, sample_regions):
        """Spot pricing is cheaper than on-demand."""
        job = Job(
            job_id="test_001",
            eta_minutes=60,
            cpu_cores=4,
            memory_gb=16,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        assignment_spot = JobAssignment(
            job_id="test_001", decision="schedule",
            region="us-east-1", instance_type="m5.xlarge", machine_type="spot",
        )
        assignment_on_demand = JobAssignment(
            job_id="test_001", decision="schedule",
            region="us-east-1", instance_type="m5.xlarge", machine_type="on-demand",
        )
        cost_spot = actual_cost(job, assignment_spot, sample_regions)
        cost_on_demand = actual_cost(job, assignment_on_demand, sample_regions)
        
        assert cost_spot < cost_on_demand


def _create_dummy_job(job_id: str) -> Job:
    """Create a minimal job for testing state calculations."""
    return Job(
        job_id=job_id,
        cpu_cores=2,
        memory_gb=8,
        sla_start=0,
        sla_end=10,
        delay_tolerant=True,
        instance_preference="spot",
    )


class TestPotentialFunction:
    """Tests for potential-based shaping."""

    def test_potential_zero_at_start(self):
        """Potential is zero when no jobs completed."""
        state = Ko2cubeState(
            all_jobs={f"j{i}": _create_dummy_job(f"j{i}") for i in range(1, 4)},
            jobs_completed=0,
        )
        assert _potential(state) == 0.0

    def test_potential_one_at_end(self):
        """Potential is 1.0 when all jobs completed."""
        state = Ko2cubeState(
            all_jobs={f"j{i}": _create_dummy_job(f"j{i}") for i in range(1, 4)},
            jobs_completed=3,
        )
        assert _potential(state) == 1.0

    def test_potential_partial(self):
        """Potential is fraction of jobs completed."""
        state = Ko2cubeState(
            all_jobs={f"j{i}": _create_dummy_job(f"j{i}") for i in range(1, 5)},
            jobs_completed=2,
        )
        assert _potential(state) == 0.5

    def test_potential_empty_jobs(self):
        """Potential is zero with no jobs."""
        state = Ko2cubeState(all_jobs={}, jobs_completed=0)
        assert _potential(state) == 0.0


class TestStepReward:
    """Tests for compute_step_reward function."""

    @pytest.fixture
    def sample_state(self) -> Ko2cubeState:
        return Ko2cubeState(
            current_step=1,
            all_jobs={},
            jobs_completed=0,
        )

    @pytest.fixture
    def sample_regions(self) -> Dict[str, RegionInfo]:
        return {
            "us-east-1": RegionInfo(
                region_name="us-east-1",
                carbon=CarbonData(current_intensity=300.0, forecast=[300.0]),
                available_instances=[
                    InstanceType(name="m5.large", cpu_cores=2, memory_gb=8,
                                spot_price=0.05, on_demand_price=0.096, available_count=10),
                    InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                                spot_price=0.10, on_demand_price=0.192, available_count=10),
                ],
            ),
        }

    def test_schedule_on_time_bonus(self, sample_state, sample_regions):
        """Valid schedule gets on-time bonus."""
        job = Job(
            job_id="test_001",
            eta_minutes=60,
            cpu_cores=2,
            memory_gb=8,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
            baseline_carbon_intensity=300.0,
            baseline_spot_price=0.05,
        )
        sample_state.current_step = 1
        assignment = JobAssignment(
            job_id="test_001", decision="schedule",
            region="us-east-1", instance_type="m5.large", machine_type="spot",
        )
        rb = compute_step_reward(
            assignments=[assignment],
            queue=[job],
            regions=sample_regions,
            state=sample_state,
            phi_before=0.0,
            phi_after=0.0,
        )
        assert rb.sla >= SCHEDULE_ON_TIME_BONUS

    def test_drop_penalty(self, sample_state, sample_regions):
        """Dropping a job incurs penalty."""
        job = Job(
            job_id="test_001",
            eta_minutes=60,
            cpu_cores=2,
            memory_gb=8,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        assignment = JobAssignment(job_id="test_001", decision="drop")
        rb = compute_step_reward(
            assignments=[assignment],
            queue=[job],
            regions=sample_regions,
            state=sample_state,
            phi_before=0.0,
            phi_after=0.0,
        )
        assert rb.sla == DROP_PENALTY

    def test_always_on_drop_penalty(self, sample_state, sample_regions):
        """Dropping always-on job incurs larger penalty."""
        job = Job(
            job_id="always_on",
            eta_minutes=None,
            cpu_cores=2,
            memory_gb=8,
            sla_start=0,
            sla_end=ALWAYS_ON,
            delay_tolerant=False,
            instance_preference="on-demand",
        )
        assignment = JobAssignment(job_id="always_on", decision="drop")
        rb = compute_step_reward(
            assignments=[assignment],
            queue=[job],
            regions=sample_regions,
            state=sample_state,
            phi_before=0.0,
            phi_after=0.0,
        )
        assert rb.sla == ALWAYS_ON_PENALTY

    def test_defer_non_deferrable_penalty(self, sample_state, sample_regions):
        """Deferring non-deferrable job incurs SLA breach penalty."""
        job = Job(
            job_id="urgent_001",
            eta_minutes=30,
            cpu_cores=4,
            memory_gb=16,
            sla_start=0,
            sla_end=1,
            delay_tolerant=False,
            instance_preference="spot",
        )
        assignment = JobAssignment(
            job_id="urgent_001",
            decision="defer",
            defer_to_step=2,
        )
        rb = compute_step_reward(
            assignments=[assignment],
            queue=[job],
            regions=sample_regions,
            state=sample_state,
            phi_before=0.0,
            phi_after=0.0,
        )
        assert rb.sla == SLA_BREACH_PENALTY


class TestTerminalReward:
    """Tests for compute_terminal_reward function."""

    def test_full_completion_bonus(self):
        """100% completion gets full bonus."""
        state = Ko2cubeState(
            all_jobs={f"j{i}": _create_dummy_job(f"j{i}") for i in range(1, 4)},
            jobs_completed=3,
            sla_violations=0,
            total_carbon_gco2=500.0,
            baseline_carbon_gco2=1000.0,
            total_cost_usd=10.0,
            baseline_cost_usd=20.0,
        )
        rb = compute_terminal_reward(state)
        assert rb.terminal > 0
        assert rb.components.get("completion_rate", 0) == 1.0

    def test_sla_violations_reduce_terminal(self):
        """SLA violations reduce terminal reward."""
        state_good = Ko2cubeState(
            all_jobs={f"j{i}": _create_dummy_job(f"j{i}") for i in range(1, 3)},
            jobs_completed=2,
            sla_violations=0,
        )
        state_bad = Ko2cubeState(
            all_jobs={f"j{i}": _create_dummy_job(f"j{i}") for i in range(1, 3)},
            jobs_completed=1,
            sla_violations=1,
        )
        rb_good = compute_terminal_reward(state_good)
        rb_bad = compute_terminal_reward(state_bad)
        
        assert rb_good.terminal > rb_bad.terminal

    def test_carbon_improvement_bonus(self):
        """Better carbon efficiency gets bonus."""
        state = Ko2cubeState(
            all_jobs={"j1": _create_dummy_job("j1")},
            jobs_completed=1,
            total_carbon_gco2=400.0,
            baseline_carbon_gco2=1000.0,
        )
        rb = compute_terminal_reward(state)
        assert rb.components.get("carbon_improvement", 0) > 0


class TestGraderScore:
    """Tests for compute_grader_score function."""

    def test_grader_score_range(self):
        """Grader score is always in [0, 1]."""
        for completed in range(0, 11):
            state = Ko2cubeState(
                all_jobs={f"j{i}": _create_dummy_job(f"j{i}") for i in range(10)},
                jobs_completed=completed,
                sla_violations=10 - completed,
                total_carbon_gco2=float(1000 - completed * 50),
                baseline_carbon_gco2=1000.0,
                min_carbon_gco2=500.0,
                total_cost_usd=float(100 - completed * 5),
                baseline_cost_usd=100.0,
                min_cost_usd=50.0,
            )
            score = compute_grader_score(state)
            assert 0.0 <= score <= 1.0

    def test_grader_perfect_score(self):
        """Perfect performance approaches 1.0."""
        state = Ko2cubeState(
            all_jobs={f"j{i}": _create_dummy_job(f"j{i}") for i in range(1, 4)},
            jobs_completed=3,
            sla_violations=0,
            total_carbon_gco2=100.0,
            baseline_carbon_gco2=500.0,
            min_carbon_gco2=100.0,
            total_cost_usd=10.0,
            baseline_cost_usd=50.0,
            min_cost_usd=10.0,
        )
        score = compute_grader_score(state)
        assert score >= 0.9

    def test_grader_worst_score(self):
        """Worst performance has low score."""
        state = Ko2cubeState(
            all_jobs={f"j{i}": _create_dummy_job(f"j{i}") for i in range(1, 4)},
            jobs_completed=0,
            sla_violations=3,
            total_carbon_gco2=1000.0,
            baseline_carbon_gco2=500.0,
            total_cost_usd=100.0,
            baseline_cost_usd=50.0,
        )
        score = compute_grader_score(state)
        assert score < 0.3

    def test_grader_empty_jobs(self):
        """Grader handles empty job list."""
        state = Ko2cubeState(all_jobs={}, jobs_completed=0)
        score = compute_grader_score(state)
        assert 0.0 <= score <= 1.0


class TestRewardBreakdown:
    """Tests for RewardBreakdown dataclass."""

    def test_total_calculation(self):
        """Total sums all components."""
        rb = RewardBreakdown(
            sla=1.0,
            carbon=0.5,
            cost=0.3,
            waste=-0.2,
            shaping=0.1,
            terminal=2.0,
        )
        assert rb.total == pytest.approx(3.7)

    def test_to_dict(self):
        """to_dict returns all components."""
        rb = RewardBreakdown(sla=1.0, carbon=0.5)
        d = rb.to_dict()
        assert "sla" in d
        assert "carbon" in d
        assert "total" in d


class TestCurrentAvgIntensity:
    """Tests for _current_avg_intensity helper."""

    def test_average_calculation(self):
        """Average is computed correctly."""
        regions = {
            "r1": RegionInfo(
                region_name="r1",
                carbon=CarbonData(current_intensity=100.0, forecast=[]),
                available_instances=[],
            ),
            "r2": RegionInfo(
                region_name="r2",
                carbon=CarbonData(current_intensity=200.0, forecast=[]),
                available_instances=[],
            ),
            "r3": RegionInfo(
                region_name="r3",
                carbon=CarbonData(current_intensity=300.0, forecast=[]),
                available_instances=[],
            ),
        }
        avg = _current_avg_intensity(regions)
        assert avg == 200.0

    def test_empty_regions(self):
        """Empty regions returns zero."""
        assert _current_avg_intensity({}) == 0.0

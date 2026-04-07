"""
Tests for scenario definitions in server/data/scenarios.py.
Validates scenario structure, region consistency, and job pool integrity.
"""
import os
import json
import pytest
from typing import List, Set

from server.data.scenarios import (
    Scenario, TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD,
    SCENARIO_REGISTRY, DIFFICULTY_MAP, get_scenario,
)
from models import Job, ALWAYS_ON


def load_infrastructure_regions() -> List[str]:
    """Load the valid region names from infrastructure.json."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    infra_path = os.path.join(root_dir, "server", "data", "infrastructure.json")
    with open(infra_path, "r") as f:
        infra = json.load(f)
    return infra["regions"]


VALID_REGIONS = load_infrastructure_regions()


class TestScenarioStructure:
    """Tests for scenario dataclass structure."""

    @pytest.mark.parametrize("scenario", [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD])
    def test_scenario_has_required_fields(self, scenario: Scenario):
        """Each scenario has all required fields."""
        assert scenario.name is not None
        assert scenario.difficulty in ["easy", "medium", "hard"]
        assert isinstance(scenario.description, str)
        assert scenario.total_steps > 0
        assert scenario.step_duration_minutes > 0
        assert scenario.lookahead_steps > 0
        assert len(scenario.regions) > 0
        assert len(scenario.job_pool) > 0

    @pytest.mark.parametrize("scenario", [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD])
    def test_scenario_difficulty_matches_name(self, scenario: Scenario):
        """Scenario difficulty is consistent with name."""
        if "easy" in scenario.name:
            assert scenario.difficulty == "easy"
        elif "medium" in scenario.name:
            assert scenario.difficulty == "medium"
        elif "hard" in scenario.name:
            assert scenario.difficulty == "hard"


class TestRegionConsistency:
    """Tests for region naming consistency with infrastructure.json."""

    @pytest.mark.parametrize("scenario", [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD])
    def test_scenario_regions_are_valid(self, scenario: Scenario):
        """All scenario regions must exist in infrastructure.json."""
        for region in scenario.regions:
            assert region in VALID_REGIONS, (
                f"Scenario '{scenario.name}' uses region '{region}' which is not "
                f"in infrastructure.json. Valid regions: {VALID_REGIONS}"
            )

    def test_easy_scenario_regions(self):
        """Easy scenario uses us-east-1."""
        assert "us-east-1" in TASK_1_EASY.regions

    def test_medium_scenario_regions(self):
        """Medium scenario uses all three regions."""
        expected = {"us-east-1", "us-west-2", "eu-west-1"}
        assert set(TASK_2_MEDIUM.regions) == expected

    def test_hard_scenario_regions(self):
        """Hard scenario uses properly named regions (not short names)."""
        for region in TASK_3_HARD.regions:
            assert "-" in region, f"Region '{region}' looks like a short name"
            assert region in VALID_REGIONS


class TestJobPoolIntegrity:
    """Tests for job pool definitions."""

    @pytest.mark.parametrize("scenario", [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD])
    def test_job_ids_are_unique(self, scenario: Scenario):
        """All job IDs within a scenario must be unique."""
        job_ids = [job.job_id for job in scenario.job_pool]
        assert len(job_ids) == len(set(job_ids)), (
            f"Scenario '{scenario.name}' has duplicate job IDs"
        )

    @pytest.mark.parametrize("scenario", [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD])
    def test_job_sla_windows_are_valid(self, scenario: Scenario):
        """Job SLA windows must be logically consistent."""
        for job in scenario.job_pool:
            if job.sla_end != ALWAYS_ON:
                assert job.sla_start <= job.sla_end, (
                    f"Job '{job.job_id}' has sla_start ({job.sla_start}) > "
                    f"sla_end ({job.sla_end})"
                )
                assert job.arrival_step <= job.sla_end, (
                    f"Job '{job.job_id}' arrives at step {job.arrival_step} "
                    f"but SLA ends at {job.sla_end}"
                )

    @pytest.mark.parametrize("scenario", [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD])
    def test_job_arrival_within_episode(self, scenario: Scenario):
        """Jobs must arrive before episode ends."""
        for job in scenario.job_pool:
            assert job.arrival_step < scenario.total_steps, (
                f"Job '{job.job_id}' arrives at step {job.arrival_step} "
                f"but episode only has {scenario.total_steps} steps"
            )

    @pytest.mark.parametrize("scenario", [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD])
    def test_job_resource_requirements_positive(self, scenario: Scenario):
        """Jobs have non-negative resource requirements."""
        for job in scenario.job_pool:
            assert job.cpu_cores >= 0, f"Job '{job.job_id}' has negative CPU"
            assert job.memory_gb >= 0, f"Job '{job.job_id}' has negative memory"

    @pytest.mark.parametrize("scenario", [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD])
    def test_job_instance_preference_valid(self, scenario: Scenario):
        """Jobs have valid instance preference."""
        for job in scenario.job_pool:
            assert job.instance_preference in ["spot", "on-demand"], (
                f"Job '{job.job_id}' has invalid instance_preference: "
                f"{job.instance_preference}"
            )


class TestJobTypeDistribution:
    """Tests for job type distribution across scenarios."""

    def test_easy_has_only_delay_tolerant_jobs(self):
        """Easy scenario should have only delay-tolerant jobs."""
        for job in TASK_1_EASY.job_pool:
            assert job.delay_tolerant, (
                f"Easy scenario job '{job.job_id}' is not delay tolerant"
            )

    def test_medium_has_non_deferrable_jobs(self):
        """Medium scenario should have CI/CD (non-deferrable) jobs."""
        non_deferrable = [j for j in TASK_2_MEDIUM.job_pool if not j.delay_tolerant]
        assert len(non_deferrable) > 0, "Medium scenario has no CI/CD jobs"

    def test_hard_has_always_on_job(self):
        """Hard scenario should have at least one always-on job."""
        always_on = [j for j in TASK_3_HARD.job_pool if j.sla_end == ALWAYS_ON]
        assert len(always_on) > 0, "Hard scenario has no always-on jobs"

    def test_hard_has_cicd_burst(self):
        """Hard scenario should have CI/CD burst (steps 10-14)."""
        cicd_in_burst = [
            j for j in TASK_3_HARD.job_pool
            if "cicd" in j.job_id and 10 <= j.arrival_step <= 14
        ]
        assert len(cicd_in_burst) >= 5, (
            f"Hard scenario should have CI/CD burst, found only {len(cicd_in_burst)} jobs"
        )


class TestScenarioRegistry:
    """Tests for scenario lookup functions."""

    def test_get_scenario_by_name(self):
        """Scenarios can be retrieved by name."""
        assert get_scenario("task1_easy") == TASK_1_EASY
        assert get_scenario("task2_medium") == TASK_2_MEDIUM
        assert get_scenario("task3_hard") == TASK_3_HARD

    def test_get_scenario_by_difficulty(self):
        """Scenarios can be retrieved by difficulty string."""
        assert get_scenario("easy") == TASK_1_EASY
        assert get_scenario("medium") == TASK_2_MEDIUM
        assert get_scenario("hard") == TASK_3_HARD

    def test_get_scenario_invalid_name(self):
        """Invalid scenario name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_scenario("nonexistent")
        assert "Unknown scenario" in str(exc_info.value)

    def test_scenario_registry_completeness(self):
        """Registry contains all defined scenarios."""
        assert "task1_easy" in SCENARIO_REGISTRY
        assert "task2_medium" in SCENARIO_REGISTRY
        assert "task3_hard" in SCENARIO_REGISTRY

    def test_difficulty_map_completeness(self):
        """Difficulty map covers all difficulty levels."""
        assert "easy" in DIFFICULTY_MAP
        assert "medium" in DIFFICULTY_MAP
        assert "hard" in DIFFICULTY_MAP


class TestScenarioTiming:
    """Tests for scenario timing configuration."""

    @pytest.mark.parametrize("scenario", [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD])
    def test_lookahead_is_reasonable(self, scenario: Scenario):
        """Lookahead should not exceed total steps."""
        assert scenario.lookahead_steps <= scenario.total_steps

    def test_easy_is_shorter_than_hard(self):
        """Easy scenario should be shorter or equal to hard."""
        assert TASK_1_EASY.total_steps <= TASK_3_HARD.total_steps

    def test_step_duration_is_hourly(self):
        """All scenarios use 60-minute steps (1 hour)."""
        for scenario in [TASK_1_EASY, TASK_2_MEDIUM, TASK_3_HARD]:
            assert scenario.step_duration_minutes == 60


class TestScenarioDifficulty:
    """Tests for difficulty progression."""

    def test_job_count_increases_with_difficulty(self):
        """Harder scenarios have more jobs."""
        easy_jobs = len(TASK_1_EASY.job_pool)
        medium_jobs = len(TASK_2_MEDIUM.job_pool)
        hard_jobs = len(TASK_3_HARD.job_pool)
        
        assert easy_jobs <= medium_jobs, "Medium should have >= jobs than easy"
        assert medium_jobs <= hard_jobs, "Hard should have >= jobs than medium"

    def test_region_count_increases_with_difficulty(self):
        """Harder scenarios may have more regions."""
        easy_regions = len(TASK_1_EASY.regions)
        medium_regions = len(TASK_2_MEDIUM.regions)
        hard_regions = len(TASK_3_HARD.regions)
        
        assert easy_regions <= medium_regions
        assert medium_regions <= hard_regions or medium_regions == hard_regions

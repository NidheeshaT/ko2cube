"""
Tests for curriculum learning and adversarial scenario generation.
"""
import pytest
from typing import Dict

from server.curriculum import (
    DifficultyLevel, DIFFICULTY_PROGRESSION, LEVEL_TO_SCENARIO,
    MASTERY_THRESHOLDS, MasteryTracker, CurriculumManager, EpisodeResult,
)
from server.adversarial import (
    AdversarialConfig, AdversarialGenerator,
)
from models import ALWAYS_ON


class TestDifficultyLevel:
    """Tests for difficulty level enum."""

    def test_difficulty_levels_exist(self):
        """All expected difficulty levels exist."""
        assert DifficultyLevel.WARMUP.value == "warmup"
        assert DifficultyLevel.BEGINNER.value == "beginner"
        assert DifficultyLevel.INTERMEDIATE.value == "intermediate"
        assert DifficultyLevel.ADVANCED.value == "advanced"
        assert DifficultyLevel.EXPERT.value == "expert"

    def test_progression_order(self):
        """Progression is in correct order."""
        assert DIFFICULTY_PROGRESSION[0] == DifficultyLevel.WARMUP
        assert DIFFICULTY_PROGRESSION[-1] == DifficultyLevel.EXPERT

    def test_level_to_scenario_mapping(self):
        """All levels map to valid scenarios."""
        for level in DifficultyLevel:
            assert level in LEVEL_TO_SCENARIO
            assert LEVEL_TO_SCENARIO[level] in ["easy", "medium", "hard"]

    def test_mastery_thresholds_increase(self):
        """Mastery thresholds increase with difficulty."""
        thresholds = [
            MASTERY_THRESHOLDS[level]
            for level in DIFFICULTY_PROGRESSION
        ]
        for i in range(len(thresholds) - 1):
            assert thresholds[i] <= thresholds[i + 1]


class TestEpisodeResult:
    """Tests for EpisodeResult dataclass."""

    def test_episode_result_creation(self):
        """Can create episode result."""
        result = EpisodeResult(
            grader_score=0.75,
            carbon_saved_gco2=500.0,
            sla_compliance_rate=0.9,
            jobs_by_type={"etl": 5, "cicd": 3},
            successful_jobs_by_type={"etl": 4, "cicd": 3},
            region_carbon_efficiency={"us-east-1": 0.8},
        )
        assert result.grader_score == 0.75
        assert result.sla_compliance_rate == 0.9


class TestMasteryTracker:
    """Tests for MasteryTracker."""

    @pytest.fixture
    def tracker(self) -> MasteryTracker:
        """Create fresh tracker."""
        return MasteryTracker()

    def test_initial_state(self, tracker: MasteryTracker):
        """Tracker starts at warmup level."""
        assert tracker.current_level == 0
        assert tracker.get_current_difficulty() == DifficultyLevel.WARMUP
        assert tracker.total_episodes == 0

    def test_get_scenario_task_id(self, tracker: MasteryTracker):
        """Can get task ID for current level."""
        task_id = tracker.get_scenario_task_id()
        assert task_id in ["easy", "medium", "hard"]

    def test_get_mastery_threshold(self, tracker: MasteryTracker):
        """Can get mastery threshold."""
        threshold = tracker.get_mastery_threshold()
        assert 0.0 < threshold < 1.0

    def test_update_mastery(self, tracker: MasteryTracker):
        """Mastery updates after episode."""
        result = EpisodeResult(
            grader_score=0.6,
            carbon_saved_gco2=300.0,
            sla_compliance_rate=0.85,
            jobs_by_type={"etl": 5},
            successful_jobs_by_type={"etl": 4},
            region_carbon_efficiency={"us-east-1": 0.7},
        )
        
        tracker.update_mastery(result)
        
        assert tracker.total_episodes == 1
        assert tracker.episodes_at_level == 1
        assert "etl" in tracker.job_type_success
        assert "us-east-1" in tracker.region_efficiency

    def test_consecutive_mastery_tracking(self, tracker: MasteryTracker):
        """Tracks consecutive successful episodes."""
        good_result = EpisodeResult(
            grader_score=0.8,
            carbon_saved_gco2=500.0,
            sla_compliance_rate=0.95,
            jobs_by_type={},
            successful_jobs_by_type={},
            region_carbon_efficiency={},
        )
        
        tracker.update_mastery(good_result)
        assert tracker.consecutive_mastery == 1
        
        tracker.update_mastery(good_result)
        assert tracker.consecutive_mastery == 2

    def test_consecutive_mastery_reset(self, tracker: MasteryTracker):
        """Consecutive mastery resets on failure."""
        good_result = EpisodeResult(
            grader_score=0.8,
            carbon_saved_gco2=500.0,
            sla_compliance_rate=0.95,
            jobs_by_type={},
            successful_jobs_by_type={},
            region_carbon_efficiency={},
        )
        bad_result = EpisodeResult(
            grader_score=0.3,
            carbon_saved_gco2=0.0,
            sla_compliance_rate=0.5,
            jobs_by_type={},
            successful_jobs_by_type={},
            region_carbon_efficiency={},
        )
        
        tracker.update_mastery(good_result)
        tracker.update_mastery(good_result)
        assert tracker.consecutive_mastery == 2
        
        tracker.update_mastery(bad_result)
        assert tracker.consecutive_mastery == 0

    def test_should_escalate_requires_minimum_episodes(self, tracker: MasteryTracker):
        """Won't escalate without minimum episodes."""
        good_result = EpisodeResult(
            grader_score=0.9,
            carbon_saved_gco2=1000.0,
            sla_compliance_rate=1.0,
            jobs_by_type={},
            successful_jobs_by_type={},
            region_carbon_efficiency={},
        )
        
        tracker.update_mastery(good_result)
        assert not tracker.should_escalate()

    def test_should_escalate_with_consecutive_mastery(self, tracker: MasteryTracker):
        """Escalates after consecutive mastery."""
        good_result = EpisodeResult(
            grader_score=0.9,
            carbon_saved_gco2=1000.0,
            sla_compliance_rate=1.0,
            jobs_by_type={},
            successful_jobs_by_type={},
            region_carbon_efficiency={},
        )
        
        for _ in range(tracker.required_consecutive + 1):
            tracker.update_mastery(good_result)
        
        assert tracker.should_escalate()

    def test_escalate_advances_level(self, tracker: MasteryTracker):
        """Escalation advances to next level."""
        initial_level = tracker.current_level
        tracker.escalate()
        assert tracker.current_level == initial_level + 1
        assert tracker.episodes_at_level == 0

    def test_escalate_capped_at_expert(self, tracker: MasteryTracker):
        """Can't escalate beyond expert."""
        tracker.current_level = len(DIFFICULTY_PROGRESSION) - 1
        tracker.escalate()
        assert tracker.get_current_difficulty() == DifficultyLevel.EXPERT

    def test_get_weakest_job_types(self, tracker: MasteryTracker):
        """Can identify weakest job types."""
        tracker.job_type_success = {
            "etl": 0.9,
            "cicd": 0.5,
            "ml_training": 0.3,
        }
        
        weak = tracker.get_weakest_job_types(2)
        assert len(weak) == 2
        assert "ml_training" in weak
        assert "cicd" in weak

    def test_get_best_regions(self, tracker: MasteryTracker):
        """Can identify best regions."""
        tracker.region_efficiency = {
            "us-east-1": 0.6,
            "us-west-2": 0.9,
            "eu-west-1": 0.7,
        }
        
        best = tracker.get_best_regions(2)
        assert len(best) == 2
        assert "us-west-2" in best

    def test_get_stats(self, tracker: MasteryTracker):
        """Can get stats summary."""
        stats = tracker.get_stats()
        assert "current_level" in stats
        assert "total_episodes" in stats
        assert "mastery_threshold" in stats


class TestCurriculumManager:
    """Tests for CurriculumManager."""

    @pytest.fixture
    def manager(self) -> CurriculumManager:
        """Create curriculum manager."""
        return CurriculumManager()

    def test_initial_task(self, manager: CurriculumManager):
        """Initial task is for warmup level."""
        task = manager.get_next_task()
        assert task in ["easy", "medium", "hard"]

    def test_record_and_progress(self, manager: CurriculumManager):
        """Recording episodes updates tracker."""
        result = EpisodeResult(
            grader_score=0.7,
            carbon_saved_gco2=400.0,
            sla_compliance_rate=0.9,
            jobs_by_type={"etl": 3},
            successful_jobs_by_type={"etl": 3},
            region_carbon_efficiency={},
        )
        
        manager.record_episode(result)
        assert manager.tracker.total_episodes == 1

    def test_set_level(self, manager: CurriculumManager):
        """Can manually set level."""
        manager.set_level(DifficultyLevel.INTERMEDIATE)
        assert manager.tracker.get_current_difficulty() == DifficultyLevel.INTERMEDIATE

    def test_get_progress_report(self, manager: CurriculumManager):
        """Can get progress report."""
        report = manager.get_progress_report()
        assert "tracker_stats" in report
        assert "weakest_job_types" in report
        assert "should_escalate" in report


class TestAdversarialConfig:
    """Tests for AdversarialConfig."""

    def test_default_config(self):
        """Default config has valid values."""
        config = AdversarialConfig()
        assert 0 < config.weakness_focus_ratio <= 1.0
        assert 0 <= config.noise_probability <= 1.0
        assert len(config.regions) > 0

    def test_custom_config(self):
        """Can customize config."""
        config = AdversarialConfig(
            weakness_focus_ratio=0.8,
            regions=["us-east-1"],
        )
        assert config.weakness_focus_ratio == 0.8
        assert config.regions == ["us-east-1"]


class TestAdversarialGenerator:
    """Tests for AdversarialGenerator."""

    @pytest.fixture
    def generator(self) -> AdversarialGenerator:
        """Create generator."""
        return AdversarialGenerator()

    def test_generate_scenario(self, generator: AdversarialGenerator):
        """Can generate adversarial scenario."""
        scenario = generator.generate_scenario(
            weak_job_types=["cicd", "api_serving"],
            weak_regions=["us-east-1"],
        )
        
        assert scenario.name.startswith("adversarial")
        assert scenario.difficulty == "adversarial"
        assert len(scenario.job_pool) > 0
        assert len(scenario.regions) > 0

    def test_scenario_has_focus_jobs(self, generator: AdversarialGenerator):
        """Generated scenario focuses on weak job types."""
        generator.config.weakness_focus_ratio = 0.9
        
        scenario = generator.generate_scenario(
            weak_job_types=["cicd"],
            weak_regions=[],
        )
        
        cicd_count = sum(1 for j in scenario.job_pool if "cicd" in j.job_id)
        assert cicd_count > 0

    def test_job_templates_valid(self, generator: AdversarialGenerator):
        """All job templates have required fields."""
        for job_type, template in generator.JOB_TEMPLATES.items():
            assert "cpu_cores" in template
            assert "memory_gb" in template
            assert "delay_tolerant" in template
            assert "instance_preference" in template

    def test_generate_with_burst(self, generator: AdversarialGenerator):
        """Can generate scenario with job burst."""
        generator.config.job_burst_probability = 1.0
        
        scenario = generator.generate_scenario(
            weak_job_types=["etl"],
            weak_regions=[],
        )
        
        cicd_jobs = [j for j in scenario.job_pool if "cicd" in j.job_id]
        assert len(cicd_jobs) >= 3

    def test_generate_with_always_on(self, generator: AdversarialGenerator):
        """Can include always-on jobs."""
        scenario = generator.generate_scenario(
            weak_job_types=["api_serving"],
            weak_regions=[],
            base_difficulty="hard",
        )
        
        always_on_jobs = [j for j in scenario.job_pool if j.sla_end == ALWAYS_ON]
        assert len(always_on_jobs) >= 1

    def test_mutate_scenario(self, generator: AdversarialGenerator):
        """Can mutate existing scenario."""
        original = generator.generate_scenario(
            weak_job_types=["etl"],
            weak_regions=[],
        )
        
        mutated = generator.mutate_scenario(original, mutation_strength=0.5)
        
        assert "mutated" in mutated.name
        assert len(mutated.job_pool) == len(original.job_pool)

    def test_generate_from_mastery(self, generator: AdversarialGenerator):
        """Can generate from mastery stats."""
        mastery_stats = {
            "job_type_success": {"etl": 0.9, "cicd": 0.3, "ml_training": 0.4},
            "region_efficiency": {"us-east-1": 0.8, "us-west-2": 0.5},
        }
        
        scenario = generator.generate_from_mastery(mastery_stats)
        
        assert scenario is not None
        assert len(scenario.job_pool) > 0

    def test_sla_windows_valid(self, generator: AdversarialGenerator):
        """Generated jobs have valid SLA windows."""
        scenario = generator.generate_scenario(
            weak_job_types=["etl", "cicd"],
            weak_regions=[],
            total_steps=24,
        )
        
        for job in scenario.job_pool:
            if job.sla_end != ALWAYS_ON:
                assert job.sla_start <= job.sla_end
                assert job.arrival_step <= job.sla_end
            assert job.arrival_step >= 0
            assert job.arrival_step < 24

    def test_unique_job_ids(self, generator: AdversarialGenerator):
        """Generated jobs have unique IDs."""
        scenario = generator.generate_scenario(
            weak_job_types=["etl"],
            weak_regions=[],
        )
        
        job_ids = [j.job_id for j in scenario.job_pool]
        assert len(job_ids) == len(set(job_ids))

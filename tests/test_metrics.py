"""
Tests for Prometheus metrics collection.
"""
import pytest
import time

from server.metrics import (
    MetricsCollector, get_job_type_from_id, episode_timer,
    PROMETHEUS_AVAILABLE,
)


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """Create fresh collector."""
        return MetricsCollector(scenario="test_scenario", difficulty="easy")

    def test_collector_creation(self, collector: MetricsCollector):
        """Can create collector with context."""
        assert collector.scenario == "test_scenario"
        assert collector.difficulty == "easy"

    def test_set_context(self, collector: MetricsCollector):
        """Can update context."""
        collector.set_context("new_scenario", "hard")
        assert collector.scenario == "new_scenario"
        assert collector.difficulty == "hard"

    def test_start_episode(self, collector: MetricsCollector):
        """Can start episode timer."""
        collector.start_episode()
        assert collector._episode_start is not None

    def test_end_episode(self, collector: MetricsCollector):
        """Can end episode and record metrics."""
        collector.start_episode()
        time.sleep(0.01)
        collector.end_episode(total_reward=5.0, steps=24, grader=0.75)
        assert collector._episode_start is None

    def test_record_carbon_savings(self, collector: MetricsCollector):
        """Records carbon savings."""
        collector.record_carbon_savings(500.0)
        metrics = collector.get_internal_metrics()
        assert metrics["carbon_savings_gco2"] == 500.0

    def test_record_carbon_savings_negative_ignored(self, collector: MetricsCollector):
        """Negative savings are clamped to 0."""
        collector.record_carbon_savings(-100.0)
        metrics = collector.get_internal_metrics()
        assert metrics["carbon_savings_gco2"] == 0.0

    def test_record_carbon_emission(self, collector: MetricsCollector):
        """Records carbon emissions."""
        collector.record_carbon_emission(200.0, "us-east-1")
        metrics = collector.get_internal_metrics()
        assert metrics["carbon_emitted_gco2"] == 200.0

    def test_record_cost(self, collector: MetricsCollector):
        """Records cost."""
        collector.record_cost(10.5)
        metrics = collector.get_internal_metrics()
        assert metrics["cost_usd"] == 10.5

    def test_record_job_scheduled(self, collector: MetricsCollector):
        """Records scheduled jobs."""
        collector.record_job_scheduled("etl", "us-east-1", "spot")
        collector.record_job_scheduled("cicd", "us-west-2", "on-demand")
        metrics = collector.get_internal_metrics()
        assert metrics["jobs_scheduled"] == 2

    def test_record_job_completed(self, collector: MetricsCollector):
        """Records completed jobs."""
        collector.record_job_completed("etl", "us-east-1")
        metrics = collector.get_internal_metrics()
        assert metrics["jobs_completed"] == 1

    def test_record_job_dropped(self, collector: MetricsCollector):
        """Records dropped jobs."""
        collector.record_job_dropped("etl", "agent_decision")
        metrics = collector.get_internal_metrics()
        assert metrics["jobs_dropped"] == 1

    def test_record_job_deferred(self, collector: MetricsCollector):
        """Records deferred jobs."""
        collector.record_job_deferred("etl")
        metrics = collector.get_internal_metrics()
        assert metrics["jobs_deferred"] == 1

    def test_record_sla_violation(self, collector: MetricsCollector):
        """Records SLA violations."""
        collector.record_sla_violation("cicd", "deadline_missed")
        metrics = collector.get_internal_metrics()
        assert metrics["sla_violations"] == 1

    def test_get_internal_metrics(self, collector: MetricsCollector):
        """Can get all internal metrics."""
        collector.record_carbon_savings(100.0)
        collector.record_cost(5.0)
        collector.record_job_scheduled("etl", "us-east-1", "spot")
        
        metrics = collector.get_internal_metrics()
        
        assert "carbon_savings_gco2" in metrics
        assert "cost_usd" in metrics
        assert "jobs_scheduled" in metrics
        assert metrics["carbon_savings_gco2"] == 100.0
        assert metrics["cost_usd"] == 5.0
        assert metrics["jobs_scheduled"] == 1

    def test_reset_internal_metrics(self, collector: MetricsCollector):
        """Can reset internal counters."""
        collector.record_carbon_savings(500.0)
        collector.record_job_scheduled("etl", "us-east-1", "spot")
        
        collector.reset_internal_metrics()
        
        metrics = collector.get_internal_metrics()
        assert metrics["carbon_savings_gco2"] == 0.0
        assert metrics["jobs_scheduled"] == 0

    def test_update_carbon_intensity(self, collector: MetricsCollector):
        """Can update carbon intensity (no-op if prometheus not available)."""
        collector.update_carbon_intensity("us-east-1", 350.0)

    def test_update_active_jobs(self, collector: MetricsCollector):
        """Can update active jobs count."""
        collector.update_active_jobs("us-east-1", 5)

    def test_update_queue_size(self, collector: MetricsCollector):
        """Can update queue size."""
        collector.update_queue_size(10)

    def test_record_training_episode(self, collector: MetricsCollector):
        """Can record training episode."""
        collector.record_training_episode("test_model")

    def test_update_curriculum_level(self, collector: MetricsCollector):
        """Can update curriculum level."""
        collector.update_curriculum_level(3)

    def test_cumulative_tracking(self, collector: MetricsCollector):
        """Metrics accumulate across calls."""
        collector.record_carbon_savings(100.0)
        collector.record_carbon_savings(200.0)
        collector.record_carbon_savings(150.0)
        
        metrics = collector.get_internal_metrics()
        assert metrics["carbon_savings_gco2"] == 450.0

    def test_episode_count(self, collector: MetricsCollector):
        """Episode count increments with each start."""
        collector.start_episode()
        collector.end_episode(1.0, 10, 0.5)
        
        collector.start_episode()
        collector.end_episode(2.0, 20, 0.6)
        
        metrics = collector.get_internal_metrics()
        assert metrics["episode_count"] == 2


class TestJobTypeExtraction:
    """Tests for job type extraction from job IDs."""

    def test_extract_etl(self):
        """Extracts etl type."""
        assert get_job_type_from_id("etl_sales_001") == "etl"

    def test_extract_cicd(self):
        """Extracts cicd type."""
        assert get_job_type_from_id("cicd_build_003") == "cicd"

    def test_extract_ml_training(self):
        """Extracts ml_training type."""
        assert get_job_type_from_id("ml_training_001") == "ml_training"

    def test_extract_video_transcode(self):
        """Extracts video_transcode type."""
        assert get_job_type_from_id("video_transcode_002") == "video_transcode"

    def test_extract_db_backup(self):
        """Extracts db_backup type."""
        assert get_job_type_from_id("db_backup_001") == "db_backup"

    def test_extract_api_serving(self):
        """Extracts api_serving type."""
        assert get_job_type_from_id("api_serving_000") == "api_serving"

    def test_extract_batch_report(self):
        """Extracts batch_report type."""
        assert get_job_type_from_id("batch_report_001") == "batch_report"

    def test_extract_dq_scan(self):
        """Extracts dq_scan type."""
        assert get_job_type_from_id("dq_scan_002") == "dq_scan"

    def test_extract_adversarial(self):
        """Extracts adv (adversarial) type."""
        assert get_job_type_from_id("adv_etl_0001") == "adv"

    def test_unknown_job_type(self):
        """Returns unknown for unrecognized types."""
        assert get_job_type_from_id("random_job_123") == "unknown"


class TestEpisodeTimer:
    """Tests for episode_timer context manager."""

    def test_episode_timer_starts(self):
        """Timer starts episode on entry."""
        collector = MetricsCollector()
        
        with episode_timer(collector):
            assert collector._episode_start is not None

    def test_episode_timer_context(self):
        """Can use as context manager."""
        collector = MetricsCollector()
        
        with episode_timer(collector):
            time.sleep(0.01)


class TestPrometheusAvailability:
    """Tests for prometheus availability handling."""

    def test_prometheus_available_check(self):
        """PROMETHEUS_AVAILABLE is a boolean."""
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_collector_works_without_prometheus(self):
        """Collector works even if prometheus is not installed."""
        collector = MetricsCollector()
        
        collector.record_carbon_savings(100.0)
        collector.record_job_scheduled("etl", "us-east-1", "spot")
        collector.record_sla_violation("cicd", "timeout")
        
        metrics = collector.get_internal_metrics()
        assert metrics["carbon_savings_gco2"] == 100.0
        assert metrics["jobs_scheduled"] == 1
        assert metrics["sla_violations"] == 1

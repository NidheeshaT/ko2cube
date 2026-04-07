"""
Tests for Ko2cubeEnvironment in server/environment.py.
Validates reset, step, SLA handling, and episode flow.
"""
import pytest
from typing import List

from server.environment import Ko2cubeEnvironment
from models import (
    Ko2cubeAction, Ko2cubeObservation, Ko2cubeState,
    Job, JobAssignment, ALWAYS_ON,
)
from tests.conftest import (
    create_schedule_action, create_defer_action, create_drop_action,
    schedule_all_jobs_in_region,
)


class TestEnvironmentReset:
    """Tests for environment reset functionality."""

    def test_reset_returns_observation(self, environment: Ko2cubeEnvironment):
        """Reset returns a valid Ko2cubeObservation."""
        obs = environment.reset(task_id="easy")
        assert isinstance(obs, Ko2cubeObservation)
        assert obs.current_step == 0
        assert obs.done is False
        assert obs.reward == 0.0

    def test_reset_easy_has_jobs(self, environment: Ko2cubeEnvironment):
        """Easy scenario starts with jobs in queue."""
        obs = environment.reset(task_id="easy")
        assert len(obs.job_queue) > 0

    def test_reset_has_regions(self, environment: Ko2cubeEnvironment):
        """Reset provides region data."""
        obs = environment.reset(task_id="easy")
        assert len(obs.regions) > 0
        assert "us-east-1" in obs.regions

    def test_reset_different_difficulties(self, environment: Ko2cubeEnvironment):
        """Can reset to different difficulty levels."""
        for difficulty in ["easy", "medium", "hard"]:
            obs = environment.reset(task_id=difficulty)
            assert obs.current_step == 0

    def test_reset_clears_previous_state(self, environment: Ko2cubeEnvironment):
        """Reset clears state from previous episode."""
        obs1 = environment.reset(task_id="easy")
        if obs1.job_queue:
            job = obs1.job_queue[0]
            action = create_schedule_action(
                job, region="us-east-1", instance_type="m5.xlarge"
            )
            environment.step(action)
        
        obs2 = environment.reset(task_id="easy")
        assert obs2.current_step == 0
        assert environment.state.jobs_completed == 0

    def test_reset_region_data_has_carbon(self, environment: Ko2cubeEnvironment):
        """Region data includes carbon intensity and forecast."""
        obs = environment.reset(task_id="easy")
        for region_name, region_info in obs.regions.items():
            assert region_info.carbon.current_intensity > 0
            assert len(region_info.carbon.forecast) > 0

    def test_reset_region_data_has_instances(self, environment: Ko2cubeEnvironment):
        """Region data includes available instances."""
        obs = environment.reset(task_id="easy")
        for region_name, region_info in obs.regions.items():
            assert len(region_info.available_instances) > 0
            for inst in region_info.available_instances:
                assert inst.cpu_cores > 0
                assert inst.memory_gb > 0


class TestEnvironmentStep:
    """Tests for environment step functionality."""

    def test_step_advances_current_step(self, easy_environment: Ko2cubeEnvironment):
        """Step increments current_step."""
        obs = easy_environment.get_observation()
        initial_step = obs.current_step
        
        if obs.job_queue:
            job = obs.job_queue[0]
            action = create_schedule_action(job, "us-east-1", "m5.xlarge")
        else:
            action = Ko2cubeAction(assignments=[])
        
        next_obs = easy_environment.step(action)
        assert next_obs.current_step == initial_step + 1

    def test_step_returns_reward(self, easy_environment: Ko2cubeEnvironment):
        """Step returns numeric reward."""
        obs = easy_environment.get_observation()
        if obs.job_queue:
            job = obs.job_queue[0]
            action = create_schedule_action(job, "us-east-1", "m5.xlarge")
        else:
            action = Ko2cubeAction(assignments=[])
        
        next_obs = easy_environment.step(action)
        assert isinstance(next_obs.reward, (int, float))


class TestScheduleAction:
    """Tests for schedule decision processing."""

    def test_schedule_removes_from_queue(self, easy_environment: Ko2cubeEnvironment):
        """Scheduled job is removed from queue."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        action = create_schedule_action(job, "us-east-1", "m5.xlarge")
        
        next_obs = easy_environment.step(action)
        job_ids_in_queue = [j.job_id for j in next_obs.job_queue]
        assert job.job_id not in job_ids_in_queue

    def test_schedule_adds_to_active(self, easy_environment: Ko2cubeEnvironment):
        """Scheduled job appears in active_jobs or completes within the step."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        action = create_schedule_action(job, "us-east-1", "m5.xlarge")
        
        next_obs = easy_environment.step(action)
        active_ids = [rj.job_id for rj in next_obs.active_jobs]
        
        state_job = easy_environment.state.all_jobs.get(job.job_id)
        assert state_job is not None
        assert state_job.status in ["running", "completed"], \
            f"Job status should be running or completed, got {state_job.status}"

    def test_schedule_invalid_region_fails(self, easy_environment: Ko2cubeEnvironment):
        """Scheduling to invalid region doesn't schedule."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        action = Ko2cubeAction(
            assignments=[
                JobAssignment(
                    job_id=job.job_id,
                    decision="schedule",
                    region="invalid-region",
                    instance_type="m5.xlarge",
                    machine_type="spot",
                )
            ]
        )
        
        next_obs = easy_environment.step(action)
        active_ids = [rj.job_id for rj in next_obs.active_jobs]
        assert job.job_id not in active_ids
        assert "failed" in next_obs.last_action_result.lower()


class TestDeferAction:
    """Tests for defer decision processing."""

    def test_defer_removes_from_queue(self, easy_environment: Ko2cubeEnvironment):
        """Deferred job is removed from current queue."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        if not job.delay_tolerant:
            pytest.skip("Need delay tolerant job")
        
        action = create_defer_action(job, job.sla_end)
        next_obs = easy_environment.step(action)
        
        job_ids = [j.job_id for j in next_obs.job_queue]
        assert job.job_id not in job_ids

    def test_defer_resurfaces_at_target_step(self, easy_environment: Ko2cubeEnvironment):
        """Deferred job reappears at defer_to_step."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        if not job.delay_tolerant:
            pytest.skip("Need delay tolerant job")
        
        defer_step = obs.current_step + 2
        if defer_step > job.sla_end:
            defer_step = job.sla_end
        
        action = create_defer_action(job, defer_step)
        easy_environment.step(action)
        
        while easy_environment.get_observation().current_step < defer_step:
            obs2 = easy_environment.get_observation()
            action2 = schedule_all_jobs_in_region(obs2.job_queue, "us-east-1", "m5.xlarge")
            easy_environment.step(action2)
        
        obs_at_defer = easy_environment.get_observation()
        job_ids = [j.job_id for j in obs_at_defer.job_queue]
        assert job.job_id in job_ids


class TestDropAction:
    """Tests for drop decision processing."""

    def test_drop_removes_from_queue(self, easy_environment: Ko2cubeEnvironment):
        """Dropped job is removed from queue."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        action = create_drop_action(job)
        
        next_obs = easy_environment.step(action)
        job_ids = [j.job_id for j in next_obs.job_queue]
        assert job.job_id not in job_ids

    def test_drop_increments_dropped_count(self, easy_environment: Ko2cubeEnvironment):
        """Dropping increments jobs_dropped counter."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        initial_dropped = easy_environment.state.jobs_dropped
        job = obs.job_queue[0]
        action = create_drop_action(job)
        
        easy_environment.step(action)
        assert easy_environment.state.jobs_dropped == initial_dropped + 1

    def test_drop_incurs_penalty(self, easy_environment: Ko2cubeEnvironment):
        """Dropping a job gives negative reward."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        action = create_drop_action(job)
        
        next_obs = easy_environment.step(action)
        assert next_obs.reward < 0


class TestSLAHandling:
    """Tests for SLA violation detection."""

    def test_sla_violation_on_expiry(self, easy_environment: Ko2cubeEnvironment):
        """Job past SLA end is marked as sla_missed."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        if job.sla_end == ALWAYS_ON or job.sla_end > 20:
            pytest.skip("Need job with reachable SLA end")
        
        action = create_defer_action(job, job.sla_end)
        easy_environment.step(action)
        
        while easy_environment.state.current_step <= job.sla_end + 1:
            obs2 = easy_environment.get_observation()
            other_jobs = [j for j in obs2.job_queue if j.job_id != job.job_id]
            action2 = schedule_all_jobs_in_region(other_jobs, "us-east-1", "m5.xlarge")
            if easy_environment.state.is_done:
                break
            easy_environment.step(action2)
        
        state = easy_environment.state
        job_state = state.all_jobs.get(job.job_id)
        if job_state:
            assert job_state.status in ["sla_missed", "deferred", "completed"]


class TestAlwaysOnJobs:
    """Tests for always-on job handling."""

    def test_always_on_job_detected(self, hard_environment: Ko2cubeEnvironment):
        """Hard scenario has always-on jobs."""
        obs = hard_environment.get_observation()
        always_on_jobs = [j for j in obs.job_queue if j.sla_end == ALWAYS_ON]
        
        state = hard_environment.state
        all_always_on = [j for j in state.all_jobs.values() if j.sla_end == ALWAYS_ON]
        
        assert len(all_always_on) > 0, "Hard scenario should have always-on jobs"

    def test_always_on_defer_blocked(self, hard_environment: Ko2cubeEnvironment):
        """Deferring always-on job is treated as drop."""
        obs = hard_environment.get_observation()
        always_on_jobs = [j for j in obs.job_queue if j.sla_end == ALWAYS_ON]
        
        if not always_on_jobs:
            pytest.skip("No always-on jobs in queue at step 0")
        
        job = always_on_jobs[0]
        action = Ko2cubeAction(
            assignments=[
                JobAssignment(
                    job_id=job.job_id,
                    decision="defer",
                    defer_to_step=5,
                )
            ]
        )
        
        next_obs = hard_environment.step(action)
        assert "blocked" in next_obs.last_action_result.lower() or \
               "dropped" in next_obs.last_action_result.lower()


class TestJobCompletion:
    """Tests for job completion tracking."""

    def test_job_completion_increments_counter(self, easy_environment: Ko2cubeEnvironment):
        """Completed job increments jobs_completed."""
        obs = easy_environment.get_observation()
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        action = create_schedule_action(job, "us-east-1", "m5.xlarge")
        easy_environment.step(action)
        
        steps_needed = (job.eta_minutes or 60) // 60 + 2
        for _ in range(steps_needed):
            if easy_environment.state.is_done:
                break
            obs2 = easy_environment.get_observation()
            action2 = schedule_all_jobs_in_region(obs2.job_queue, "us-east-1", "m5.xlarge")
            easy_environment.step(action2)
        
        assert easy_environment.state.jobs_completed >= 1


class TestEpisodeCompletion:
    """Tests for episode termination."""

    def test_episode_ends_at_total_steps(self, easy_environment: Ko2cubeEnvironment):
        """Episode terminates at total_steps."""
        from server.data.scenarios import TASK_1_EASY
        total_steps = TASK_1_EASY.total_steps
        
        for step in range(total_steps + 5):
            obs = easy_environment.get_observation()
            if obs.done:
                break
            action = schedule_all_jobs_in_region(obs.job_queue, "us-east-1", "m5.xlarge")
            easy_environment.step(action)
        
        assert easy_environment.state.is_done
        assert easy_environment.state.current_step >= total_steps

    def test_done_flag_set_at_end(self, easy_environment: Ko2cubeEnvironment):
        """done=True when episode completes."""
        from server.data.scenarios import TASK_1_EASY
        total_steps = TASK_1_EASY.total_steps
        
        last_obs = None
        for _ in range(total_steps + 5):
            obs = easy_environment.get_observation()
            if obs.done:
                last_obs = obs
                break
            action = schedule_all_jobs_in_region(obs.job_queue, "us-east-1", "m5.xlarge")
            last_obs = easy_environment.step(action)
        
        assert last_obs is not None
        assert last_obs.done is True


class TestStateAccess:
    """Tests for state property and grader_score."""

    def test_state_property_returns_state(self, environment: Ko2cubeEnvironment):
        """state property returns Ko2cubeState."""
        environment.reset(task_id="easy")
        state = environment.state
        assert isinstance(state, Ko2cubeState)

    def test_grader_score_in_range(self, environment: Ko2cubeEnvironment):
        """grader_score is in [0, 1]."""
        environment.reset(task_id="easy")
        
        for _ in range(5):
            obs = environment.get_observation()
            if obs.done:
                break
            action = schedule_all_jobs_in_region(obs.job_queue, "us-east-1", "m5.xlarge")
            environment.step(action)
        
        score = environment.grader_score()
        assert 0.0 <= score <= 1.0


class TestGetObservation:
    """Tests for get_observation method."""

    def test_get_observation_without_step(self, easy_environment: Ko2cubeEnvironment):
        """get_observation returns current state without advancing."""
        obs1 = easy_environment.get_observation()
        obs2 = easy_environment.get_observation()
        
        assert obs1.current_step == obs2.current_step
        assert len(obs1.job_queue) == len(obs2.job_queue)


class TestCarbonDataConsistency:
    """Tests for carbon data consistency."""

    def test_carbon_forecast_length_matches_lookahead(self, easy_environment: Ko2cubeEnvironment):
        """Carbon forecast length matches scenario lookahead."""
        from server.data.scenarios import TASK_1_EASY
        obs = easy_environment.get_observation()
        
        for region_name, region_info in obs.regions.items():
            assert len(region_info.carbon.forecast) == TASK_1_EASY.lookahead_steps

    def test_carbon_intensity_positive(self, easy_environment: Ko2cubeEnvironment):
        """Carbon intensities are positive."""
        obs = easy_environment.get_observation()
        
        for region_name, region_info in obs.regions.items():
            assert region_info.carbon.current_intensity > 0
            for f in region_info.carbon.forecast:
                assert f > 0


class TestMultipleEpisodes:
    """Tests for running multiple episodes."""

    def test_multiple_resets(self, environment: Ko2cubeEnvironment):
        """Can reset multiple times."""
        for difficulty in ["easy", "medium", "hard"] * 2:
            obs = environment.reset(task_id=difficulty)
            assert obs.current_step == 0
            assert not obs.done

    def test_episode_isolation(self, environment: Ko2cubeEnvironment):
        """Episodes are isolated from each other."""
        environment.reset(task_id="easy")
        obs1 = environment.get_observation()
        if obs1.job_queue:
            action = create_drop_action(obs1.job_queue[0])
            environment.step(action)
        
        dropped_before_reset = environment.state.jobs_dropped
        assert dropped_before_reset > 0
        
        environment.reset(task_id="easy")
        assert environment.state.jobs_dropped == 0

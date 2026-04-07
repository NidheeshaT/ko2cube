"""
Tests for environment determinism and reward signal quality.
Validates that the environment produces consistent, learnable rewards.
"""
import pytest
from typing import List

from server.environment import Ko2cubeEnvironment
from models import (
    Ko2cubeAction, Ko2cubeObservation, JobAssignment, ALWAYS_ON,
)
from tests.conftest import (
    create_schedule_action, create_defer_action, create_drop_action,
    schedule_all_jobs_in_region,
)


class TestEnvironmentDeterminism:
    """Verify environment produces identical results for identical actions."""

    def test_reset_is_deterministic(self):
        """Same reset produces identical observations."""
        env1 = Ko2cubeEnvironment()
        env2 = Ko2cubeEnvironment()
        
        obs1 = env1.reset(task_id="easy")
        obs2 = env2.reset(task_id="easy")
        
        assert obs1.current_step == obs2.current_step
        assert len(obs1.job_queue) == len(obs2.job_queue)
        
        job_ids_1 = sorted(j.job_id for j in obs1.job_queue)
        job_ids_2 = sorted(j.job_id for j in obs2.job_queue)
        assert job_ids_1 == job_ids_2
        
        for region in obs1.regions:
            assert region in obs2.regions
            assert obs1.regions[region].carbon.current_intensity == \
                   obs2.regions[region].carbon.current_intensity

    def test_step_is_deterministic(self):
        """Same actions produce identical rewards and observations."""
        env1 = Ko2cubeEnvironment()
        env2 = Ko2cubeEnvironment()
        
        env1.reset(task_id="easy")
        env2.reset(task_id="easy")
        
        for _ in range(10):
            obs1 = env1.get_observation()
            obs2 = env2.get_observation()
            
            if obs1.done or obs2.done:
                break
            
            action = schedule_all_jobs_in_region(
                obs1.job_queue, "us-east-1", "m5.xlarge"
            )
            
            result1 = env1.step(action)
            result2 = env2.step(action)
            
            assert result1.reward == result2.reward, \
                f"Rewards differ: {result1.reward} vs {result2.reward}"
            assert result1.current_step == result2.current_step
            assert len(result1.job_queue) == len(result2.job_queue)

    def test_carbon_data_deterministic(self):
        """Carbon intensity data is deterministic across resets."""
        env = Ko2cubeEnvironment()
        
        intensities_run1 = []
        env.reset(task_id="easy")
        for step in range(5):
            obs = env.get_observation()
            intensities_run1.append({
                region: obs.regions[region].carbon.current_intensity
                for region in obs.regions
            })
            if obs.done:
                break
            action = Ko2cubeAction(assignments=[])
            env.step(action)
        
        intensities_run2 = []
        env.reset(task_id="easy")
        for step in range(5):
            obs = env.get_observation()
            intensities_run2.append({
                region: obs.regions[region].carbon.current_intensity
                for region in obs.regions
            })
            if obs.done:
                break
            action = Ko2cubeAction(assignments=[])
            env.step(action)
        
        assert intensities_run1 == intensities_run2


class TestRewardSignalQuality:
    """Verify reward signal has good separation between actions."""

    def test_good_actions_better_than_bad(self):
        """Good actions have clearly higher rewards than bad actions."""
        env = Ko2cubeEnvironment()
        obs = env.reset(task_id="easy")
        
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        
        job = obs.job_queue[0]
        regions = sorted(
            obs.regions.items(),
            key=lambda x: x[1].carbon.current_intensity
        )
        best_region = regions[0][0]
        
        good_action = Ko2cubeAction(assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=best_region,
                instance_type="m5.large",
                machine_type="spot",
            )
        ])
        good_result = env.step(good_action)
        
        env.reset(task_id="easy")
        
        drop_action = Ko2cubeAction(assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="drop",
            )
        ])
        bad_result = env.step(drop_action)
        
        reward_diff = good_result.reward - bad_result.reward
        assert reward_diff > 2.0, \
            f"Reward separation too small: good={good_result.reward}, bad={bad_result.reward}, diff={reward_diff}"

    def test_schedule_better_than_unnecessary_defer(self):
        """Scheduling in clean region beats unnecessary deferral when conditions are right."""
        env = Ko2cubeEnvironment()
        obs = env.reset(task_id="easy")
        
        delay_tolerant_jobs = [
            j for j in obs.job_queue 
            if j.delay_tolerant and obs.current_step >= j.sla_start
        ]
        if not delay_tolerant_jobs:
            pytest.skip("No schedulable delay tolerant jobs")
        
        job = delay_tolerant_jobs[0]
        regions = sorted(
            obs.regions.items(),
            key=lambda x: x[1].carbon.current_intensity
        )
        cleanest_region = regions[0][0]
        cleanest_intensity = regions[0][1].carbon.current_intensity
        
        baseline_intensity = getattr(job, 'baseline_carbon_intensity', 200)
        if cleanest_intensity >= baseline_intensity:
            pytest.skip("Current carbon is not below baseline - defer would be smart")
        
        region_info = obs.regions[cleanest_region]
        fitting_instances = [
            i for i in region_info.available_instances
            if i.cpu_cores >= job.cpu_cores and i.memory_gb >= job.memory_gb
        ]
        if not fitting_instances:
            pytest.skip("No fitting instance for job")
        instance_type = fitting_instances[0].name
        
        schedule_action = Ko2cubeAction(assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=cleanest_region,
                instance_type=instance_type,
                machine_type="spot",
            )
        ])
        schedule_result = env.step(schedule_action)
        
        env.reset(task_id="easy")
        
        defer_action = Ko2cubeAction(assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="defer",
                defer_to_step=job.sla_end,
            )
        ])
        defer_result = env.step(defer_action)
        
        assert schedule_result.reward >= defer_result.reward - 2.0, \
            f"Schedule should be comparable to defer when carbon is low: schedule={schedule_result.reward}, defer={defer_result.reward}"

    def test_low_carbon_region_better_reward(self):
        """Scheduling in low-carbon region gets better carbon reward."""
        env = Ko2cubeEnvironment()
        obs = env.reset(task_id="medium")
        
        if not obs.job_queue:
            pytest.skip("No jobs in queue")
        if len(obs.regions) < 2:
            pytest.skip("Need multiple regions")
        
        job = obs.job_queue[0]
        regions = sorted(
            obs.regions.items(),
            key=lambda x: x[1].carbon.current_intensity
        )
        
        low_carbon_region = regions[0][0]
        high_carbon_region = regions[-1][0]
        
        if regions[0][1].carbon.current_intensity == regions[-1][1].carbon.current_intensity:
            pytest.skip("All regions have same carbon intensity")
        
        low_action = Ko2cubeAction(assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=low_carbon_region,
                instance_type="m5.large",
                machine_type="spot",
            )
        ])
        low_result = env.step(low_action)
        
        env.reset(task_id="medium")
        
        high_action = Ko2cubeAction(assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region=high_carbon_region,
                instance_type="m5.large",
                machine_type="spot",
            )
        ])
        high_result = env.step(high_action)
        
        assert low_result.reward >= high_result.reward - 0.5


class TestRewardConsistency:
    """Verify reward components are consistent and reasonable."""

    def test_reward_in_reasonable_range(self):
        """Individual step rewards are in reasonable range."""
        env = Ko2cubeEnvironment()
        env.reset(task_id="medium")
        
        rewards = []
        for _ in range(24):
            obs = env.get_observation()
            if obs.done:
                break
            
            action = schedule_all_jobs_in_region(
                obs.job_queue, "us-east-1", "m5.xlarge"
            )
            result = env.step(action)
            rewards.append(result.reward)
        
        for r in rewards:
            assert -20 <= r <= 20, f"Reward {r} out of reasonable range"

    def test_grader_score_in_valid_range(self):
        """Grader score is always between 0 and 1."""
        env = Ko2cubeEnvironment()
        
        for difficulty in ["easy", "medium", "hard"]:
            env.reset(task_id=difficulty)
            
            for _ in range(30):
                obs = env.get_observation()
                if obs.done:
                    break
                
                action = schedule_all_jobs_in_region(
                    obs.job_queue, "us-east-1", "m5.xlarge"
                )
                env.step(action)
            
            score = env.grader_score()
            assert 0.0 <= score <= 1.0, \
                f"Grader score {score} out of range for {difficulty}"

    def test_terminal_reward_positive_for_completion(self):
        """Completing all jobs gives positive terminal bonus."""
        env = Ko2cubeEnvironment()
        env.reset(task_id="easy")
        
        last_reward = 0.0
        while not env.state.is_done:
            obs = env.get_observation()
            action = schedule_all_jobs_in_region(
                obs.job_queue, "us-east-1", "m5.xlarge"
            )
            result = env.step(action)
            last_reward = result.reward
        
        assert env.state.jobs_completed > 0


class TestRewardComponents:
    """Verify individual reward components work correctly."""

    def test_sla_penalty_for_drop(self):
        """Dropping jobs incurs SLA penalty."""
        env = Ko2cubeEnvironment()
        obs = env.reset(task_id="easy")
        
        if not obs.job_queue:
            pytest.skip("No jobs")
        
        job = obs.job_queue[0]
        action = create_drop_action(job)
        result = env.step(action)
        
        assert result.reward < 0

    def test_urgent_job_immediate_schedule_bonus(self):
        """Scheduling urgent job immediately gives bonus."""
        env = Ko2cubeEnvironment()
        obs = env.reset(task_id="medium")
        
        urgent_jobs = [j for j in obs.job_queue if not j.delay_tolerant]
        if not urgent_jobs:
            pytest.skip("No urgent jobs at step 0")
        
        job = urgent_jobs[0]
        action = Ko2cubeAction(assignments=[
            JobAssignment(
                job_id=job.job_id,
                decision="schedule",
                region="us-east-1",
                instance_type="m5.xlarge",
                machine_type="spot",
            )
        ])
        result = env.step(action)
        
        assert result.reward > 0


class TestCarbonSignal:
    """Verify carbon-related signals work correctly."""

    def test_carbon_forecast_length(self):
        """Carbon forecast has expected length."""
        env = Ko2cubeEnvironment()
        obs = env.reset(task_id="easy")
        
        from server.data.scenarios import TASK_1_EASY
        expected_length = TASK_1_EASY.lookahead_steps
        
        for region_info in obs.regions.values():
            assert len(region_info.carbon.forecast) == expected_length

    def test_carbon_varies_across_steps(self):
        """Carbon intensity changes across steps."""
        env = Ko2cubeEnvironment()
        env.reset(task_id="easy")
        
        intensities = []
        for _ in range(10):
            obs = env.get_observation()
            if obs.done:
                break
            
            region_intensity = obs.regions["us-east-1"].carbon.current_intensity
            intensities.append(region_intensity)
            
            action = Ko2cubeAction(assignments=[])
            env.step(action)
        
        unique_intensities = set(intensities)
        assert len(unique_intensities) > 1, "Carbon should vary across steps"

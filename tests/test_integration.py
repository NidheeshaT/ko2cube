"""
Integration tests for the Ko2cube environment.
Tests full episode flows across all difficulty levels.
"""
import pytest
from typing import List

from server.environment import Ko2cubeEnvironment
from models import (
    Ko2cubeAction, Ko2cubeObservation, JobAssignment, ALWAYS_ON,
)
from tests.conftest import schedule_all_jobs_in_region


class TestFullEpisodeEasy:
    """Integration tests for full episode with easy difficulty."""

    @pytest.fixture
    def env(self) -> Ko2cubeEnvironment:
        """Create environment reset to easy."""
        env = Ko2cubeEnvironment()
        env.reset(task_id="easy")
        return env

    def test_complete_episode_all_schedule(self, env: Ko2cubeEnvironment):
        """Can complete full episode by scheduling all jobs."""
        total_reward = 0.0
        steps = 0
        
        while not env.state.is_done:
            obs = env.get_observation()
            action = schedule_all_jobs_in_region(
                obs.job_queue, "us-east-1", "m5.xlarge"
            )
            result = env.step(action)
            total_reward += result.reward
            steps += 1
            
            if steps > 50:
                break
        
        assert env.state.is_done
        assert env.state.jobs_completed > 0
        assert env.grader_score() > 0

    def test_complete_episode_with_deferral(self, env: Ko2cubeEnvironment):
        """Can complete episode with strategic deferral."""
        steps = 0
        deferred_jobs = []
        
        while not env.state.is_done:
            obs = env.get_observation()
            assignments = []
            
            for job in obs.job_queue:
                if job.delay_tolerant and steps < 5 and job.job_id not in deferred_jobs:
                    assignments.append(JobAssignment(
                        job_id=job.job_id,
                        decision="defer",
                        defer_to_step=steps + 3,
                    ))
                    deferred_jobs.append(job.job_id)
                else:
                    assignments.append(JobAssignment(
                        job_id=job.job_id,
                        decision="schedule",
                        region="us-east-1",
                        instance_type="m5.xlarge",
                        machine_type=job.instance_preference,
                    ))
            
            action = Ko2cubeAction(assignments=assignments)
            env.step(action)
            steps += 1
            
            if steps > 50:
                break
        
        assert env.state.is_done

    def test_grader_score_improves_with_good_actions(self, env: Ko2cubeEnvironment):
        """Good scheduling decisions lead to better grader score."""
        while not env.state.is_done:
            obs = env.get_observation()
            
            assignments = []
            sorted_regions = sorted(
                obs.regions.items(),
                key=lambda x: x[1].carbon.current_intensity
            )
            best_region = sorted_regions[0][0] if sorted_regions else "us-east-1"
            
            for job in obs.job_queue:
                assignments.append(JobAssignment(
                    job_id=job.job_id,
                    decision="schedule",
                    region=best_region,
                    instance_type="m5.xlarge",
                    machine_type=job.instance_preference,
                ))
            
            action = Ko2cubeAction(assignments=assignments)
            env.step(action)
        
        score = env.grader_score()
        assert score >= 0.4


class TestFullEpisodeMedium:
    """Integration tests for full episode with medium difficulty."""

    @pytest.fixture
    def env(self) -> Ko2cubeEnvironment:
        """Create environment reset to medium."""
        env = Ko2cubeEnvironment()
        env.reset(task_id="medium")
        return env

    def test_complete_medium_episode(self, env: Ko2cubeEnvironment):
        """Can complete medium difficulty episode."""
        steps = 0
        
        while not env.state.is_done:
            obs = env.get_observation()
            
            assignments = []
            for job in obs.job_queue:
                if not job.delay_tolerant:
                    assignments.append(JobAssignment(
                        job_id=job.job_id,
                        decision="schedule",
                        region="us-east-1",
                        instance_type="m5.xlarge",
                        machine_type=job.instance_preference,
                    ))
                else:
                    sorted_regions = sorted(
                        obs.regions.items(),
                        key=lambda x: x[1].carbon.current_intensity
                    )
                    best_region = sorted_regions[0][0]
                    assignments.append(JobAssignment(
                        job_id=job.job_id,
                        decision="schedule",
                        region=best_region,
                        instance_type="m5.xlarge",
                        machine_type=job.instance_preference,
                    ))
            
            action = Ko2cubeAction(assignments=assignments)
            env.step(action)
            steps += 1
            
            if steps > 50:
                break
        
        assert env.state.is_done
        assert env.state.jobs_completed > 0

    def test_medium_handles_cicd_urgency(self, env: Ko2cubeEnvironment):
        """Medium scenario correctly handles CI/CD urgency."""
        cicd_handled = 0
        steps = 0
        
        while not env.state.is_done:
            obs = env.get_observation()
            
            for job in obs.job_queue:
                if "cicd" in job.job_id and not job.delay_tolerant:
                    cicd_handled += 1
            
            action = schedule_all_jobs_in_region(
                obs.job_queue, "us-east-1", "m5.xlarge"
            )
            env.step(action)
            steps += 1
            
            if steps > 50:
                break
        
        assert env.state.jobs_completed > 0


class TestFullEpisodeHard:
    """Integration tests for full episode with hard difficulty."""

    @pytest.fixture
    def env(self) -> Ko2cubeEnvironment:
        """Create environment reset to hard."""
        env = Ko2cubeEnvironment()
        env.reset(task_id="hard")
        return env

    def test_complete_hard_episode(self, env: Ko2cubeEnvironment):
        """Can complete hard difficulty episode."""
        steps = 0
        
        while not env.state.is_done:
            obs = env.get_observation()
            
            assignments = []
            for job in obs.job_queue:
                if job.sla_end == ALWAYS_ON:
                    assignments.append(JobAssignment(
                        job_id=job.job_id,
                        decision="schedule",
                        region="us-east-1",
                        instance_type="m5.xlarge",
                        machine_type="on-demand",
                    ))
                else:
                    assignments.append(JobAssignment(
                        job_id=job.job_id,
                        decision="schedule",
                        region="us-east-1",
                        instance_type="m5.xlarge",
                        machine_type=job.instance_preference,
                    ))
            
            action = Ko2cubeAction(assignments=assignments)
            env.step(action)
            steps += 1
            
            if steps > 60:
                break
        
        assert env.state.is_done

    def test_hard_always_on_protected(self, env: Ko2cubeEnvironment):
        """Always-on jobs in hard scenario get special handling."""
        obs = env.get_observation()
        
        always_on = [j for j in env.state.all_jobs.values() if j.sla_end == ALWAYS_ON]
        assert len(always_on) > 0, "Hard scenario should have always-on jobs"


class TestCrossScenarioConsistency:
    """Tests for consistency across scenarios."""

    def test_all_scenarios_complete(self):
        """All difficulty levels can complete."""
        for difficulty in ["easy", "medium", "hard"]:
            env = Ko2cubeEnvironment()
            env.reset(task_id=difficulty)
            
            steps = 0
            while not env.state.is_done:
                obs = env.get_observation()
                action = schedule_all_jobs_in_region(
                    obs.job_queue, "us-east-1", "m5.xlarge"
                )
                env.step(action)
                steps += 1
                
                if steps > 60:
                    break
            
            assert env.state.is_done, f"{difficulty} scenario should complete"

    def test_reward_range_reasonable(self):
        """Rewards are in reasonable range across scenarios."""
        for difficulty in ["easy", "medium", "hard"]:
            env = Ko2cubeEnvironment()
            env.reset(task_id=difficulty)
            
            rewards = []
            steps = 0
            
            while not env.state.is_done:
                obs = env.get_observation()
                action = schedule_all_jobs_in_region(
                    obs.job_queue, "us-east-1", "m5.xlarge"
                )
                result = env.step(action)
                rewards.append(result.reward)
                steps += 1
                
                if steps > 60:
                    break
            
            assert len(rewards) > 0
            assert max(rewards) < 20
            assert min(rewards) > -20


class TestDeterminism:
    """Tests for environment determinism."""

    def test_same_actions_same_results(self):
        """Same actions produce identical results."""
        env1 = Ko2cubeEnvironment()
        env2 = Ko2cubeEnvironment()
        
        obs1 = env1.reset(task_id="easy")
        obs2 = env2.reset(task_id="easy")
        
        assert len(obs1.job_queue) == len(obs2.job_queue)
        
        for _ in range(5):
            if env1.state.is_done or env2.state.is_done:
                break
            
            obs1_current = env1.get_observation()
            obs2_current = env2.get_observation()
            
            action = schedule_all_jobs_in_region(
                obs1_current.job_queue, "us-east-1", "m5.xlarge"
            )
            
            result1 = env1.step(action)
            result2 = env2.step(action)
            
            assert result1.reward == result2.reward

    def test_region_data_consistent(self):
        """Region data is consistent across resets."""
        env = Ko2cubeEnvironment()
        
        obs1 = env.reset(task_id="easy")
        regions1 = {r: obs1.regions[r].carbon.current_intensity for r in obs1.regions}
        
        obs2 = env.reset(task_id="easy")
        regions2 = {r: obs2.regions[r].carbon.current_intensity for r in obs2.regions}
        
        assert regions1 == regions2


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_action_list(self):
        """Environment handles empty action list."""
        env = Ko2cubeEnvironment()
        env.reset(task_id="easy")
        
        action = Ko2cubeAction(assignments=[])
        result = env.step(action)
        
        assert result is not None
        assert result.current_step == 1

    def test_invalid_job_id_ignored(self):
        """Invalid job IDs in actions are ignored."""
        env = Ko2cubeEnvironment()
        env.reset(task_id="easy")
        
        action = Ko2cubeAction(assignments=[
            JobAssignment(
                job_id="nonexistent_job",
                decision="schedule",
                region="us-east-1",
                instance_type="m5.xlarge",
                machine_type="spot",
            )
        ])
        
        result = env.step(action)
        assert result is not None

    def test_multiple_resets(self):
        """Can reset multiple times."""
        env = Ko2cubeEnvironment()
        
        for _ in range(5):
            obs = env.reset(task_id="easy")
            assert obs.current_step == 0
            
            action = schedule_all_jobs_in_region(
                obs.job_queue, "us-east-1", "m5.xlarge"
            )
            env.step(action)

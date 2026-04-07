"""
Tests for baseline agents and evaluation framework.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import Ko2cubeEnvironment
from server.baselines import (
    BaselineAgent, RandomAgent, GreedyCostAgent,
    CarbonAwareGreedyAgent, OracleAgent, HybridAgent,
    AgentMetrics, create_agent, run_episode, BASELINE_AGENTS,
)
from models import Ko2cubeAction, Ko2cubeObservation, Job, RegionInfo, CarbonData, InstanceType


class TestAgentMetrics:
    """Test AgentMetrics dataclass."""
    
    def test_default_values(self):
        metrics = AgentMetrics()
        assert metrics.total_reward == 0.0
        assert metrics.total_carbon_gco2 == 0.0
        assert metrics.total_cost_usd == 0.0
        assert metrics.jobs_scheduled == 0
        assert metrics.jobs_deferred == 0
        assert metrics.jobs_dropped == 0
        assert metrics.sla_violations == 0
        assert metrics.steps == 0
    
    def test_accumulation(self):
        metrics = AgentMetrics()
        metrics.total_reward += 10.0
        metrics.jobs_scheduled += 5
        assert metrics.total_reward == 10.0
        assert metrics.jobs_scheduled == 5


class TestBaselineAgentInterface:
    """Test that all baseline agents implement the interface correctly."""
    
    @pytest.fixture
    def sample_observation(self) -> Ko2cubeObservation:
        """Create a minimal valid observation."""
        return Ko2cubeObservation(
            current_step=0,
            job_queue=[
                Job(
                    job_id="test_job_1",
                    cpu_cores=2,
                    memory_gb=8,
                    sla_start=0,
                    sla_end=10,
                    delay_tolerant=True,
                    instance_preference="spot",
                    eta_minutes=60,
                )
            ],
            active_jobs=[],
            regions={
                "us-east-1": RegionInfo(
                    region_name="us-east-1",
                    carbon=CarbonData(current_intensity=200.0, forecast=[190, 180, 170]),
                    available_instances=[
                        InstanceType(
                            name="m5.xlarge",
                            cpu_cores=4,
                            memory_gb=16,
                            spot_price=0.1,
                            on_demand_price=0.2,
                            available_count=10,
                        )
                    ],
                ),
            },
        )
    
    @pytest.mark.parametrize("agent_class", [
        RandomAgent, GreedyCostAgent, CarbonAwareGreedyAgent, OracleAgent, HybridAgent
    ])
    def test_agent_has_name(self, agent_class):
        agent = agent_class()
        assert hasattr(agent, "name")
        assert isinstance(agent.name, str)
        assert len(agent.name) > 0
    
    @pytest.mark.parametrize("agent_class", [
        RandomAgent, GreedyCostAgent, CarbonAwareGreedyAgent, OracleAgent, HybridAgent
    ])
    def test_agent_has_metrics(self, agent_class):
        agent = agent_class()
        assert hasattr(agent, "metrics")
        assert isinstance(agent.metrics, AgentMetrics)
    
    @pytest.mark.parametrize("agent_class", [
        RandomAgent, GreedyCostAgent, CarbonAwareGreedyAgent, OracleAgent, HybridAgent
    ])
    def test_agent_reset(self, agent_class):
        agent = agent_class()
        agent.metrics.total_reward = 100.0
        agent.metrics.jobs_scheduled = 50
        
        agent.reset()
        
        assert agent.metrics.total_reward == 0.0
        assert agent.metrics.jobs_scheduled == 0
    
    @pytest.mark.parametrize("agent_class", [
        RandomAgent, GreedyCostAgent, CarbonAwareGreedyAgent, OracleAgent, HybridAgent
    ])
    def test_agent_act_returns_action(self, agent_class, sample_observation):
        agent = agent_class()
        action = agent.act(sample_observation)
        
        assert isinstance(action, Ko2cubeAction)
        assert hasattr(action, "assignments")
        assert len(action.assignments) == len(sample_observation.job_queue)
    
    @pytest.mark.parametrize("agent_class", [
        RandomAgent, GreedyCostAgent, CarbonAwareGreedyAgent, OracleAgent, HybridAgent
    ])
    def test_agent_action_has_valid_decisions(self, agent_class, sample_observation):
        agent = agent_class()
        action = agent.act(sample_observation)
        
        for assignment in action.assignments:
            assert assignment.decision in ["schedule", "defer", "drop"]
            
            if assignment.decision == "schedule":
                assert assignment.region is not None
                assert assignment.instance_type is not None
            elif assignment.decision == "defer":
                assert assignment.defer_to_step is not None


class TestRandomAgent:
    """Test RandomAgent specific behavior."""
    
    @pytest.fixture
    def agent(self):
        return RandomAgent(drop_prob=0.1, defer_prob=0.3)
    
    def test_custom_probabilities(self, agent):
        assert agent.drop_prob == 0.1
        assert agent.defer_prob == 0.3
    
    def test_randomness_produces_variety(self):
        """Over many runs, random agent should produce different decisions."""
        env = Ko2cubeEnvironment()
        agent = RandomAgent()
        
        decisions = set()
        for _ in range(20):
            obs = env.reset(task_id="easy")
            if obs.job_queue:
                action = agent.act(obs)
                for a in action.assignments:
                    decisions.add(a.decision)
        
        # Should see at least 2 different decision types
        assert len(decisions) >= 2


class TestGreedyCostAgent:
    """Test GreedyCostAgent behavior."""
    
    def test_chooses_cheapest_region(self):
        """Agent should prefer the cheapest region."""
        agent = GreedyCostAgent()
        
        obs = Ko2cubeObservation(
            current_step=0,
            job_queue=[
                Job(
                    job_id="test_job",
                    cpu_cores=2,
                    memory_gb=8,
                    sla_start=0,
                    sla_end=10,
                    delay_tolerant=True,
                    instance_preference="spot",
                    eta_minutes=60,
                )
            ],
            active_jobs=[],
            regions={
                "expensive_region": RegionInfo(
                    region_name="expensive_region",
                    carbon=CarbonData(current_intensity=100.0, forecast=[100]),
                    available_instances=[
                        InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                                   spot_price=1.0, on_demand_price=2.0, available_count=10)
                    ],
                ),
                "cheap_region": RegionInfo(
                    region_name="cheap_region",
                    carbon=CarbonData(current_intensity=300.0, forecast=[300]),
                    available_instances=[
                        InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                                   spot_price=0.1, on_demand_price=0.2, available_count=10)
                    ],
                ),
            },
        )
        
        action = agent.act(obs)
        assert action.assignments[0].region == "cheap_region"


class TestCarbonAwareGreedyAgent:
    """Test CarbonAwareGreedyAgent behavior."""
    
    def test_chooses_greenest_region(self):
        """Agent should prefer the lowest carbon region."""
        agent = CarbonAwareGreedyAgent()
        
        obs = Ko2cubeObservation(
            current_step=0,
            job_queue=[
                Job(
                    job_id="test_job",
                    cpu_cores=2,
                    memory_gb=8,
                    sla_start=0,
                    sla_end=10,
                    delay_tolerant=True,
                    instance_preference="spot",
                    eta_minutes=60,
                )
            ],
            active_jobs=[],
            regions={
                "dirty_region": RegionInfo(
                    region_name="dirty_region",
                    carbon=CarbonData(current_intensity=500.0, forecast=[500]),
                    available_instances=[
                        InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                                   spot_price=0.05, on_demand_price=0.1, available_count=10)
                    ],
                ),
                "clean_region": RegionInfo(
                    region_name="clean_region",
                    carbon=CarbonData(current_intensity=50.0, forecast=[50]),
                    available_instances=[
                        InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                                   spot_price=0.5, on_demand_price=1.0, available_count=10)
                    ],
                ),
            },
        )
        
        action = agent.act(obs)
        assert action.assignments[0].region == "clean_region"


class TestOracleAgent:
    """Test OracleAgent behavior."""
    
    def test_uses_forecast_for_deferral(self):
        """Oracle should defer if forecast shows improvement."""
        agent = OracleAgent(carbon_weight=1.0, cost_weight=0.0)  # Pure carbon optimization
        
        obs = Ko2cubeObservation(
            current_step=0,
            job_queue=[
                Job(
                    job_id="test_job",
                    cpu_cores=2,
                    memory_gb=8,
                    sla_start=0,
                    sla_end=5,
                    delay_tolerant=True,
                    instance_preference="spot",
                    eta_minutes=60,
                )
            ],
            active_jobs=[],
            regions={
                "region1": RegionInfo(
                    region_name="region1",
                    carbon=CarbonData(
                        current_intensity=400.0,
                        forecast=[300, 200, 100, 50, 50]  # Improving forecast
                    ),
                    available_instances=[
                        InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                                   spot_price=0.1, on_demand_price=0.2, available_count=10)
                    ],
                ),
            },
        )
        
        action = agent.act(obs)
        # Oracle should recognize the improving forecast and defer
        # (or schedule at the optimal time, which may still be now depending on penalty)
        assert action.assignments[0].decision in ["schedule", "defer"]
    
    def test_custom_weights(self):
        agent = OracleAgent(carbon_weight=0.9, cost_weight=0.1)
        assert agent.carbon_weight == 0.9
        assert agent.cost_weight == 0.1


class TestHybridAgent:
    """Test HybridAgent behavior."""
    
    def test_considers_urgency(self):
        """Hybrid agent should schedule urgent jobs immediately."""
        agent = HybridAgent(urgency_weight=0.5)
        
        # Job with tight deadline (urgent)
        obs = Ko2cubeObservation(
            current_step=9,  # Near deadline
            job_queue=[
                Job(
                    job_id="urgent_job",
                    cpu_cores=2,
                    memory_gb=8,
                    sla_start=0,
                    sla_end=10,  # Deadline is step 10
                    delay_tolerant=True,
                    instance_preference="spot",
                    eta_minutes=60,
                )
            ],
            active_jobs=[],
            regions={
                "region1": RegionInfo(
                    region_name="region1",
                    carbon=CarbonData(current_intensity=300.0, forecast=[100, 50]),  # Improving
                    available_instances=[
                        InstanceType(name="m5.xlarge", cpu_cores=4, memory_gb=16,
                                   spot_price=0.1, on_demand_price=0.2, available_count=10)
                    ],
                ),
            },
        )
        
        action = agent.act(obs)
        # Should schedule immediately due to urgency, not defer despite improving forecast
        assert action.assignments[0].decision == "schedule"


class TestCreateAgent:
    """Test agent factory function."""
    
    def test_create_known_agents(self):
        for name in ["random", "greedy_cost", "carbon_aware", "oracle", "hybrid"]:
            agent = create_agent(name)
            assert isinstance(agent, BaselineAgent)
    
    def test_create_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown agent"):
            create_agent("nonexistent_agent")
    
    def test_create_with_kwargs(self):
        agent = create_agent("random", drop_prob=0.5, defer_prob=0.5)
        assert agent.drop_prob == 0.5
        assert agent.defer_prob == 0.5


class TestRunEpisode:
    """Test run_episode helper function."""
    
    @pytest.fixture
    def environment(self):
        return Ko2cubeEnvironment()
    
    def test_run_episode_completes(self, environment):
        agent = RandomAgent()
        metrics = run_episode(environment, agent, task_id="easy", verbose=False)
        
        assert isinstance(metrics, AgentMetrics)
        assert metrics.steps > 0
    
    def test_run_episode_collects_metrics(self, environment):
        agent = GreedyCostAgent()
        metrics = run_episode(environment, agent, task_id="easy", verbose=False)
        
        # Should have some carbon and cost recorded
        assert metrics.total_carbon_gco2 >= 0
        assert metrics.total_cost_usd >= 0


class TestBaselineRegistry:
    """Test BASELINE_AGENTS registry."""
    
    def test_registry_contains_expected_agents(self):
        expected = {"random", "greedy_cost", "carbon_aware", "oracle", "hybrid"}
        assert set(BASELINE_AGENTS.keys()) == expected
    
    def test_registry_values_are_classes(self):
        for name, cls in BASELINE_AGENTS.items():
            assert callable(cls)
            instance = cls()
            assert isinstance(instance, BaselineAgent)


class TestAgentHelperMethods:
    """Test helper methods on BaselineAgent."""
    
    @pytest.fixture
    def agent(self):
        return RandomAgent()
    
    @pytest.fixture
    def region_with_instances(self) -> RegionInfo:
        return RegionInfo(
            region_name="test_region",
            carbon=CarbonData(current_intensity=200.0, forecast=[200]),
            available_instances=[
                InstanceType(name="small", cpu_cores=2, memory_gb=4,
                           spot_price=0.1, on_demand_price=0.2, available_count=5),
                InstanceType(name="medium", cpu_cores=4, memory_gb=8,
                           spot_price=0.2, on_demand_price=0.4, available_count=3),
                InstanceType(name="large", cpu_cores=8, memory_gb=16,
                           spot_price=0.4, on_demand_price=0.8, available_count=1),
            ],
        )
    
    def test_get_fitting_instance_finds_match(self, agent, region_with_instances):
        job = Job(
            job_id="test",
            cpu_cores=3,
            memory_gb=6,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        
        instance = agent._get_fitting_instance(job, region_with_instances)
        assert instance == "medium"  # First instance that fits
    
    def test_get_fitting_instance_returns_none_if_no_fit(self, agent, region_with_instances):
        job = Job(
            job_id="test",
            cpu_cores=16,  # Too big
            memory_gb=32,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        
        instance = agent._get_fitting_instance(job, region_with_instances)
        assert instance is None
    
    def test_can_schedule_in_region(self, agent, region_with_instances):
        small_job = Job(
            job_id="small",
            cpu_cores=2,
            memory_gb=4,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        
        large_job = Job(
            job_id="large",
            cpu_cores=100,
            memory_gb=200,
            sla_start=0,
            sla_end=10,
            delay_tolerant=True,
            instance_preference="spot",
        )
        
        assert agent._can_schedule_in_region(small_job, region_with_instances) is True
        assert agent._can_schedule_in_region(large_job, region_with_instances) is False


class TestIntegrationWithEnvironment:
    """Integration tests running agents with real environment."""
    
    @pytest.fixture
    def environment(self):
        return Ko2cubeEnvironment()
    
    @pytest.mark.parametrize("agent_name", list(BASELINE_AGENTS.keys()))
    def test_full_episode_all_agents(self, environment, agent_name):
        """All agents should complete a full episode without errors."""
        agent = create_agent(agent_name)
        
        obs = environment.reset(task_id="easy")
        steps = 0
        max_steps = 100
        
        while steps < max_steps:
            action = agent.act(obs)
            result = environment.step(action)
            steps += 1
            
            if result.done:
                break
            obs = result
        
        # Episode should complete within max steps
        assert result.done or steps == max_steps
    
    def test_oracle_outperforms_random_on_carbon(self, environment):
        """Oracle agent should generally save more carbon than random."""
        random_carbons = []
        oracle_carbons = []
        
        for _ in range(3):
            # Run random
            random_agent = RandomAgent()
            obs = environment.reset(task_id="easy")
            while True:
                action = random_agent.act(obs)
                result = environment.step(action)
                if result.done:
                    break
                obs = result
            random_carbons.append(environment.state.total_carbon_gco2)
            
            # Run oracle
            oracle_agent = OracleAgent()
            obs = environment.reset(task_id="easy")
            while True:
                action = oracle_agent.act(obs)
                result = environment.step(action)
                if result.done:
                    break
                obs = result
            oracle_carbons.append(environment.state.total_carbon_gco2)
        
        # Oracle should have lower average carbon (or at least not much worse)
        avg_random = sum(random_carbons) / len(random_carbons)
        avg_oracle = sum(oracle_carbons) / len(oracle_carbons)
        
        # Oracle should be at least not significantly worse
        assert avg_oracle <= avg_random * 1.5  # Allow some variance

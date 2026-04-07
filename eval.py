#!/usr/bin/env python3
"""
Ko2cube Evaluation Framework

This module provides comprehensive evaluation tools for assessing agent performance
on carbon-aware scheduling tasks. It supports:

1. Baseline comparison against multiple reference agents
2. Multi-scenario evaluation across difficulty levels
3. Statistical analysis with confidence intervals
4. Detailed reporting with carbon/cost/SLA metrics
5. Curriculum-based progressive evaluation

Usage:
    python eval.py --agent <agent_type> --scenarios easy,medium,hard --episodes 10
    
    # Evaluate all baselines
    python eval.py --all-baselines --scenarios easy --episodes 5
    
    # Run with curriculum progression
    python eval.py --agent oracle --curriculum --episodes 20
"""

import os
import sys
import json
import argparse
import random
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import Ko2cubeEnvironment
from server.baselines import (
    BaselineAgent, RandomAgent, GreedyCostAgent, 
    CarbonAwareGreedyAgent, OracleAgent, HybridAgent,
    create_agent, run_episode, AgentMetrics,
)
from server.curriculum import CurriculumManager, DifficultyLevel
from server.data.scenarios import SCENARIO_REGISTRY


@dataclass
class EpisodeResult:
    """Result from a single evaluation episode."""
    agent_name: str
    scenario: str
    difficulty: str
    episode_num: int
    total_reward: float
    total_carbon_gco2: float
    total_cost_usd: float
    baseline_carbon_gco2: float
    baseline_cost_usd: float
    carbon_savings_pct: float
    cost_savings_pct: float
    sla_violations: int
    jobs_completed: int
    jobs_dropped: int
    steps: int
    grader_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AggregatedResults:
    """Aggregated results across multiple episodes."""
    agent_name: str
    scenario: str
    num_episodes: int
    
    # Reward statistics
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    
    # Carbon statistics
    mean_carbon_gco2: float
    std_carbon_gco2: float
    mean_carbon_savings_pct: float
    
    # Cost statistics
    mean_cost_usd: float
    std_cost_usd: float
    mean_cost_savings_pct: float
    
    # Quality metrics
    mean_sla_violations: float
    mean_grader_score: float
    
    # Success rate
    success_rate: float  # % of episodes with positive reward
    
    # Confidence intervals (95%)
    reward_ci_lower: float
    reward_ci_upper: float


def calculate_statistics(values: List[float]) -> Tuple[float, float, float, float]:
    """Calculate mean, std, min, max for a list of values."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    
    n = len(values)
    mean = sum(values) / n
    
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    
    return mean, std, min(values), max(values)


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for the mean."""
    if len(values) < 2:
        mean = values[0] if values else 0.0
        return mean, mean
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std_error = math.sqrt(variance / n)
    
    # Use 1.96 for 95% CI (z-score approximation)
    z_score = 1.96 if confidence == 0.95 else 2.576  # 99% fallback
    margin = z_score * std_error
    
    return mean - margin, mean + margin


class Evaluator:
    """Main evaluation class for running agent assessments."""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.env = Ko2cubeEnvironment()
        self.output_dir = Path(output_dir) if output_dir else Path("eval_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if seed is not None:
            random.seed(seed)
        
        self.episode_results: List[EpisodeResult] = []
    
    def evaluate_agent(
        self,
        agent: BaselineAgent,
        scenarios: List[str],
        num_episodes: int = 10,
        verbose: bool = True,
    ) -> Dict[str, AggregatedResults]:
        """
        Evaluate an agent across multiple scenarios.
        
        Args:
            agent: The agent to evaluate
            scenarios: List of scenario names to test
            num_episodes: Number of episodes per scenario
            verbose: Print progress
        
        Returns:
            Dictionary mapping scenario -> AggregatedResults
        """
        results = {}
        
        for scenario in scenarios:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating {agent.name} on {scenario}")
                print('='*60)
            
            scenario_results = self._evaluate_scenario(
                agent, scenario, num_episodes, verbose
            )
            results[scenario] = scenario_results
        
        return results
    
    def _evaluate_scenario(
        self,
        agent: BaselineAgent,
        scenario: str,
        num_episodes: int,
        verbose: bool,
    ) -> AggregatedResults:
        """Run evaluation episodes for a single scenario."""
        episode_results = []
        
        for ep in range(num_episodes):
            agent.reset()
            obs = self.env.reset(task_id=scenario)
            total_reward = 0.0
            steps = 0
            
            while True:
                action = agent.act(obs)
                result = self.env.step(action)
                total_reward += result.reward
                steps += 1
                
                if result.done:
                    break
                obs = result
            
            # Collect final state metrics
            state = self.env.state
            
            # Calculate savings percentages
            carbon_savings = 0.0
            if state.baseline_carbon_gco2 > 0:
                carbon_savings = (
                    (state.baseline_carbon_gco2 - state.total_carbon_gco2) 
                    / state.baseline_carbon_gco2 * 100
                )
            
            cost_savings = 0.0
            if state.baseline_cost_usd > 0:
                cost_savings = (
                    (state.baseline_cost_usd - state.total_cost_usd)
                    / state.baseline_cost_usd * 100
                )
            
            # Get grader score
            from server.rewards import compute_grader_score
            grader_score = compute_grader_score(state)
            
            ep_result = EpisodeResult(
                agent_name=agent.name,
                scenario=scenario,
                difficulty=state.difficulty,
                episode_num=ep + 1,
                total_reward=total_reward,
                total_carbon_gco2=state.total_carbon_gco2,
                total_cost_usd=state.total_cost_usd,
                baseline_carbon_gco2=state.baseline_carbon_gco2,
                baseline_cost_usd=state.baseline_cost_usd,
                carbon_savings_pct=carbon_savings,
                cost_savings_pct=cost_savings,
                sla_violations=state.sla_violations,
                jobs_completed=state.jobs_completed,
                jobs_dropped=state.jobs_dropped,
                steps=steps,
                grader_score=grader_score,
            )
            
            episode_results.append(ep_result)
            self.episode_results.append(ep_result)
            
            if verbose:
                print(f"  Episode {ep+1}/{num_episodes}: "
                      f"reward={total_reward:.2f}, "
                      f"carbon_savings={carbon_savings:.1f}%, "
                      f"grader={grader_score:.2f}")
        
        # Aggregate results
        return self._aggregate_results(agent.name, scenario, episode_results)
    
    def _aggregate_results(
        self,
        agent_name: str,
        scenario: str,
        results: List[EpisodeResult],
    ) -> AggregatedResults:
        """Aggregate episode results into summary statistics."""
        rewards = [r.total_reward for r in results]
        carbons = [r.total_carbon_gco2 for r in results]
        costs = [r.total_cost_usd for r in results]
        carbon_savings = [r.carbon_savings_pct for r in results]
        cost_savings = [r.cost_savings_pct for r in results]
        violations = [r.sla_violations for r in results]
        grader_scores = [r.grader_score for r in results]
        
        mean_r, std_r, min_r, max_r = calculate_statistics(rewards)
        mean_c, std_c, _, _ = calculate_statistics(carbons)
        mean_cost, std_cost, _, _ = calculate_statistics(costs)
        
        ci_lower, ci_upper = calculate_confidence_interval(rewards)
        
        success_rate = sum(1 for r in rewards if r > 0) / len(rewards) * 100
        
        return AggregatedResults(
            agent_name=agent_name,
            scenario=scenario,
            num_episodes=len(results),
            mean_reward=mean_r,
            std_reward=std_r,
            min_reward=min_r,
            max_reward=max_r,
            mean_carbon_gco2=mean_c,
            std_carbon_gco2=std_c,
            mean_carbon_savings_pct=sum(carbon_savings) / len(carbon_savings),
            mean_cost_usd=mean_cost,
            std_cost_usd=std_cost,
            mean_cost_savings_pct=sum(cost_savings) / len(cost_savings),
            mean_sla_violations=sum(violations) / len(violations),
            mean_grader_score=sum(grader_scores) / len(grader_scores),
            success_rate=success_rate,
            reward_ci_lower=ci_lower,
            reward_ci_upper=ci_upper,
        )
    
    def compare_baselines(
        self,
        scenarios: List[str],
        num_episodes: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, AggregatedResults]]:
        """
        Compare all baseline agents across scenarios.
        
        Returns:
            Nested dict: agent_name -> scenario -> AggregatedResults
        """
        agents = [
            RandomAgent(),
            GreedyCostAgent(),
            CarbonAwareGreedyAgent(),
            OracleAgent(),
            HybridAgent(),
        ]
        
        all_results = {}
        
        for agent in agents:
            results = self.evaluate_agent(
                agent, scenarios, num_episodes, verbose
            )
            all_results[agent.name] = results
        
        return all_results
    
    def evaluate_with_curriculum(
        self,
        agent: BaselineAgent,
        num_episodes: int = 20,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate agent with curriculum progression.
        
        The agent starts with easy scenarios and progresses to harder
        ones as it demonstrates mastery.
        """
        curriculum = CurriculumManager()
        progression_log = []
        
        for ep in range(num_episodes):
            task_id = curriculum.get_task()
            agent.reset()
            
            obs = self.env.reset(task_id=task_id)
            total_reward = 0.0
            
            while True:
                action = agent.act(obs)
                result = self.env.step(action)
                total_reward += result.reward
                
                if result.done:
                    break
                obs = result
            
            state = self.env.state
            from server.rewards import compute_grader_score
            grader_score = compute_grader_score(state)
            
            # Record result with curriculum
            carbon_savings = 0.0
            if state.baseline_carbon_gco2 > 0:
                carbon_savings = (
                    (state.baseline_carbon_gco2 - state.total_carbon_gco2)
                    / state.baseline_carbon_gco2 * 100
                )
            
            curriculum.record_result(
                task_id=task_id,
                reward=total_reward,
                grader_score=grader_score,
                carbon_savings=carbon_savings,
            )
            
            current_level = curriculum.tracker.current_level.value
            progression_log.append({
                "episode": ep + 1,
                "task": task_id,
                "level": current_level,
                "reward": total_reward,
                "grader_score": grader_score,
                "carbon_savings": carbon_savings,
            })
            
            if verbose:
                print(f"Episode {ep+1}: task={task_id}, level={current_level}, "
                      f"reward={total_reward:.2f}, grader={grader_score:.2f}")
        
        return {
            "agent_name": agent.name,
            "num_episodes": num_episodes,
            "final_level": curriculum.tracker.current_level.value,
            "progression": progression_log,
            "mastery_stats": {
                jt.value: curriculum.tracker.job_type_mastery.get(jt, 0.0)
                for jt in curriculum.tracker.job_type_mastery
            },
        }
    
    def generate_report(
        self,
        results: Dict[str, Dict[str, AggregatedResults]],
        output_file: Optional[str] = None,
    ) -> str:
        """Generate a formatted evaluation report."""
        lines = [
            "=" * 80,
            "KO2CUBE EVALUATION REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 80,
            "",
        ]
        
        # Summary table per scenario
        for scenario in list(results.values())[0].keys():
            lines.append(f"\n--- Scenario: {scenario.upper()} ---\n")
            lines.append(f"{'Agent':<25} {'Reward':>10} {'Carbon%':>10} {'Cost%':>10} {'SLA Viol':>10} {'Grader':>10}")
            lines.append("-" * 75)
            
            scenario_results = [
                (agent, r[scenario]) 
                for agent, r in results.items()
            ]
            
            # Sort by grader score descending
            scenario_results.sort(key=lambda x: x[1].mean_grader_score, reverse=True)
            
            for agent_name, r in scenario_results:
                lines.append(
                    f"{agent_name:<25} "
                    f"{r.mean_reward:>10.2f} "
                    f"{r.mean_carbon_savings_pct:>9.1f}% "
                    f"{r.mean_cost_savings_pct:>9.1f}% "
                    f"{r.mean_sla_violations:>10.1f} "
                    f"{r.mean_grader_score:>10.2f}"
                )
        
        # Best performers summary
        lines.extend([
            "",
            "=" * 80,
            "BEST PERFORMERS BY METRIC",
            "=" * 80,
        ])
        
        for scenario in list(results.values())[0].keys():
            lines.append(f"\n{scenario}:")
            
            # Best by grader score
            best_grader = max(
                results.items(),
                key=lambda x: x[1][scenario].mean_grader_score
            )
            lines.append(f"  Best Grader Score: {best_grader[0]} "
                        f"({best_grader[1][scenario].mean_grader_score:.2f})")
            
            # Best by carbon savings
            best_carbon = max(
                results.items(),
                key=lambda x: x[1][scenario].mean_carbon_savings_pct
            )
            lines.append(f"  Best Carbon Savings: {best_carbon[0]} "
                        f"({best_carbon[1][scenario].mean_carbon_savings_pct:.1f}%)")
            
            # Best by reward
            best_reward = max(
                results.items(),
                key=lambda x: x[1][scenario].mean_reward
            )
            lines.append(f"  Best Reward: {best_reward[0]} "
                        f"({best_reward[1][scenario].mean_reward:.2f})")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
        
        return report
    
    def save_results(self, filename: Optional[str] = None):
        """Save all episode results to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"eval_results_{timestamp}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "num_episodes": len(self.episode_results),
            "episodes": [asdict(r) for r in self.episode_results],
        }
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filename}")


def main():
    """Main entry point for evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Ko2cube Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Oracle agent on easy and medium scenarios
  python eval.py --agent oracle --scenarios easy,medium --episodes 10
  
  # Compare all baselines on all difficulties
  python eval.py --all-baselines --scenarios easy,medium,hard --episodes 5
  
  # Run curriculum evaluation
  python eval.py --agent hybrid --curriculum --episodes 30
  
  # Evaluate with specific seed for reproducibility
  python eval.py --agent random --scenarios easy --episodes 20 --seed 42
        """
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        choices=["random", "greedy_cost", "carbon_aware", "oracle", "hybrid"],
        help="Agent type to evaluate",
    )
    parser.add_argument(
        "--all-baselines",
        action="store_true",
        help="Compare all baseline agents",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="easy",
        help="Comma-separated list of scenarios (easy,medium,hard)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes per scenario",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum-based evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    
    args = parser.parse_args()
    verbose = args.verbose and not args.quiet
    
    # Parse scenarios
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    
    # Validate scenarios
    for s in scenarios:
        if s not in SCENARIO_REGISTRY:
            print(f"Error: Unknown scenario '{s}'. Available: {list(SCENARIO_REGISTRY.keys())}")
            sys.exit(1)
    
    # Create evaluator
    evaluator = Evaluator(output_dir=args.output_dir, seed=args.seed)
    
    if args.all_baselines:
        # Compare all baselines
        print("\n" + "="*60)
        print("COMPARING ALL BASELINE AGENTS")
        print("="*60)
        
        results = evaluator.compare_baselines(
            scenarios=scenarios,
            num_episodes=args.episodes,
            verbose=verbose,
        )
        
        # Generate and print report
        report = evaluator.generate_report(
            results,
            output_file=Path(args.output_dir) / "baseline_comparison.txt"
        )
        print("\n" + report)
        
    elif args.curriculum:
        # Curriculum evaluation
        if not args.agent:
            print("Error: --agent required for curriculum evaluation")
            sys.exit(1)
        
        agent = create_agent(args.agent)
        
        print("\n" + "="*60)
        print(f"CURRICULUM EVALUATION: {agent.name}")
        print("="*60)
        
        results = evaluator.evaluate_with_curriculum(
            agent=agent,
            num_episodes=args.episodes,
            verbose=verbose,
        )
        
        # Print summary
        print("\n" + "="*60)
        print("CURRICULUM SUMMARY")
        print("="*60)
        print(f"Final Level: {results['final_level']}")
        print(f"Episodes: {results['num_episodes']}")
        
        # Save curriculum results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(args.output_dir) / f"curriculum_{args.agent}_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
    else:
        # Single agent evaluation
        if not args.agent:
            print("Error: --agent or --all-baselines required")
            parser.print_help()
            sys.exit(1)
        
        agent = create_agent(args.agent)
        
        print("\n" + "="*60)
        print(f"EVALUATING: {agent.name}")
        print("="*60)
        
        results = evaluator.evaluate_agent(
            agent=agent,
            scenarios=scenarios,
            num_episodes=args.episodes,
            verbose=verbose,
        )
        
        # Print summary
        for scenario, agg in results.items():
            print(f"\n{scenario.upper()}:")
            print(f"  Mean Reward: {agg.mean_reward:.2f} ± {agg.std_reward:.2f}")
            print(f"  Carbon Savings: {agg.mean_carbon_savings_pct:.1f}%")
            print(f"  Cost Savings: {agg.mean_cost_savings_pct:.1f}%")
            print(f"  SLA Violations: {agg.mean_sla_violations:.1f}")
            print(f"  Grader Score: {agg.mean_grader_score:.2f}")
            print(f"  95% CI: [{agg.reward_ci_lower:.2f}, {agg.reward_ci_upper:.2f}]")
    
    # Save all episode results
    evaluator.save_results()
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ko2cube Training Script

This script trains an LLM agent to optimize carbon-aware Kubernetes job scheduling
using Group Relative Policy Optimization (GRPO) from the TRL library.

The training process:
1. Loads a pre-trained LLM (e.g., Qwen-2.5-3B-Instruct)
2. Generates rollouts in the Ko2cube environment
3. Uses the deterministic grader score as the reward signal
4. Optimizes the policy using GRPO with curriculum learning

Usage:
    # Basic training
    python train.py --model Qwen/Qwen2.5-3B-Instruct --episodes 100
    
    # Training with curriculum
    python train.py --model Qwen/Qwen2.5-3B-Instruct --curriculum --episodes 200
    
    # Resume from checkpoint
    python train.py --model Qwen/Qwen2.5-3B-Instruct --resume checkpoints/latest
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import Ko2cubeEnvironment
from server.curriculum import CurriculumManager, DifficultyLevel
from server.data.scenarios import SCENARIO_REGISTRY
from server.rewards import compute_grader_score
from models import Ko2cubeObservation, Ko2cubeAction, JobAssignment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    output_dir: str = "checkpoints"
    
    # Training hyperparameters
    num_episodes: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_steps_per_episode: int = 100
    gradient_accumulation_steps: int = 4
    
    # GRPO specific
    num_generations: int = 4  # Number of completions per prompt for GRPO
    temperature: float = 0.7
    max_new_tokens: int = 512
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Curriculum settings
    use_curriculum: bool = False
    curriculum_threshold: float = 0.7
    
    # Checkpointing
    save_steps: int = 50
    eval_steps: int = 25
    
    # Device settings
    device: str = "auto"
    bf16: bool = True
    
    # Logging
    log_with: str = "tensorboard"
    seed: int = 42


@dataclass
class RolloutData:
    """Data from a single rollout."""
    prompt: str
    completion: str
    reward: float
    grader_score: float
    carbon_savings: float
    cost_savings: float
    steps: int
    task_id: str


def format_observation_prompt(obs: Ko2cubeObservation, task_context: str = "") -> str:
    """
    Format an observation into a prompt for the LLM.
    
    The prompt includes:
    - Task context and constraints
    - Current job queue with requirements
    - Region information with carbon/cost data
    - Active jobs status
    """
    prompt_parts = [
        "You are a carbon-aware Kubernetes scheduler. Your goal is to minimize carbon emissions while meeting SLAs and managing costs.",
        "",
        f"=== Current State (Step {obs.current_step}) ===",
        "",
    ]
    
    # Add job queue
    prompt_parts.append("## Jobs Waiting to be Scheduled:")
    if obs.job_queue:
        for job in obs.job_queue[:10]:  # Limit to 10 jobs for context length
            sla_info = f"SLA: steps {job.sla_start}-{job.sla_end}" if job.sla_end != -1 else "Always-on (no deadline)"
            prompt_parts.append(
                f"- {job.job_id}: {job.cpu_cores} cores, {job.memory_gb}GB RAM, "
                f"ETA: {job.eta_minutes or 'continuous'}min, {sla_info}, "
                f"{'delay-tolerant' if job.delay_tolerant else 'time-sensitive'}, "
                f"prefers {job.instance_preference}"
            )
    else:
        prompt_parts.append("(No jobs in queue)")
    
    prompt_parts.append("")
    
    # Add region information
    prompt_parts.append("## Available Regions:")
    for region_name, region in obs.regions.items():
        carbon = region.carbon.current_intensity
        forecast_trend = ""
        if region.carbon.forecast:
            avg_future = sum(region.carbon.forecast[:3]) / min(3, len(region.carbon.forecast))
            if avg_future < carbon * 0.9:
                forecast_trend = " (improving)"
            elif avg_future > carbon * 1.1:
                forecast_trend = " (worsening)"
        
        prompt_parts.append(f"- {region_name}: Carbon={carbon:.0f} gCO2/kWh{forecast_trend}")
        for inst in region.available_instances[:3]:  # Limit instances shown
            prompt_parts.append(
                f"    {inst.name}: {inst.cpu_cores} cores, {inst.memory_gb}GB, "
                f"spot=${inst.spot_price:.3f}/hr, on-demand=${inst.on_demand_price:.3f}/hr, "
                f"{inst.available_count} available"
            )
    
    prompt_parts.append("")
    
    # Add active jobs
    if obs.active_jobs:
        prompt_parts.append("## Currently Running Jobs:")
        for aj in obs.active_jobs[:5]:
            prompt_parts.append(f"- {aj.job_id} in {aj.region}: {aj.steps_remaining} steps remaining")
        prompt_parts.append("")
    
    # Instructions
    prompt_parts.extend([
        "## Your Task:",
        "For each job in the queue, decide:",
        "1. 'schedule' - Run it now (specify region and instance type)",
        "2. 'defer' - Wait for better conditions (specify defer_to_step within SLA)",
        "3. 'drop' - Abandon the job (SLA violation, use sparingly)",
        "",
        "Respond in JSON format with 'assignments' array:",
        '{"assignments": [{"job_id": "...", "decision": "schedule|defer|drop", "region": "...", "machine_type": "spot|on-demand", "instance_type": "...", "defer_to_step": N}]}',
        "",
        "Optimize for: LOW CARBON > SLA compliance > Cost efficiency",
    ])
    
    return "\n".join(prompt_parts)


def parse_llm_response(response: str, obs: Ko2cubeObservation) -> Optional[Ko2cubeAction]:
    """
    Parse LLM response into a Ko2cubeAction.
    
    Returns None if parsing fails.
    """
    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[^{}]*"assignments"[^{}]*\[.*?\][^{}]*\}', response, re.DOTALL)
        if not json_match:
            # Try to find any JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if not json_match:
            return None
        
        data = json.loads(json_match.group())
        
        if "assignments" not in data:
            return None
        
        assignments = []
        job_ids_in_queue = {job.job_id for job in obs.job_queue}
        
        for item in data["assignments"]:
            job_id = item.get("job_id")
            if job_id not in job_ids_in_queue:
                continue
            
            decision = item.get("decision", "defer")
            if decision not in ["schedule", "defer", "drop"]:
                decision = "defer"
            
            assignment = JobAssignment(
                job_id=job_id,
                decision=decision,
                region=item.get("region"),
                machine_type=item.get("machine_type"),
                instance_type=item.get("instance_type"),
                defer_to_step=item.get("defer_to_step"),
            )
            assignments.append(assignment)
        
        # Add default assignments for any jobs not mentioned
        mentioned_jobs = {a.job_id for a in assignments}
        for job in obs.job_queue:
            if job.job_id not in mentioned_jobs:
                assignments.append(JobAssignment(
                    job_id=job.job_id,
                    decision="defer",
                    defer_to_step=obs.current_step + 1 if job.sla_end > obs.current_step + 1 else job.sla_end,
                ))
        
        return Ko2cubeAction(assignments=assignments)
    
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug(f"Failed to parse LLM response: {e}")
        return None


def create_default_action(obs: Ko2cubeObservation) -> Ko2cubeAction:
    """Create a safe default action (defer all jobs)."""
    assignments = []
    for job in obs.job_queue:
        if job.sla_end > obs.current_step + 1:
            defer_to = obs.current_step + 1
        else:
            defer_to = job.sla_end
        
        assignments.append(JobAssignment(
            job_id=job.job_id,
            decision="defer",
            defer_to_step=defer_to,
        ))
    return Ko2cubeAction(assignments=assignments)


class Ko2cubeTrainer:
    """
    Trainer for Ko2cube using GRPO.
    
    This trainer implements the core training loop:
    1. Generate prompts from environment observations
    2. Sample multiple completions from the LLM
    3. Execute actions and collect rewards
    4. Update policy using GRPO
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.env = Ko2cubeEnvironment()
        self.curriculum = CurriculumManager() if config.use_curriculum else None
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        
        # Metrics tracking
        self.training_history: List[Dict[str, Any]] = []
        
        # Model and tokenizer (initialized lazily)
        self._model = None
        self._tokenizer = None
        self._trainer = None
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._init_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._init_model()
        return self._tokenizer
    
    def _init_model(self):
        """Initialize model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            import torch
            
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float16,
            }
            
            if self.config.device == "auto":
                model_kwargs["device_map"] = "auto"
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Apply LoRA if configured
            if self.config.use_lora:
                logger.info("Applying LoRA configuration")
                lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                )
                self._model = get_peft_model(self._model, lora_config)
                self._model.print_trainable_parameters()
            
            logger.info("Model loaded successfully")
            
        except ImportError as e:
            logger.warning(f"Training dependencies not installed: {e}")
            logger.warning("Install with: pip install 'ko2cube[train]'")
            raise
    
    def generate_completion(self, prompt: str) -> str:
        """Generate a single completion from the LLM."""
        try:
            import torch
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            completion = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            
            return completion
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def run_episode(self, task_id: str) -> RolloutData:
        """
        Run a single episode and collect rollout data.
        
        Returns RolloutData with the full trajectory.
        """
        obs = self.env.reset(task_id=task_id)
        total_reward = 0.0
        steps = 0
        
        prompts = []
        completions = []
        
        while steps < self.config.max_steps_per_episode:
            # Generate prompt
            prompt = format_observation_prompt(obs, task_id)
            prompts.append(prompt)
            
            # Get LLM completion
            completion = self.generate_completion(prompt)
            completions.append(completion)
            
            # Parse action
            action = parse_llm_response(completion, obs)
            if action is None:
                action = create_default_action(obs)
            
            # Execute action
            result = self.env.step(action)
            total_reward += result.reward
            steps += 1
            
            if result.done:
                break
            
            obs = result
        
        # Calculate final metrics
        state = self.env.state
        grader_score = compute_grader_score(state)
        
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
        
        # Combine prompts and completions for GRPO
        full_prompt = prompts[0] if prompts else ""
        full_completion = completions[0] if completions else ""
        
        return RolloutData(
            prompt=full_prompt,
            completion=full_completion,
            reward=total_reward,
            grader_score=grader_score,
            carbon_savings=carbon_savings,
            cost_savings=cost_savings,
            steps=steps,
            task_id=task_id,
        )
    
    def train(self):
        """
        Main training loop using GRPO.
        
        The training process:
        1. For each episode:
           a. Get task from curriculum (or fixed difficulty)
           b. Run episode and collect rollout
           c. Record results and update curriculum
        2. Periodically:
           a. Update model using GRPO
           b. Evaluate on held-out scenarios
           c. Save checkpoints
        """
        logger.info("Starting training...")
        logger.info(f"Config: {asdict(self.config)}")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        rollouts = []
        
        for episode in range(self.config.num_episodes):
            # Get task
            if self.curriculum:
                task_id = self.curriculum.get_task()
            else:
                task_id = "easy"
            
            # Run episode
            rollout = self.run_episode(task_id)
            rollouts.append(rollout)
            
            # Update curriculum
            if self.curriculum:
                self.curriculum.record_result(
                    task_id=task_id,
                    reward=rollout.reward,
                    grader_score=rollout.grader_score,
                    carbon_savings=rollout.carbon_savings,
                )
            
            # Log progress
            self.episode_count += 1
            self.training_history.append({
                "episode": self.episode_count,
                "task_id": task_id,
                "reward": rollout.reward,
                "grader_score": rollout.grader_score,
                "carbon_savings": rollout.carbon_savings,
                "cost_savings": rollout.cost_savings,
                "steps": rollout.steps,
                "curriculum_level": self.curriculum.tracker.current_level.value if self.curriculum else "N/A",
            })
            
            if episode % 10 == 0:
                recent_rewards = [h["reward"] for h in self.training_history[-10:]]
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                logger.info(
                    f"Episode {episode}/{self.config.num_episodes}: "
                    f"task={task_id}, reward={rollout.reward:.2f}, "
                    f"grader={rollout.grader_score:.2f}, "
                    f"carbon_savings={rollout.carbon_savings:.1f}%, "
                    f"avg_reward(10)={avg_reward:.2f}"
                )
            
            # Periodic model update
            if len(rollouts) >= self.config.batch_size:
                self._update_model(rollouts)
                rollouts = []
            
            # Periodic evaluation
            if (episode + 1) % self.config.eval_steps == 0:
                self._run_evaluation()
            
            # Periodic checkpoint
            if (episode + 1) % self.config.save_steps == 0:
                self._save_checkpoint(output_dir / f"checkpoint-{episode+1}")
        
        # Final save
        self._save_checkpoint(output_dir / "final")
        self._save_training_history(output_dir / "training_history.json")
        
        logger.info("Training complete!")
        return self.training_history
    
    def _update_model(self, rollouts: List[RolloutData]):
        """
        Update model using GRPO.
        
        GRPO (Group Relative Policy Optimization):
        1. Group rollouts by prompt
        2. Compute relative rewards within each group
        3. Use clipped objective similar to PPO
        """
        try:
            from trl import GRPOConfig, GRPOTrainer
            from datasets import Dataset
            
            # Prepare dataset for GRPO
            data = {
                "prompt": [r.prompt for r in rollouts],
                "completion": [r.completion for r in rollouts],
                "reward": [r.grader_score for r in rollouts],  # Use grader score as reward
            }
            dataset = Dataset.from_dict(data)
            
            if self._trainer is None:
                # Initialize GRPO trainer
                grpo_config = GRPOConfig(
                    output_dir=self.config.output_dir,
                    per_device_train_batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    num_generations=self.config.num_generations,
                    temperature=self.config.temperature,
                    max_new_tokens=self.config.max_new_tokens,
                    logging_steps=10,
                    bf16=self.config.bf16,
                )
                
                self._trainer = GRPOTrainer(
                    model=self.model,
                    config=grpo_config,
                    train_dataset=dataset,
                    processing_class=self.tokenizer,
                )
            
            # Run training step
            self._trainer.train()
            self.global_step += 1
            
        except ImportError:
            logger.warning("TRL not installed, skipping model update")
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def _run_evaluation(self):
        """Run evaluation on held-out scenarios."""
        logger.info("Running evaluation...")
        
        eval_results = {}
        for scenario in ["easy", "medium"]:
            rollout = self.run_episode(scenario)
            eval_results[scenario] = {
                "reward": rollout.reward,
                "grader_score": rollout.grader_score,
                "carbon_savings": rollout.carbon_savings,
            }
            logger.info(
                f"Eval {scenario}: reward={rollout.reward:.2f}, "
                f"grader={rollout.grader_score:.2f}"
            )
        
        # Track best model
        avg_grader = sum(r["grader_score"] for r in eval_results.values()) / len(eval_results)
        if avg_grader > self.best_reward:
            self.best_reward = avg_grader
            logger.info(f"New best model: grader_score={avg_grader:.2f}")
    
    def _save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        
        try:
            if self._model is not None:
                self._model.save_pretrained(path)
                self._tokenizer.save_pretrained(path)
                logger.info(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "best_reward": self.best_reward,
        }
        with open(path / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
    
    def _save_training_history(self, path: Path):
        """Save training history to file."""
        with open(path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {path}")


def train_without_llm(config: TrainingConfig) -> List[Dict]:
    """
    Lightweight training loop without LLM dependencies.
    
    Uses baseline agents for collecting trajectories, useful for:
    - Testing the training pipeline
    - Debugging environment interactions
    - Establishing baseline metrics
    """
    from server.baselines import create_agent, run_episode
    
    logger.info("Running lightweight training (no LLM)...")
    
    env = Ko2cubeEnvironment()
    curriculum = CurriculumManager() if config.use_curriculum else None
    
    # Use hybrid agent as a stand-in for LLM behavior
    agent = create_agent("hybrid")
    
    history = []
    
    for episode in range(config.num_episodes):
        if curriculum:
            task_id = curriculum.get_task()
        else:
            task_id = "easy"
        
        metrics = run_episode(env, agent, task_id=task_id, verbose=False)
        
        state = env.state
        grader_score = compute_grader_score(state)
        
        carbon_savings = 0.0
        if state.baseline_carbon_gco2 > 0:
            carbon_savings = (
                (state.baseline_carbon_gco2 - state.total_carbon_gco2)
                / state.baseline_carbon_gco2 * 100
            )
        
        if curriculum:
            curriculum.record_result(
                task_id=task_id,
                reward=metrics.total_reward,
                grader_score=grader_score,
                carbon_savings=carbon_savings,
            )
        
        history.append({
            "episode": episode + 1,
            "task_id": task_id,
            "reward": metrics.total_reward,
            "grader_score": grader_score,
            "carbon_savings": carbon_savings,
            "steps": metrics.steps,
            "curriculum_level": curriculum.tracker.current_level.value if curriculum else "N/A",
        })
        
        if (episode + 1) % 10 == 0:
            recent_rewards = [h["reward"] for h in history[-10:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            logger.info(
                f"Episode {episode+1}/{config.num_episodes}: "
                f"avg_reward(10)={avg_reward:.2f}, "
                f"grader={grader_score:.2f}"
            )
    
    return history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train an LLM agent for carbon-aware scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training (requires train dependencies)
  python train.py --model Qwen/Qwen2.5-3B-Instruct --episodes 100
  
  # Lightweight training without LLM (for testing)
  python train.py --no-llm --episodes 50 --curriculum
  
  # Full training with curriculum
  python train.py --model Qwen/Qwen2.5-3B-Instruct --curriculum --episodes 500
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without LLM (use baseline agent)",
    )
    
    # Training arguments
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    
    # Curriculum arguments
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use curriculum learning",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint path",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create config
    config = TrainingConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_curriculum=args.curriculum,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        seed=args.seed,
    )
    
    if args.no_llm:
        # Lightweight training without LLM
        history = train_without_llm(config)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to {output_dir / 'training_history.json'}")
    else:
        # Full training with LLM
        trainer = Ko2cubeTrainer(config)
        
        if args.resume:
            logger.info(f"Resuming from {args.resume}")
            # Load checkpoint would go here
        
        trainer.train()
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

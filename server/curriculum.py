"""
Curriculum learning system for Ko2cube environment.

Implements progressive difficulty escalation based on agent mastery,
allowing gradual skill acquisition from simple to complex scenarios.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import random


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum progression."""
    WARMUP = "warmup"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# Difficulty progression order
DIFFICULTY_PROGRESSION: List[DifficultyLevel] = [
    DifficultyLevel.WARMUP,
    DifficultyLevel.BEGINNER,
    DifficultyLevel.INTERMEDIATE,
    DifficultyLevel.ADVANCED,
    DifficultyLevel.EXPERT,
]

# Mapping from curriculum levels to scenario task IDs
LEVEL_TO_SCENARIO: Dict[DifficultyLevel, str] = {
    DifficultyLevel.WARMUP: "easy",
    DifficultyLevel.BEGINNER: "easy",
    DifficultyLevel.INTERMEDIATE: "medium",
    DifficultyLevel.ADVANCED: "medium",
    DifficultyLevel.EXPERT: "hard",
}

# Thresholds for mastery at each level
MASTERY_THRESHOLDS: Dict[DifficultyLevel, float] = {
    DifficultyLevel.WARMUP: 0.5,
    DifficultyLevel.BEGINNER: 0.55,
    DifficultyLevel.INTERMEDIATE: 0.6,
    DifficultyLevel.ADVANCED: 0.7,
    DifficultyLevel.EXPERT: 0.8,
}


@dataclass
class EpisodeResult:
    """Results from a completed episode."""
    grader_score: float
    carbon_saved_gco2: float
    sla_compliance_rate: float
    jobs_by_type: Dict[str, int]
    successful_jobs_by_type: Dict[str, int]
    region_carbon_efficiency: Dict[str, float]


@dataclass
class MasteryTracker:
    """
    Tracks agent mastery across job types and regions.
    
    Used to determine when to escalate difficulty and to identify
    areas where the agent needs more practice.
    """
    job_type_success: Dict[str, float] = field(default_factory=dict)
    region_efficiency: Dict[str, float] = field(default_factory=dict)
    current_level: int = 0
    episodes_at_level: int = 0
    total_episodes: int = 0
    level_scores: List[float] = field(default_factory=list)
    
    consecutive_mastery: int = 0
    required_consecutive: int = 3
    
    def get_current_difficulty(self) -> DifficultyLevel:
        """Get the current difficulty level."""
        if self.current_level >= len(DIFFICULTY_PROGRESSION):
            return DIFFICULTY_PROGRESSION[-1]
        return DIFFICULTY_PROGRESSION[self.current_level]
    
    def get_scenario_task_id(self) -> str:
        """Get the task ID for the current curriculum level."""
        level = self.get_current_difficulty()
        return LEVEL_TO_SCENARIO[level]
    
    def get_mastery_threshold(self) -> float:
        """Get the mastery threshold for current level."""
        level = self.get_current_difficulty()
        return MASTERY_THRESHOLDS[level]
    
    def update_mastery(self, result: EpisodeResult) -> None:
        """
        Update mastery metrics after episode completion.
        
        Args:
            result: Results from the completed episode
        """
        self.total_episodes += 1
        self.episodes_at_level += 1
        self.level_scores.append(result.grader_score)
        
        for job_type, count in result.jobs_by_type.items():
            if count > 0:
                success_count = result.successful_jobs_by_type.get(job_type, 0)
                success_rate = success_count / count
                
                if job_type in self.job_type_success:
                    alpha = 0.3
                    self.job_type_success[job_type] = (
                        alpha * success_rate +
                        (1 - alpha) * self.job_type_success[job_type]
                    )
                else:
                    self.job_type_success[job_type] = success_rate
        
        for region, efficiency in result.region_carbon_efficiency.items():
            if region in self.region_efficiency:
                alpha = 0.3
                self.region_efficiency[region] = (
                    alpha * efficiency +
                    (1 - alpha) * self.region_efficiency[region]
                )
            else:
                self.region_efficiency[region] = efficiency
        
        if result.grader_score >= self.get_mastery_threshold():
            self.consecutive_mastery += 1
        else:
            self.consecutive_mastery = 0
    
    def should_escalate(self) -> bool:
        """
        Check if agent has mastered current level and should move up.
        
        Returns:
            True if agent should progress to next difficulty level
        """
        if self.current_level >= len(DIFFICULTY_PROGRESSION) - 1:
            return False
        
        if self.episodes_at_level < 3:
            return False
        
        if self.consecutive_mastery >= self.required_consecutive:
            return True
        
        if self.episodes_at_level >= 10:
            recent_scores = self.level_scores[-5:]
            avg_score = sum(recent_scores) / len(recent_scores)
            if avg_score >= self.get_mastery_threshold():
                return True
        
        return False
    
    def escalate(self) -> DifficultyLevel:
        """
        Move to the next difficulty level.
        
        Returns:
            The new difficulty level
        """
        if self.current_level < len(DIFFICULTY_PROGRESSION) - 1:
            self.current_level += 1
            self.episodes_at_level = 0
            self.level_scores = []
            self.consecutive_mastery = 0
        
        return self.get_current_difficulty()
    
    def get_weakest_job_types(self, n: int = 3) -> List[str]:
        """
        Get job types where the agent struggles most.
        
        Args:
            n: Number of job types to return
            
        Returns:
            List of job type names with lowest success rates
        """
        if not self.job_type_success:
            return []
        
        sorted_types = sorted(
            self.job_type_success.items(),
            key=lambda x: x[1]
        )
        return [t[0] for t in sorted_types[:n]]
    
    def get_best_regions(self, n: int = 2) -> List[str]:
        """
        Get regions where the agent performs best.
        
        Args:
            n: Number of regions to return
            
        Returns:
            List of region names with highest efficiency
        """
        if not self.region_efficiency:
            return []
        
        sorted_regions = sorted(
            self.region_efficiency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [r[0] for r in sorted_regions[:n]]
    
    def get_stats(self) -> Dict:
        """Get summary statistics for debugging/logging."""
        return {
            "current_level": self.get_current_difficulty().value,
            "episodes_at_level": self.episodes_at_level,
            "total_episodes": self.total_episodes,
            "consecutive_mastery": self.consecutive_mastery,
            "mastery_threshold": self.get_mastery_threshold(),
            "job_type_success": dict(self.job_type_success),
            "region_efficiency": dict(self.region_efficiency),
            "recent_avg_score": (
                sum(self.level_scores[-5:]) / len(self.level_scores[-5:])
                if self.level_scores else 0.0
            ),
        }


class CurriculumManager:
    """
    Manages curriculum progression for training.
    
    Coordinates between the MasteryTracker and environment
    to provide appropriate difficulty scenarios.
    """
    
    def __init__(self, start_level: DifficultyLevel = DifficultyLevel.WARMUP):
        """
        Initialize the curriculum manager.
        
        Args:
            start_level: Initial difficulty level
        """
        self.tracker = MasteryTracker()
        self.tracker.current_level = DIFFICULTY_PROGRESSION.index(start_level)
    
    def get_next_task(self) -> str:
        """
        Get the task ID for the next episode.
        
        Checks if escalation should happen and returns appropriate task.
        
        Returns:
            Task ID string for environment reset
        """
        if self.tracker.should_escalate():
            self.tracker.escalate()
        
        return self.tracker.get_scenario_task_id()
    
    def record_episode(self, result: EpisodeResult) -> None:
        """
        Record episode results and update mastery.
        
        Args:
            result: Results from completed episode
        """
        self.tracker.update_mastery(result)
    
    def get_progress_report(self) -> Dict:
        """Get detailed progress report."""
        return {
            "tracker_stats": self.tracker.get_stats(),
            "weakest_job_types": self.tracker.get_weakest_job_types(),
            "best_regions": self.tracker.get_best_regions(),
            "should_escalate": self.tracker.should_escalate(),
        }
    
    def set_level(self, level: DifficultyLevel) -> None:
        """
        Manually set the difficulty level.
        
        Args:
            level: Difficulty level to set
        """
        self.tracker.current_level = DIFFICULTY_PROGRESSION.index(level)
        self.tracker.episodes_at_level = 0
        self.tracker.level_scores = []
        self.tracker.consecutive_mastery = 0

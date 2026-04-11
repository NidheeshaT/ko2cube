# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Ko2cube Env Environment Client."""

from typing import Dict, List, Optional
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    Ko2cubeAction, Ko2cubeObservation, Ko2cubeState,
    Job, RunningJob, RegionInfo
)

class Ko2cubeEnv(EnvClient[Ko2cubeAction, Ko2cubeObservation, Ko2cubeState]):
    """
    Client for the Ko2cube Carbon-Aware Scheduling Environment.
    """

    def _step_payload(self, action: Ko2cubeAction) -> Dict:
        """Convert Ko2cubeAction to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[Ko2cubeObservation]:
        """Parse server response into StepResult[Ko2cubeObservation]."""
        obs_data = payload.get("observation", {})
        
        # Reconstruct the observation from the raw dict
        observation = Ko2cubeObservation.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> Ko2cubeState:
        """Parse server response into Ko2cubeState object."""
        return Ko2cubeState.model_validate(payload)

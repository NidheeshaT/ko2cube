# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Ko2cube Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import Ko2cubeAction, Ko2cubeObservation, Ko2cubeState, Job, RegionInfo, CarbonData, PriceData

class Ko2cubeEnvironment(Environment):
    """
    Ko2cube: A carbon-aware cloud job scheduler environment.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = Ko2cubeEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "Ko2cube Env environment ready!"
        >>>
        >>> obs = env.step(Ko2cubeAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the ko2cube_env environment."""
        self._state = Ko2cubeState(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> Ko2cubeObservation:
        """
        Reset the environment.
        """
        self._state = Ko2cubeState(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return Ko2cubeObservation(
            current_step=0,
            job_queue=[],
            active_jobs=[],
            regions={},
            done=False,
            reward=0.0,
            metadata={"msg": "Environment reset"}
        )

    def step(self, action: Ko2cubeAction) -> Ko2cubeObservation:  # type: ignore[override]
        """
        Execute a scheduling step in the environment.
        """
        self._state.step_count += 1

        return Ko2cubeObservation(
            current_step=self._state.step_count,
            job_queue=[],
            active_jobs=[],
            regions={},
            done=False,
            reward=0.0,
            metadata={"msg": "Step executed"},
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

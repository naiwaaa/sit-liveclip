from __future__ import annotations

from typing import TYPE_CHECKING

import copy
import random

import gym
from gym import spaces

from sit_liveclip.core.action import Action


if TYPE_CHECKING:
    from typing import Any

    from sit_liveclip.core import Observation, PlayerConnection


class BaseLiveClipEnv(gym.Env):
    # pylint: disable=abstract-method
    def __init__(
        self,
        history_size: int,
        sliding_window_size: int,
        segment_size: int,
    ) -> None:
        self.segment_size = segment_size
        self.state: Observation | None = None

        self.observation_space = spaces.Dict(
            {
                "download_speed": spaces.Box(
                    0,
                    17000,
                    shape=(1, history_size),
                    dtype="float32",
                ),
                "user_staying_time": spaces.Box(
                    0,
                    120,
                    shape=(1, history_size),
                    dtype="float32",
                ),
                #
                # current playback state of the player
                "play_progress": spaces.Box(0, 60, shape=(1,), dtype="float32"),
                "current_staying_time": spaces.Box(0, 120, shape=(1,), dtype="float32"),
                "replay_round": spaces.Box(0, 0, shape=(1,), dtype="float32"),
                #
                # videos in the sliding window
                "list_video_bitrate": spaces.Box(
                    1000,
                    1000,
                    shape=(sliding_window_size,),
                    dtype="float32",
                ),
                "list_video_length": spaces.Box(
                    60,
                    60,
                    shape=(sliding_window_size,),
                    dtype="float32",
                ),
                "list_buffered_content": spaces.Box(
                    0,
                    60,
                    shape=(sliding_window_size,),
                    dtype="float32",
                ),
                "list_time_spent_downloading_videos": spaces.Box(
                    0,
                    120,
                    shape=(sliding_window_size,),
                    dtype="float32",
                ),
                "list_completed_videos": spaces.Box(
                    0,
                    1,
                    shape=(sliding_window_size,),
                    dtype="float32",
                ),
            },
        )
        self.action_space = spaces.Discrete(4)


class DummyLiveClipEnv(BaseLiveClipEnv):
    # pylint: disable=abstract-method
    def reset(self) -> Observation:
        obs: Observation = self.observation_space.sample()
        return obs

    def step(self, _: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        return (
            self.observation_space.sample(),
            -10 * random.random(),
            bool(round(random.random())),
            {},
        )


class LiveClipEnv(BaseLiveClipEnv):
    # pylint: disable=abstract-method
    def __init__(
        self,
        player_conn: PlayerConnection,
        history_size: int,
        sliding_window_size: int,
        segment_size: int,
    ) -> None:
        super().__init__(history_size, sliding_window_size, segment_size)
        self.player_conn = player_conn

    def reset(self) -> Observation:
        self.player_conn.post_action(Action.RESET)
        self.state, *_ = self.player_conn.get_state()
        return copy.deepcopy(self.state)

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        assert self.action_space.contains(int(action))
        assert self.state is not None, "Call reset before using step method."

        self.player_conn.post_action(action)
        self.state, done, wastage_cost = self.player_conn.get_state()
        reward = self._compute_reward(self.state, wastage_cost)

        return copy.deepcopy(self.state), reward, done, {}

    def _compute_reward(self, obs: Observation, wastage_cost: float) -> float:
        current_video = 0

        # QoE
        no_rebuffering_event = (
            obs["list_buffered_content"][current_video] - obs["play_progress"]
            >= self.segment_size
            or obs["list_buffered_content"][current_video]
            == obs["list_video_length"][current_video]
        )
        penalty_qoe = 0 if no_rebuffering_event else 1

        # Wastage cost
        penalty_wastage = wastage_cost

        # total penalty
        total_penalty = -penalty_qoe + penalty_wastage

        return -total_penalty

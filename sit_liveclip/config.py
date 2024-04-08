from __future__ import annotations

from typing import NamedTuple


class Config(NamedTuple):
    player_conn_base_url: str = "http://localhost:1996"
    history_size: int = 8
    sliding_window_size: int = 3
    segment_size: int = 2
    conv1d_out: int = 4
    conv1d_kernel_size: int = 3
    actor_hidden_size: list[int] = [128, 128]
    critic_hidden_size: list[int] = [128, 128]
    learning_rate: float = 0.0001
    reward_threshold: float = 0.1

    max_epoch: int = 100
    step_per_epoch: int | None = 1000
    episode_per_collect: int | None = None
    episode_per_test: int = 1
    repeat_per_collect: int = 5
    batch_size: int = 256
    step_per_collect: int = 2000

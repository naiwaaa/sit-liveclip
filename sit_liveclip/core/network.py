from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn


if TYPE_CHECKING:
    from typing import Any, Literal


class PreprocessNet(nn.Module):
    def __init__(
        self,
        history_size: int,
        sliding_window_size: int,
        conv1d_out: int,
        conv1d_kernel_size: int,
        device: Literal["cpu", "cuda"],
    ):
        super().__init__()
        self.device = device
        self.output_dim = (
            2 * conv1d_out * (history_size - conv1d_kernel_size + 1)
            + 3
            + sliding_window_size
        )

        self.download_speed_conv = nn.Sequential(
            nn.utils.weight_norm(  # type:ignore
                nn.Conv1d(
                    in_channels=1,
                    out_channels=conv1d_out,
                    kernel_size=conv1d_kernel_size,
                ),
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.user_behavior_conv = nn.Sequential(
            nn.utils.weight_norm(  # type:ignore
                nn.Conv1d(
                    in_channels=1,
                    out_channels=conv1d_out,
                    kernel_size=conv1d_kernel_size,
                ),
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(
        self,
        obs: Any,
        state: Any,
        _: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:

        # historical information
        obs_download_speed = torch.as_tensor(  # noqa
            obs.download_speed,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )
        obs_user_staying_time = torch.as_tensor(  # noqa
            obs.user_staying_time,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )

        batch_size = obs_download_speed.shape[0]

        download_speed_feat = self.download_speed_conv(obs_download_speed).reshape(
            batch_size,
            -1,
        )
        user_behavior_feat = self.user_behavior_conv(obs_user_staying_time).reshape(
            batch_size,
            -1,
        )

        # current playback state
        obs_play_progress = torch.as_tensor(  # noqa
            obs.play_progress,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )
        obs_current_stay_time = torch.as_tensor(  # noqa
            obs.current_staying_time,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )
        obs_current_video_length = torch.as_tensor(  # noqa
            obs.list_video_length[:, 0].reshape(-1, 1),
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )

        # videos in the sliding window
        obs_buffered_content = torch.as_tensor(  # noqa
            obs.list_buffered_content,
            device=self.device,  # type: ignore
            dtype=torch.float32,
        )

        return (
            torch.cat(
                (
                    download_speed_feat,
                    user_behavior_feat,
                    obs_play_progress,
                    obs_current_stay_time,
                    obs_current_video_length,
                    obs_buffered_content,
                ),
                1,
            ),
            state,
        )

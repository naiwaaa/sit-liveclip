from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict


if TYPE_CHECKING:
    from sit_liveclip.types import NDArrayFloat32


class Observation(TypedDict):
    download_speed: NDArrayFloat32  # past k seconds, Continuous([0, 17000])
    user_staying_time: NDArrayFloat32  # past k videos, Continuous([0, 120])

    # current playback state of the player
    play_progress: NDArrayFloat32  # seconds, Continuous([0, 60])
    current_staying_time: NDArrayFloat32  # seconds, Continuous([0, 120])
    replay_round: NDArrayFloat32  # how many times?, Constant(0)

    # videos in the sliding window
    list_video_bitrate: NDArrayFloat32  # Constant(1000)
    list_video_length: NDArrayFloat32  # seconds, Constant(60)
    list_buffered_content: NDArrayFloat32  # seconds, Continuous([0, 60])
    list_time_spent_downloading_videos: NDArrayFloat32  # ???
    list_completed_videos: NDArrayFloat32  # ???

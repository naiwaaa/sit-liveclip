from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from urllib.parse import urljoin

import numpy as np
import requests
from requests.adapters import Retry
from requests.sessions import HTTPAdapter

from sit_liveclip.core.observation import Observation


if TYPE_CHECKING:
    from typing import Any, Literal

    from sit_liveclip.core.action import Action


class PlayerResponse(NamedTuple):
    obs: Observation
    done: bool
    wastage_cost: float


class PlayerConnection:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        adapter = HTTPAdapter(
            max_retries=Retry(
                total=50,
                backoff_factor=0.1,
                status_forcelist=[429, 500, 502, 503, 504, 404],
            ),
        )
        self.session.mount("http://", adapter)

        self.timeout = 10 * 60

    def close(self) -> None:
        self.session.close()

    def get_state(self) -> PlayerResponse:
        res = self._do_request("get", "/state")
        data = res.json()

        done = data["status"]
        wastage_cost = data["wastage_cost"]
        obs = Observation(
            download_speed=np.asarray(
                data["download_speed"],
                dtype="float32",
            ).reshape(1, -1),
            user_staying_time=np.asarray(
                data["user_staying_time"],
                dtype="float32",
            ).reshape(1, -1),
            #
            # current playback state of the player
            play_progress=np.asarray(
                [data["play_progress"]],
                dtype="float32",
            ),
            current_staying_time=np.asarray(
                [data["current_staying_time"]],
                dtype="float32",
            ),
            replay_round=np.asarray(
                [data["replay_round"]],
                dtype="float32",
            ),
            #
            # videos in the sliding window
            list_video_bitrate=np.asarray(
                data["list_video_bitrate"],
                dtype="float32",
            ),
            list_video_length=np.asarray(
                data["list_video_length"],
                dtype="float32",
            ),
            list_buffered_content=np.asarray(
                data["list_buffered_content"],
                dtype="float32",
            ),
            list_time_spent_downloading_videos=np.asarray(
                data["list_time_spent_downloading_videos"],
                dtype="float32",
            ),
            list_completed_videos=np.asarray(
                data["list_completed_videos"],
                dtype="float32",
            ),
        )

        return PlayerResponse(obs=obs, done=done, wastage_cost=wastage_cost)

    def post_action(self, action: Action) -> None:
        player_action = int(action) - 1
        self._do_request("post", "/action", data={"action": player_action})

    def _do_request(
        self,
        method: Literal["get", "post"],
        route: str,
        data: dict[str, Any] | None = None,
    ) -> requests.Response:
        url = urljoin(self.base_url, route)
        res = self.session.request(
            method,
            url,
            json=data,
            timeout=self.timeout,
        )
        res.raise_for_status()
        return res

#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Union

from flask import Flask, request


ResponseType = Union[dict[str, Any], tuple[Any, int]]

app = Flask(__name__)

action_queue: list[int] = []
state_queue: list[dict[str, Any]] = []


@app.route("/")
def index() -> str:
    return "Hello, World!"


@app.route("/state", methods=["POST", "GET"])
def handle_state_route() -> ResponseType:
    if request.method == "GET":
        if len(state_queue) == 0:
            return "", 404
        return state_queue.pop()

    state = _preprocess_player_state(request.json)
    state_queue.append(state)
    return {"len_state_queue": len(state_queue)}


@app.route("/action/<action>", methods=["POST", "GET"])
def handle_action_route(action: int) -> ResponseType:
    if request.method == "GET":
        if len(action_queue) == 0:
            return "", 404
        return {"action": action}

    action_queue.append(action)
    return {"action_queue": action_queue}


def _preprocess_player_state(state: Any) -> dict[str, Any]:
    return {"state": state}


if __name__ == "__main__":
    app.run(port=1234, debug=False)

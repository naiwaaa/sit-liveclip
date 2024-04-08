from __future__ import annotations

from typing import TYPE_CHECKING

import json
import pickle as pkl

import numpy as np


if TYPE_CHECKING:
    from typing import Any

    from pathlib import Path


def serialize(out_file: Path, data: Any) -> None:
    """Save data to a binary file in numpy `.npy`, pickle `.pkl`, or `.json` format."""
    file_format = out_file.suffix

    if file_format == ".npy":
        np.save(out_file, data)
    elif file_format == ".pkl":
        with open(out_file, "wb") as file:
            pkl.dump(data, file)
    elif file_format == ".json":
        out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        raise ValueError(
            f"`{file_format}` is not supported. Supported formats: .npy, .pkl, .json",
        )


def deserialize(input_file: Path) -> Any:
    """Load data from numpy `.npy`, pickle `.pkl`, or `.json` files."""
    file_format = input_file.suffix

    if file_format == ".npy":
        return np.load(input_file)

    if file_format == ".pkl":
        with open(input_file, "rb") as file:
            return pkl.load(file)

    if file_format == ".json":
        return json.loads(input_file.read_text())

    raise ValueError(
        f"`{file_format}` is not supported. Supported formats: .npy, .pkl, .json",
    )

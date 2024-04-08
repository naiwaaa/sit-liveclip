from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import TypeAlias

    import numpy as np
    from numpy.typing import NDArray

    NDArrayInt: TypeAlias = NDArray[np.int_]
    NDArrayUInt8: TypeAlias = NDArray[np.uint8]
    NDArrayInt32: TypeAlias = NDArray[np.int32]
    NDArrayInt64: TypeAlias = NDArray[np.int64]

    NDArrayFloat: TypeAlias = NDArray[np.float_]
    NDArrayFloat32: TypeAlias = NDArray[np.float32]
    NDArrayFloat64: TypeAlias = NDArray[np.float64]

    NDArrayStr: TypeAlias = NDArray[np.str_]

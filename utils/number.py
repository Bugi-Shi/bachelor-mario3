from __future__ import annotations

from typing import Any, Optional

import numpy as np


def to_python_int(value: Any) -> int:
    """Convert values like `np.int64` / 0-d arrays into a Python `int`.

    Many gym/retro envs put values into `info` as numpy scalars
    (or 0-d arrays).
    This helper makes comparisons and logging predictable.
    """

    try:
        return int(value)
    except Exception:
        return int(np.asarray(value).item())


def to_python_int_or_none(value: Any) -> Optional[int]:
    """Like `to_python_int`, but returns None when input is None."""

    if value is None:
        return None
    return to_python_int(value)

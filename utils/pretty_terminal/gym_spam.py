"""Gym banner suppression.

Upstream `gym` prints a large banner by importing `gym_notices.notices` and
reading `notices.notices[__version__]`.

We keep the mapping empty to suppress the spam.
"""

from __future__ import annotations

notices: dict[str, str] = {}

from __future__ import annotations

import signal


def ignore_sigint() -> None:
    """Best-effort ignore Ctrl+C (SIGINT) in the current process.

    Useful for subprocess workers: the main process should handle Ctrl+C and
    coordinate shutdown, while workers should exit quietly.
    """

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

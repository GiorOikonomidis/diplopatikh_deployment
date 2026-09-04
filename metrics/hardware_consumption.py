"""Hardware resource usage tracking (CPU, memory) for pipeline inference
jobs, backed by psutil.
"""

from __future__ import annotations

import time

import psutil


class HardwareMonitor:
    """Context manager that tracks CPU utilization and RSS memory usage
    of the code block it wraps.

    Usage:
        with HardwareMonitor() as monitor:
            ... run inference ...
        report = monitor.summary()
    """

    def __init__(self) -> None:
        """Set up the underlying psutil process handle.

        Input:
            None.
        Output:
            None.
        """
        self._process: psutil.Process = psutil.Process()
        self._rss_before_mb: float | None = None
        self._rss_after_mb: float | None = None
        self._cpu_percent: float | None = None
        self._t_start: float | None = None
        self._t_end: float | None = None

    def __enter__(self) -> "HardwareMonitor":
        """Start tracking: record the RSS baseline and prime psutil's
        internal CPU-usage counter.

        Input:
            None.
        Output:
            This HardwareMonitor instance, bound to the `as` target.
        """
        self._process.cpu_percent(interval=None)  # prime the internal counter
        self._rss_before_mb = self._process.memory_info().rss / 1e6
        self._t_start = time.perf_counter()
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_val: BaseException | None, exc_tb: object) -> None:
        """Stop tracking and record the memory/CPU/elapsed-time deltas.

        Input:
            exc_type, exc_val, exc_tb: standard context-manager exception
                info, unused (the block's exception, if any, still
                propagates normally).
        Output:
            None.
        """
        self._t_end = time.perf_counter()
        self._rss_after_mb = self._process.memory_info().rss / 1e6
        self._cpu_percent = self._process.cpu_percent(interval=None)

    def summary(self) -> dict[str, float]:
        """Return the measured hardware usage for the tracked block.

        Input:
            None.
        Output:
            Dict with keys "elapsed_s", "rss_before_mb", "rss_after_mb",
            "rss_delta_mb", "cpu_percent".
        """
        if self._t_end is None or self._t_start is None:
            raise RuntimeError(
                "HardwareMonitor.summary() called before the tracked block finished "
                "(the `with` block must exit first)")
        return {
            "elapsed_s": self._t_end - self._t_start,
            "rss_before_mb": self._rss_before_mb,
            "rss_after_mb": self._rss_after_mb,
            "rss_delta_mb": self._rss_after_mb - self._rss_before_mb,
            "cpu_percent": self._cpu_percent,
        }



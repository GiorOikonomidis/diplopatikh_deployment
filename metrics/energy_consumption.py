"""Energy consumption and CO2 emissions tracking for pipeline inference
jobs, backed by codecarbon's EmissionsTracker.
"""

from __future__ import annotations

from pathlib import Path

from codecarbon import EmissionsTracker


class EnergyMonitor:
    """Context manager that tracks energy consumption and CO2 emissions
    of the code block it wraps, via codecarbon.

    Usage:
        with EnergyMonitor(project_name="router_inference") as monitor:
            ... run inference ...
        report = monitor.summary()
    """

    def __init__(self, project_name: str, output_dir: str = "metrics/energy_logs",
                 measure_power_secs: int = 1) -> None:
        """Set up the underlying codecarbon EmissionsTracker.

        Input:
            project_name: name used to tag this run's emissions record.
            output_dir: directory codecarbon writes its emissions.csv log
                to (created if it doesn't exist).
            measure_power_secs: sampling interval, in seconds, codecarbon
                uses to poll hardware power draw.
        Output:
            None.
        """
        self.project_name: str = project_name
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tracker: EmissionsTracker = EmissionsTracker(
            project_name=project_name,
            output_dir=str(self.output_dir),
            measure_power_secs=measure_power_secs,
            log_level="error",
            save_to_file=True,
        )
        self._emissions_kg: float | None = None

    def __enter__(self) -> "EnergyMonitor":
        """Start tracking.

        Input:
            None.
        Output:
            This EnergyMonitor instance, bound to the `as` target.
        """
        self._tracker.start()
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_val: BaseException | None, exc_tb: object) -> None:
        """Stop tracking and record the measured emissions.

        Input:
            exc_type, exc_val, exc_tb: standard context-manager exception
                info, unused (the block's exception, if any, still
                propagates normally).
        Output:
            None.
        """
        self._emissions_kg = self._tracker.stop()

    def summary(self) -> dict[str, float]:
        """Return the measured energy/emissions for the tracked block.

        Input:
            None.
        Output:
            Dict with keys "emissions_kg_co2" and "energy_kwh".
        """
        if self._emissions_kg is None:
            raise RuntimeError(
                "EnergyMonitor.summary() called before the tracked block finished "
                "(the `with` block must exit first)")
        energy_kwh = self._tracker.final_emissions_data.energy_consumed
        return {"emissions_kg_co2": self._emissions_kg, "energy_kwh": energy_kwh}


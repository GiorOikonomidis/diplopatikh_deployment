"""Formats a pipeline run's timing/hardware/energy/classification metrics
into one structured, fixed-width terminal report -- table-like output
suitable for pasting directly into a paper's results section or appendix.
"""

from __future__ import annotations

import numpy as np

_WIDTH = 80


def _rule(char: str = "-") -> str:
    """Build one horizontal divider line.

    Input:
        char: the character to repeat.
    Output:
        A string of `char` repeated _WIDTH times.
    """
    return char * _WIDTH


def _section(title: str, rows: list[tuple[str, str]]) -> str:
    """Format one titled section as a divider, a header, then aligned
    "label : value" rows.

    Input:
        title: section header text (upper-cased on output).
        rows: (label, formatted_value) pairs, printed in order with the
            label left-justified and the value right-aligned.
    Output:
        The section's formatted text, without a trailing newline.
    """
    label_width = max((len(label) for label, _ in rows), default=0)
    lines = [_rule(), title.upper(), _rule()]
    lines += [f"{label:<{label_width}} : {value:>15}" for label, value in rows]
    return "\n".join(lines)


def format_pipeline_report(
    dataset_name: str,
    n_rows: int,
    router_ms: list[float],
    expert_ms_by_group: dict[str, list[float]],
    hardware_summary: dict[str, float],
    energy_summary: dict[str, float] | None,
    accuracy: float,
    macro_f1: float,
    weighted_f1: float,
) -> str:
    """Assemble the full TIMING / HARDWARE / ENERGY / CLASSIFICATION
    report as one printable string.

    Input:
        dataset_name: label for the scored dataset, shown in the header.
        n_rows: number of rows classified.
        router_ms: per-row router inference time, in milliseconds, one
            entry per classified row.
        expert_ms_by_group: per-row expert inference time, in
            milliseconds, keyed by expert group name (e.g. "G1"/"G3"/
            "G2") -- each list holds one entry per row that reached that
            specific expert (a group with no rows routed to it gets an
            empty list and is skipped in the report).
        hardware_summary: HardwareMonitor.summary()'s return value.
        energy_summary: EnergyMonitor.summary()'s return value, or None
            if energy wasn't tracked for this run (the ENERGY section is
            omitted in that case).
        accuracy: overall accuracy.
        macro_f1: macro-averaged F1 score.
        weighted_f1: support-weighted F1 score.
    Output:
        The full multi-section report as one string, ready to print.
    """
    timing_rows = [
        ("rows classified", f"{n_rows:,}"),
        ("router, mean", f"{np.mean(router_ms):.4f} ms/row"),
        ("router, p95", f"{np.percentile(router_ms, 95):.4f} ms/row"),
    ]
    for group, times in expert_ms_by_group.items():
        if not times:
            continue
        timing_rows += [
            (f"expert {group}, mean", f"{np.mean(times):.4f} ms/row"),
            (f"expert {group}, p95", f"{np.percentile(times, 95):.4f} ms/row"),
            (f"expert {group}, rows", f"{len(times):,} ({100 * len(times) / n_rows:.2f}%)"),
        ]
    total_expert_rows = sum(len(times) for times in expert_ms_by_group.values())
    if total_expert_rows:
        timing_rows.append(
            ("rows reaching any expert",
             f"{total_expert_rows:,} ({100 * total_expert_rows / n_rows:.2f}%)"))
    timing_rows.append(("total wall-clock", f"{hardware_summary['elapsed_s']:.3f} s"))

    hardware_rows = [
        ("RSS before", f"{hardware_summary['rss_before_mb']:.1f} MB"),
        ("RSS after", f"{hardware_summary['rss_after_mb']:.1f} MB"),
        ("RSS delta", f"{hardware_summary['rss_delta_mb']:.1f} MB"),
        ("CPU utilization", f"{hardware_summary['cpu_percent']:.1f} %"),
    ]

    classification_rows = [
        ("accuracy", f"{accuracy:.4f}"),
        ("macro F1", f"{macro_f1:.4f}"),
        ("weighted F1", f"{weighted_f1:.4f}"),
    ]

    sections = [
        _rule("="),
        f"INFERENCE PIPELINE REPORT -- {dataset_name}",
        _rule("="),
        "",
        _section("timing", timing_rows),
        "",
        _section("hardware", hardware_rows),
        "",
    ]
    if energy_summary is not None:
        energy_rows = [
            ("energy consumed", f"{energy_summary['energy_kwh']:.6e} kWh"),
            ("CO2 emissions", f"{energy_summary['emissions_kg_co2']:.6e} kg"),
        ]
        sections += [_section("energy", energy_rows), ""]
    sections += [_section("classification", classification_rows), _rule("=")]

    return "\n".join(sections)

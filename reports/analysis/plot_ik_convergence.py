#!/usr/bin/env python3
"""Generate the thesis-ready generalized-IK convergence figure.

The script reads the validated experiment JSON, carries each converged sample's
final error forward to the remaining checkpoints, and writes:

* ``assets/ik_convergence.png`` — 300 dpi publication figure;
* ``assets/ik_convergence.svg`` — vector version of the same figure;
* ``assets/ik_solve_times.csv`` — all 60 per-target timing records.

The figure legend reports the median and P95 single-target solve time.  Each
stored target time is itself the median of the timing repeats configured in the
experiment JSON.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = [1, *range(10, 251, 10)]
CURVE_COLOR = "#0077BB"
THRESHOLD_COLOR = "#A52A2A"


def esc(value: Any) -> str:
    return html.escape(str(value))


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def select_generalized_combined_rows(
    results: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in results["numerical_ik"]["rows"]
        if row["method"] == "generalized" and row["kind"] == "combined"
    ]
    if len(rows) != 60:
        raise ValueError(f"Expected 60 generalized combined-target rows, got {len(rows)}.")
    return rows


def median_trace(
    rows: list[dict[str, Any]], metric: str
) -> list[tuple[int, float]]:
    """Compute an n-constant median curve with last-observation carried forward."""
    checkpoint_values: dict[int, list[float]] = {
        checkpoint: [] for checkpoint in CHECKPOINTS
    }
    for row in rows:
        trace = sorted(row["trace"], key=lambda point: point["iteration"])
        if not trace:
            raise ValueError(f"Missing trace for target {row['target_id']}.")
        trace_index = 0
        last_value = float(trace[0][metric])
        for checkpoint in CHECKPOINTS:
            while (
                trace_index + 1 < len(trace)
                and trace[trace_index + 1]["iteration"] <= checkpoint
            ):
                trace_index += 1
                last_value = float(trace[trace_index][metric])
            checkpoint_values[checkpoint].append(last_value)
    if any(len(values) != len(rows) for values in checkpoint_values.values()):
        raise ValueError("A convergence checkpoint does not contain all samples.")
    return [
        (checkpoint, float(np.median(values)))
        for checkpoint, values in checkpoint_values.items()
    ]


def solve_time_statistics(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = np.asarray([row["solve_time_ms"] for row in rows], dtype=float)
    return {
        "n": int(values.size),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def legend_label(timing: dict[str, float]) -> str:
    return (
        "广义雅可比 IK"
        f"（单目标求解时间中位数：{timing['median_ms']:.2f} ms；"
        f"P95 = {timing['p95_ms']:.2f} ms）"
    )


def _power_label(exponent: int) -> str:
    superscript = str(exponent).translate(
        str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    )
    return f"10{superscript}"


def academic_convergence_svg(
    results_or_rows: dict[str, Any] | list[dict[str, Any]]
) -> str:
    """Return the self-contained SVG used by the HTML report."""
    rows = (
        select_generalized_combined_rows(results_or_rows)
        if isinstance(results_or_rows, dict)
        else [
            row
            for row in results_or_rows
            if row["method"] == "generalized" and row["kind"] == "combined"
        ]
    )
    if len(rows) != 60:
        raise ValueError(f"Expected 60 generalized combined-target rows, got {len(rows)}.")
    timing = solve_time_statistics(rows)
    panels = [
        {
            "title": "位置误差收敛曲线",
            "ylabel": "位置误差 / mm",
            "points": median_trace(rows, "position_error_mm"),
            "domain": (0.1, 100.0),
            "ticks": [(10.0**power, _power_label(power)) for power in (-1, 0, 1, 2)],
            "threshold": 1.0,
            "threshold_label": "收敛阈值：1 mm",
            "panel_label": "（a）位置误差",
        },
        {
            "title": "姿态误差收敛曲线",
            "ylabel": "姿态误差 / (°)",
            "points": median_trace(rows, "rotation_error_deg"),
            "domain": (0.01, 10.0),
            "ticks": [(10.0**power, _power_label(power)) for power in (-2, -1, 0, 1)],
            "threshold": 0.1,
            "threshold_label": "收敛阈值：0.1°",
            "panel_label": "（b）姿态误差",
        },
    ]
    width, height = 900, 382
    svg = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="广义雅可比逆运动学误差收敛曲线" '
        'xmlns="http://www.w3.org/2000/svg">',
        "<style>"
        ".ap-title{font:15px 'Times New Roman','Songti SC','STSong',serif;fill:#111}"
        ".ap-axis{stroke:#111;stroke-width:1.15;shape-rendering:crispEdges}"
        ".ap-grid{stroke:#d9d9d9;stroke-width:.8;shape-rendering:crispEdges}"
        ".ap-tick{font:12px 'Times New Roman','Songti SC','STSong',serif;fill:#111}"
        ".ap-label{font:13px 'Times New Roman','Songti SC','STSong',serif;fill:#111}"
        ".ap-panel{font:12.5px 'Times New Roman','Songti SC','STSong',serif;fill:#111}"
        ".ap-legend{font:12px 'Times New Roman','Songti SC','STSong',serif;fill:#111}"
        "</style>",
        '<defs><clipPath id="clip-a"><rect x="70" y="43" width="340" height="220"/></clipPath>'
        '<clipPath id="clip-b"><rect x="520" y="43" width="340" height="220"/></clipPath></defs>',
    ]
    for index, panel in enumerate(panels):
        x0 = 70 + index * 450
        y0, plot_w, plot_h = 43, 340, 220
        bottom = y0 + plot_h
        ymin, ymax = panel["domain"]
        lo, hi = math.log10(ymin), math.log10(ymax)

        def xmap(value: float) -> float:
            return x0 + plot_w * value / 250.0

        def ymap(value: float) -> float:
            return y0 + plot_h * (
                hi - math.log10(max(float(value), ymin))
            ) / (hi - lo)

        svg.append(
            f'<text class="ap-title" x="{x0 + plot_w / 2}" y="20" '
            f'text-anchor="middle">{panel["title"]}</text>'
        )
        for tick_value, tick_label in panel["ticks"]:
            y = ymap(tick_value)
            svg.append(
                f'<line class="ap-grid" x1="{x0}" y1="{y:.1f}" '
                f'x2="{x0 + plot_w}" y2="{y:.1f}"/>'
            )
            svg.append(
                f'<line class="ap-axis" x1="{x0}" y1="{y:.1f}" '
                f'x2="{x0 + 5}" y2="{y:.1f}"/>'
            )
            svg.append(
                f'<text class="ap-tick" x="{x0 - 9}" y="{y + 4:.1f}" '
                f'text-anchor="end">{tick_label}</text>'
            )
        for tick in (0, 50, 100, 150, 200, 250):
            x = xmap(tick)
            svg.append(
                f'<line class="ap-axis" x1="{x:.1f}" y1="{bottom}" '
                f'x2="{x:.1f}" y2="{bottom - 5}"/>'
            )
            svg.append(
                f'<text class="ap-tick" x="{x:.1f}" y="{bottom + 18}" '
                f'text-anchor="middle">{tick}</text>'
            )
        svg.extend(
            [
                f'<line class="ap-axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{bottom}"/>',
                f'<line class="ap-axis" x1="{x0}" y1="{bottom}" x2="{x0 + plot_w}" y2="{bottom}"/>',
                f'<line class="ap-axis" x1="{x0}" y1="{y0}" x2="{x0 + plot_w}" y2="{y0}"/>',
                f'<line class="ap-axis" x1="{x0 + plot_w}" y1="{y0}" x2="{x0 + plot_w}" y2="{bottom}"/>',
            ]
        )
        threshold_y = ymap(panel["threshold"])
        svg.append(
            f'<line x1="{x0}" y1="{threshold_y:.1f}" x2="{x0 + plot_w}" '
            f'y2="{threshold_y:.1f}" stroke="{THRESHOLD_COLOR}" stroke-width="1.15" '
            'stroke-dasharray="6 4"/>'
        )
        svg.append(
            f'<text x="{x0 + 8}" y="{threshold_y - 7:.1f}" '
            'font-size="11.5" font-family="Times New Roman,Songti SC,STSong,serif" fill="#8b2727">'
            f'{panel["threshold_label"]}</text>'
        )
        path = " ".join(
            ("M" if point_index == 0 else "L")
            + f"{xmap(iteration):.1f},{ymap(value):.1f}"
            for point_index, (iteration, value) in enumerate(panel["points"])
        )
        clip_id = "clip-a" if index == 0 else "clip-b"
        svg.append(
            f'<path d="{path}" clip-path="url(#{clip_id})" fill="none" '
            f'stroke="{CURVE_COLOR}" stroke-width="2.35" stroke-linejoin="round"/>'
        )
        for point_index, (iteration, value) in enumerate(panel["points"]):
            if point_index % 3:
                continue
            svg.append(
                f'<circle cx="{xmap(iteration):.1f}" cy="{ymap(value):.1f}" '
                f'r="2.5" fill="white" stroke="{CURVE_COLOR}" stroke-width="1.35">'
                f'<title>迭代 {iteration}: {fmt(value, 3)}</title></circle>'
            )
        svg.append(
            f'<text class="ap-label" x="{x0 + plot_w / 2}" y="{bottom + 43}" '
            'text-anchor="middle">迭代次数</text>'
        )
        svg.append(
            f'<text class="ap-label" transform="translate({x0 - 49},'
            f'{y0 + plot_h / 2}) rotate(-90)" text-anchor="middle">'
            f'{panel["ylabel"]}</text>'
        )
        svg.append(
            f'<text class="ap-panel" x="{x0 + plot_w / 2}" y="323" '
            f'text-anchor="middle">{panel["panel_label"]}</text>'
        )
    legend_y = 359
    svg.append(
        f'<line x1="207" y1="{legend_y}" x2="247" y2="{legend_y}" '
        f'stroke="{CURVE_COLOR}" stroke-width="2.5"/>'
    )
    svg.append(
        f'<circle cx="227" cy="{legend_y}" r="2.8" fill="white" '
        f'stroke="{CURVE_COLOR}" stroke-width="1.4"/>'
    )
    svg.append(
        f'<text class="ap-legend" x="258" y="{legend_y + 4}">'
        f'{esc(legend_label(timing))}</text>'
    )
    svg.append("</svg>")
    return "".join(svg)


def _configure_matplotlib() -> tuple[Any, Any, Any]:
    cache_dir = Path(tempfile.gettempdir()) / "my_simulation_matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    songti_path = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
    chinese_font_path: str | None = None
    if songti_path.exists():
        font_manager.fontManager.addfont(str(songti_path))
        chinese_font_path = str(songti_path)
    chinese_font = font_manager.FontProperties(
        fname=chinese_font_path, size=9
    )
    matplotlib.rcParams.update(
        {
            "font.family": [
                "Times New Roman",
                chinese_font.get_name(),
                "STIXGeneral",
                "serif",
            ],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )
    return matplotlib, plt, chinese_font


def render_png(results: dict[str, Any], output_path: Path) -> dict[str, float]:
    rows = select_generalized_combined_rows(results)
    timing = solve_time_statistics(rows)
    position = median_trace(rows, "position_error_mm")
    rotation = median_trace(rows, "rotation_error_deg")
    _, plt, chinese_font = _configure_matplotlib()
    from matplotlib.ticker import LogFormatterMathtext, NullLocator

    figure, axes = plt.subplots(1, 2, figsize=(6.9, 3.35))
    panel_data = [
        (
            axes[0],
            position,
            "位置误差收敛曲线",
            "位置误差 / mm",
            (0.1, 100),
            1.0,
            "收敛阈值：1 mm",
            "（a）位置误差",
        ),
        (
            axes[1],
            rotation,
            "姿态误差收敛曲线",
            "姿态误差 / (°)",
            (0.01, 10),
            0.1,
            "收敛阈值：0.1°",
            "（b）姿态误差",
        ),
    ]
    line_handle = None
    for (
        axis,
        points,
        title,
        ylabel,
        ylim,
        threshold,
        threshold_text,
        panel_label,
    ) in panel_data:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        (line_handle,) = axis.plot(
            x_values,
            y_values,
            color=CURVE_COLOR,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgewidth=0.9,
            markevery=3,
            zorder=3,
        )
        axis.set_yscale("log")
        axis.set_xlim(0, 250)
        axis.set_ylim(*ylim)
        axis.set_xticks([0, 50, 100, 150, 200, 250])
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.55)
        axis.axhline(
            threshold,
            color=THRESHOLD_COLOR,
            linewidth=0.9,
            linestyle=(0, (6, 4)),
        )
        axis.text(
            6,
            threshold * 1.18,
            threshold_text,
            color=THRESHOLD_COLOR,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        axis.set_title(
            title,
            pad=12,
            fontweight="normal",
        )
        axis.set_xlabel(
            "迭代次数",
            labelpad=8,
        )
        axis.set_ylabel(
            ylabel,
            labelpad=7,
        )
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            length=4,
            width=0.8,
            top=True,
            right=True,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#111111")
        axis.text(
            0.5,
            -0.27,
            panel_label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    figure.subplots_adjust(left=0.095, right=0.985, top=0.87, bottom=0.29, wspace=0.30)
    assert line_handle is not None
    figure.legend(
        [line_handle],
        [legend_label(timing)],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        handlelength=3.0,
        numpoints=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return timing


def write_timing_csv(
    rows: list[dict[str, Any]], output_path: Path, timing_repeats: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "configuration",
                "target_id",
                "solve_time_ms",
                "timing_repeats",
            ],
        )
        writer.writeheader()
        for row in sorted(
            rows, key=lambda item: (item["configuration"], item["target_id"])
        ):
            writer.writerow(
                {
                    "configuration": row["configuration"],
                    "target_id": row["target_id"],
                    "solve_time_ms": f"{row['solve_time_ms']:.6f}",
                    "timing_repeats": timing_repeats,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="reports/ik_validation/results.json")
    parser.add_argument("--png-output", default="assets/ik_convergence.png")
    parser.add_argument("--svg-output", default="assets/ik_convergence.svg")
    parser.add_argument("--timing-output", default="assets/ik_solve_times.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = (REPO_ROOT / args.results).resolve()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows = select_generalized_combined_rows(results)
    repeats = results["experiment_config"]["budget"]["timing_repeats"]

    png_path = (REPO_ROOT / args.png_output).resolve()
    timing = render_png(results, png_path)
    svg_path = (REPO_ROOT / args.svg_output).resolve()
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(academic_convergence_svg(results), encoding="utf-8")
    timing_path = (REPO_ROOT / args.timing_output).resolve()
    write_timing_csv(rows, timing_path, repeats)
    print(f"PNG: {png_path}")
    print(f"SVG: {svg_path}")
    print(f"timings: {timing_path}")
    print(
        "single-target solve time: "
        f"median={timing['median_ms']:.3f} ms, "
        f"P95={timing['p95_ms']:.3f} ms, n={timing['n']}"
    )


if __name__ == "__main__":
    main()

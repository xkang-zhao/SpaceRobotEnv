#!/usr/bin/env python3
"""Plot eight task-level success-rate comparisons from a validated CSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "reports/record/auto_grasp_success_rate.csv"
DEFAULT_PNG = PROJECT_ROOT / "assets/auto_grasp_success_rate.png"
DEFAULT_SVG = PROJECT_ROOT / "assets/auto_grasp_success_rate.svg"
EXPECTED_TASKS = (
    "Cube",
    "Debris panel",
    "Debris truss",
    "Sat-2 left truss",
    "Sat-3 left panel",
    "Sat-3 lift",
    "Sat handle",
    "Sat left panel",
)
SERIES_COLORS = ("#0077BB", "#EE7733", "#009988")
SERIES_HATCHES = ("", "///", "\\\\\\")
EDGE_COLOR = "#111111"
PLACEHOLDER_COLOR = "#9B2C2C"


@dataclass(frozen=True)
class SuccessRateRow:
    task: str
    method: str
    success: int
    trials: int
    rate_percent: float
    data_status: str


def parse_rate(value: str) -> float:
    text = value.strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid percentage value: {value!r}.") from exc


def load_rows(path: Path) -> list[SuccessRateRow]:
    path = Path(path)
    expected_fields = [
        "Task",
        "Method",
        "Success",
        "Trials",
        "Rate",
        "DataStatus",
    ]
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"Expected CSV fields {expected_fields}, "
                f"got {reader.fieldnames}."
            )
        rows = []
        for line_number, item in enumerate(reader, start=2):
            try:
                success = int(item["Success"])
                trials = int(item["Trials"])
            except ValueError as exc:
                raise ValueError(
                    f"Success and Trials must be integers at line "
                    f"{line_number}."
                ) from exc
            rows.append(
                SuccessRateRow(
                    task=item["Task"].strip(),
                    method=item["Method"].strip(),
                    success=success,
                    trials=trials,
                    rate_percent=parse_rate(item["Rate"]),
                    data_status=item["DataStatus"].strip().lower(),
                )
            )
    validate_rows(rows)
    return rows


def group_rows(
    rows: list[SuccessRateRow],
) -> OrderedDict[str, list[SuccessRateRow]]:
    grouped: OrderedDict[str, list[SuccessRateRow]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row.task, []).append(row)
    return grouped


def validate_rows(rows: list[SuccessRateRow]) -> None:
    if len(rows) != 24:
        raise ValueError(
            f"Expected 24 rows (8 tasks x 3 methods), got {len(rows)}."
        )
    grouped = group_rows(rows)
    if tuple(grouped) != EXPECTED_TASKS:
        raise ValueError(
            "Task names or order do not match the expected table: "
            f"{tuple(grouped)}."
        )

    reference_methods = tuple(row.method for row in grouped[EXPECTED_TASKS[0]])
    if len(reference_methods) != 3 or len(set(reference_methods)) != 3:
        raise ValueError("Each task must contain three distinct methods.")

    for task, task_rows in grouped.items():
        methods = tuple(row.method for row in task_rows)
        if methods != reference_methods:
            raise ValueError(
                f"Method names or order for {task!r} are {methods}; "
                f"expected {reference_methods}."
            )
        for row in task_rows:
            if row.trials <= 0:
                raise ValueError(
                    f"Trials must be positive for {task!r}/{row.method!r}."
                )
            if not 0 <= row.success <= row.trials:
                raise ValueError(
                    f"Success must be between 0 and Trials for "
                    f"{task!r}/{row.method!r}."
                )
            expected_rate = row.success / row.trials * 100.0
            if abs(row.rate_percent - expected_rate) > 1e-9:
                raise ValueError(
                    f"Rate for {task!r}/{row.method!r} is "
                    f"{row.rate_percent:g}%, but Success/Trials gives "
                    f"{expected_rate:g}%."
                )
            if row.data_status not in {"measured", "placeholder"}:
                raise ValueError(
                    f"DataStatus for {task!r}/{row.method!r} must be "
                    "'measured' or 'placeholder'."
                )


def configure_matplotlib():
    cache_dir = Path(tempfile.gettempdir()) / "my_simulation_matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": [
                "Times New Roman",
                "STIXGeneral",
                "DejaVu Serif",
                "serif",
            ],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
        }
    )
    return plt


def render_chart(
    rows: list[SuccessRateRow],
    *,
    png_output: Path,
    svg_output: Path,
) -> None:
    plt = configure_matplotlib()
    from matplotlib.patches import Patch

    grouped = group_rows(rows)
    methods = tuple(row.method for row in grouped[EXPECTED_TASKS[0]])
    has_placeholders = any(
        row.data_status == "placeholder" for row in rows
    )
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(12.6, 6.15),
        sharey=True,
        facecolor="white",
    )

    for task_index, (axis, (task, task_rows)) in enumerate(
        zip(axes.flat, grouped.items())
    ):
        x_values = list(range(3))
        bars = axis.bar(
            x_values,
            [row.rate_percent for row in task_rows],
            width=0.64,
            color=SERIES_COLORS,
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            hatch=SERIES_HATCHES,
            zorder=3,
        )
        axis.set_title(task, pad=8, fontweight="normal")
        axis.set_xlim(-0.6, 2.6)
        axis.set_ylim(0, 112)
        axis.set_yticks([0, 20, 40, 60, 80, 100])
        axis.set_xticks([])
        axis.grid(
            axis="y",
            color="#D9D9D9",
            linewidth=0.55,
            zorder=0,
        )
        axis.tick_params(
            axis="y",
            direction="out",
            length=3.5,
            width=0.7,
            color=EDGE_COLOR,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.75)
        axis.spines["bottom"].set_linewidth(0.75)
        axis.spines["left"].set_color(EDGE_COLOR)
        axis.spines["bottom"].set_color(EDGE_COLOR)
        if task_index % 4 == 0:
            axis.set_ylabel("Success Rate / %", labelpad=7)

        for bar, row in zip(bars, task_rows):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{row.success}/{row.trials}\n{row.rate_percent:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
                linespacing=0.85,
                color=EDGE_COLOR,
            )

    legend_handles = [
        Patch(
            facecolor=SERIES_COLORS[index],
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            hatch=SERIES_HATCHES[index],
            label=method,
        )
        for index, method in enumerate(methods)
    ]
    figure.suptitle(
        "Automatic Grasp Success Rate Comparison",
        y=0.985,
        fontsize=16,
        fontweight="normal",
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        handlelength=2.4,
        columnspacing=2.2,
    )
    if has_placeholders:
        figure.text(
            0.5,
            0.018,
            "Baseline values are placeholders; replace them in the CSV "
            "before publication.",
            ha="center",
            va="bottom",
            color=PLACEHOLDER_COLOR,
            fontsize=9,
        )

    figure.subplots_adjust(
        left=0.065,
        right=0.99,
        top=0.84,
        bottom=0.085 if has_placeholders else 0.065,
        wspace=0.18,
        hspace=0.32,
    )
    png_output = Path(png_output)
    svg_output = Path(svg_output)
    png_output.parent.mkdir(parents=True, exist_ok=True)
    svg_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        png_output,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.06,
    )
    figure.savefig(
        svg_output,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--png-output",
        type=Path,
        default=DEFAULT_PNG,
    )
    parser.add_argument(
        "--svg-output",
        type=Path,
        default=DEFAULT_SVG,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = load_rows(args.csv.resolve())
        render_chart(
            rows,
            png_output=args.png_output.resolve(),
            svg_output=args.svg_output.resolve(),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    placeholders = sum(
        row.data_status == "placeholder" for row in rows
    )
    print(f"rows: {len(rows)}")
    print(f"methods: {len(set(row.method for row in rows))}")
    print(f"placeholder_rows: {placeholders}")
    print(f"png: {args.png_output.resolve()}")
    print(f"svg: {args.svg_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

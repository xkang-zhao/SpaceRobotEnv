#!/usr/bin/env python3
"""Render the validated IK JSON into one self-contained, print-ready HTML file."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_ik_convergence import academic_convergence_svg  # noqa: E402

COLORS = {
    "generalized": "#176b87",
    "fixed": "#c26535",
    "translation": "#176b87",
    "rotation": "#6f5aa8",
    "combined": "#c26535",
}
METHOD_LABEL = {"generalized": "广义雅可比 IK", "fixed": "固定基座 IK"}
KIND_LABEL = {"translation": "纯平移", "rotation": "纯旋转", "combined": "复合位姿"}
TASK_LABELS = {
    "cube": "立方体",
    "debris_antenna_panel": "碎片天线板",
    "debris_truss": "碎片桁架",
    "satellite2_left_truss_connection": "卫星2左桁架",
    "satellite3_left_antenna_panel": "卫星3左天线板",
    "satellite3_upper_rod": "卫星3上杆",
    "satellite_handle": "卫星把手",
    "satellite_left_antenna_panel": "卫星左天线板",
}


def f(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def esc(value: Any) -> str:
    return html.escape(str(value))


def summary_map(rows: list[dict[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    return {row[key]: row for row in rows}


def _svg_open(width: int, height: int, label: str) -> list[str]:
    return [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}" '
        'xmlns="http://www.w3.org/2000/svg">',
        "<style>"
        ".axis{stroke:#64727d;stroke-width:1}.grid{stroke:#dbe3e7;stroke-width:1}"
        ".tick{font:11px system-ui;fill:#44515b}.label{font:12px system-ui;fill:#26343d}"
        ".title{font:bold 13px system-ui;fill:#17242c}.legend{font:11px system-ui;fill:#26343d}"
        "</style>",
    ]


def grouped_bars(
    panels: list[dict[str, Any]], width: int = 900, height: int = 330
) -> str:
    """Two or three compact panels with grouped bars."""
    svg = _svg_open(width, height, "分组柱状图")
    panel_gap = 26
    outer = 34
    panel_width = (width - outer * 2 - panel_gap * (len(panels) - 1)) / len(panels)
    for panel_index, panel in enumerate(panels):
        x0 = outer + panel_index * (panel_width + panel_gap)
        y0, plot_h = 46, height - 105
        values = [v for group in panel["groups"] for v in group["values"]]
        vmax = max(values) * 1.12 if max(values) > 0 else 1
        svg.append(
            f'<text class="title" x="{x0 + panel_width / 2:.1f}" y="19" '
            f'text-anchor="middle">{esc(panel["title"])}</text>'
        )
        for tick in range(5):
            value = vmax * tick / 4
            y = y0 + plot_h - plot_h * tick / 4
            svg.append(
                f'<line class="grid" x1="{x0}" y1="{y:.1f}" '
                f'x2="{x0 + panel_width}" y2="{y:.1f}"/>'
            )
            svg.append(
                f'<text class="tick" x="{x0 - 5}" y="{y + 4:.1f}" '
                f'text-anchor="end">{f(value, panel.get("digits", 2))}</text>'
            )
        group_count = len(panel["groups"])
        group_span = panel_width / max(group_count, 1)
        for group_index, group in enumerate(panel["groups"]):
            n = len(group["values"])
            bar_width = min(28, group_span * 0.65 / n)
            center = x0 + group_span * (group_index + 0.5)
            for value_index, value in enumerate(group["values"]):
                x = center + (value_index - (n - 1) / 2) * bar_width - bar_width * 0.42
                bar_h = plot_h * value / vmax
                color = panel["colors"][value_index]
                svg.append(
                    f'<rect x="{x:.1f}" y="{y0 + plot_h - bar_h:.1f}" '
                    f'width="{bar_width * .84:.1f}" height="{bar_h:.1f}" '
                    f'rx="2" fill="{color}"><title>{esc(group["label"])}: {f(value)}</title></rect>'
                )
            svg.append(
                f'<text class="tick" x="{center:.1f}" y="{y0 + plot_h + 17}" '
                f'text-anchor="middle">{esc(group["label"])}</text>'
            )
        svg.append(
            f'<text class="label" x="{x0 + panel_width / 2:.1f}" y="{height - 41}" '
            f'text-anchor="middle">{esc(panel.get("x_label", ""))}</text>'
        )
        svg.append(
            f'<text class="label" transform="translate({x0 - 29:.1f},'
            f'{y0 + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">'
            f'{esc(panel["y_label"])}</text>'
        )
        legend_y = height - 15
        legend_width = panel_width / len(panel["legend"])
        for index, label in enumerate(panel["legend"]):
            lx = x0 + index * legend_width + 4
            svg.append(
                f'<rect x="{lx:.1f}" y="{legend_y - 9}" width="10" height="10" '
                f'fill="{panel["colors"][index]}"/>'
            )
            svg.append(
                f'<text class="legend" x="{lx + 14:.1f}" y="{legend_y}">{esc(label)}</text>'
            )
    svg.append("</svg>")
    return "".join(svg)


def line_panels(
    panels: list[dict[str, Any]], width: int = 900, height: int = 340
) -> str:
    svg = _svg_open(width, height, "折线图")
    gap, outer = 42, 48
    panel_width = (width - outer * 2 - gap * (len(panels) - 1)) / len(panels)
    for panel_index, panel in enumerate(panels):
        x0 = outer + panel_index * (panel_width + gap)
        y0, plot_h = 48, height - 138
        all_x = [x for series in panel["series"] for x, _ in series["points"]]
        all_y = [y for series in panel["series"] for _, y in series["points"]]
        xmin, xmax = panel.get("x_domain", (min(all_x), max(all_x)))
        if panel.get("log_y"):
            positives = [max(y, 1e-6) for y in all_y]
            ymin, ymax = panel.get("y_domain", (min(positives), max(positives)))
            lo, hi = math.log10(ymin), math.log10(ymax)
            ymap = lambda y: y0 + plot_h * (hi - math.log10(max(y, 1e-6))) / max(hi - lo, 1e-9)
            ticks = panel.get("y_ticks", np.geomspace(ymin, ymax, 5))
        else:
            ymin, ymax = panel.get(
                "y_domain", (min(0, min(all_y)), max(all_y) * 1.08)
            )
            ymap = lambda y: y0 + plot_h * (ymax - y) / max(ymax - ymin, 1e-9)
            ticks = panel.get("y_ticks", np.linspace(ymin, ymax, 5))
        xmap = lambda x: x0 + panel_width * (x - xmin) / max(xmax - xmin, 1e-9)
        svg.append(
            f'<text class="title" x="{x0 + panel_width / 2:.1f}" y="18" '
            f'text-anchor="middle">{esc(panel["title"])}</text>'
        )
        if "phase_boundary" in panel:
            boundary_x = xmap(panel["phase_boundary"])
            svg.append(
                f'<rect x="{x0:.1f}" y="{y0}" width="{boundary_x - x0:.1f}" '
                f'height="{plot_h}" fill="#176b87" opacity=".025"/>'
            )
            svg.append(
                f'<rect x="{boundary_x:.1f}" y="{y0}" '
                f'width="{x0 + panel_width - boundary_x:.1f}" height="{plot_h}" '
                f'fill="#c26535" opacity=".055"/>'
            )
            svg.append(
                f'<line x1="{boundary_x:.1f}" y1="{y0}" x2="{boundary_x:.1f}" '
                f'y2="{y0 + plot_h}" stroke="#8b969c" stroke-width="1.2" '
                'stroke-dasharray="4 4"/>'
            )
            svg.append(
                f'<text class="tick" x="{(x0 + boundary_x) / 2:.1f}" y="{y0 + 14}" '
                f'text-anchor="middle">分段插值（1–200）</text>'
            )
            svg.append(
                f'<text class="tick" x="{(boundary_x + x0 + panel_width) / 2:.1f}" '
                f'y="{y0 + 14}" text-anchor="middle">精调</text>'
            )
        for tick in ticks:
            y = ymap(float(tick))
            svg.append(
                f'<line class="grid" x1="{x0}" y1="{y:.1f}" '
                f'x2="{x0 + panel_width}" y2="{y:.1f}"/>'
            )
            svg.append(
                f'<text class="tick" x="{x0 - 6}" y="{y + 4:.1f}" '
                f'text-anchor="end">{float(tick):g}</text>'
            )
        for threshold in panel.get("thresholds", []):
            value = float(threshold["value"])
            if ymin <= value <= ymax:
                y = ymap(value)
                svg.append(
                    f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_width}" '
                    f'y2="{y:.1f}" stroke="#a23b36" stroke-width="1.4" '
                    'stroke-dasharray="7 5"/>'
                )
                svg.append(
                    f'<text x="{x0 + panel_width - 4:.1f}" y="{y - 5:.1f}" '
                    f'text-anchor="end" font-size="10.5" font-family="system-ui" '
                    f'fill="#8f302c">{esc(threshold["label"])}</text>'
                )
        for tick in panel.get("x_ticks", np.linspace(xmin, xmax, 5)):
            x = xmap(float(tick))
            svg.append(
                f'<line class="axis" x1="{x:.1f}" y1="{y0 + plot_h}" '
                f'x2="{x:.1f}" y2="{y0 + plot_h + 4}"/>'
            )
            svg.append(
                f'<text class="tick" x="{x:.1f}" y="{y0 + plot_h + 17}" '
                f'text-anchor="middle">{float(tick):g}</text>'
            )
        for series in panel["series"]:
            path = " ".join(
                ("M" if index == 0 else "L") + f"{xmap(x):.1f},{ymap(y):.1f}"
                for index, (x, y) in enumerate(series["points"])
            )
            dash = (
                f' stroke-dasharray="{series["dash"]}"'
                if series.get("dash")
                else ""
            )
            svg.append(
                f'<path d="{path}" fill="none" stroke="{series["color"]}" '
                f'stroke-width="{series.get("width", 2.2)}" stroke-linejoin="round"'
                f'{dash}/>'
            )
            marker_every = series.get("marker_every", 1)
            for point_index, (x, y) in enumerate(series["points"]):
                if point_index % marker_every:
                    continue
                fill = "white" if series.get("hollow") else series["color"]
                svg.append(
                    f'<circle cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" '
                    f'r="{series.get("marker_radius", 2.2)}" fill="{fill}" '
                    f'stroke="{series["color"]}" stroke-width="1.3">'
                    f'<title>{esc(series["label"])}: {f(y)}</title></circle>'
                )
        svg.append(
            f'<text class="label" x="{x0 + panel_width / 2:.1f}" '
            f'y="{y0 + plot_h + 42:.1f}" '
            f'text-anchor="middle">{esc(panel["x_label"])}</text>'
        )
        svg.append(
            f'<text class="label" transform="translate({x0 - 36:.1f},'
            f'{y0 + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">'
            f'{esc(panel["y_label"])}</text>'
        )
        legend_width = panel_width / max(len(panel["series"]), 1)
        for index, series in enumerate(panel["series"]):
            lx = x0 + index * legend_width + 4
            dash = (
                f' stroke-dasharray="{series["dash"]}"'
                if series.get("dash")
                else ""
            )
            svg.append(
                f'<line x1="{lx:.1f}" y1="{height - 15}" x2="{lx + 15:.1f}" '
                f'y2="{height - 15}" stroke="{series["color"]}" stroke-width="3"'
                f'{dash}/>'
            )
            svg.append(
                f'<text class="legend" x="{lx + 19:.1f}" y="{height - 11}">'
                f'{esc(series["label"])}</text>'
            )
    svg.append("</svg>")
    return "".join(svg)


def academic_convergence_figure(rows: list[dict[str, Any]]) -> str:
    """Publication-style convergence plot for the proposed generalized IK only."""
    position_series = median_trace_series(
        rows, "combined", "position_error_mm"
    )[0]["points"]
    rotation_series = median_trace_series(
        rows, "combined", "rotation_error_deg"
    )[0]["points"]
    panels = [
        {
            "title": "位置误差收敛曲线",
            "ylabel": "位置误差中位数 / mm",
            "points": position_series,
            "domain": (0.1, 100.0),
            "ticks": [(0.1, "10⁻¹"), (1.0, "10⁰"), (10.0, "10¹"), (100.0, "10²")],
            "threshold": 1.0,
            "threshold_label": "收敛阈值：1 mm",
            "panel_label": "（a）位置误差",
        },
        {
            "title": "姿态误差收敛曲线",
            "ylabel": "姿态误差中位数 / (°)",
            "points": rotation_series,
            "domain": (0.01, 10.0),
            "ticks": [(0.01, "10⁻²"), (0.1, "10⁻¹"), (1.0, "10⁰"), (10.0, "10¹")],
            "threshold": 0.1,
            "threshold_label": "收敛阈值：0.1°",
            "panel_label": "（b）姿态误差",
        },
    ]
    width, height = 900, 350
    svg = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="广义雅可比逆运动学误差收敛曲线" '
        'xmlns="http://www.w3.org/2000/svg">',
        "<style>"
        ".ap-title{font:600 15px 'Songti SC','STSong',serif;fill:#111}"
        ".ap-axis{stroke:#111;stroke-width:1.15;shape-rendering:crispEdges}"
        ".ap-grid{stroke:#d9d9d9;stroke-width:.8;shape-rendering:crispEdges}"
        ".ap-tick{font:12px 'Times New Roman','Songti SC',serif;fill:#111}"
        ".ap-label{font:13px 'Songti SC','STSong',serif;fill:#111}"
        ".ap-note{font:11.5px 'Songti SC','STSong',serif;fill:#454545}"
        ".ap-panel{font:12.5px 'Songti SC','STSong',serif;fill:#111}"
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
        svg.append(
            f'<text class="ap-note" x="{x0 + plot_w}" y="37" '
            'text-anchor="end">n = 60</text>'
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
            f'y2="{threshold_y:.1f}" stroke="#9d2f2f" stroke-width="1.15" '
            'stroke-dasharray="6 4"/>'
        )
        svg.append(
            f'<text x="{x0 + 8}" y="{threshold_y - 7:.1f}" '
            'font-size="11.5" font-family="Songti SC,STSong,serif" fill="#8b2727">'
            f'{panel["threshold_label"]}</text>'
        )
        boundary_x = xmap(200)
        svg.append(
            f'<line x1="{boundary_x:.1f}" y1="{y0}" x2="{boundary_x:.1f}" '
            f'y2="{bottom}" stroke="#555" stroke-width="1" stroke-dasharray="3 4"/>'
        )
        svg.append(
            f'<text class="ap-note" x="{boundary_x - 7:.1f}" y="{y0 + 16}" '
            'text-anchor="end">插值阶段</text>'
        )
        svg.append(
            f'<text class="ap-note" x="{boundary_x + 7:.1f}" y="{y0 + 16}">'
            '精调阶段</text>'
        )
        path = " ".join(
            ("M" if point_index == 0 else "L")
            + f"{xmap(iteration):.1f},{ymap(value):.1f}"
            for point_index, (iteration, value) in enumerate(panel["points"])
        )
        clip_id = "clip-a" if index == 0 else "clip-b"
        svg.append(
            f'<path d="{path}" clip-path="url(#{clip_id})" fill="none" '
            'stroke="#0b6f8a" stroke-width="2.35" stroke-linejoin="round"/>'
        )
        for point_index, (iteration, value) in enumerate(panel["points"]):
            if point_index % 3:
                continue
            svg.append(
                f'<circle cx="{xmap(iteration):.1f}" cy="{ymap(value):.1f}" '
                f'r="2.5" fill="white" stroke="#0b6f8a" stroke-width="1.35">'
                f'<title>迭代 {iteration}: {f(value)}</title></circle>'
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
            f'<text class="ap-panel" x="{x0 + plot_w / 2}" y="338" '
            f'text-anchor="middle">{panel["panel_label"]}</text>'
        )
    svg.append("</svg>")
    return "".join(svg)


def algorithm_diagram() -> str:
    return """
<svg viewBox="0 0 900 310" role="img" aria-label="广义雅可比逆运动学计算流程"
 xmlns="http://www.w3.org/2000/svg">
<defs><marker id="a" markerWidth="9" markerHeight="7" refX="8" refY="3.5"
 orient="auto"><polygon points="0 0,9 3.5,0 7" fill="#52636d"/></marker></defs>
<style>.b{fill:#f4f7f8;stroke:#176b87;stroke-width:1.5}.o{fill:#fff4ed;stroke:#c26535;stroke-width:1.5}
.t{font:14px system-ui;fill:#1c2b33}.s{font:12px system-ui;fill:#52636d}.l{stroke:#52636d;stroke-width:1.4;marker-end:url(#a)}</style>
<rect class="b" x="28" y="45" width="165" height="64" rx="9"/><text class="t" x="110" y="71" text-anchor="middle">目标位姿 T*</text><text class="s" x="110" y="93" text-anchor="middle">位置线性插值 + 姿态 SLERP</text>
<rect class="b" x="250" y="45" width="174" height="64" rx="9"/><text class="t" x="337" y="70" text-anchor="middle">SE(3) 误差 e</text><text class="s" x="337" y="93" text-anchor="middle">平移 [m] / 旋转 [rad]</text>
<rect class="b" x="482" y="22" width="185" height="74" rx="9"/><text class="t" x="574" y="49" text-anchor="middle">质量矩阵分块</text><text class="s" x="574" y="72" text-anchor="middle">Mbb 与 Mbm（CRBA）</text>
<rect class="b" x="482" y="122" width="185" height="74" rx="9"/><text class="t" x="574" y="149" text-anchor="middle">末端雅可比分块</text><text class="s" x="574" y="172" text-anchor="middle">Jb 与 Jm</text>
<rect class="o" x="712" y="78" width="160" height="72" rx="9"/><text class="t" x="792" y="105" text-anchor="middle">广义雅可比</text><text class="s" x="792" y="129" text-anchor="middle">Jg = Jm − Jb Mbb⁻¹Mbm</text>
<rect class="o" x="250" y="225" width="208" height="64" rx="9"/><text class="t" x="354" y="251" text-anchor="middle">阻尼最小二乘</text><text class="s" x="354" y="274" text-anchor="middle">vₘ=(JgᵀJg+λ²I)⁻¹Jgᵀe</text>
<rect class="o" x="520" y="225" width="180" height="64" rx="9"/><text class="t" x="610" y="251" text-anchor="middle">动量耦合</text><text class="s" x="610" y="274" text-anchor="middle">vᵦ=−Mbb⁻¹Mbm vₘ</text>
<rect class="b" x="752" y="225" width="120" height="64" rx="9"/><text class="t" x="812" y="251" text-anchor="middle">流形积分</text><text class="s" x="812" y="274" text-anchor="middle">q⁺=integrate(q,vΔt)</text>
<line class="l" x1="193" y1="77" x2="250" y2="77"/><path class="l" d="M424 77 L455 77 L455 257 L458 257"/>
<line class="l" x1="667" y1="59" x2="712" y2="102"/><line class="l" x1="667" y1="159" x2="712" y2="127"/>
<path class="l" d="M792 150 L792 196 L420 196 L420 225"/><line class="l" x1="458" y1="257" x2="520" y2="257"/><line class="l" x1="700" y1="257" x2="752" y2="257"/>
</svg>"""


def median_trace_series(rows: list[dict[str, Any]], kind: str, metric: str) -> list[dict[str, Any]]:
    checkpoints = [1, *range(10, 251, 10)]
    series = []
    for method in ("generalized", "fixed"):
        selected = [r for r in rows if r["kind"] == kind and r["method"] == method]
        iteration_values: dict[int, list[float]] = {
            checkpoint: [] for checkpoint in checkpoints
        }
        for row in selected:
            trace = sorted(row["trace"], key=lambda point: point["iteration"])
            trace_index = 0
            last_value = float(trace[0][metric])
            for checkpoint in checkpoints:
                while (
                    trace_index + 1 < len(trace)
                    and trace[trace_index + 1]["iteration"] <= checkpoint
                ):
                    trace_index += 1
                    last_value = float(trace[trace_index][metric])
                iteration_values[checkpoint].append(last_value)
        points = [
            (iteration, float(np.median(values)))
            for iteration, values in sorted(iteration_values.items())
        ]
        series.append(
            {
                "label": METHOD_LABEL[method],
                "color": COLORS[method],
                "points": points,
                "dash": "7 4" if method == "fixed" else None,
                "hollow": method == "fixed",
                "marker_every": 3,
                "marker_radius": 2.5 if method == "fixed" else 2.0,
                "width": 2.5 if method == "generalized" else 2.2,
            }
        )
    return series


def frame_montage(dataset_root: Path, task_rows: list[dict[str, Any]]) -> str:
    tiles = []
    for index, task in enumerate(task_rows):
        video = (
            dataset_root
            / f"{task['task']}_lerobot_v21"
            / "videos"
            / "chunk-000"
            / "observation.images.cam1"
            / "episode_000000.mp4"
        )
        capture = cv2.VideoCapture(str(video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            continue
        frame = cv2.resize(frame, (240, 180), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            continue
        payload = base64.b64encode(encoded).decode("ascii")
        x = (index % 4) * 220
        y = (index // 4) * 190
        tiles.append(
            f'<image href="data:image/jpeg;base64,{payload}" x="{x}" y="{y}" '
            'width="210" height="157.5" preserveAspectRatio="xMidYMid slice"/>'
            f'<rect x="{x}" y="{y + 135}" width="210" height="22" fill="#10222c" opacity=".78"/>'
            f'<text x="{x + 8}" y="{y + 151}" fill="white" font-size="12" '
            f'font-family="system-ui">{esc(TASK_LABELS.get(task["task"], task["task"]))}</text>'
        )
    return (
        '<svg viewBox="0 0 870 370" role="img" aria-label="八类任务代表性中间帧" '
        'xmlns="http://www.w3.org/2000/svg">'
        + "".join(tiles)
        + "</svg>"
    )


def render(results: dict[str, Any], result_sha: str, dataset_root: Path) -> str:
    metadata = results["metadata"]
    config = results["experiment_config"]
    numerical = results["numerical_ik"]
    dynamic = results["dynamic_validation"]
    ablation = results["damping_ablation"]
    tasks = results["task_summary"]
    methods = summary_map(numerical["summary_by_method"], "method")
    by_kind = {
        (row["method"], row["kind"]): row
        for row in numerical["summary_by_method_and_kind"]
    }
    dynamics = summary_map(dynamic["summary_by_method"], "method")

    fig2 = grouped_bars(
        [
            {
                "title": "位置误差 P95",
                "y_label": "位置误差 / mm",
                "x_label": "目标类别（每种方法 n=240）",
                "groups": [
                    {
                        "label": KIND_LABEL[kind],
                        "values": [
                            by_kind[(method, kind)]["position_error_mm"]["p95"]
                            for method in ("generalized", "fixed")
                        ],
                    }
                    for kind in ("translation", "rotation", "combined")
                ],
                "colors": [COLORS["generalized"], COLORS["fixed"]],
                "legend": [METHOD_LABEL["generalized"], METHOD_LABEL["fixed"]],
            },
            {
                "title": "姿态误差 P95",
                "y_label": "姿态误差 / (°)",
                "x_label": "目标类别（每种方法 n=240）",
                "groups": [
                    {
                        "label": KIND_LABEL[kind],
                        "values": [
                            by_kind[(method, kind)]["rotation_error_deg"]["p95"]
                            for method in ("generalized", "fixed")
                        ],
                    }
                    for kind in ("translation", "rotation", "combined")
                ],
                "colors": [COLORS["generalized"], COLORS["fixed"]],
                "legend": [METHOD_LABEL["generalized"], METHOD_LABEL["fixed"]],
            },
        ]
    )
    fig3 = academic_convergence_svg(numerical["rows"])
    fig4 = grouped_bars(
        [
            {
                "title": "阈值达标率",
                "y_label": "达标率 / %",
                "groups": [
                    {
                        "label": "全部目标",
                        "values": [
                            methods[m]["success_rate"] * 100
                            for m in ("generalized", "fixed")
                        ],
                    }
                ],
                "colors": [COLORS["generalized"], COLORS["fixed"]],
                "legend": [METHOD_LABEL["generalized"], METHOD_LABEL["fixed"]],
            },
            {
                "title": "单目标求解时间中位数",
                "y_label": "时间 / ms",
                "groups": [
                    {
                        "label": "全部目标",
                        "values": [
                            methods[m]["solve_time_ms"]["median"]
                            for m in ("generalized", "fixed")
                        ],
                    }
                ],
                "colors": [COLORS["generalized"], COLORS["fixed"]],
                "legend": [METHOD_LABEL["generalized"], METHOD_LABEL["fixed"]],
            },
            {
                "title": "关节运动量中位数",
                "y_label": "‖Δqₘ‖₂ / rad",
                "groups": [
                    {
                        "label": "全部目标",
                        "values": [
                            methods[m]["joint_motion_l2_rad"]["median"]
                            for m in ("generalized", "fixed")
                        ],
                    }
                ],
                "colors": [COLORS["generalized"], COLORS["fixed"]],
                "legend": [METHOD_LABEL["generalized"], METHOD_LABEL["fixed"]],
            },
        ]
    )
    fig5 = grouped_bars(
        [
            {
                "title": "0.05 s 后实际末端误差中位数",
                "y_label": "位置误差 / mm",
                "groups": [
                    {
                        "label": "12 类运动",
                        "values": [
                            dynamics[m]["position_error_mm"]["median"]
                            for m in ("generalized", "fixed")
                        ],
                    }
                ],
                "colors": [COLORS["generalized"], COLORS["fixed"]],
                "legend": [METHOD_LABEL["generalized"], METHOD_LABEL["fixed"]],
            },
            {
                "title": "实际基座响应中位数",
                "y_label": "基座平移 / mm",
                "groups": [
                    {
                        "label": "12 类运动",
                        "values": [
                            dynamics[m]["actual_base_translation_mm"]["median"]
                            for m in ("generalized", "fixed")
                        ],
                    }
                ],
                "colors": [COLORS["generalized"], COLORS["fixed"]],
                "legend": [METHOD_LABEL["generalized"], METHOD_LABEL["fixed"]],
            },
        ]
    )
    damping_rows = ablation["summary_by_damping"]
    fig6 = line_panels(
        [
            {
                "title": "阻尼—收敛率",
                "x_label": "阻尼系数 λ",
                "y_label": "达标率 / %",
                "series": [
                    {
                        "label": "广义雅可比 IK",
                        "color": COLORS["generalized"],
                        "points": [
                            (row["damping"], row["success_rate"] * 100)
                            for row in damping_rows
                        ],
                    }
                ],
            },
            {
                "title": "阻尼—位置误差",
                "x_label": "阻尼系数 λ",
                "y_label": "位置误差中位数 / mm",
                "series": [
                    {
                        "label": "广义雅可比 IK",
                        "color": COLORS["generalized"],
                        "points": [
                            (row["damping"], row["position_error_mm"]["median"])
                            for row in damping_rows
                        ],
                    }
                ],
            },
            {
                "title": "阻尼—最大关节变化",
                "x_label": "阻尼系数 λ",
                "y_label": "样本最大值 / rad",
                "series": [
                    {
                        "label": "广义雅可比 IK",
                        "color": COLORS["generalized"],
                        "points": [
                            (row["damping"], row["max_joint_change_rad"]["max"])
                            for row in damping_rows
                        ],
                    }
                ],
            },
        ]
    )
    fig7 = grouped_bars(
        [
            {
                "title": "八类任务保存时长",
                "y_label": "时长 / s",
                "x_label": "每类 2 条成功示范",
                "groups": [
                    {
                        "label": str(index + 1),
                        "values": [row["duration_s"]],
                    }
                    for index, row in enumerate(tasks["task_rows"])
                ],
                "colors": [COLORS["generalized"]],
                "legend": ["两条示范合计"],
            },
            {
                "title": "八类任务末端路径",
                "y_label": "路径长度 / m",
                "x_label": "编号与表 6 对应",
                "groups": [
                    {
                        "label": str(index + 1),
                        "values": [row["ee_path_length_m"]],
                    }
                    for index, row in enumerate(tasks["task_rows"])
                ],
                "colors": [COLORS["fixed"]],
                "legend": ["两条示范合计"],
            },
        ]
    )
    montage = frame_montage(dataset_root, tasks["task_rows"])

    setup_rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in [
            ("运动学模型", f"{config['model_path']}（SHA-256: {metadata['model_sha256'][:16]}…）"),
            ("动力学场景", f"{config['scene_path']}（SHA-256: {metadata['scene_sha256'][:16]}…）"),
            ("模型维度", f"nq={config['nq']}，nv={config['nv']}，机械臂 {config['arm_dof']} 自由度"),
            ("末端坐标系", config["end_effector_frame"]),
            ("软件环境", f"Python {metadata['python_version']}；Pinocchio {metadata['pinocchio_version']}；MuJoCo {metadata['mujoco_version']}；NumPy {metadata['numpy_version']}"),
            ("运行平台", metadata["platform"]),
            ("随机性控制", f"固定随机种子 {config['seed']}；5 组确定性初始构型"),
            ("版本状态", f"Git {metadata['git']['commit'][:12]}；工作区{'含未提交修改' if metadata['git']['dirty'] else '干净'}"),
        ]
    )
    method_table = "".join(
        "<tr>"
        f"<td>{METHOD_LABEL[m]}</td>"
        f"<td>{methods[m]['success_count']}/{methods[m]['n']}（{methods[m]['success_rate']*100:.2f}%）</td>"
        f"<td>{f(methods[m]['position_error_mm']['median'])} / {f(methods[m]['position_error_mm']['p95'])} / {f(methods[m]['position_error_mm']['max'])}</td>"
        f"<td>{f(methods[m]['rotation_error_deg']['median'])} / {f(methods[m]['rotation_error_deg']['p95'])} / {f(methods[m]['rotation_error_deg']['max'])}</td>"
        f"<td>{f(methods[m]['iterations']['median'],1)}</td>"
        f"<td>{f(methods[m]['solve_time_ms']['median'])}</td>"
        "</tr>"
        for m in ("generalized", "fixed")
    )
    failure_table = "".join(
        "<tr>"
        f"<td>{row['configuration']}</td><td>{esc(row['target_id'])}</td>"
        f"<td>{METHOD_LABEL[row['method']]}</td>"
        f"<td>{f(row['position_error_mm'])}</td>"
        f"<td>{f(row['rotation_error_deg'])}</td>"
        f"<td>{row['iterations']}</td></tr>"
        for row in numerical["threshold_failures"]
    )
    dynamic_table = "".join(
        "<tr>"
        f"<td>{METHOD_LABEL[m]}</td>"
        f"<td>{f(dynamics[m]['position_error_mm']['median'])} / {f(dynamics[m]['position_error_mm']['p95'])}</td>"
        f"<td>{f(dynamics[m]['rotation_error_deg']['median'])} / {f(dynamics[m]['rotation_error_deg']['p95'])}</td>"
        f"<td>{f(dynamics[m]['actual_base_translation_mm']['median'])} / {f(dynamics[m]['actual_base_translation_mm']['p95'])}</td>"
        f"<td>{f(dynamics[m]['actual_base_rotation_deg']['median'])} / {f(dynamics[m]['actual_base_rotation_deg']['p95'])}</td>"
        "</tr>"
        for m in ("generalized", "fixed")
    )
    damping_table = "".join(
        "<tr>"
        f"<td>{row['damping']:.2f}</td><td>{row['success_count']}/{row['n']}（{row['success_rate']*100:.1f}%）</td>"
        f"<td>{f(row['position_error_mm']['median'])}</td>"
        f"<td>{f(row['rotation_error_deg']['median'])}</td>"
        f"<td>{f(row['solve_time_ms']['median'])}</td>"
        f"<td>{f(row['max_joint_change_rad']['max'])}</td></tr>"
        for row in damping_rows
    )
    task_table = "".join(
        "<tr>"
        f"<td>{index + 1}</td><td>{esc(TASK_LABELS.get(row['task'], row['task']))}</td>"
        f"<td>{row['episodes']}</td><td>{row['frames']}</td>"
        f"<td>{f(row['duration_s'],2)}</td><td>{f(row['ee_path_length_m'])}</td>"
        f"<td>{f(row['max_position_step_mm'])}</td>"
        f"<td>{f(row['max_rotation_step_deg'])}</td></tr>"
        for index, row in enumerate(tasks["task_rows"])
    )

    generalized = methods["generalized"]
    fixed = methods["fixed"]
    momentum = generalized["momentum_residual"]
    base_t = generalized["predicted_base_translation_mm"]
    base_r = generalized["predicted_base_rotation_deg"]
    damping_005 = next(row for row in damping_rows if row["damping"] == 0.05)
    damping_02 = next(row for row in damping_rows if row["damping"] == 0.2)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>自由漂浮空间机器人逆运动学算法设计与验证</title>
<style>
:root{{--ink:#17242c;--muted:#52636d;--line:#d8e0e4;--paper:#fff;--blue:#176b87;--orange:#c26535;--wash:#f4f7f8}}
*{{box-sizing:border-box}} html{{background:#e8edef}} body{{margin:0;color:var(--ink);font-family:"Noto Serif SC","Songti SC","STSong",serif;line-height:1.82;font-size:15px}}
main{{max-width:1040px;margin:28px auto;background:var(--paper);padding:62px 76px;box-shadow:0 12px 34px #26343d1a}}
h1{{font-size:30px;line-height:1.35;text-align:center;margin:0 0 12px;letter-spacing:.05em}}
h2{{font-size:22px;border-bottom:2px solid var(--blue);padding-bottom:7px;margin:2.2em 0 .8em}}
h3{{font-size:18px;margin:1.6em 0 .5em}} p{{text-align:justify;margin:.65em 0}}
.subtitle,.meta{{text-align:center;color:var(--muted)}} .meta{{font-size:13px;margin-bottom:34px}}
.abstract{{border:1px solid var(--line);background:var(--wash);padding:18px 22px;margin:26px 0}}
.keywords{{font-size:14px}} .eq{{font-family:"Times New Roman",serif;text-align:center;margin:15px 0;font-size:17px}}
.eqno{{float:right}} figure{{margin:24px 0 29px;break-inside:avoid}} figure svg{{width:100%;height:auto;display:block}}
figcaption{{font-size:13px;text-align:center;color:#34454e;margin-top:8px}} table{{width:100%;border-collapse:collapse;margin:16px 0 25px;font-size:13px;break-inside:avoid}}
caption{{font-weight:700;margin-bottom:8px}} th,td{{border:1px solid #bdc9cf;padding:7px 8px;text-align:center;vertical-align:middle}}
th{{background:#edf3f5}} td:first-child{{font-weight:500}} .note{{font-size:13px;color:var(--muted)}}
.finding{{border-left:4px solid var(--orange);background:#fff7f1;padding:10px 15px;margin:16px 0}}
.toc{{columns:2;column-gap:42px;background:var(--wash);padding:16px 25px;margin:24px 0}} .toc a{{color:var(--ink);text-decoration:none}}
.refs{{font-size:13px}} code{{font-family:ui-monospace,monospace;font-size:.92em}} a{{color:#145e76}} a,code,td{{overflow-wrap:anywhere}}
@media(max-width:720px){{main{{margin:0;padding:28px 18px;box-shadow:none}}h1{{font-size:25px}}.toc{{columns:1}}table{{display:block;overflow-x:auto;white-space:nowrap}}}}
@page{{size:A4;margin:18mm 16mm}} @media print{{html{{background:white}}body{{font-size:10.5pt}}main{{max-width:none;margin:0;padding:0;box-shadow:none}}h2{{break-before:auto}}a{{color:inherit;text-decoration:none}}figure,table{{break-inside:avoid}}}}
</style>
</head>
<body><main>
<h1>自由漂浮空间机器人逆运动学算法设计与验证</h1>
<p class="subtitle">基于广义雅可比、阻尼最小二乘与无渲染动力学仿真的可复现实验</p>
<p class="meta">生成时间：{esc(metadata['generated_at'])}　结果文件 SHA-256：{result_sha[:20]}…</p>

<div class="abstract"><strong>摘要：</strong>面向零重力自由漂浮空间机器人在轨抓取任务，本章针对仓库现有逆运动学实现建立一套可复现、分层次的验证方法。算法层验证采用 5 组初始构型与 48 类相对目标，共形成 240 个目标；广义雅可比逆运动学与固定基座逆运动学使用完全一致的 20 个插值点、每点 10 次迭代及最多 50 次精调预算。评价时重新执行正运动学，将位置误差和姿态误差分别以毫米和角度表示，并以“位置不超过 1 mm 且姿态不超过 0.1°”作为联合判据。结果表明，广义方法和固定基座方法均有 239/240 个目标达标，二者的静态精度相近；广义方法预测的基座平移中位数为 {f(base_t['median'])} mm，基座转动中位数为 {f(base_r['median'])}°，归一化动量残差中位数为 {f(momentum['median'])}，说明其核心价值在于满足自由漂浮耦合关系，而非必然降低离线末端误差。进一步开展 12 类 MuJoCo 无渲染动力学验证、5 组阻尼消融以及 16 条成功示范的任务级统计。实验揭示：单次 0.05 s 关节位置控制尚不能使大幅目标完全到位；过大的阻尼显著降低固定预算内的收敛率；已保存示范只能证明数据质量与任务覆盖，不能反推规划器成功率。</div>
<p class="keywords"><strong>关键词：</strong>自由漂浮空间机器人；广义雅可比矩阵；逆运动学；阻尼最小二乘；动量守恒；MuJoCo</p>

<div class="toc"><strong>本章结构</strong><br>
<a href="#s1">1 问题定义与研究对象</a><br><a href="#s2">2 算法建模与求解方法</a><br>
<a href="#s3">3 实验设计与评价指标</a><br><a href="#s4">4 数值求解验证</a><br>
<a href="#s5">5 空间机器人特性与动力学验证</a><br><a href="#s6">6 阻尼参数消融</a><br>
<a href="#s7">7 任务级示范验证</a><br><a href="#s8">8 讨论与局限性</a><br>
<a href="#s9">9 本章小结</a></div>

<h2 id="s1">1　问题定义与研究对象</h2>
<h3>1.1　自由漂浮系统假设</h3>
<p>研究对象为搭载 6 自由度 UR10e 机械臂和 Robotiq 2F85 夹爪的航天器—机械臂组合体。与地面固定机械臂不同，基座通过自由关节与惯性坐标系连接；机械臂关节运动产生的反作用会改变基座位置与姿态，进而改变末端实际运动。本文采用三个基本假设：外部重力为零；系统初始线动量与角动量为零；逆运动学迭代期间不施加基座位置或姿态控制。该假设对应模型文件中的零重力设置和自由基座关节，也与广义雅可比矩阵（Generalized Jacobian Matrix, GJM）的经典建模条件一致 [1]。</p>
<p>Pinocchio 配置变量维度为 <em>nq</em>={config['nq']}，速度变量维度为 <em>nv</em>={config['nv']}。二者不同的原因是自由基座姿态在配置空间中采用四元数表示，需要 7 个配置量描述 6 个速度自由度；机械臂包含 6 个转动关节。因此，配置向量写为 <em>q</em>=[<em>q</em><sub>b</sub>,<em>q</em><sub>m</sub>]，其中 <em>q</em><sub>b</sub>∈ℝ³×S³，<em>q</em><sub>m</sub>∈ℝ⁶；广义速度写为 <em>v</em>=[<em>v</em><sub>b</sub>,<em>v</em><sub>m</sub>]∈ℝ¹²。</p>
<h3>1.2　验证问题的层次</h3>
<p>本章将“算法是否正确”拆分为三个彼此不可替代的问题。第一层是数值逆运动学：给定起始构型和目标位姿，重新正运动学后是否满足精度阈值。第二层是动力学跟踪：把求解所得机械臂关节角作为 MuJoCo 位置执行器目标后，在有限仿真时间内末端实际到达何处、基座如何响应。第三层是任务示范：自动抓取流程保存的轨迹是否覆盖不同对象，并具有可接受的长度与连续性。只有第一层直接评价求解器精度；第二层同时受执行器动态和控制周期影响；第三层还包含规划、接触、成功筛选和数据写入机制。</p>

<h2 id="s2">2　算法建模与求解方法</h2>
<h3>2.1　基座—机械臂惯性耦合</h3>
<p>系统质量矩阵按基座与机械臂速度分块。无外力且初始动量为零时，基座广义动量满足</p>
<div class="eq">M<sub>bb</sub>v<sub>b</sub> + M<sub>bm</sub>v<sub>m</sub> = 0，<span class="eqno">（1）</span></div>
<p>其中 M<sub>bb</sub>∈ℝ⁶×⁶ 为基座惯性块，M<sub>bm</sub>∈ℝ⁶×⁶ 为基座—机械臂耦合块。由式（1）可得被动基座速度</p>
<div class="eq">v<sub>b</sub> = −M<sub>bb</sub><sup>−1</sup>M<sub>bm</sub>v<sub>m</sub>。<span class="eqno">（2）</span></div>
<p>末端空间速度满足 v<sub>e</sub>=J<sub>b</sub>v<sub>b</sub>+J<sub>m</sub>v<sub>m</sub>。将式（2）代入可得</p>
<div class="eq">v<sub>e</sub> = J<sub>g</sub>v<sub>m</sub>，　J<sub>g</sub>=J<sub>m</sub>−J<sub>b</sub>M<sub>bb</sub><sup>−1</sup>M<sub>bm</sub>。<span class="eqno">（3）</span></div>
<p>式（3）是本实现采用的广义雅可比。质量矩阵由复合刚体算法（Composite Rigid Body Algorithm, CRBA）计算，末端雅可比采用世界坐标对齐表达。固定基座对照方法直接丢弃 J<sub>b</sub> 对应的六列，仅以 J<sub>m</sub> 求解关节速度。该对照用于分离“考虑耦合”与“忽略耦合”的影响，而不是将固定基座方法视为自由漂浮系统的物理真值。</p>
<h3>2.2　阻尼最小二乘与流形积分</h3>
<p>为抑制接近奇异构型时雅可比逆的数值放大，采用阻尼最小二乘（Damped Least Squares, DLS）求解 [2]：</p>
<div class="eq">v<sub>m</sub>=(J<sub>g</sub><sup>T</sup>J<sub>g</sub>+λ²I)<sup>−1</sup>J<sub>g</sub><sup>T</sup>e。<span class="eqno">（4）</span></div>
<p>误差 e=[e<sub>p</sub>,e<sub>R</sub>] 由世界坐标系位置差和旋转对数映射构成。程序内部仍以一个六维向量参与更新，但验证阶段不使用米与弧度混合范数判断成功。速度总范数被限制为 {config['budget']['velocity_norm_limit']}，并通过 Pinocchio 的 <code>integrate</code> 在包含四元数的配置流形上积分，步长 Δt={config['budget']['dt']}。Pinocchio 提供了正运动学、雅可比和刚体动力学算法的统一实现 [3]。</p>
<h3>2.3　位姿插值</h3>
<p>从起始位姿 T₀ 到目标位姿 T* 设置 {config['budget']['interpolation_points']} 个插值点。位置采用线性插值 p(α)=p₀+α(p*−p₀)，姿态采用球面线性插值（Spherical Linear Interpolation, SLERP），α∈(0,1]。每个插值点执行 {config['budget']['iterations_per_point']} 次局部迭代，随后对最终目标最多精调 {config['budget']['fine_iterations']} 次。两种方法共享相同插值、阻尼、积分步长、速度上限和提前终止判据，避免原生产封装中精调次数不同造成的预算偏差。</p>
<figure>{algorithm_diagram()}<figcaption>图 1　自由漂浮空间机器人广义雅可比逆运动学计算流程</figcaption></figure>

<h2 id="s3">3　实验设计与评价指标</h2>
<h3>3.1　模型、构型与目标集合</h3>
<table><caption>表 1　实验对象与可复现环境</caption><tbody>{setup_rows}</tbody></table>
<p>在基准构型基础上施加四组幅值适中的确定性关节偏置，共得到 5 组初始构型。目标均相对于各构型的实际末端起始位姿生成，以避免不同构型起点不一致引入绝对工作空间偏差。每组测试 48 个目标，总计 240 个目标；每个目标分别由广义方法和固定基座方法求解，因此主实验包含 480 次有效求解。计时前分别对两种代码路径预热，并对每个样本重复 {config['budget']['timing_repeats']} 次，报告中位数。计时只用于本机实现开销比较，不作为跨平台实时性结论。</p>
<p>目标集合由三部分组成。纯平移部分沿 x、y、z 三轴施加 ±20、±50 和 ±100 mm 偏移，每个构型 18 个、合计 90 个；纯旋转部分绕三轴施加 ±5°、±10° 和 ±15° 转动，每个构型 18 个、合计 90 个；复合位姿部分以随机种子 {config['seed']} 生成 12 个平移 30–100 mm、转动 4–15° 的随机方向偏移，5 组构型合计 60 个。三类目标既覆盖单一自由度扰动，也覆盖平移与旋转同时存在的局部操作需求。</p>
<h3>3.2　独立量纲的误差与成功判据</h3>
<p>求解结束后重新执行正运动学。位置误差定义为 e<sub>p</sub>=‖p*−p(q)‖₂，姿态误差定义为 e<sub>R</sub>=‖Log(R(q)<sup>T</sup>R*)‖₂。成功条件为 e<sub>p</sub>≤1 mm 且 e<sub>R</sub>≤0.1°。该定义避免了直接将米和弧度相加，并修正了生产单步接口返回“更新前误差”的时序问题。除误差外还记录迭代次数、求解时间、关节运动二范数、最大单关节变化、预测基座平移与转动，以及式（1）的归一化残差。</p>
<p>动力学验证使用 {config['scene_path']}，不创建 Viewer、Renderer 或任何图像传感器。12 类运动为沿三轴 ±50 mm 和绕三轴 ±10°。每个场景把求解后的 6 个机械臂关节角写入位置执行器，推进 {dynamic['mujoco_steps_per_command']} 个步长，即 {dynamic['simulated_seconds_per_command']:.3f} s。MuJoCo 是面向模型控制的多体动力学引擎，其广义坐标动力学和接触求解适合此类闭环验证 [4]。</p>
<h3>3.3　可复现性与证据分级</h3>
<p>实验脚本在开始计算前记录 Git 提交与工作区状态，并计算运动学模型和动力学场景的 SHA-256 摘要。这样即使同一提交下存在尚未提交的模型修改，也可以依据文件哈希判断实验对象是否一致。目标集合完全由显式规则和固定种子生成；除计时外，同一软件栈重复执行应获得相同的构型、目标、迭代次数与误差结果。JSON 写盘前还检查顶层字段、240 个数值目标、12 个动力学场景和每个阻尼 60 个消融目标，任何数量缺失都会使命令失败。</p>
<p>本章将证据分为四级：动量方程残差用于检验代数实现；最终正运动学误差用于检验数值求解；MuJoCo 步进用于检验有限控制周期内的动力学响应；已保存数据用于描述任务覆盖。四类证据回答的问题不同，不能相互替代。例如，极小的动量残差不保证目标可达，亚毫米正运动学误差不保证执行器在 0.05 s 内到位，成功示范也不能提供未保存失败尝试的分母。明确证据边界可避免把一个层面的良好结果外推为整个系统的成功率。</p>

<h2 id="s4">4　数值求解验证</h2>
<h3>4.1　总体精度与收敛率</h3>
<table><caption>表 2　主实验总体结果（误差列依次为中位数 / P95 / 最大值）</caption>
<thead><tr><th>方法</th><th>达标样本</th><th>位置误差 / mm</th><th>姿态误差 / (°)</th><th>迭代中位数</th><th>时间中位数 / ms</th></tr></thead>
<tbody>{method_table}</tbody></table>
<p>两种方法都达到 {generalized['success_count']}/{generalized['n']} 的联合阈值达标数，即 {generalized['success_rate']*100:.2f}%。广义方法位置误差中位数为 {f(generalized['position_error_mm']['median'])} mm，固定基座方法为 {f(fixed['position_error_mm']['median'])} mm；姿态误差中位数分别为 {f(generalized['rotation_error_deg']['median'])}° 与 {f(fixed['rotation_error_deg']['median'])}°。差异远小于所设阈值，不能据此宣称广义方法具有更高的离线末端精度。相反，广义方法单目标时间中位数为 {f(generalized['solve_time_ms']['median'])} ms，约为固定基座方法的 {generalized['solve_time_ms']['median']/fixed['solve_time_ms']['median']:.2f} 倍，额外开销来自每步 CRBA、质量矩阵分块和基座耦合计算。</p>
<figure>{fig2}<figcaption>图 2　三类目标的位置与姿态误差 P95（每种方法 n=240）</figcaption></figure>
<p>图 2 表明误差形态与目标类型有关。纯平移样本的位置误差接近 1 mm 阈值，而姿态误差极小；纯旋转样本的位置误差很小，但姿态误差接近 0.1° 阈值。这并非求解失败，而是统一联合停止条件的直接结果：只要两个独立条件都满足，精调即终止。复合位姿同时消耗位置与姿态收敛预算，其 P95 仍低于阈值。</p>
<h3>4.2　收敛过程</h3>
<figure>{fig3}<figcaption>图 3　广义雅可比逆运动学在 60 个复合目标上的误差中位收敛曲线（对数坐标）</figcaption></figure>
<p>图 3 只展示本文研究对象——广义雅可比逆运动学；固定基座方法保留在总体对照表和性能对比图中，不参与本图的算法收敛过程表达。目标位姿随迭代逐步接近最终值，因此曲线表达的是“当前解相对最终目标”的剩余误差。红色水平虚线分别表示 1 mm 和 0.1° 收敛阈值。对于提前达到阈值的样本，后续检查点保持其最终误差，因而每个点始终由完整的 60 个目标计算中位数，避免只保留未收敛难样本所造成的曲线末端回升。横轴采用统一的 50 次间隔，纵轴采用以 10 为底的幂次刻度，位置与姿态误差的数量级下降过程可以直接辨识。</p>
<p>从曲线形态看，位置误差与姿态误差均连续下降，并在约 240 次累计迭代处分别进入对应阈值。两条曲线没有出现跨数量级振荡或发散，说明当前阻尼和速度上限能够在所测复合目标范围内保持平稳更新。由于图中展示的是样本中位数，曲线用于描述典型收敛趋势；最大误差、P95 以及唯一未达标样本仍应结合表 2 和表 3 阅读，不能仅凭中位曲线替代尾部风险分析。</p>
<p>图例同时给出单目标求解时间的中位数与 P95，用于反映典型计算开销及其尾部分布。每个目标的计时数值均由三次重复运行取中位数，因此可降低偶然系统负载对单次测量的影响。</p>
<figure>{fig4}<figcaption>图 4　广义雅可比 IK 与固定基座 IK 的达标率、耗时和关节运动量对比（各 n=240）</figcaption></figure>
<h3>4.3　未满足阈值样本</h3>
<table><caption>表 3　固定预算内未满足联合阈值的样本</caption>
<thead><tr><th>构型</th><th>目标</th><th>方法</th><th>位置误差 / mm</th><th>姿态误差 / (°)</th><th>迭代数</th></tr></thead>
<tbody>{failure_table}</tbody></table>
<p>异常集中在第 5 组构型绕 y 轴正向旋转 15° 的同一目标。广义与固定基座方法均耗尽 250 次预算，姿态误差分别为 {f(numerical['threshold_failures'][0]['rotation_error_deg'])}° 和 {f(numerical['threshold_failures'][1]['rotation_error_deg'])}°，而位置误差均小于 1 mm。这种“同一目标、两种方法同时轻微越界”的模式更支持局部收敛速度和固定预算不足的解释，而不是某一种雅可比定义独有的故障。报告保留该样本，没有通过放宽阈值或增加迭代隐藏异常。</p>

<h2 id="s5">5　空间机器人特性与动力学验证</h2>
<h3>5.1　预测基座反向响应与动量残差</h3>
<p>广义方法在 240 个目标上预测的基座平移中位数/P95/最大值为 {f(base_t['median'])}/{f(base_t['p95'])}/{f(base_t['max'])} mm，基座转动为 {f(base_r['median'])}/{f(base_r['p95'])}/{f(base_r['max'])}°。其归一化动量守恒残差中位数为 {f(momentum['median'])}，P95 为 {f(momentum['p95'])}，最大值为 {f(momentum['max'])}。残差接近双精度浮点舍入量级，验证了每次速度分解满足式（1）。这是一项代数一致性验证，不等同于真实硬件上的动量测量。</p>
<p>基座响应方向由耦合矩阵和具体构型共同决定，不能简单概括为“末端向正 x 运动，基座一定沿负 x 平移”。更准确的表述是：机械臂广义动量由基座相反的广义动量补偿，基座线速度与角速度是一个六维耦合结果。广义雅可比把这一结果反馈到末端速度映射中；固定基座方法则明确设 v<sub>b</sub>=0。</p>
<p>还应区分“瞬时速度约束”和“有限位姿变化”。式（1）在每个迭代点约束广义速度，随后通过流形积分累积为基座位置与姿态变化；因此最终位姿变化不是一次静态质量比例换算，而是沿整条构型路径积分的结果。随着关节角改变，M<sub>bb</sub>、M<sub>bm</sub>、J<sub>b</sub> 和 J<sub>m</sub> 都会更新。当前实现每一步重新计算这些矩阵，避免了把初始构型的耦合关系冻结到整个轨迹中。</p>
<h3>5.2　无渲染 MuJoCo 跟踪结果</h3>
<table><caption>表 4　12 类典型运动的动力学结果（中位数 / P95）</caption>
<thead><tr><th>方法</th><th>末端位置误差 / mm</th><th>末端姿态误差 / (°)</th><th>实际基座平移 / mm</th><th>实际基座转动 / (°)</th></tr></thead>
<tbody>{dynamic_table}</tbody></table>
<figure>{fig5}<figcaption>图 5　一次控制周期后的实际末端误差与基座响应（每种方法 n=12，仿真时长 0.05 s）</figcaption></figure>
<p>在 0.05 s 后，广义方法实际位置误差中位数为 {f(dynamics['generalized']['position_error_mm']['median'])} mm，固定基座方法为 {f(dynamics['fixed']['position_error_mm']['median'])} mm；两者都明显大于离线 IK 误差。主要原因是本实验给出 ±50 mm 或 ±10° 的一步目标，但执行器仅推进一个 20 Hz 控制周期，关节尚未到达命令角。该结果说明“逆运动学解满足几何目标”与“执行器在有限时间内跟踪到位”是不同命题。若应用要求单周期完成大幅运动，应设计轨迹级关节控制或延长执行时间，而不能仅增加 IK 精调次数。</p>
<div class="finding"><strong>关键边界：</strong>生产环境当前只把求解后的机械臂关节角写入 MuJoCo 控制量，并未把广义 IK 内部预测的基座位姿直接写回仿真。表 5 的“实际基座响应”完全来自 MuJoCo 自由基座动力学；前述“预测基座变化”来自离线流形积分。二者用途不同，本章不把前者冒充后者，也不声称它们应逐点相等。</div>

<h2 id="s6">6　阻尼参数消融</h2>
<p>在 5 组构型的 60 个复合目标上，对 λ∈{{0.01,0.03,0.05,0.10,0.20}} 分别执行相同的 250 次最大预算，总计 {ablation['solve_count']} 次广义 IK 求解。阻尼越大，矩阵求解越稳定、单步速度越保守；在固定预算下，过度保守会表现为尚未进入阈值，而不一定是发散。</p>
<table><caption>表 5　阻尼系数消融结果（每组 n=60）</caption>
<thead><tr><th>λ</th><th>达标样本</th><th>位置误差中位 / mm</th><th>姿态误差中位 / (°)</th><th>时间中位 / ms</th><th>最大关节变化样本最大值 / rad</th></tr></thead>
<tbody>{damping_table}</tbody></table>
<figure>{fig6}<figcaption>图 6　阻尼系数对精度、达标率与最大关节变化的影响（每个 λ 的 n=60）</figcaption></figure>
<p>λ=0.01、0.03 和 0.05 时 60 个目标全部达标；λ=0.10 时达标率降至 {next(row for row in damping_rows if row['damping']==0.1)['success_rate']*100:.1f}%，λ=0.20 时仅为 {damping_02['success_rate']*100:.1f}%。λ=0.05 的位置误差中位数为 {f(damping_005['position_error_mm']['median'])} mm，姿态误差中位数为 {f(damping_005['rotation_error_deg']['median'])}°，并保持 100% 达标。虽然 λ=0.03 的位置误差中位数略低，但差异很小；结合生产实现已有参数、对奇异附近速度放大的抑制需求以及本次全样本达标结果，保留 λ=0.05 是合理的工程折中。该结论适用于当前构型与目标范围，不意味着它是所有任务的全局最优值。</p>

<h2 id="s7">7　任务级示范验证</h2>
<h3>7.1　数据范围与统计口径</h3>
<p>现有 LeRobot v2.1 数据包含 {tasks['task_count']} 类任务、{tasks['episode_count']} 条轨迹、{tasks['frame_count']} 帧。按每帧 0.05 s 计，保存数据覆盖 {f(tasks['total_duration_s'],2)} s；单条示范时长中位数为 {f(tasks['episode_duration_s']['median'],2)} s，末端路径长度中位数为 {f(tasks['episode_path_length_m']['median'])} m。每条 Parquet 轨迹的终止帧均标记成功，这是采集器筛选后写盘的结果。由于未保存失败尝试总数，16 条轨迹应被称为“16 条成功示范”，不能写成“成功率 100%”。</p>
<table><caption>表 6　八类抓取任务的已保存示范统计</caption>
<thead><tr><th>编号</th><th>任务</th><th>轨迹数</th><th>帧数</th><th>覆盖时长 / s</th><th>末端路径 / m</th><th>最大位置步长 / mm</th><th>最大姿态步长 / (°)</th></tr></thead>
<tbody>{task_table}</tbody></table>
<figure>{fig7}<figcaption>图 7　八类任务的保存时长与末端路径长度（共 16 条成功示范、{tasks['frame_count']} 帧）</figcaption></figure>
<figure>{montage}<figcaption>图 8　八类任务第一条成功示范的中间帧（cam1，图像内嵌，离线可用）</figcaption></figure>
<h3>7.2　轨迹连续性</h3>
<p>逐帧末端位置最大步长的单轨迹中位数为 {f(tasks['max_position_step_mm']['median'])} mm，P95 为 {f(tasks['max_position_step_mm']['p95'])} mm，跨全部轨迹最大值为 {f(tasks['max_position_step_mm']['max'])} mm；姿态最大步长对应数值为 {f(tasks['max_rotation_step_deg']['median'])}°、{f(tasks['max_rotation_step_deg']['p95'])}° 和 {f(tasks['max_rotation_step_deg']['max'])}°。这些统计能够发现跳变和写入错位，但不能替代速度、加速度或冲击约束。由于数据状态同时包含 6 个关节角和 7 维末端位姿，后续可进一步从任务阶段、接触前后和夹爪闭合时刻分层分析连续性。</p>
<p>任务级结果证明当前算法已经在立方体、把手、天线板、桁架和杆件等多种几何目标上生成可保存的成功轨迹。其证据强度低于独立成功率评测：自动采集器会丢弃未成功尝试，数据集只保留达到终止条件的 episode。因此，本节只讨论覆盖范围、时长、路径和连续性，不将样本筛选后的数据解释为规划器鲁棒性。</p>
<p>示范时长按帧周期计算，即一条包含 N 帧的数据覆盖 N×0.05 s。若改用“末帧时间戳减首帧时间戳”，数值会少一个采样周期；两种口径都可用于描述序列，但必须保持一致。本报告选择帧覆盖口径，因为它与数据集 20 FPS 标注和模型每个环境动作 0.05 s 的仿真周期一致。该口径不依赖采集脚本在墙钟时间中的运行速度，因此删除自动采集循环的 <code>sleep</code> 不会改变仿真时间和数据集时间轴。</p>

<h2 id="s8">8　讨论与局限性</h2>
<h3>8.1　广义雅可比的实际价值</h3>
<p>主实验中两种方法静态误差几乎相同，是因为二者都在各自模型假设下重新执行正运动学；固定基座方法通过六个关节也能覆盖这些局部目标。广义方法的优势不是“任何情况下末端误差都更小”，而是把无外力条件下基座反作用纳入速度映射，并给出与动量约束一致的联合运动。对于基座质量较大、机械臂运动幅值有限的当前模型，预测漂移仅为毫米和亚角度量级，因此离线误差差异不显著是合理结果。若降低基座惯量、扩大工作空间或接近奇异构型，耦合项影响可能增大，需要另行验证。</p>
<h3>8.2　当前实现的四项边界</h3>
<p>第一，生产求解器内部仍用位置米数与旋转弧度数拼接后的二范数作为局部误差，并在封装中以 10<sup>−3</sup> 判断精调；本报告改用独立量纲判据，但没有修改生产算法。第二，单步接口返回的是积分前误差，本实验通过最终正运动学重新计算，避免结果时序偏一拍。第三，求解未显式处理关节限位、自碰撞、环境碰撞和可操作度约束；本次小范围目标未出现数值爆炸，并不证明更大工作空间内始终可行。第四，广义求解器积分得到的预测基座位姿未直接写入 MuJoCo；实际仿真通过关节执行器力矩自然产生基座运动，因此预测与实际之间还隔着执行器动力学和离散控制。</p>
<h3>8.3　外部有效性与后续工作</h3>
<p>240 个数值目标覆盖局部平移、局部旋转和有限随机复合位姿，但不是完整工作空间采样；5 组构型也未覆盖全部奇异邻域。动力学验证只有 12 类一步运动，尚未评估长时轨迹、接触冲量和夹爪闭合后的动量交换。计时来自单台 Apple arm64 平台并受解释器、BLAS 和系统负载影响，只能比较本次相对开销。现有 16 条示范经过成功筛选，不能估计失败概率。后续应增加基座惯量梯度实验、关节限位与碰撞约束、未筛选尝试日志、轨迹级闭环控制，以及仿真—实物的一致性验证。</p>
<p>从报告可复现性看，本次运行记录了 Git 提交、脏工作区清单、模型与场景哈希、随机种子和全部软件版本。除计时外，目标生成与数值输出由固定随机种子决定。HTML 中的表格和图均由结果 JSON 生成，源 JSON 的 SHA-256 为 <code>{result_sha}</code>。工作区在实验时包含未提交修改，因此模型哈希比单独的提交号更能标识实际实验对象。</p>

<h2 id="s9">9　本章小结</h2>
<p>本章围绕自由漂浮空间机器人的逆运动学建立了数值求解、动力学响应和任务示范三层验证链。首先从零初始动量约束推导了基座被动速度和广义雅可比，结合阻尼最小二乘、位姿分段插值及配置流形积分形成求解流程。随后以 5 组构型、240 个目标和统一迭代预算比较广义方法与固定基座方法。两者均有 239 个目标满足 1 mm 与 0.1° 联合阈值，唯一异常目标在两种方法下同时轻微超出姿态阈值。由此可见，当前局部目标范围内两者静态精度相当。</p>
<p>广义方法的关键证据来自空间机器人特性：预测基座平移和转动非零，归一化动量残差保持在 {f(momentum['p95'])} 的 P95 水平，表明耦合速度符合模型动量约束。MuJoCo 无渲染实验进一步显示，一次 0.05 s 控制周期不足以完成 50 mm 或 10° 的大幅动作，动力学跟踪不能由离线 IK 精度替代。阻尼消融支持 λ=0.05 作为当前预算下兼顾稳定性和精度的工程选择。最后，8 类任务的 16 条成功示范提供了任务覆盖和轨迹连续性证据，但因缺少失败尝试分母，不构成成功率估计。</p>
<p>综合而言，现有广义雅可比 IK 在所测局部工作域内具备亚毫米/亚 0.1° 级数值收敛能力，并能正确表达基座—机械臂动量耦合；其下一阶段改进重点应从单纯增加迭代次数转向关节与碰撞约束、轨迹级执行器控制、预测基座状态与动力学状态的一致性，以及未筛选任务尝试的统计评测。</p>

<h2>参考文献</h2>
<ol class="refs">
<li>UMETANI Y, YOSHIDA K. Resolved Motion Rate Control of Space Manipulators with Generalized Jacobian Matrix[J]. IEEE Transactions on Robotics and Automation, 1989, 5(3): 303-314. DOI: <a href="https://doi.org/10.1109/70.34766">10.1109/70.34766</a>.</li>
<li>WAMPLER C W. Manipulator Inverse Kinematic Solutions Based on Vector Formulations and Damped Least-Squares Methods[J]. IEEE Transactions on Systems, Man, and Cybernetics, 1986, 16(1): 93-101. DOI: <a href="https://doi.org/10.1109/TSMC.1986.289285">10.1109/TSMC.1986.289285</a>.</li>
<li>CARPENTIER J, SAUREL G, BUONDONNO G, et al. The Pinocchio C++ Library—A Fast and Flexible Implementation of Rigid Body Dynamics Algorithms and Their Analytical Derivatives[C]//IEEE International Symposium on System Integrations. 2019. <a href="https://hal.science/hal-01866228">HAL-01866228</a>.</li>
<li>TODOROV E, EREZ T, TASSA Y. MuJoCo: A Physics Engine for Model-Based Control[C]//2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. 2012: 5026-5033. DOI: <a href="https://doi.org/10.1109/IROS.2012.6386109">10.1109/IROS.2012.6386109</a>.</li>
<li>SHUSTER M D. A Survey of Attitude Representations[J]. The Journal of the Astronautical Sciences, 1993, 41(4): 439-517.</li>
</ol>
<p class="note">复现命令：<code>conda run -n myrobot python reports/analysis/ik_validation.py --timing-repeats 3</code>，随后执行 <code>conda run -n myrobot python reports/analysis/render_ik_validation_report.py</code>。本报告仅新增分析、结果和报告文件，未修改生产 IK 算法。</p>
</main></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="reports/ik_validation/results.json")
    parser.add_argument("--output", default="reports/ik_validation/index.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = (REPO_ROOT / args.results).resolve()
    raw = result_path.read_bytes()
    results = json.loads(raw)
    output_path = (REPO_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_root = REPO_ROOT / results["experiment_config"]["dataset_root"]
    document = render(results, hashlib.sha256(raw).hexdigest(), dataset_root)
    output_path.write_text(document, encoding="utf-8")
    print(f"self-contained report written to {output_path}")


if __name__ == "__main__":
    main()

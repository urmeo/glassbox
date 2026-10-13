"""Regenerate README figures from the offline study and scenario presentations.

The study uses simulated readers and a simulated preference judge. Run from the
repository root::

    python3 docs/make_assets.py

Outputs (all committed):

============================  =========================================================
``img/cards.png``             the three interfaces under test, rendered by the same
``img/table.png``             HTML the readers screenshot (needs Chrome; skipped with a
``img/annotated.png``         warning if it is unavailable)
``img/divergence-*.svg``      preference vs comprehension — the H1 reversal
``img/accuracy-*.svg``        per-reader accuracy on each interface
``img/figures.json``          the numbers behind both charts, for citation
============================  =========================================================

Charts ship as a ``-light`` and a ``-dark`` pair; the README selects between them with
``<picture>`` so they follow the reader's GitHub theme.

Standard library only, like the rest of the package.
"""

from __future__ import annotations

import json
import os
import shutil
from statistics import fmean
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG = os.path.join(HERE, "img")
sys.path.insert(0, os.path.join(ROOT, "src"))

from glassbox import interfaces, render, studies  # noqa: E402
from glassbox.schema import load_all_scenarios  # noqa: E402

# The scenario shown in the README's figures, and the study behind its numbers.
FIGURE_SCENARIO = "loans"
FIGURE_STUDY = "offline_demo"

READER_ORDER = ["simulated:literal", "simulated:diligent", "simulated:careless"]
VARIANT_ORDER = ["cards", "table", "annotated"]

# Series colours, identical in both themes (they carry meaning, so they must not shift).
GOOD = "#2ea043"         # comprehension gained
BAD = "#d1495b"          # comprehension lost
PREF = "#8957e5"         # preference


# --------------------------------------------------------------------------- interfaces


def render_interfaces() -> bool:
    """Screenshot the three interfaces. Return False if Chrome is unavailable."""
    if not render.chrome_available():
        print("! Chrome not found — skipping the interface PNGs (charts still regenerate)")
        return False
    scenario = load_all_scenarios()[FIGURE_SCENARIO]
    with tempfile.TemporaryDirectory(prefix="glassbox_images_") as directory:
        for name in VARIANT_ORDER:
            stimulus = render.render(interfaces.variant(scenario, name), directory)
            png = os.path.join(IMG, name + ".png")
            shutil.copyfile(stimulus.image_path, png)
            print("wrote %s" % os.path.relpath(png, ROOT))
    return True


# ------------------------------------------------------------------------------ numbers


def measure() -> dict:
    """Run the offline demo study and return the figures' numbers."""
    config = studies.load_study(FIGURE_STUDY)
    with tempfile.TemporaryDirectory(prefix="glassbox_figures_") as out:
        studies.run_study(config, out)
        with open(os.path.join(out, "results.json"), "r", encoding="utf-8") as fh:
            results = json.load(fh)

    metrics = results["metrics"]
    scenarios = sorted(metrics)

    aggregate = {}
    for variant in VARIANT_ORDER:
        preference = [row["pref_score"]
                      for s in scenarios
                      for row in metrics[s]["h1"]["rows"] if row["variant"] == variant]
        aggregate[variant] = {
            "lift": fmean(metrics[s]["aggregate"][variant]["mean_lift"] for s in scenarios),
            "accuracy": fmean(metrics[s]["aggregate"][variant]["mean_accuracy"]
                              for s in scenarios),
            "preference": fmean(preference),
        }

    per_reader = {
        reader: {
            variant: fmean(metrics[s]["readers"][reader]["variants"][variant]["accuracy"]
                           for s in scenarios)
            for variant in VARIANT_ORDER
        }
        for reader in READER_ORDER
    }

    return {
        "study": config.id,
        "scenarios": scenarios,
        "readers": READER_ORDER,
        "questions_per_reader": len({
            (row["scenario_id"], row["question_id"])
            for row in results["per_question"]
        }),
        "mean_spearman": results["h1_summary"]["mean_spearman"],
        "simulated": results["h1_summary"]["simulated"],
        "aggregate": aggregate,
        "per_reader_accuracy": per_reader,
    }


# ------------------------------------------------------------------------------- charts

# GitHub honours <picture> + prefers-color-scheme, so each chart ships as a light and a
# dark file with explicit colours — no media query inside the SVG, which would follow the
# operating system rather than the reader's GitHub theme.
THEMES = {
    "light": {"text": "#1f2328", "muted": "#59636e", "track": "#d1d9e0", "rule": "#8b949e"},
    "dark": {"text": "#e6edf3", "muted": "#9198a1", "track": "#30363d", "rule": "#6e7681"},
}


def _svg_open(width: int, height: int, title: str, desc: str) -> list:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" '
        'height="%d" role="img" aria-labelledby="chart-title chart-desc" '
        'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,'
        'sans-serif">' % (width, height, width, height),
        '<title id="chart-title">%s</title>' % title,
        '<desc id="chart-desc">%s</desc>' % desc,
    ]


def _text(x, y, s, fill, size=13, anchor="start", weight=None):
    extra = ' font-weight="%s"' % weight if weight else ""
    return ('<text x="%.1f" y="%.1f" fill="%s" font-size="%d" text-anchor="%s"%s>%s</text>'
            % (x, y, fill, size, anchor, extra, s))


def _rect(x, y, w, h, fill, rx=3):
    return ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%d" fill="%s"/>'
            % (x, y, max(w, 0.0), h, rx, fill))


def _line(x1, y1, x2, y2, stroke, opacity=1.0):
    return ('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1" '
            'opacity="%.2f"/>' % (x1, y1, x2, y2, stroke, opacity))


def divergence_chart(numbers: dict, theme: str) -> str:
    """Show preference scores and accuracy lift for each interface."""
    c = THEMES[theme]
    width, row_h, top = 780, 50, 92
    height = top + row_h * len(VARIANT_ORDER) + 56
    left_x, left_w = 160, 190          # preference panel: 0 -> 1
    right_x, right_w = 440, 280        # lift panel: -100 -> +100 pts, zero in the middle
    zero_x = right_x + right_w / 2.0

    out = _svg_open(width, height, "Preference score and accuracy lift",
                    "Simulated judge scores on the left; simulated reader accuracy "
                    "lift on the right. Cards score higher than the table on preference "
                    "and lower on answer accuracy. These are deterministic fixture results.")
    out.append(_text(20, 28, "PREFERENCE SCORE", c["muted"], size=11, weight="700"))
    out.append(_text(20, 46, "fixed judge weights, 0 → 1", c["muted"], size=11))
    out.append(_text(right_x, 28, "ANSWER ACCURACY LIFT", c["muted"], size=11, weight="700"))
    out.append(_text(right_x, 46, "percentage points vs plain text",
                     c["muted"], size=11))
    out.append(_line(20, 62, width - 20, 62, c["rule"], 0.35))
    out.append(_line(zero_x, top - 18, zero_x, top + row_h * len(VARIANT_ORDER) - 12,
                     c["rule"], 0.55))
    out.append(_text(zero_x, top - 24, "0", c["muted"], size=10, anchor="middle"))

    for i, variant in enumerate(VARIANT_ORDER):
        cell = numbers["aggregate"][variant]
        y = top + i * row_h
        bar_y, bar_h = y - 3, 22
        out.append(_text(20, y + 13, variant, c["text"], size=14, weight="700"))

        out.append(_rect(left_x, bar_y, left_w, bar_h, c["track"]))
        out.append(_rect(left_x, bar_y, left_w * cell["preference"], bar_h, PREF))
        out.append(_text(left_x + left_w + 12, y + 13, "%.2f" % cell["preference"],
                         c["text"], size=13, weight="600"))

        pts = cell["lift"] * 100.0
        span = (right_w / 2.0) * min(abs(pts) / 100.0, 1.0)
        if pts >= 0:
            out.append(_rect(zero_x, bar_y, span, bar_h, GOOD))
            out.append(_text(zero_x + span + 12, y + 13, "%+.0f pts" % pts, c["text"],
                             size=13, weight="600"))
        else:
            out.append(_rect(zero_x - span, bar_y, span, bar_h, BAD))
            out.append(_text(zero_x - span - 12, y + 13, "%+.0f pts" % pts, c["text"],
                             size=13, weight="600", anchor="end"))

    gap = (numbers["aggregate"]["table"]["lift"] -
           numbers["aggregate"]["cards"]["lift"]) * 100.0
    out.append(_line(20, height - 46, width - 20, height - 46, c["rule"], 0.35))
    out.append(_text(20, height - 26, "Cards have a higher preference score than the "
                                      "table, with %.0f points lower answer accuracy." % gap,
                     c["text"], size=12, weight="600"))
    out.append(_text(20, height - 9, "Simulated readers and judge; no model or human study.", c["muted"],
                     size=11))
    out.append("</svg>")
    return "\n".join(out)


def accuracy_chart(numbers: dict, theme: str) -> str:
    """Grouped bars: each reader's accuracy on each interface."""
    c = THEMES[theme]
    width, group_h, top = 780, 100, 96
    height = top + group_h * len(READER_ORDER) + 34
    bar_x, bar_w, bar_h = 230, 380, 19
    colors = {"cards": BAD, "table": GOOD, "annotated": PREF}
    bottom = top + group_h * (len(READER_ORDER) - 1) + 2 * (bar_h + 6) + bar_h - 4

    out = _svg_open(width, height, "Accuracy by reader and interface",
                    "Each simulated reader's accuracy on the cards, table and annotated "
                    "interfaces, averaged over three scenarios. These are deterministic "
                    "fixture results, not measured model or human comprehension.")
    out.append(_text(20, 28, "ACCURACY BY READER", c["muted"], size=11, weight="700"))
    out.append(_text(20, 46, "%d questions per reader and interface across %d scenarios"
                     % (numbers["questions_per_reader"], len(numbers["scenarios"])),
                     c["muted"], size=11))
    out.append(_line(20, 62, width - 20, 62, c["rule"], 0.35))
    for pct in (0, 50, 100):
        x = bar_x + bar_w * pct / 100.0
        out.append(_line(x, top - 14, x, bottom, c["rule"], 0.3))
        out.append(_text(x, top - 20, "%d%%" % pct, c["muted"], size=10, anchor="middle"))

    for r, reader in enumerate(READER_ORDER):
        base = top + r * group_h
        out.append(_text(20, base + 13, reader.split(":")[1], c["text"], size=14,
                         weight="700"))
        out.append(_text(20, base + 30, "simulated", c["muted"], size=10))
        for v, variant in enumerate(VARIANT_ORDER):
            y = base + v * (bar_h + 6) - 4
            accuracy = numbers["per_reader_accuracy"][reader][variant]
            out.append(_text(bar_x - 12, y + 14, variant, c["muted"], size=12,
                             anchor="end"))
            out.append(_rect(bar_x, y, bar_w, bar_h, c["track"]))
            out.append(_rect(bar_x, y, bar_w * accuracy, bar_h, colors[variant]))
            out.append(_text(bar_x + bar_w * accuracy + 12, y + 14,
                             "%.0f%%" % (accuracy * 100), c["text"], size=12,
                             weight="600"))

    out.append(_text(20, height - 10, "Simulated readers; scores follow configured rules.", c["muted"],
                     size=11))
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    os.makedirs(IMG, exist_ok=True)
    render_interfaces()
    numbers = measure()
    for stem, build in (("divergence", divergence_chart), ("accuracy", accuracy_chart)):
        for theme in THEMES:
            path = os.path.join(IMG, "%s-%s.svg" % (stem, theme))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(build(numbers, theme) + "\n")
            print("wrote %s" % os.path.relpath(path, ROOT))
    path = os.path.join(IMG, "figures.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote %s" % os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()

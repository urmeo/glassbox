"""Draw the README pipeline. Run python3 docs/make_pipeline.py from the checkout."""

from pathlib import Path
from xml.etree import ElementTree as ET

WIDTH, HEIGHT = 1680, 1180
COLORS = {
    "background": "#14191c",
    "panel": "#1b2327",
    "rule": "#344046",
    "text": "#e0e7e9",
    "muted": "#a2afb6",
    "cyan": "#4ebbd7",
    "gray": "#aab6bc",
    "coral": "#e18b7e",
}
DESCRIPTION = (
    "GlassBox study: numeric records, formulas, questions and settings feed reference "
    "and test views. Input capture checks numeric precision and copies the presentation. "
    "Simulated readers use shown fields; API readers use PNG or text. Answers are scored "
    "against authored keys and the plain-text baseline. A separate judge scores test "
    "views using fixed polish ratings or both pairwise presentation orders. Judge scores "
    "join answer scores only for rank comparison and saved Markdown/JSON results. "
    "Separate optimize and evaluate commands freeze a feature selection before scoring "
    "disjoint recorded scenarios without another search. Validation, repair and human "
    "accuracy comparison are separate commands. The offline demo and human scores are "
    "synthetic; text runs do not evaluate visual layout."
)


def draw() -> ET.Element:
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg", "width": str(WIDTH), "height": str(HEIGHT),
        "viewBox": "0 0 %d %d" % (WIDTH, HEIGHT), "role": "img",
        "aria-labelledby": "pipeline-title pipeline-desc",
    })
    ET.SubElement(svg, "title", id="pipeline-title").text = "GlassBox pipeline"
    ET.SubElement(svg, "desc", id="pipeline-desc").text = DESCRIPTION
    defs = ET.SubElement(svg, "defs")
    for color in ("cyan", "gray"):
        marker = ET.SubElement(defs, "marker", {
            "id": color + "-arrow", "viewBox": "0 0 10 10", "refX": "9", "refY": "5",
            "markerWidth": "7", "markerHeight": "7", "orient": "auto",
        })
        ET.SubElement(marker, "path", d="M1 1 L9 5 L1 9 Z", fill=COLORS[color])

    def text(x, y, value, size=20, color="muted", bold=False, mono=False):
        node = ET.SubElement(svg, "text", {
            "x": str(x), "y": str(y), "fill": COLORS[color], "font-size": str(size),
            "font-family": "Courier New, monospace" if mono else "Arial, sans-serif",
            "font-weight": "700" if bold else "400",
        })
        node.text = value

    def rectangle(x, y, width, height, stroke="rule", fill="panel"):
        attrs = {
            "x": str(x), "y": str(y), "width": str(width), "height": str(height),
            "rx": "8", "fill": COLORS[fill], "stroke": COLORS[stroke], "stroke-width": "2",
        }
        ET.SubElement(svg, "rect", attrs)

    def path(points, color="cyan", arrow=True):
        attrs = {
            "d": points, "fill": "none", "stroke": COLORS[color], "stroke-width": "2.5",
            "stroke-linecap": "round", "stroke-linejoin": "round",
        }
        if arrow:
            attrs["marker-end"] = "url(#%s-arrow)" % color
        ET.SubElement(svg, "path", attrs)

    rectangle(0, 0, WIDTH, HEIGHT, fill="background", stroke="background")
    text(48, 62, "GLASSBOX PIPELINE", 32, "text", bold=True)
    text(48, 104, "Compare interface ratings with answer accuracy", 25)
    path("M48 126 H1632", "gray", arrow=False)
    rectangle(20, 144, 1640, 1016)
    text(48, 184, "INPUT", 17, "cyan", bold=True)
    text(122, 184, "Numeric records + formulas · questions + answer keys · study settings", 22)

    columns = (48, 374, 700, 1026, 1352)
    titles = ("1  Build views", "2  Capture input", "3  Collect answers",
              "4  Score answers", "5  Compare & save")
    for x, title in zip(columns, titles):
        rectangle(x, 218, 280, 224, stroke="cyan")
        text(x + 22, 258, title, 25, "text", bold=True)
    for x in columns[:-1]:
        path("M%d 330 H%d" % (x + 284, x + 316))

    text(70, 296, "REFERENCES", 14, bold=True)
    text(70, 324, "Raw table · plain text", 20, "text")
    text(70, 366, "DEFAULT TEST VIEWS", 14, bold=True)
    text(70, 394, "Cards · table · annotated", 18, "text")

    for y, value in ((300, "Check numeric display"), (336, "Copy the presentation"),
                     (372, "Text + optional PNG")):
        text(396, y, value, 20, "text")
    text(396, 414, "Keep input hashes", 18, "cyan", mono=True)

    text(722, 300, "Question + choices", 20, "text")
    text(722, 336, "Simulated: shown fields", 19)
    text(722, 372, "API: PNG or text", 20)
    text(722, 414, "Repeat as configured", 18)

    text(1048, 300, "Choices vs. stored keys", 19, "text")
    text(1048, 340, "Answer accuracy", 20)
    text(1048, 378, "Lift vs. plain text", 20)
    text(1048, 414, "Raw-table accuracy check", 17)

    text(1374, 300, "Judge rank vs. lift rank", 19, "text")
    text(1374, 338, "Reversals + correlations", 18)
    text(1374, 382, "report.md", 22, "cyan", mono=True)
    text(1374, 418, "results.json", 22, "cyan", mono=True)

    path("M188 446 V574 H442", "gray")
    text(220, 554, "Test views only", 20, "gray")
    rectangle(456, 496, 670, 160, stroke="gray")
    text(480, 536, "Separate judge", 27, "text", bold=True)
    text(480, 574, "Simulated feature scores or API text judgments", 21)
    text(480, 610, "Pairwise: both orders; invalid / abstained replies excluded", 20)
    text(480, 640, "Pairwise mode: 3 views → 6 judgments", 18, "gray")
    path("M1130 574 H1492 V452", "gray")
    text(1178, 554, "Judge scores + coverage", 20, "gray")

    path("M48 696 H1632", "gray", arrow=False)
    text(48, 734, "FROZEN FEATURE EVALUATION", 20, "text", bold=True)
    text(425, 734, "Separate commands · selection scores remain in sample", 20)

    for x in (48, 456, 864, 1272):
        rectangle(x, 768, 360, 174, stroke="gray")
    for x in (48, 456, 864):
        path("M%d 854 H%d" % (x + 364, x + 398), "gray")

    text(70, 810, "optimize", 26, "text", bold=True, mono=True)
    for y, value in ((850, "Search feature subsets"), (884, "Use selection scenarios"),
                     (918, "Report in-sample lift")):
        text(70, y, value, 20)
    text(478, 810, "selection.json", 26, "text", bold=True, mono=True)
    for y, value in ((850, "Freeze chosen features"), (884, "Save readers + settings"),
                     (918, "Record task hashes")):
        text(478, y, value, 20)
    text(886, 810, "evaluate", 26, "text", bold=True, mono=True)
    for y, value in ((850, "Score frozen features"), (884, "Use disjoint scenarios"),
                     (918, "No new search")):
        text(886, y, value, 20)
    text(1294, 810, "Evaluation outputs", 26, "text", bold=True)
    text(1294, 848, "evaluation.md", 21, "cyan", mono=True)
    text(1294, 880, "evaluation.json", 21, "cyan", mono=True)
    text(1294, 916, "Accuracy + lift vs. baseline", 19)

    text(48, 980, "OVERLAP CHECK", 16, "coral", bold=True)
    text(210, 980, "Reject reused scenario IDs, task content and source data before evaluation", 21, "coral")
    path("M48 1012 H1632", "gray", arrow=False)

    for x, command, caption in (
        (48, "validate", "Recompute source answer keys"),
        (592, "repair", "Revise one question’s view"),
        (1136, "anchor", "Compare supplied human accuracy"),
    ):
        text(x, 1048, command, 23, "coral" if command == "validate" else "text", mono=True)
        text(x, 1078, caption, 20)
    text(48, 1120, "Offline demo: 3 synthetic scenarios · 9 questions · simulated readers and judge.", 19)
    text(48, 1148, "Text runs do not test visual layout. Shipped human scores are synthetic; held-out fixtures do not establish generalization.", 18)
    return svg


def main():
    destination = Path(__file__).resolve().parent / "img" / "pipeline.svg"
    svg = draw()
    ET.indent(svg, space="  ")
    destination.write_text(ET.tostring(svg, encoding="unicode") + "\n", encoding="utf-8")
    print("wrote %s" % destination)


if __name__ == "__main__":
    main()

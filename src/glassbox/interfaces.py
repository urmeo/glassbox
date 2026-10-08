"""Build escaped, ordered presentations and optional question-specific repair views."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

from . import compute
from .schema import Question, Scenario, raw_field_names, validate_structure

VARIANT_FEATURES: Dict[str, FrozenSet[str]] = {
    "raw": frozenset({"all_raw"}),
    "baseline": frozenset({"all_raw", "as_text"}),
    "cards": frozenset({"headline", "cards_context", "polished"}),
    "table": frozenset({"detail", "derived"}),
    "annotated": frozenset({"detail", "derived", "highlight", "sorted", "polished"}),
}

REFERENCE_VARIANTS = ("raw", "baseline")
INTERFACE_VARIANTS = ("cards", "table", "annotated")
ALL_VARIANTS = REFERENCE_VARIANTS + INTERFACE_VARIANTS


@dataclass(frozen=True)
class Presentation:
    """What a concrete interface visibly presents for one scenario."""

    scenario_id: str
    variant: str
    features: FrozenSet[str]
    order: List[str]
    shown: Dict[str, Dict[str, Any]]
    emphasized_field: Optional[str]
    highlight_item: Optional[str]
    primary_metric: str
    primary_extreme: str
    unit: str = ""
    value_label: str = ""
    title: str = ""
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    field_units: Dict[str, str] = field(default_factory=dict)
    field_labels: Dict[str, str] = field(default_factory=dict)
    highlight_label: str = "best"
    target_question_id: Optional[str] = None
    target_operation: Optional[str] = None
    target_stem: str = ""

    def unit_for(self, name: str) -> str:
        return self.field_units.get(name, self.unit)

    @property
    def polished(self) -> bool:
        return "polished" in self.features

    def shows_field(self, item_id: str, field_name: str) -> bool:
        return field_name in self.shown.get(item_id, {})

    def to_html(self) -> str:
        return render_html(self)

    def to_text(self) -> str:
        return render_text(self)




def _num(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, int):
        return "{:,}".format(v)
    if isinstance(v, float):
        return "{:,.2f}".format(v).rstrip("0").rstrip(".")
    return str(v)


def format_value(v: Any, unit: str) -> str:
    if unit == "$":
        return "$" + _num(v)
    if unit == "$k":
        return "$" + _num(v) + "k"
    if unit:
        return _num(v) + " " + unit
    return _num(v)




def build(
    scenario: Scenario,
    features: FrozenSet[str],
    variant: str = "custom",
    target: Optional[Question] = None,
) -> Presentation:
    validate_structure(scenario)
    if target is not None and (
        not isinstance(target, Question) or target not in scenario.questions
    ):
        raise ValueError("target must be a question from this scenario")
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError("presentation variant must be nonempty text")
    if not isinstance(features, (set, frozenset)) or any(not isinstance(f, str) for f in features):
        raise ValueError("presentation features must be a set of strings")
    pres = scenario.presentation
    metric = target.target_field if target is not None else pres.primary_metric
    op = target.compute["op"] if target is not None else pres.primary_extreme
    direction = op if op in ("argmin", "argmax") else "argmin"
    if op == "rank":
        direction = "argmax" if target.compute.get("order", "asc") == "desc" else "argmin"
    details = pres.detail_fields
    if target is not None:
        inputs = (
            compute.expr_fields(scenario.derived[metric])
            if metric in scenario.derived
            else {metric}
        )
        details = [name for name in pres.detail_fields if name in inputs]
        details += sorted(inputs - set(details))
    visible: List[str] = []

    def add_fields(names):
        for name in names:
            if name not in visible:
                visible.append(name)

    if "all_raw" in features:
        add_fields(sorted(raw_field_names(scenario.items)))
    if "detail" in features:
        add_fields(details)
    if "cards_context" in features:
        add_fields(pres.cards_fields)
    if "headline" in features and pres.headline_field:
        add_fields([pres.headline_field])
    show_metric = "derived" in features
    values = dict(
        zip(
            (item["id"] for item in scenario.items),
            compute._numeric(scenario.items, metric, scenario.derived),
        )
    )
    shown: Dict[str, Dict[str, Any]] = {}
    for item in scenario.items:
        row = {name: item[name] for name in visible if name in item}
        if show_metric:
            row[metric] = values[item["id"]]
        shown[item["id"]] = row
    if show_metric:
        emphasized = metric
    elif ("headline" in features or "all_raw" in features) and pres.headline_field:
        emphasized = pres.headline_field
    else:
        emphasized = None
    highlight_item, highlight_label = None, "best"
    if "highlight" in features and op in ("argmin", "argmax", "rank"):
        if target is not None:
            highlight_item = compute.compute_answer_value(scenario.data, target.compute)
        else:
            chooser = min if direction == "argmin" else max
            highlight_item = chooser(values, key=lambda iid: values[iid])
        if op == "rank":
            highlight_label = "rank %d" % target.compute["k"]
    order = [item["id"] for item in scenario.items]
    if "sorted" in features:
        order = sorted(order, key=lambda iid: values[iid], reverse=direction == "argmax")
    unit = scenario.data.get("unit", "") if metric == pres.primary_metric else ""
    label = (
        scenario.data.get("value_label", "")
        if metric == pres.primary_metric
        else _field_label(metric)
    )
    display = scenario.data.get("presentation", {}).get("field_display", {})
    units = {metric: unit}
    if pres.headline_field and pres.headline_field != metric:
        units[pres.headline_field] = scenario.data.get("unit", "")
    labels = {metric: label}
    for name, hints in display.items():
        units[name] = hints.get("unit", units.get(name, ""))
        labels[name] = hints.get("value_label", _field_label(name))
    unit, label = units[metric], labels[metric]
    return Presentation(
        scenario_id=scenario.id,
        variant=variant,
        features=frozenset(features),
        order=order,
        shown=shown,
        emphasized_field=emphasized,
        highlight_item=highlight_item,
        primary_metric=metric,
        primary_extreme=direction,
        unit=unit,
        value_label=label,
        title=scenario.title,
        description=scenario.description,
        labels={item["id"]: item["label"] for item in scenario.items},
        field_units=units,
        field_labels=labels,
        highlight_label=highlight_label,
        target_question_id=target.id if target is not None else None,
        target_operation=op if target is not None else None,
        target_stem=target.stem if target is not None else "",
    )


def variant(scenario: Scenario, name: str, target: Optional[Question] = None) -> Presentation:
    if name not in VARIANT_FEATURES:
        raise KeyError("unknown interface variant %r" % name)
    return build(scenario, VARIANT_FEATURES[name], variant=name, target=target)


def with_features(
    scenario: Scenario,
    base: Presentation,
    added: FrozenSet[str],
    new_name: Optional[str] = None,
    target: Optional[Question] = None,
) -> Presentation:
    if target is None and base.target_question_id is not None:
        target = next((q for q in scenario.questions if q.id == base.target_question_id), None)
        if target is None:
            raise ValueError("presentation target question no longer exists")
    combined = frozenset(base.features | added)
    name = new_name or (base.variant + "+" + "+".join(sorted(added)))
    return build(scenario, combined, variant=name, target=target)



_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 28px; background: #ffffff; color: #16181d; width: 760px; }
h1 { font-size: 22px; margin: 0 0 4px; }
p.desc { color: #55606e; margin: 0 0 20px; font-size: 14px; max-width: 60ch; }
table { border-collapse: collapse; table-layout: fixed; width: 100%; font-size: 15px; }
th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid #e6e8ec; }
th { font-weight: 600; color: #55606e; font-size: 13px; text-transform: uppercase; letter-spacing: .03em;
     white-space: normal; overflow-wrap: break-word; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.cards { display: flex; gap: 16px; flex-wrap: wrap; }
.card { flex: 1 1 210px; border: 1px solid #e6e8ec; border-radius: 14px; padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(16,24,40,.06); background: #fbfcfe; }
.card h2 { font-size: 16px; margin: 0 0 12px; }
.card .headline { font-size: 26px; font-weight: 700; margin: 0 0 4px; }
.card .headline-label { font-size: 12px; color: #55606e; text-transform: uppercase; letter-spacing: .03em; }
.card ul { list-style: none; padding: 0; margin: 14px 0 0; font-size: 13px; color: #55606e; }
.card li { display: flex; justify-content: space-between; padding: 2px 0; }
.mark { outline: 2px solid #1f7a45; background: #eafaf0; }
.badge { display: inline-block; margin-left: 8px; font-size: 11px; font-weight: 600; color: #1f7a45; }
pre.plain { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 14px;
            white-space: pre-wrap; line-height: 1.5; color: #16181d; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
@media (prefers-color-scheme: dark) {
  body { background: #16181d; color: #e6e8ec; }
  p.desc, th, .card .headline-label, .card ul { color: #9aa4b2; }
  th, td { border-bottom-color: #2a2e37; }
  .card { background: #1c1f26; border-color: #2a2e37; }
  .mark { background: #133024; outline-color: #3ecf8e; }
  .badge { color: #3ecf8e; }
}
"""


def _doc(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=760">'
        "<title>" + html.escape(title) + "</title><style>" + _CSS + "</style></head>"
        "<body>" + body + "</body></html>"
    )


def _field_label(name: str) -> str:
    return name.replace("_", " ")


def render_html(p: Presentation) -> str:
    """Render a presentation to a self-contained HTML document."""
    head = "<h1>" + html.escape(p.title) + "</h1>"
    desc = '<p class="desc">' + html.escape(p.description) + "</p>"
    if p.target_stem:
        desc += '<p class="desc">Target: ' + html.escape(p.target_stem) + "</p>"

    if "as_text" in p.features:
        return _doc(
            p.title,
            head
            + desc
            + '<pre class="plain">'
            + html.escape(render_text(p, include_header=False))
            + "</pre>",
        )

    if "polished" in p.features and "detail" not in p.features and "all_raw" not in p.features:
        return _doc(p.title, head + desc + _render_cards(p))

    if "polished" in p.features:
        return _doc(p.title, head + desc + _render_cards(p, show_metric=True))

    return _doc(p.title, head + desc + _render_table(p))


def _render_cards(p: Presentation, show_metric: bool = False) -> str:
    cards = []
    for iid in p.order:
        row = p.shown[iid]
        marked = " mark" if iid == p.highlight_item else ""
        badge = (
            '<span class="badge">' + html.escape(p.highlight_label) + "</span>"
            if iid == p.highlight_item
            else ""
        )
        headline_html = ""
        if p.emphasized_field and p.emphasized_field in row:
            label = (
                p.value_label
                if p.emphasized_field == p.primary_metric
                else p.field_labels.get(p.emphasized_field, _field_label(p.emphasized_field))
            )
            headline_html = (
                '<div class="headline">'
                + html.escape(format_value(row[p.emphasized_field], p.unit_for(p.emphasized_field)))
                + "</div>"
                '<div class="headline-label">' + html.escape(label) + "</div>"
            )
        lis = []
        for fname, val in row.items():
            if fname == p.emphasized_field:
                continue
            lis.append(
                "<li><span>"
                + html.escape(_field_label(fname))
                + "</span><span>"
                + html.escape(_num(val))
                + "</span></li>"
            )
        cards.append(
            '<div class="card'
            + marked
            + '"><h2>'
            + html.escape(p.labels.get(iid, iid))
            + badge
            + "</h2>"
            + headline_html
            + "<ul>"
            + "".join(lis)
            + "</ul></div>"
        )
    return '<div class="cards">' + "".join(cards) + "</div>"


def _render_table(p: Presentation) -> str:
    raw_cols: List[str] = []
    for iid in p.order:
        for f in p.shown[iid]:
            if f != p.primary_metric and f not in raw_cols:
                raw_cols.append(f)
    metric_shown = any(p.primary_metric in p.shown[iid] for iid in p.order)
    header = "<tr><th>option</th>" + "".join(
        "<th>" + html.escape(_field_label(c)) + "</th>" for c in raw_cols
    )
    if metric_shown:
        header += "<th>" + html.escape(p.value_label or _field_label(p.primary_metric)) + "</th>"
    header += "</tr>"

    rows = []
    for iid in p.order:
        row = p.shown[iid]
        marked = ' class="mark"' if iid == p.highlight_item else ""
        cells = "<td>" + html.escape(p.labels.get(iid, iid)) + "</td>"
        for c in raw_cols:
            cells += '<td class="num">' + (html.escape(_num(row[c])) if c in row else "") + "</td>"
        if metric_shown:
            mval = row.get(p.primary_metric)
            cells += (
                '<td class="num">'
                + (
                    html.escape(format_value(mval, p.unit_for(p.primary_metric)))
                    if mval is not None
                    else ""
                )
                + "</td>"
            )
        rows.append("<tr" + marked + ">" + cells + "</tr>")
    return "<table>" + header + "".join(rows) + "</table>"


def render_text(p: Presentation, include_header: bool = True) -> str:
    """Serialize the actual ordered visible values as plain text."""
    lines: List[str] = []
    if include_header:
        lines.append(p.title)
        lines.append(p.description)
        if p.target_stem:
            lines.append("Target: " + p.target_stem)
        lines.append("")
    for iid in p.order:
        row = p.shown[iid]
        parts = []
        for fname, val in row.items():
            if fname == p.primary_metric:
                parts.append(
                    "%s = %s"
                    % (p.value_label or _field_label(fname), format_value(val, p.unit_for(fname)))
                )
            elif fname == p.emphasized_field:
                parts.append("%s: %s" % (_field_label(fname), format_value(val, p.unit_for(fname))))
            else:
                parts.append("%s %s" % (_field_label(fname), _num(val)))
        marker = ("  <- highlighted as " + p.highlight_label) if iid == p.highlight_item else ""
        lines.append("- %s: %s%s" % (p.labels.get(iid, iid), ", ".join(parts), marker))
    return "\n".join(lines)

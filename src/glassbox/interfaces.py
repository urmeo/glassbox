"""Interface variants — how one scenario's data is turned into something a reader sees.

An interface is a **feature set** applied to a scenario. The named variants are
presets over a small vocabulary of flags:

    all_raw       show every raw field (the reference conditions)
    detail        show the raw inputs of the primary metric
    derived       compute and show the primary metric itself (the honest number)
    headline      emphasize the scenario's headline field (which can mislead)
    cards_context show the polished card's few context fields
    highlight     mark the argmin/argmax of the primary metric
    sorted        order items by the primary metric
    polished / as_text   rendering hints (affect HTML + the preference judge only)

The same vocabulary drives the H4 repair loop: repairing a ``cards`` interface just
*adds* flags (``detail``, ``derived``, ``highlight`` …). One `Presentation` object is
the single source of truth for the rendered HTML (what a real VLM reader screenshots)
and the structured "shown values" (what a deterministic simulated reader reads).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

from . import compute
from .schema import Scenario, raw_field_names

# Preset feature sets for the named variants.
VARIANT_FEATURES: Dict[str, FrozenSet[str]] = {
    "raw": frozenset({"all_raw"}),
    "baseline": frozenset({"all_raw", "as_text"}),
    "cards": frozenset({"headline", "cards_context", "polished"}),
    "table": frozenset({"detail", "derived"}),
    "annotated": frozenset({"detail", "derived", "highlight", "sorted", "polished"}),
}

# The two reference conditions and the interfaces under test (H1 comparison set).
REFERENCE_VARIANTS = ("raw", "baseline")
INTERFACE_VARIANTS = ("cards", "table", "annotated")
ALL_VARIANTS = REFERENCE_VARIANTS + INTERFACE_VARIANTS


@dataclass(frozen=True)
class Presentation:
    """What a concrete interface visibly presents for one scenario."""
    scenario_id: str
    variant: str
    features: FrozenSet[str]
    order: List[str]                       # item ids in display order
    shown: Dict[str, Dict[str, Any]]       # item id -> {field: value} the reader can read
    emphasized_field: Optional[str]        # the value the interface presents as "the answer"
    highlight_item: Optional[str]          # item id the interface visually marks, or None
    primary_metric: str
    primary_extreme: str
    unit: str = ""
    value_label: str = ""
    title: str = ""
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)

    @property
    def polished(self) -> bool:
        return "polished" in self.features

    def shows_field(self, item_id: str, field_name: str) -> bool:
        return field_name in self.shown.get(item_id, {})

    def to_html(self) -> str:
        return render_html(self)

    def to_text(self) -> str:
        return render_text(self)


# --- value formatting -------------------------------------------------------

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


# --- building a presentation from a feature set -----------------------------

def _primary_value(scenario: Scenario, item: Dict[str, Any]) -> Any:
    return compute.resolve_field(item, scenario.presentation.primary_metric, scenario.derived)


def build(scenario: Scenario, features: FrozenSet[str], variant: str = "custom") -> Presentation:
    """Build a :class:`Presentation` for ``scenario`` from a feature set."""
    pres = scenario.presentation
    metric = pres.primary_metric
    all_raw = sorted(raw_field_names(scenario.items))

    # Which raw fields are visible.
    visible: set = set()
    if "all_raw" in features:
        visible |= set(all_raw)
    if "detail" in features:
        visible |= set(pres.detail_fields)
    if "cards_context" in features:
        visible |= set(pres.cards_fields)
    if "headline" in features and pres.headline_field:
        visible.add(pres.headline_field)

    show_derived = "derived" in features

    shown: Dict[str, Dict[str, Any]] = {}
    for it in scenario.items:
        row: Dict[str, Any] = {f: it[f] for f in visible if f in it}
        if show_derived:
            row[metric] = _primary_value(scenario, it)
        shown[it["id"]] = row

    # What the interface presents as "the answer value" (reader fallback target).
    if show_derived:
        emphasized: Optional[str] = metric
    elif "headline" in features and pres.headline_field:
        emphasized = pres.headline_field
    elif "all_raw" in features and pres.headline_field:
        emphasized = pres.headline_field
    else:
        emphasized = None

    # Highlight and sort by the primary metric.
    values = {it["id"]: _primary_value(scenario, it) for it in scenario.items}
    extreme_item: Optional[str] = None
    if values:
        chooser = min if pres.primary_extreme == "argmin" else max
        extreme_item = chooser(values, key=lambda k: values[k])
    highlight_item = extreme_item if "highlight" in features else None

    order = [it["id"] for it in scenario.items]
    if "sorted" in features:
        reverse = pres.primary_extreme == "argmax"
        order = sorted(order, key=lambda k: values[k], reverse=reverse)

    return Presentation(
        scenario_id=scenario.id,
        variant=variant,
        features=frozenset(features),
        order=order,
        shown=shown,
        emphasized_field=emphasized,
        highlight_item=highlight_item,
        primary_metric=metric,
        primary_extreme=pres.primary_extreme,
        unit=scenario.data.get("unit", ""),
        value_label=scenario.data.get("value_label", ""),
        title=scenario.title,
        description=scenario.description,
        labels={it["id"]: it["label"] for it in scenario.items},
    )


def variant(scenario: Scenario, name: str) -> Presentation:
    """Build one of the named preset variants."""
    if name not in VARIANT_FEATURES:
        raise KeyError("unknown interface variant %r" % name)
    return build(scenario, VARIANT_FEATURES[name], variant=name)


def with_features(scenario: Scenario, base: Presentation,
                  added: FrozenSet[str], new_name: Optional[str] = None) -> Presentation:
    """Return a new presentation that layers ``added`` flags onto ``base`` (repair)."""
    combined = frozenset(base.features | added)
    return build(scenario, combined, variant=new_name or (base.variant + "+" + "+".join(sorted(added))))


# --- rendering --------------------------------------------------------------

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 28px; background: #ffffff; color: #16181d; width: 760px; }
h1 { font-size: 22px; margin: 0 0 4px; }
p.desc { color: #55606e; margin: 0 0 20px; font-size: 14px; max-width: 60ch; }
table { border-collapse: collapse; width: 100%; font-size: 15px; }
th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid #e6e8ec; }
th { font-weight: 600; color: #55606e; font-size: 13px; text-transform: uppercase; letter-spacing: .03em; }
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
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=760\">"
        "<title>" + html.escape(title) + "</title><style>" + _CSS + "</style></head>"
        "<body>" + body + "</body></html>"
    )


def _field_label(name: str) -> str:
    return name.replace("_", " ")


def render_html(p: Presentation) -> str:
    """Render a presentation to a self-contained HTML document."""
    head = "<h1>" + html.escape(p.title) + "</h1>"
    desc = "<p class=\"desc\">" + html.escape(p.description) + "</p>"

    if "as_text" in p.features:
        return _doc(p.title, head + desc + "<pre class=\"plain\">"
                    + html.escape(render_text(p, include_header=False)) + "</pre>")

    if "polished" in p.features and "detail" not in p.features and "all_raw" not in p.features:
        return _doc(p.title, head + desc + _render_cards(p))

    if "polished" in p.features:  # annotated: polished cards WITH the computed metric
        return _doc(p.title, head + desc + _render_cards(p, show_metric=True))

    return _doc(p.title, head + desc + _render_table(p))


def _render_cards(p: Presentation, show_metric: bool = False) -> str:
    cards = []
    for iid in p.order:
        row = p.shown[iid]
        marked = " mark" if iid == p.highlight_item else ""
        badge = "<span class=\"badge\">best</span>" if iid == p.highlight_item else ""
        headline_html = ""
        if p.emphasized_field and p.emphasized_field in row:
            label = p.value_label if p.emphasized_field == p.primary_metric else _field_label(p.emphasized_field)
            headline_html = (
                "<div class=\"headline\">" + html.escape(format_value(row[p.emphasized_field], p.unit)) + "</div>"
                "<div class=\"headline-label\">" + html.escape(label) + "</div>"
            )
        lis = []
        for fname, val in row.items():
            if fname == p.emphasized_field:
                continue
            lis.append("<li><span>" + html.escape(_field_label(fname)) + "</span><span>"
                       + html.escape(_num(val)) + "</span></li>")
        cards.append(
            "<div class=\"card" + marked + "\"><h2>" + html.escape(p.labels.get(iid, iid))
            + badge + "</h2>" + headline_html + "<ul>" + "".join(lis) + "</ul></div>"
        )
    return "<div class=\"cards\">" + "".join(cards) + "</div>"


def _render_table(p: Presentation) -> str:
    # Column order: label, then shown raw fields (stable), then the metric last if shown.
    raw_cols: List[str] = []
    for iid in p.order:
        for f in p.shown[iid]:
            if f != p.primary_metric and f not in raw_cols:
                raw_cols.append(f)
    metric_shown = any(p.primary_metric in p.shown[iid] for iid in p.order)
    header = "<tr><th>option</th>" + "".join("<th>" + html.escape(_field_label(c)) + "</th>" for c in raw_cols)
    if metric_shown:
        header += "<th>" + html.escape(p.value_label or _field_label(p.primary_metric)) + "</th>"
    header += "</tr>"

    rows = []
    for iid in p.order:
        row = p.shown[iid]
        marked = " class=\"mark\"" if iid == p.highlight_item else ""
        cells = "<td>" + html.escape(p.labels.get(iid, iid)) + "</td>"
        for c in raw_cols:
            cells += "<td class=\"num\">" + (html.escape(_num(row[c])) if c in row else "") + "</td>"
        if metric_shown:
            mval = row.get(p.primary_metric)
            cells += "<td class=\"num\">" + (html.escape(format_value(mval, p.unit)) if mval is not None else "") + "</td>"
        rows.append("<tr" + marked + ">" + cells + "</tr>")
    return "<table>" + header + "".join(rows) + "</table>"


def render_text(p: Presentation, include_header: bool = True) -> str:
    """Plain-text serialization — the plain-text baseline and the skip-render view."""
    lines: List[str] = []
    if include_header:
        lines.append(p.title)
        lines.append(p.description)
        lines.append("")
    for iid in p.order:
        row = p.shown[iid]
        parts = []
        for fname, val in row.items():
            if fname == p.primary_metric:
                parts.append("%s = %s" % (p.value_label or _field_label(fname), format_value(val, p.unit)))
            elif fname == p.emphasized_field:
                parts.append("%s: %s" % (_field_label(fname), format_value(val, p.unit)))
            else:
                parts.append("%s %s" % (_field_label(fname), _num(val)))
        marker = "  <- highlighted as best" if iid == p.highlight_item else ""
        lines.append("- %s: %s%s" % (p.labels.get(iid, iid), ", ".join(parts), marker))
    return "\n".join(lines)

"""Visible numeric precision and fallback labels in text and HTML."""

import math
import re
import unittest
from dataclasses import replace
from html.parser import HTMLParser
from unittest.mock import patch

from glassbox import interfaces, schema


NUMBER = re.compile(r"[+-]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.numeric = []
        self.plain = []
        self.visible = []
        self.capture = None
        self.depth = 0
        self.body = False

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "body":
            self.body = True
        if self.depth:
            self.depth += 1
        elif (tag == "td" and attributes.get("class") == "num"
              or tag == "div" and attributes.get("class") == "headline"):
            self.capture, self.depth = self.numeric, 1
        elif tag == "pre":
            self.capture, self.depth = self.plain, 1

    def handle_endtag(self, tag):
        if self.depth:
            self.depth -= 1
            if not self.depth:
                self.capture = None
        if tag == "body":
            self.body = False

    def handle_data(self, data):
        if self.capture is not None:
            self.capture.append(data)
        if self.body:
            self.visible.append(data)


def parsed_number(text):
    match = NUMBER.search(text)
    if match is None:
        raise AssertionError("no visible numeric token in %r" % text)
    token = match.group().replace(",", "")
    return float(token) if "." in token or "e" in token.lower() else int(token)


def scenario(values=(0.001, 0.002), unit="", label=None, field_label=None):
    data = {
        "unit": unit,
        "items": [{"id": "A", "label": "Alpha", "measured_value": values[0]},
                  {"id": "B", "label": "Beta", "measured_value": values[1]}],
        "presentation": {"primary_metric": "measured_value", "primary_extreme": "argmin",
                         "headline_field": "measured_value", "detail_fields": ["measured_value"]},
    }
    if label is not None:
        data["value_label"] = label
    if field_label is not None:
        data["presentation"]["field_display"] = {
            "measured_value": {"value_label": field_label}
        }
    return schema.parse_scenario({
        "id": "display", "title": "Measured values", "description": "Compare the readings.",
        "domain": "test", "data": data,
        "questions": [{"id": "minimum", "stem": "Which reading is smaller?",
                       "compute": {"op": "argmin", "field": "measured_value"},
                       "choices": [{"id": "a", "text": "Alpha", "value": "A"},
                                   {"id": "b", "text": "Beta", "value": "B"}],
                       "answer": "a" if values[0] <= values[1] else "b"}],
    })


class TestDisplayPrecision(unittest.TestCase):
    def test_visible_text_and_html_values_round_trip_across_units_and_layouts(self):
        pairs = [
            (0.001, 0.002), (1e-200, 2e-200), (5e-324, 1e-323),
            (-0.001, -0.002), (1.0, math.nextafter(1.0, math.inf)),
            (1e16, math.nextafter(1e16, math.inf)), (1e308, math.nextafter(1e308, math.inf)),
            (2**60, 2**60 + 1), (-0.0, 0.0),
        ]
        for values in pairs:
            for unit in ("", "$", "$k", "%", "ms"):
                source = scenario(values, unit)
                for variant in interfaces.ALL_VARIANTS:
                    with self.subTest(values=values, unit=unit, variant=variant):
                        presentation = interfaces.variant(source, variant)
                        interfaces.validate_display_values(presentation)
                        text_tokens = [line.split(" = ", 1)[1] for line in
                                       presentation.to_text().splitlines() if line.startswith("- ")]
                        html = VisibleHTML()
                        html.feed(presentation.to_html())
                        html_tokens = html.numeric or [line.split(" = ", 1)[1] for line in
                                                      "".join(html.plain).splitlines()
                                                      if line.startswith("- ")]
                        expected = [source.item(item_id)["measured_value"]
                                    for item_id in presentation.order]
                        for tokens in (text_tokens, html_tokens):
                            actual = [parsed_number(token) for token in tokens]
                            self.assertEqual(actual, expected)
                            for before, after in zip(expected, actual):
                                if isinstance(before, float) and before == 0:
                                    self.assertEqual(math.copysign(1, after), math.copysign(1, before))

    def test_integer_fixture_appearance_is_preserved(self):
        self.assertEqual(interfaces.format_value(305.0, "$"), "$305")
        self.assertEqual(interfaces.format_value(11280.0, "$"), "$11,280")
        self.assertEqual(interfaces.format_value(2**60 + 1, ""), "1,152,921,504,606,846,977")
        loans = schema.load_all_scenarios()["loans"]
        presentation = interfaces.variant(loans, "table")
        html = VisibleHTML()
        html.feed(presentation.to_html())
        self.assertEqual(html.numeric, ["305", "36", "300", "$11,280",
                                        "299", "36", "150", "$10,914",
                                        "289", "42", "0", "$12,138"])

    def test_validator_rejects_distinct_but_inaccurate_numeric_tokens(self):
        presentation = interfaces.variant(scenario(), "table")
        with patch("glassbox.interfaces._num", side_effect=lambda value: str(value * 1000)):
            with self.assertRaisesRegex(ValueError, "numeric display"):
                interfaces.validate_display_values(presentation)

    def test_validator_checks_the_formatted_value_including_units(self):
        presentation = interfaces.variant(scenario(unit="$k"), "table")
        with patch("glassbox.interfaces.format_value", return_value="$0k"):
            with self.assertRaisesRegex(ValueError, "numeric display"):
                interfaces.validate_display_values(presentation)

    def test_nonfinite_visible_values_reject(self):
        presentation = interfaces.variant(scenario(), "table")
        for value in (float("nan"), float("inf"), float("-inf")):
            shown = {item: dict(row) for item, row in presentation.shown.items()}
            shown["A"]["measured_value"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "numeric display"):
                interfaces.validate_display_values(replace(presentation, shown=shown))

    def test_validator_checks_table_context_units_as_well_as_text_emphasis(self):
        presentation = interfaces.variant(scenario(unit="$"), "table")
        shown = {item_id: dict(row, advertised_value=100)
                 for item_id, row in presentation.shown.items()}
        presentation = replace(presentation, shown=shown, emphasized_field="advertised_value")
        original = interfaces.format_value
        with patch("glassbox.interfaces.format_value", side_effect=lambda value, unit:
                   "0" if unit == "" else original(value, unit)):
            self.assertIn("advertised value: $100", presentation.to_text())
            html = VisibleHTML()
            html.feed(presentation.to_html())
            self.assertEqual(html.numeric[0], "0")
            with self.assertRaisesRegex(ValueError, "numeric display"):
                interfaces.validate_display_values(presentation)

    def test_intentionally_hidden_task_operands_are_allowed(self):
        source = scenario()
        data = dict(source.data)
        data["items"] = [dict(item, advertised_value=100 + index)
                         for index, item in enumerate(source.items)]
        data["presentation"] = dict(data["presentation"], headline_field="advertised_value")
        presentation = interfaces.variant(replace(source, data=data), "cards")
        self.assertTrue(all("measured_value" not in row for row in presentation.shown.values()))
        interfaces.validate_display_values(presentation)


class TestDisplayLabels(unittest.TestCase):
    def test_omitted_and_blank_labels_fall_back_in_every_layout(self):
        for label in (None, "", " \t "):
            for field_label in (None, "", " \t "):
                for variant in interfaces.ALL_VARIANTS:
                    with self.subTest(label=label, field_label=field_label, variant=variant):
                        presentation = interfaces.variant(scenario(label=label, field_label=field_label),
                                                          variant)
                        self.assertIn("measured value = 0.001", presentation.to_text())
                        html = VisibleHTML()
                        html.feed(presentation.to_html())
                        self.assertIn("measured value", " ".join(html.visible))
                        self.assertNotIn(":  =", presentation.to_text())

    def test_explicit_labels_are_retained_and_escaped_in_html(self):
        for variant in interfaces.ALL_VARIANTS:
            presentation = interfaces.variant(scenario(label="Reading", field_label="Signal <rate>"),
                                              variant)
            self.assertIn("Signal <rate> = 0.001", presentation.to_text())
            html = VisibleHTML()
            html.feed(presentation.to_html())
            self.assertIn("Signal <rate>", " ".join(html.visible))
            self.assertIn("Signal &lt;rate&gt;", presentation.to_html())

    def test_direct_presentation_blank_labels_use_normal_field_name(self):
        presentation = interfaces.variant(scenario(), "annotated")
        presentation = replace(presentation, value_label="  ", field_labels={"measured_value": ""})
        self.assertIn("measured value = 0.001", presentation.to_text())
        self.assertIn('class="headline-label">measured value<', presentation.to_html())


if __name__ == "__main__":
    unittest.main()

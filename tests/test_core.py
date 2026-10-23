import copy
import math
import re
import unittest
from dataclasses import replace
from fractions import Fraction
from html.parser import HTMLParser
from unittest.mock import patch

from glassbox import core, evaluation
from glassbox.core import (
    Answer,
    Reader,
    ResourceError,
    build_reader,
    build_readers,
    decode_yaml,
    resource_names,
    resource_text,
    validate_scenario,
)
from glassbox.evaluation import Judge, SimulatedJudge


class TestExpressions(unittest.TestCase):
    def test_arithmetic_and_fields(self):
        fields = {"monthly_payment": 305.0, "term_months": 36, "fees": 300}
        for expression, expected in (
            ("2 + 3 * 4", 14), ("(2 + 3) * 4", 20),
            ("10 / 4", 2.5), ("-5 + 2", -3),
            ("monthly_payment * term_months + fees", 305.0 * 36 + 300),
        ):
            with self.subTest(expression=expression):
                self.assertEqual(core.evaluate_expr(expression, fields), expected)

    def test_unsupported_expressions(self):
        for expression, fields in (
            ("2 ** 3", {}), ("x / 0", {"x": 5}),
            ("mystery + 1", {"known": 1}), ("flag + 1", {"flag": True}),
            ("__import__('os').system('echo hi')", {}), ("().__class__", {}),
            ("open('x')", {}), ("[i for i in range(3)]", {}),
            ("value.attr", {"value": 1}), ("1 if True else 2", {}),
            ("1e309", {}), ("x + 1", {"x": float("nan")}),
            ("x * 2", {"x": 1e308}), ("x", {"x": True}),
            ("x", {"x": 1 << 2000}), ("1" * 10001, {}),
        ):
            with self.subTest(expression=expression), self.assertRaises(core.ExpressionError):
                core.evaluate_expr(expression, fields)


class TestComputeOps(unittest.TestCase):
    def test_operations(self):
        data = {
            "items": [{"id": "A", "x": 11280}, {"id": "B", "x": 10914}, {"id": "C", "x": 12138}],
            "derived": {"half": "x / 2"},
        }
        for spec, expected in (
            ({"op": "argmin", "field": "x"}, "B"),
            ({"op": "argmax", "field": "x"}, "C"),
            ({"op": "rank", "field": "x", "k": 2, "order": "asc"}, "A"),
            ({"op": "rank", "field": "x", "k": 1, "order": "desc"}, "C"),
            ({"op": "count_gt", "field": "x", "threshold": 11000}, 2),
            ({"op": "count_le", "field": "x", "threshold": 10914}, 1),
            ({"op": "argmin", "field": "half"}, "B"),
        ):
            with self.subTest(spec=spec):
                self.assertEqual(core.compute_answer_value(data, spec), expected)

    def test_recompute_choice_mapping(self):
        data = {"items": [{"id": "A", "x": 2}, {"id": "B", "x": 1}]}
        question = {"id": "q", "compute": {"op": "argmin", "field": "x"}}
        choices = [{"id": "a", "value": "A"}, {"id": "b", "value": "B"}]
        self.assertEqual(core.recompute(data, dict(question, choices=choices)), "b")
        for choices, message in (
            ([{"id": "a", "value": "A"}], ""),
            ([{"id": "a", "value": "B"}, {"id": "b", "value": "B"}], "exactly one choice"),
            ([{"id": "a", "value": "A"}, {"id": "a", "value": "B"}], "unique nonempty"),
            ([{"id": "", "value": "B"}], "unique nonempty"),
            ([{"id": " ", "value": "B"}], "unique nonempty"),
        ):
            with self.subTest(choices=choices), self.assertRaisesRegex(ValueError, message):
                core.recompute(data, dict(question, choices=choices))

    def test_direct_specs_reject_rank_coercion_invalid_order_and_bad_targets(self):
        data = {"items": [{"id": "A", "x": 1}, {"id": "B", "x": 2}]}
        with self.assertRaisesRegex(ValueError, "nonempty items"):
            core.compute_answer_value({"items": []}, {"op": "argmin", "field": "x"})
        for k in (True, 0, 1.5, "1", 3):
            with self.subTest(k=k), self.assertRaises(ValueError):
                core.compute_answer_value(data, {"op": "rank", "field": "x", "k": k})
        with self.assertRaises(ValueError):
            core.compute_answer_value(data, {"op": "rank", "field": "x", "k": 1, "order": "bad"})
        for value in (True, "1", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                core.compute_answer_value(
                    {"items": [{"id": "A", "x": value}]}, {"op": "argmin", "field": "x"}
                )
            with self.assertRaises(ValueError):
                core.compute_answer_value(
                    data, {"op": "count_gt", "field": "x", "threshold": value}
                )


SCENARIO_DATA = {
    "id": "t",
    "title": "T",
    "domain": "d",
    "description": "desc",
    "data": {
        "items": [
            {"id": "A", "label": "Apple", "x": 2, "z": 0},
            {"id": "B", "label": "Banana", "x": 1, "z": 4},
        ],
    },
    "questions": [
        {
            "id": "q",
            "stem": "which has the smallest x?",
            "compute": {"op": "argmin", "field": "x"},
            "choices": [
                {"id": "a", "text": "Apple", "value": "A"},
                {"id": "b", "text": "Banana", "value": "B"},
            ],
            "answer": "b",
        }
    ],
}


class TestSchema(unittest.TestCase):
    def test_invalid_choice_answer_and_presentation(self):
        for section, change in (
            ("question", {"choices": [{"id": "a", "text": "Banana", "value": "A"},
                                      {"id": "b", "text": "Banana", "value": "B"}]}),
            ("question", {"answer": "zzz"}),
            ("question", {"compute": {"op": "unknown", "field": "x"}}),
            ("question", {"compute": {"op": "argmin", "field": ""}}),
            ("question", {"answer": []}),
            ("data", {"presentation": {"headline_field": "not_a_field"}}),
        ):
            raw = copy.deepcopy(SCENARIO_DATA)
            target = raw["questions"][0] if section == "question" else raw["data"]
            target.update(change)
            with self.subTest(change=change), self.assertRaises(core.SchemaError):
                core.parse_scenario(raw)

    def test_primary_metric_requires_numeric_values_even_when_questions_use_another_field(self):
        for explicit in (True, False):
            for value in ("fixed", None, True):
                raw = copy.deepcopy(SCENARIO_DATA)
                for item in raw["data"]["items"]:
                    item["a_metric"] = 1
                raw["data"]["items"][0]["a_metric"] = value
                if explicit:
                    raw["data"]["presentation"] = {"primary_metric": "a_metric"}
                with self.subTest(explicit=explicit, value=value):
                    with self.assertRaisesRegex(core.SchemaError, "primary_metric"):
                        core.parse_scenario(raw)

    def test_primary_metric_must_exist_on_every_item(self):
        for explicit in (True, False):
            raw = copy.deepcopy(SCENARIO_DATA)
            raw["data"]["items"][0]["a_metric"] = 1
            if explicit:
                raw["data"]["presentation"] = {"primary_metric": "a_metric"}
            with self.subTest(explicit=explicit):
                with self.assertRaisesRegex(core.SchemaError, "primary_metric"):
                    core.parse_scenario(raw)

    def test_division_by_zero_derived_reported_not_crashed(self):
        raw = copy.deepcopy(SCENARIO_DATA)
        raw["data"]["derived"] = {"ratio": "x / z"}
        s = core.parse_scenario(raw)
        errors = validate_scenario(s)
        self.assertTrue(any("ratio" in e for e in errors), errors)
        self.assertIn("questions", " ".join(validate_scenario(replace(s, questions=[]))))


class TestStrictShapes(unittest.TestCase):
    def _count_scenario(self):
        raw = copy.deepcopy(SCENARIO_DATA)
        raw["questions"][0] = {
            "id": "count",
            "stem": "How many values exceed one?",
            "compute": {"op": "count_gt", "field": "x", "threshold": 1},
            "choices": [
                {"id": "n0", "text": "0", "value": 0},
                {"id": "n1", "text": "1", "value": 1},
                {"id": "n2", "text": "2", "value": 2},
            ],
            "answer": "n1",
        }
        return raw

    def test_count_labels_match_exact_nonnegative_integer_values(self):
        self.assertEqual(core.parse_scenario(self._count_scenario()).questions[0].answer, "n1")
        for value, text in ((0, "2"), (True, "1"), (0.5, "0.5"), (-1, "-1"), (1, "1 or 2")):
            with self.subTest(value=value, text=text):
                raw = self._count_scenario()
                raw["questions"][0]["choices"][0].update(value=value, text=text)
                with self.assertRaises(core.SchemaError):
                    core.parse_scenario(raw)

    def test_duplicate_visible_answers_and_values_are_rejected(self):
        for key in ("text", "value"):
            raw = self._count_scenario()
            raw["questions"][0]["choices"][1][key] = raw["questions"][0]["choices"][0][key]
            with self.assertRaises(core.SchemaError):
                core.parse_scenario(raw)
        raw = copy.deepcopy(SCENARIO_DATA)
        raw["questions"][0]["choices"][1] = {"id": "other", "text": "Apple", "value": "A"}
        with self.assertRaises(core.SchemaError):
            core.parse_scenario(raw)


class TestResources(unittest.TestCase):
    def test_resource_names_reject_traversal_and_yaml_is_strict(self):
        for name in ("../reader_mcq", "/tmp/file", "..\\file", "", "C:escape"):
            with self.subTest(name=name), self.assertRaises(ResourceError):
                resource_text("scenarios", name)
        for kind in ("other", None, []):
            with self.subTest(kind=kind), self.assertRaises(ResourceError):
                resource_names(kind)
        for text in (
            "x: .nan", "x: .inf", "x: -.inf", "x: 1.0e+999", "x: " + str(10**1000),
            "x: 1\nx: 2",
            "x: !!python/object:builtins.object {}", "x: !!binary YQ==", "x: 2026-10-24",
            "1: value", "x: !!set {value: null}", "x: &ref [*ref]",
            "x: &ref [1]\ny: *ref", "x: " + "[" * 101 + "0" + "]" * 101,
        ):
            with self.subTest(yaml=text), self.assertRaises(ResourceError):
                decode_yaml(text)
        valid = {"id": "reading", "values": [0.001, 0.002], "enabled": True, "missing": None}
        self.assertEqual(decode_yaml(core.encode_yaml(valid)), valid)
        self.assertEqual(decode_yaml("id: reading\nvalues:\n  - 0.001\n  - 0.002"),
                         {"id": "reading", "values": [0.001, 0.002]})
        with self.assertRaises(ResourceError):
            resource_text("scenarios", "missing")


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


def display_scenario(values=(0.001, 0.002), unit="", label=None, field_label=None):
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
    return core.parse_scenario({
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
                source = display_scenario(values, unit)
                for variant in core.ALL_VARIANTS:
                    with self.subTest(values=values, unit=unit, variant=variant):
                        presentation = core.build_variant(source, variant)
                        core.validate_display_values(presentation)
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

    def test_validator_rejects_distinct_but_inaccurate_numeric_tokens(self):
        presentation = core.build_variant(display_scenario(), "table")
        with patch("glassbox.core._num", side_effect=lambda value: str(value * 1000)):
            with self.assertRaisesRegex(ValueError, "numeric display"):
                core.validate_display_values(presentation)

    def test_validator_checks_the_formatted_value_including_units(self):
        presentation = core.build_variant(display_scenario(unit="$k"), "table")
        with patch("glassbox.core.format_value", return_value="$0k"):
            with self.assertRaisesRegex(ValueError, "numeric display"):
                core.validate_display_values(presentation)

    def test_nonfinite_visible_values_reject(self):
        presentation = core.build_variant(display_scenario(), "table")
        for value in (float("nan"), float("inf"), float("-inf")):
            shown = {item: dict(row) for item, row in presentation.shown.items()}
            shown["A"]["measured_value"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "numeric display"):
                core.validate_display_values(replace(presentation, shown=shown))

    def test_validator_checks_table_context_units_as_well_as_text_emphasis(self):
        presentation = core.build_variant(display_scenario(unit="$"), "table")
        shown = {item_id: dict(row, advertised_value=100)
                 for item_id, row in presentation.shown.items()}
        presentation = replace(presentation, shown=shown, emphasized_field="advertised_value")
        original = core.format_value
        with patch("glassbox.core.format_value", side_effect=lambda value, unit:
                   "0" if unit == "" else original(value, unit)):
            self.assertIn("advertised value: $100", presentation.to_text())
            html = VisibleHTML()
            html.feed(presentation.to_html())
            self.assertEqual(html.numeric[0], "0")
            with self.assertRaisesRegex(ValueError, "numeric display"):
                core.validate_display_values(presentation)


class TestDisplayLabels(unittest.TestCase):
    def test_omitted_and_blank_labels_fall_back_in_every_layout(self):
        for label in (None, "", " \t "):
            for field_label in (None, "", " \t "):
                for variant in core.ALL_VARIANTS:
                    with self.subTest(label=label, field_label=field_label, variant=variant):
                        presentation = core.build_variant(display_scenario(label=label, field_label=field_label),
                                                          variant)
                        self.assertIn("measured value = 0.001", presentation.to_text())
                        html = VisibleHTML()
                        html.feed(presentation.to_html())
                        self.assertIn("measured value", " ".join(html.visible))
                        self.assertNotIn(":  =", presentation.to_text())

    def test_explicit_labels_are_retained_and_escaped_in_html(self):
        for variant in core.ALL_VARIANTS:
            presentation = core.build_variant(display_scenario(label="Reading", field_label="Signal <rate>"),
                                              variant)
            self.assertIn("Signal <rate> = 0.001", presentation.to_text())
            html = VisibleHTML()
            html.feed(presentation.to_html())
            self.assertIn("Signal <rate>", " ".join(html.visible))
            self.assertIn("Signal &lt;rate&gt;", presentation.to_html())

    def test_direct_presentation_blank_labels_use_normal_field_name(self):
        presentation = core.build_variant(display_scenario(), "annotated")
        presentation = replace(presentation, value_label="  ", field_labels={"measured_value": ""})
        self.assertIn("measured value = 0.001", presentation.to_text())
        self.assertIn('class="headline-label">measured value<', presentation.to_html())


class _CyclingReader(Reader):
    simulated = True
    family = "synthetic"
    name = "synthetic:cycling"

    def __init__(self):
        self._answer_count = 0

    def answer(self, scenario, question, stimulus):
        self._answer_count += 1
        choice = question.choices[self._answer_count % len(question.choices)]
        return Answer(choice_id=choice.id, method="synthetic")


class ScoringCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(core.load_all_scenarios().values())
        cls.readers = build_readers(["simulated"])
        cls.book = evaluation.ScoreBook(evaluation.read_all(cls.scenarios, cls.readers, skip_render=True))


class TestScoreBook(ScoringCase):
    def test_fixture_accuracy_and_lift(self):
        variants = ["raw", "baseline", "cards", "table", "annotated"]
        readers = ["simulated:careless", "simulated:diligent", "simulated:literal"]
        correct = {
            "growth": [(3, 3, 0, 3, 3), (3, 3, 0, 3, 3), (0, 0, 0, 2, 2)],
            "loans": [(2, 2, 1, 2, 3), (3, 3, 0, 3, 3), (0, 0, 0, 3, 3)],
            "plans": [(3, 3, 0, 3, 3), (3, 3, 0, 3, 3), (0, 0, 0, 3, 3)],
        }
        self.assertEqual(self.book.variants, variants)
        self.assertEqual(self.book.readers, readers)
        self.assertEqual(self.book.scenarios, list(correct))
        for scenario, counts in correct.items():
            for reader, expected in zip(readers, counts):
                with self.subTest(scenario=scenario, reader=reader):
                    self.assertEqual([self.book.accuracy(reader, scenario, variant)
                                      for variant in variants], [count / 3 for count in expected])
                    for variant in variants:
                        self.assertAlmostEqual(self.book.lift(reader, scenario, variant),
                                               self.book.accuracy(reader, scenario, variant)
                                               - self.book.baseline(reader, scenario))
        self.assertTrue(self.book.all_simulated())
        self.assertFalse(self.book.any_real())

    def test_iterator_preserves_rows_and_reader_identity(self):
        book = evaluation.ScoreBook(iter(self.book.results))
        self.assertEqual(book.results, self.book.results)
        self.assertEqual((book.readers, book.scenarios, book.variants),
                         (self.book.readers, self.book.scenarios, self.book.variants))
        self.assertEqual(book.accuracy("simulated:literal", "loans", "table"), 1)
        conflicting = [replace(row, reader_simulated=False) if index == 0 else row
                       for index, row in enumerate(self.book.results)]
        for rows in (conflicting, iter(conflicting)):
            with self.subTest(container=type(rows).__name__), self.assertRaises(ValueError):
                evaluation.ScoreBook(rows)

    def test_invalid_inputs_fail_before_rendering(self):
        scenario = self.scenarios[0]
        cases = [(None, self.readers, {}), (self.scenarios, None, {}),
                 (self.scenarios, self.readers, {"variants": 1})]
        for name in ([], None, "", "   "):
            reader = build_reader("simulated:literal")
            reader.name = name
            cases.extend([([scenario], [reader], {}),
                          ([replace(scenario, id=name)], self.readers, {})])
        cases.extend(([scenario], self.readers, options) for options in (
            {"replicates": True}, {"replicates": 1.5}, {"skip_render": "false"},
            {"variants": []}, {"variants": ["table", "table"]}, {"variants": "table"},
        ))
        for index, (scenarios, readers, options) in enumerate(cases):
            with self.subTest(case=index), patch("glassbox.core.render_stimulus") as render:
                with self.assertRaises(ValueError):
                    evaluation.read_all(scenarios, readers, **options)
                render.assert_not_called()


class TestReplicates(unittest.TestCase):
    def setUp(self):
        self.loans = core.load_all_scenarios()["loans"]
        self.cell = ("simulated:literal", "loans", "table")

    def book(self, specs=("simulated:literal",), replicates=1, readers=None):
        return evaluation.ScoreBook(evaluation.read_all(
            [self.loans], readers if readers is not None else build_readers(specs),
            variants=["table"], skip_render=True, replicates=replicates,
        ))

    def test_deterministic_reader_has_zero_spread(self):
        book = self.book(replicates=3)
        self.assertEqual(book.replicate_count(), 3)
        self.assertEqual(book.accuracy_std(*self.cell), 0.0)
        self.assertEqual(book.answer_stability(*self.cell), 1.0)

    def test_missing_and_single_observation_spread(self):
        book = self.book()
        self.assertTrue(math.isnan(book.accuracy_std("simulated:literal", "loans", "cards")))
        self.assertEqual(book.accuracy_std(*self.cell), 0.0)
        self.assertEqual(book.replicate_count(), 1)
        self.assertTrue(math.isnan(book.baseline("simulated:literal", "loans")))
        self.assertTrue(math.isnan(book.lift(*self.cell)))
        empty = evaluation.ScoreBook([])
        self.assertEqual(empty.replicate_count(), 0)
        self.assertFalse(empty.all_simulated())
        self.assertTrue(math.isnan(empty.mean_accuracy("missing", "table")))

    def test_varying_reader_shows_spread(self):
        book = self.book(readers=[_CyclingReader()], replicates=4)
        cell = ("synthetic:cycling", "loans", "table")
        self.assertEqual(book.replicate_count(), 4)
        self.assertEqual(book.accuracy_by_replicate(*cell), [1.0, 2 / 3, 2 / 3, 2 / 3])
        self.assertAlmostEqual(book.accuracy_std(*cell), math.sqrt(3) / 12)
        self.assertEqual(book.answer_stability(*cell), 0.75)


class _TruthValue:
    def __init__(self, value):
        self.value = bool(value)

    def __bool__(self):
        return self.value

    def __int__(self):
        return int(self.value)


class _ScalarFloat(float):
    def __gt__(self, other):
        return _TruthValue(super().__gt__(other))

    def __lt__(self, other):
        return _TruthValue(super().__lt__(other))


class _FixedWidthInt(int):
    def __sub__(self, other):
        difference = int(self) - int(other)
        return (difference + 2**63) % 2**64 - 2**63


class _CheckedFloat(float):
    def __sub__(self, other):
        difference = super().__sub__(other)
        if math.isinf(difference):
            raise FloatingPointError("scalar subtraction overflow")
        return difference


class TestCorrelations(unittest.TestCase):
    def test_kendall_accepts_scalar_truth_values(self):
        xs = [_ScalarFloat(value) for value in [1, 2, 3]]
        ys = [_ScalarFloat(value) for value in [2, 1, 3]]
        self.assertAlmostEqual(evaluation.kendall_tau(xs, ys), 1 / 3)

    def test_pearson_centers_fixed_width_integers_without_overflow(self):
        values = [-9_000_000_000_000_000_000, 0, 9_000_000_000_000_000_000]
        scalars = [_FixedWidthInt(value) for value in values]
        target = [-2.0, 0.0, 1.0]
        self.assertAlmostEqual(
            evaluation.pearson(scalars, target), evaluation.pearson(values, target)
        )

    def test_pearson_avoids_fixed_width_floating_scalar_arithmetic(self):
        values = [_CheckedFloat(value) for value in (-1e308, 0.0, 1e308)]
        self.assertAlmostEqual(evaluation.pearson(values, [-1, 0, 1]), 1.0)
        self.assertAlmostEqual(evaluation.pearson(values, [1, 0, -1]), -1.0)

    def test_correlations_preserve_close_rational_values(self):
        values = [Fraction(2**60) + Fraction(value, 10) for value in (1, 2, 3)]
        for correlation in (evaluation.pearson, evaluation.spearman, evaluation.kendall_tau):
            with self.subTest(correlation=correlation.__name__):
                self.assertAlmostEqual(correlation(values, [1, 2, 3]), 1.0)
                self.assertAlmostEqual(correlation(values, [3, 2, 1]), -1.0)

    def test_tied_and_extreme_correlations(self):
        for function, expected in ((evaluation.pearson, 9 / math.sqrt(95)),
                                   (evaluation.spearman, 3 / math.sqrt(10)),
                                   (evaluation.kendall_tau, 5 / math.sqrt(30))):
            self.assertAlmostEqual(function([1, 2, 2, 4], [1, 2, 3, 4]), expected)
        vector = [-1e308, 0.0, 1e308]
        self.assertAlmostEqual(evaluation.pearson(vector, vector), 1)
        self.assertAlmostEqual(evaluation.pearson(vector, vector[::-1]), -1)
        self.assertAlmostEqual(evaluation.kendall_tau(vector, vector), 1)
        self.assertAlmostEqual(evaluation.spearman(vector, vector), 1)
        close_integers = [2**60 + 1, 2**60 + 2, 2**60 + 3]
        for function in (evaluation.pearson, evaluation.spearman, evaluation.kendall_tau):
            self.assertAlmostEqual(function(close_integers, close_integers), 1)

    def test_invalid_correlations_reject_and_constant_is_undefined(self):
        for function in (evaluation.pearson, evaluation.spearman, evaluation.kendall_tau):
            for invalid in (True, "1", 1j, float("nan"), float("inf"), 10**1000):
                with (
                    self.subTest(function=function.__name__, invalid=type(invalid)),
                    self.assertRaises(ValueError),
                ):
                    function([invalid, 2], [1, 2])
            with self.assertRaises(ValueError):
                function([1], [1, 2])
            self.assertTrue(math.isnan(function([1, 1], [2, 3])))
            self.assertTrue(math.isnan(function([], [])))



class _ScoreJudge(Judge):
    simulated = True
    deterministic = True

    def __init__(self, scores):
        self.scores = scores

    def preference(self, scenario, presentation):
        return self.scores.get(scenario.id, {}).get(presentation.variant, float("nan"))


class TestMissingJudgeScores(ScoringCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.scenarios = {scenario.id: scenario for scenario in cls.scenarios}

    def test_partial_scores_rank_and_correlate_only_finite_pairs(self):
        judge = _ScoreJudge({"loans": {"cards": 0.9, "annotated": 0.1}})
        result = evaluation.analyze_h1(self.book, [self.scenarios["loans"]], judge)
        scenario = result.scenarios[0]
        rows = {row.variant: row for row in scenario.rows}
        self.assertEqual(rows["cards"].pref_rank, 1)
        self.assertIsNone(rows["table"].pref_rank)
        self.assertEqual(rows["annotated"].pref_rank, 2)
        self.assertAlmostEqual(scenario.spearman, -1.0)
        self.assertEqual(
            [(reversal.preferred, reversal.understood) for reversal in scenario.reversals],
            [("cards", "annotated")],
        )

    def test_constant_finite_scores_stay_tied_with_undefined_correlation(self):
        judge = _ScoreJudge({"loans": {"cards": 0.5, "annotated": 0.5}})
        result = evaluation.analyze_h1(self.book, [self.scenarios["loans"]], judge)
        scenario = result.scenarios[0]
        self.assertEqual([row.pref_rank for row in scenario.rows], [1, None, 1])
        self.assertTrue(math.isnan(scenario.spearman))
        self.assertEqual(scenario.reversals, [])


    def test_missing_baseline_has_no_lift_rank(self):
        book = evaluation.ScoreBook([
            row for row in self.book.results if row.variant != "baseline"
        ])
        scenarios = list(self.scenarios.values())
        result = evaluation.analyze_h1(book, scenarios, SimulatedJudge())
        for scenario in result.scenarios:
            self.assertTrue(math.isnan(scenario.spearman))
            self.assertEqual(scenario.reversals, [])
            self.assertTrue(all(row.comp_rank is None for row in scenario.rows))

    def test_missing_reader_cell_does_not_break_agreement(self):
        book = evaluation.ScoreBook([
            row for row in self.book.results
            if not (row.scenario_id == "loans"
                    and row.reader == "simulated:literal"
                    and row.variant == "cards")
        ])
        result = evaluation.analyze_h1(book, [self.scenarios["loans"]], SimulatedJudge())
        self.assertTrue(math.isfinite(result.scenarios[0].reader_agreement))


if __name__ == "__main__":
    unittest.main()

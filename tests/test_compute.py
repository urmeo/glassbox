"""The recompute engine: safe arithmetic, compute ops, and answer recomputation."""

import unittest

from glassbox import compute


class TestSafeExpression(unittest.TestCase):
    def test_arithmetic_and_precedence(self):
        self.assertEqual(compute.evaluate_expr("2 + 3 * 4", {}), 14)
        self.assertEqual(compute.evaluate_expr("(2 + 3) * 4", {}), 20)
        self.assertEqual(compute.evaluate_expr("10 / 4", {}), 2.5)
        self.assertEqual(compute.evaluate_expr("-5 + 2", {}), -3)

    def test_power_operator_rejected(self):
        # Exponentiation is excluded — `9**9**9` must not be evaluable (compute-DoS).
        with self.assertRaises(compute.ExpressionError):
            compute.evaluate_expr("2 ** 3", {})

    def test_division_by_zero_is_expression_error(self):
        # In-whitelist arithmetic errors are reported cleanly, never a raw traceback.
        with self.assertRaises(compute.ExpressionError):
            compute.evaluate_expr("x / 0", {"x": 5})

    def test_field_lookup(self):
        fields = {"monthly_payment": 305.0, "term_months": 36, "fees": 300}
        self.assertEqual(
            compute.evaluate_expr("monthly_payment * term_months + fees", fields),
            305.0 * 36 + 300,
        )

    def test_unknown_field_rejected(self):
        with self.assertRaises(compute.ExpressionError):
            compute.evaluate_expr("mystery + 1", {"known": 1})

    def test_code_injection_rejected(self):
        # None of these are arithmetic — the evaluator must refuse them all.
        for evil in (
            "__import__('os').system('echo hi')",
            "().__class__",
            "open('x')",
            "[i for i in range(3)]",
            "value.attr",
            "1 if True else 2",
        ):
            with self.assertRaises(compute.ExpressionError):
                compute.evaluate_expr(evil, {"value": 1})

    def test_bool_is_not_numeric(self):
        with self.assertRaises(compute.ExpressionError):
            compute.evaluate_expr("flag + 1", {"flag": True})


class TestComputeOps(unittest.TestCase):
    def setUp(self):
        self.data = {
            "items": [
                {"id": "A", "x": 11280},
                {"id": "B", "x": 10914},
                {"id": "C", "x": 12138},
            ],
            "derived": {"half": "x / 2"},
        }

    def test_argmin_argmax(self):
        self.assertEqual(compute.compute_answer_value(self.data, {"op": "argmin", "field": "x"}), "B")
        self.assertEqual(compute.compute_answer_value(self.data, {"op": "argmax", "field": "x"}), "C")

    def test_rank(self):
        self.assertEqual(
            compute.compute_answer_value(self.data, {"op": "rank", "field": "x", "k": 2, "order": "asc"}),
            "A",
        )
        self.assertEqual(
            compute.compute_answer_value(self.data, {"op": "rank", "field": "x", "k": 1, "order": "desc"}),
            "C",
        )

    def test_count_ops(self):
        self.assertEqual(
            compute.compute_answer_value(self.data, {"op": "count_gt", "field": "x", "threshold": 11000}), 2)
        self.assertEqual(
            compute.compute_answer_value(self.data, {"op": "count_le", "field": "x", "threshold": 10914}), 1)

    def test_derived_field_in_op(self):
        self.assertEqual(compute.compute_answer_value(self.data, {"op": "argmin", "field": "half"}), "B")


class TestRecompute(unittest.TestCase):
    def test_maps_value_to_choice_id(self):
        data = {"items": [{"id": "A", "x": 2}, {"id": "B", "x": 1}]}
        question = {
            "id": "q",
            "compute": {"op": "argmin", "field": "x"},
            "choices": [{"id": "a", "value": "A"}, {"id": "b", "value": "B"}],
        }
        self.assertEqual(compute.recompute(data, question), "b")

    def test_unrepresentable_answer_raises(self):
        data = {"items": [{"id": "A", "x": 2}, {"id": "B", "x": 1}]}
        question = {
            "id": "q",
            "compute": {"op": "argmin", "field": "x"},
            "choices": [{"id": "a", "value": "A"}],  # no choice for B
        }
        with self.assertRaises(ValueError):
            compute.recompute(data, question)


if __name__ == "__main__":
    unittest.main()

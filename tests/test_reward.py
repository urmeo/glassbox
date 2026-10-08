"""In-sample feature-search fixtures."""

import unittest

from glassbox import reward, schema
from glassbox.readers import build_readers


class TestReward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = list(schema.load_all_scenarios().values())
        cls.readers = build_readers(["simulated"])

    def test_honest_interface_has_positive_reward(self):
        r = reward.scenario_set_reward(
            self.scenarios, frozenset({"detail", "derived"}), self.readers
        )
        self.assertGreater(r, 0.0)

    def test_cards_have_negative_reward(self):
        from glassbox.interfaces import VARIANT_FEATURES

        r = reward.scenario_set_reward(
            self.scenarios, VARIANT_FEATURES["cards"], self.readers
        )
        self.assertLess(r, 0.0)

    def test_search_selects_the_computed_metric(self):
        ranked = reward.search_best_interface(self.scenarios, self.readers)
        best_features, best_reward = ranked[0]
        self.assertIn("derived", best_features)
        self.assertGreater(best_reward, 0.0)

    def test_generator_beats_baselines(self):
        ev = reward.evaluate_generator(self.scenarios, self.readers)
        self.assertTrue(ev.beats_preference_tuned)
        self.assertTrue(ev.beats_plaintext)
        self.assertGreater(ev.generator_reward, ev.cards_reward)

    def test_pool_excludes_family(self):
        self.assertFalse(
            reward.pool_excludes_family(
                ["anthropic:claude-sonnet-5", "openrouter:qwen/qwen3-vl-8b-instruct"],
                "qwen",
            )
        )
        self.assertTrue(
            reward.pool_excludes_family(
                ["anthropic:claude-sonnet-5", "anthropic:claude-opus-4-8"], "qwen"
            )
        )

    def test_model_family_is_structured_not_substring(self):
        self.assertEqual(reward.model_family("anthropic:claude-sonnet-5"), "anthropic")
        self.assertEqual(
            reward.model_family("openrouter:qwen/qwen3-vl-8b-instruct"), "qwen"
        )
        self.assertEqual(reward.model_family("simulated:literal"), "simulated")
        self.assertEqual(reward.model_family("openrouter:alibaba/vl-max"), "unknown")
        self.assertEqual(reward.model_family("anthropic:claude-3-qwenish"), "anthropic")

    def test_optimize_report_labels_simulated(self):
        from glassbox import report as rp

        ev = reward.evaluate_generator(self.scenarios, self.readers)
        md = rp.render_optimize_md(
            ev, ["simulated:literal", "simulated:diligent"], None
        )
        self.assertIn("Simulated readers only", md)


if __name__ == "__main__":
    unittest.main()

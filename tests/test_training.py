"""M4 training configs — declared, validated, gated (never run)."""

import copy
import os
import unittest

import glassbox
from glassbox import training

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(glassbox.__file__))))
SHIPPED = os.path.join(ROOT, "data", "training", "sft_grpo_qwen3vl.json")

VALID = {
    "id": "t", "description": "d", "framework": "tinker",
    "base_model": "Qwen/Qwen3-VL-8B-Instruct", "generator_family": "qwen",
    "methods": ["sft", "grpo"], "reward": "comprehension_lift",
    "reader_pool": ["anthropic:claude-sonnet-5"], "status": "not run",
}


class TestTrainingConfig(unittest.TestCase):
    def test_load_shipped(self):
        cfg = training.load_training_config(SHIPPED)
        self.assertEqual(cfg.generator_family, "qwen")
        self.assertEqual(cfg.reward, "comprehension_lift")

    def test_shipped_config_validates_clean(self):
        cfg = training.load_training_config(SHIPPED)
        self.assertEqual(training.validate_training_config(cfg), [])

    def test_reader_pool_with_generator_family_rejected(self):
        bad = copy.deepcopy(VALID)
        bad["reader_pool"] = ["anthropic:claude-sonnet-5", "openrouter:qwen/qwen3-vl-8b-instruct"]
        errors = training.validate_training_config(training.parse_training_config(bad))
        self.assertTrue(any("anti-gaming" in e or "exclude" in e for e in errors), errors)

    def test_bad_reward_rejected(self):
        bad = copy.deepcopy(VALID)
        bad["reward"] = "preference"
        errors = training.validate_training_config(training.parse_training_config(bad))
        self.assertTrue(any("comprehension_lift" in e for e in errors), errors)

    def test_bad_method_rejected(self):
        bad = copy.deepcopy(VALID)
        bad["methods"] = ["sft", "ppo"]
        errors = training.validate_training_config(training.parse_training_config(bad))
        self.assertTrue(any("method" in e for e in errors), errors)

    def test_missing_field_rejected(self):
        bad = copy.deepcopy(VALID)
        del bad["base_model"]
        with self.assertRaises(training.TrainingConfigError):
            training.parse_training_config(bad)


if __name__ == "__main__":
    unittest.main()

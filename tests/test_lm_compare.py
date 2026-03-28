from __future__ import annotations

import unittest

try:
    import torch

    from FunctorFlow.lm_compare import (
        GTLanguageModel,
        GTLanguageModelConfig,
        LMComparisonConfig,
        TransformerLanguageModel,
        TransformerLanguageModelConfig,
        build_default_model_suite,
        compare_language_models,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this interpreter")
class FunctorFlowLMCompareTests(unittest.TestCase):
    def test_build_default_model_suite_contains_transformer_gt_and_ket(self) -> None:
        suite = build_default_model_suite("ptb", vocab_size=64, profile="smoke")
        self.assertEqual(set(suite), {"transformer", "gt", "ket"})
        wiki103_suite = build_default_model_suite("wiki-103", vocab_size=64, profile="smoke")
        self.assertEqual(set(wiki103_suite), {"transformer", "gt", "ket"})

    def test_transformer_language_model_forward_shape(self) -> None:
        model = TransformerLanguageModel(
            vocab_size=50,
            config=TransformerLanguageModelConfig(
                d_model=16,
                n_layers=1,
                max_positions=64,
            ),
        )
        token_ids = torch.randint(0, 50, (4, 12))
        logits = model(token_ids)
        self.assertEqual(tuple(logits.shape), (4, 12, 50))

    def test_gt_language_model_forward_shape(self) -> None:
        model = GTLanguageModel(
            vocab_size=50,
            config=GTLanguageModelConfig(
                d_model=16,
                n_layers=1,
                max_positions=64,
            ),
        )
        token_ids = torch.randint(0, 50, (4, 12))
        logits = model(token_ids)
        self.assertEqual(tuple(logits.shape), (4, 12, 50))

    def test_compare_language_models_smoke(self) -> None:
        result = compare_language_models(
            "ptb",
            comparison_config=LMComparisonConfig(
                steps=1,
                block_size=8,
                batch_size=2,
                lr=1e-3,
                model_profile="smoke",
                train_tokens=256,
                valid_tokens=128,
                test_tokens=128,
                seed=0,
            ),
            device=torch.device("cpu"),
        )
        self.assertEqual(result["corpus"], "ptb")
        self.assertEqual(result["model_profile"], "smoke")
        self.assertEqual(set(result["models"]), {"transformer", "gt", "ket"})
        for model_result in result["models"].values():
            self.assertEqual(len(model_result["history"]["train_loss"]), 1)
            self.assertEqual(len(model_result["history"]["valid_ppl"]), 1)
            self.assertGreater(model_result["valid_ppl"], 0.0)


if __name__ == "__main__":
    unittest.main()

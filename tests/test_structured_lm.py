from __future__ import annotations

import unittest

try:
    import torch

    from FunctorFlow.structured_lm import (
        KETStructuredLanguageModelConfig,
        StructuredLMComparisonConfig,
        build_structured_language_model_diagram,
        compare_structured_language_models,
        run_structured_language_model_experiment,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this interpreter")
class FunctorFlowStructuredLMTests(unittest.TestCase):
    def test_structured_language_model_diagram_exposes_duality_ports(self) -> None:
        diagram = build_structured_language_model_diagram()
        self.assertIn("predict__aggregate_context", diagram.operations)
        self.assertIn("repair__complete_block", diagram.operations)
        self.assertEqual(diagram.port_type("context"), "contextualized_messages")
        self.assertEqual(diagram.port_type("completed"), "completed_state")

    def test_structured_language_model_experiment_runs(self) -> None:
        result = run_structured_language_model_experiment(
            KETStructuredLanguageModelConfig(
                corpus_name="ptb",
                task="denoise",
                seq_len=24,
                batch_size=2,
                steps=1,
                block_size=2,
                num_denoise_steps=4,
                eval_batches=1,
                lm_config=KETStructuredLanguageModelConfig.historical_smoke(
                    "ptb", task="denoise"
                ).lm_config,
            ),
            device=torch.device("cpu"),
        )
        self.assertEqual(result["corpus"], "ptb")
        self.assertEqual(result["task"], "denoise")
        self.assertIn("reconstruction_accuracy", result["eval"])
        self.assertGreater(result["eval"]["first_offset_ppl"], 0.0)
        self.assertEqual(len(result["history"]["train_loss"]), result["config"].steps)

    def test_structured_language_model_comparison_runs(self) -> None:
        result = compare_structured_language_models(
            StructuredLMComparisonConfig.historical_smoke("ptb"),
            device=torch.device("cpu"),
        )
        self.assertEqual(result["corpus"], "ptb")
        self.assertEqual(
            set(result["models"]),
            {"TF-Block-4", "KET-Block-4", "TF-Denoise-4", "KET-Denoise-4"},
        )
        for payload in result["models"].values():
            self.assertIn("first_offset_ppl", payload["eval"])
            self.assertIn("block_ppl", payload["eval"])


if __name__ == "__main__":
    unittest.main()

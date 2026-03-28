from __future__ import annotations

import unittest

try:
    import torch

    from FunctorFlow.ket_block_duality import KETBlockDualityConfig, run_ket_block_duality_demo

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this interpreter")
class FunctorFlowKETBlockDualityTests(unittest.TestCase):
    def test_ket_block_duality_demo_runs(self) -> None:
        result = run_ket_block_duality_demo(
            KETBlockDualityConfig(
                corpus_name="ptb",
                seq_len=32,
                batch_size=4,
                steps=2,
                block_size=3,
                num_denoise_steps=4,
                seed=0,
            ),
            device=torch.device("cpu"),
        )
        self.assertEqual(result["corpus"], "ptb")
        self.assertIn("FunctorFlowLeftKanBlock", result["left_kan"]["diagram"].summary())
        self.assertIn("FunctorFlowRightKanDenoise", result["right_kan"]["diagram"].summary())
        self.assertEqual(len(result["left_kan"]["history"]["train_loss"]), 2)
        self.assertEqual(len(result["right_kan"]["history"]["train_loss"]), 2)
        self.assertIn(1, result["left_kan"]["eval"]["offset_accuracy"])
        self.assertIn(1, result["right_kan"]["eval"]["offset_accuracy"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

try:
    import torch

    from FunctorFlow.predict_detach_demo import PredictDetachDemoConfig, run_predict_detach_regime_demo

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this interpreter")
class FunctorFlowPredictDetachDemoTests(unittest.TestCase):
    def test_predict_detach_regime_demo_runs(self) -> None:
        result = run_predict_detach_regime_demo(
            PredictDetachDemoConfig(
                steps=2,
                batch_size=8,
                train_sequences=64,
                test_sequences=32,
                embed_dim=16,
                sequence_length=12,
                vocab_size=20,
                seed=0,
            ),
            device=torch.device("cpu"),
        )
        self.assertEqual(set(result["regimes"]), {"causal", "leaky_noncausal", "predict_detach"})
        for payload in result["regimes"].values():
            self.assertEqual(len(payload["losses"]), 2)
            self.assertGreater(payload["final_train_loss"], 0.0)
            self.assertGreater(payload["eval_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()

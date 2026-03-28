from __future__ import annotations

import importlib.util
import unittest

from FunctorFlow import Diagram, compile_to_torch


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this interpreter")
class FunctorFlowTorchTests(unittest.TestCase):
    def test_compile_to_torch_with_module_morphism(self) -> None:
        import torch
        import torch.nn as nn

        class Shift(nn.Module):
            def forward(self, value):
                return value + 1

        diagram = Diagram("TorchShift")
        x = diagram.object("x", kind="tensor")
        diagram.morphism("shift", x, x, implementation=Shift())

        model = compile_to_torch(diagram)
        outputs = model({"x": torch.tensor([1.0, 2.0])})

        self.assertTrue(torch.equal(outputs["shift"], torch.tensor([2.0, 3.0])))


if __name__ == "__main__":
    unittest.main()

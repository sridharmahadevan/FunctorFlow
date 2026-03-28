from __future__ import annotations

import unittest

from FunctorFlow import Diagram, Morphism, Object, compile_to_callable


class FunctorFlowTests(unittest.TestCase):
    def test_detached_objects_and_morphisms_register(self) -> None:
        tokens = Object("Tokens", kind="sequence")
        nbrs = Object("Nbrs", kind="index")
        incidence = Morphism("incidence", tokens, nbrs)

        diagram = Diagram("KET")
        diagram.add(incidence)

        self.assertIn("Tokens", diagram.objects)
        self.assertIn("Nbrs", diagram.objects)
        self.assertIn("incidence", diagram.operations)

    def test_left_kan_sum_executes(self) -> None:
        diagram = Diagram("KETBlock")
        diagram.object("Values", kind="messages")
        diagram.object("Incidence", kind="relation")
        diagram.left_kan(source="Values", along="Incidence", name="aggregate", reducer="sum")

        compiled = compile_to_callable(diagram)
        result = compiled.run(
            {
                "Values": {0: 1.0, 1: 2.0, 2: 4.0},
                "Incidence": {"left": [0, 1], "right": [1, 2]},
            }
        )

        self.assertEqual(result.values["aggregate"], {"left": 3.0, "right": 6.0})

    def test_obstruction_loss_on_noncommuting_paths(self) -> None:
        diagram = Diagram("DBSquare")
        x = diagram.object("x", kind="scalar")
        diagram.morphism("double", x, x, implementation=lambda value: 2 * value)
        diagram.morphism("shift", x, x, implementation=lambda value: value + 1)
        diagram.compose("double", "shift", name="p1")
        diagram.compose("shift", "double", name="p2")
        diagram.obstruction_loss(paths=[("p1", "p2")], name="commutator")

        compiled = compile_to_callable(diagram)
        result = compiled.run({"x": 3.0})

        self.assertEqual(result.values["p1"], 7.0)
        self.assertEqual(result.values["p2"], 8.0)
        self.assertAlmostEqual(result.losses["commutator"], 1.0)


if __name__ == "__main__":
    unittest.main()

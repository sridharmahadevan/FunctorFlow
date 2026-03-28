from __future__ import annotations

import json

from FunctorFlow import Diagram, compile_to_callable


def ket_example() -> None:
    diagram = Diagram("KETBlock")
    diagram.object("Values", kind="messages")
    diagram.object("Incidence", kind="relation")
    diagram.left_kan(source="Values", along="Incidence", name="aggregate", reducer="sum")

    compiled = compile_to_callable(diagram)
    result = compiled.run(
        {
            "Values": {
                0: 1.0,
                1: 2.0,
                2: 4.0,
            },
            "Incidence": {
                "left_context": [0, 1],
                "right_context": [1, 2],
            },
        }
    )

    print("KET-style aggregation")
    print(json.dumps(result.values["aggregate"], indent=2, sort_keys=True))


def consistency_example() -> None:
    diagram = Diagram("ConsistencyAwareLM")
    x = diagram.object("x", kind="scalar")
    diagram.morphism("double", x, x, implementation=lambda value: 2 * value)
    diagram.morphism("shift", x, x, implementation=lambda value: value + 1)
    diagram.compose("double", "shift", name="p1")
    diagram.compose("shift", "double", name="p2")
    diagram.obstruction_loss(paths=[("p1", "p2")], name="commutator")

    compiled = compile_to_callable(diagram)
    result = compiled.run({"x": 3.0})

    print()
    print("DB-style obstruction")
    print(json.dumps({"p1": result.values["p1"], "p2": result.values["p2"]}, indent=2))
    print(json.dumps(result.losses, indent=2, sort_keys=True))


if __name__ == "__main__":
    ket_example()
    consistency_example()

from __future__ import annotations

import json

from FunctorFlow import (
    MACRO_LIBRARY,
    BasketRocketPipelineConfig,
    KETBlockConfig,
    basket_rocket_pipeline,
    build_macro,
    compile_to_callable,
    ket_block,
)


def main() -> None:
    print("Available macros:")
    print(json.dumps(sorted(MACRO_LIBRARY), indent=2))

    ket = ket_block(
        KETBlockConfig(
            name="TutorialKET",
            source_object="EdgeValues",
            relation_object="TokenIncidence",
            target_object="TokenStates",
            aggregate_name="gather",
            reducer="sum",
        )
    )
    ket_result = compile_to_callable(ket).run(
        {
            "EdgeValues": {0: 1.0, 1: 3.0, 2: 5.0},
            "TokenIncidence": {"ctx_a": [0, 1], "ctx_b": [1, 2]},
        }
    )
    print()
    print("Typed KET macro result:")
    print(json.dumps(ket_result.values["gather"], indent=2, sort_keys=True))

    rocket = build_macro("rocket_repair", name="TutorialRepair")
    rocket_result = compile_to_callable(rocket).run(
        {
            "CandidateFragments": {
                "draft_a": "observe -> analyze",
                "draft_b": "observe -> analyze -> optimize",
            },
            "EditNeighborhood": {
                "best_plan": ["draft_b", "draft_a"],
            },
        }
    )
    print()
    print("ROCKET macro result:")
    print(json.dumps(rocket_result.values["repair"], indent=2, sort_keys=True))

    pipeline = basket_rocket_pipeline(
        BasketRocketPipelineConfig(
            name="TutorialPlanningStack",
            fragments_object="WorkflowFragments",
            observation_object="ObservationContexts",
            edit_relation_object="RepairNeighborhood",
            repaired_plan_object="FinalPlan",
        )
    )
    pipeline_result = compile_to_callable(pipeline).run(
        {
            "WorkflowFragments": {
                "f0": ["observe", "analyze"],
                "f1": ["optimize"],
            },
            "ObservationContexts": {
                "plan_draft": ["f0", "f1"],
            },
            "RepairNeighborhood": {
                "best_plan": ["plan_draft"],
            },
        }
    )
    print()
    print("Composite BASKET -> ROCKET result:")
    print(json.dumps(pipeline_result.values[pipeline.port("output")], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

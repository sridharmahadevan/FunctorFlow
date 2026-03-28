from __future__ import annotations

import unittest

from FunctorFlow import (
    ADAPTER_LIBRARIES,
    FOUNDATIONS_TUTORIAL_LIBRARY,
    PLANNING_TUTORIAL_LIBRARY,
    STANDARD_ADAPTER_LIBRARY,
    BASKETWorkflowConfig,
    BasketRocketPipelineConfig,
    CompletionBlockConfig,
    Diagram,
    IncludedDiagram,
    KETBlockConfig,
    MACRO_LIBRARY,
    StructuredLMDualityConfig,
    build_macro,
    basket_rocket_pipeline,
    compile_to_callable,
    completion_block,
    db_square,
    democritus_gluing_block,
    get_adapter_library,
    get_tutorial_library,
    ket_block,
    rocket_repair_block,
    structured_lm_duality,
)


class FunctorFlowMacroTests(unittest.TestCase):
    def test_macro_registry_contains_named_blocks(self) -> None:
        self.assertIn("ket", MACRO_LIBRARY)
        self.assertIn("completion", MACRO_LIBRARY)
        self.assertIn("structured_lm_duality", MACRO_LIBRARY)
        self.assertIn("db_square", MACRO_LIBRARY)
        self.assertIn("rocket_repair", MACRO_LIBRARY)
        self.assertIn("standard", ADAPTER_LIBRARIES)
        self.assertEqual(get_tutorial_library("planning").name, "planning")

    def test_ket_macro_executes(self) -> None:
        diagram = ket_block()
        result = build_macro("ket").to_ir().as_dict()
        self.assertEqual(result["name"], "KETBlock")
        self.assertIn("aggregate", [op["name"] for op in result["operations"]])
        self.assertEqual(diagram.operations["aggregate"].metadata["macro"], "KETBlock")
        self.assertEqual(diagram.port("output"), "aggregate")
        self.assertEqual(diagram.port_type("output"), "contextualized_messages")

    def test_build_macro_accepts_diagram_name_kwarg(self) -> None:
        diagram = build_macro("ket", name="TutorialKET")
        self.assertEqual(diagram.name, "TutorialKET")

    def test_typed_ket_config_customizes_ports(self) -> None:
        diagram = ket_block(
            KETBlockConfig(
                name="EdgeAggregator",
                source_object="EdgeValues",
                relation_object="TokenIncidence",
                target_object="TokenStates",
                aggregate_name="gather",
                reducer="mean",
            )
        )
        self.assertEqual(diagram.name, "EdgeAggregator")
        self.assertIn("EdgeValues", diagram.objects)
        self.assertIn("gather", diagram.operations)

    def test_include_namespaces_subdiagram(self) -> None:
        parent = Diagram("Parent")
        include = parent.include(ket_block(), namespace="encoder")
        self.assertIsInstance(include, IncludedDiagram)
        self.assertEqual(include.object("Values"), "encoder__Values")
        self.assertEqual(include.operation("aggregate"), "encoder__aggregate")
        self.assertEqual(include.port("output"), "encoder__aggregate")
        self.assertEqual(include.port_type("output"), "contextualized_messages")
        self.assertIn("encoder__aggregate", parent.operations)

    def test_db_square_macro_wires_obstruction(self) -> None:
        diagram = db_square(first_impl=lambda value: 2 * value, second_impl=lambda value: value + 1)
        compiled = compile_to_callable(diagram)
        result = compiled.run({"State": 3.0})
        self.assertAlmostEqual(result.losses["obstruction"], 1.0)

    def test_rocket_macro_prefers_first_non_null_candidate(self) -> None:
        diagram = rocket_repair_block()
        compiled = compile_to_callable(diagram)
        result = compiled.run(
            {
                "CandidateFragments": {"a": None, "b": "observe -> optimize"},
                "EditNeighborhood": {"plan": ["a", "b"]},
            }
        )
        self.assertEqual(result.values["repair"], {"plan": "observe -> optimize"})

    def test_completion_macro_prefers_first_non_null_candidate(self) -> None:
        diagram = completion_block(
            CompletionBlockConfig(
                source_object="PartialBlocks",
                relation_object="Compatibility",
                target_object="CompletedBlocks",
                completion_name="complete",
            )
        )
        compiled = compile_to_callable(diagram)
        result = compiled.run(
            {
                "PartialBlocks": {"a": None, "b": "future tuple"},
                "Compatibility": {"block_0": ["a", "b"]},
            }
        )
        self.assertEqual(result.values["complete"], {"block_0": "future tuple"})

    def test_democritus_macro_unions_local_sections(self) -> None:
        diagram = democritus_gluing_block()
        compiled = compile_to_callable(diagram)
        result = compiled.run(
            {
                "LocalClaims": {"s1": {"growth", "brand"}, "s2": {"brand", "margin"}},
                "OverlapRegions": {"global": ["s1", "s2"]},
            }
        )
        self.assertEqual(result.values["glue"], {"global": {"growth", "brand", "margin"}})

    def test_basket_rocket_pipeline_composes_subblocks(self) -> None:
        diagram = basket_rocket_pipeline(
            BasketRocketPipelineConfig(
                name="PlanningStack",
                fragments_object="WorkflowFragments",
                observation_object="ObservationContexts",
                edit_relation_object="RepairNeighborhood",
                repaired_plan_object="FinalPlan",
            )
        )
        self.assertIn("draft__draft_plan", diagram.operations)
        self.assertIn("repair__repair", diagram.operations)
        self.assertEqual(diagram.port("draft_output"), "draft__draft_plan")
        self.assertEqual(diagram.port("output"), "repair__repair")
        self.assertEqual(diagram.port_type("draft_output"), "plan_candidates")
        self.assertEqual(diagram.port_type("output"), "plan")
        compiled = compile_to_callable(diagram)
        result = compiled.run(
            {
                "WorkflowFragments": {"a": ["observe"], "b": ["optimize"]},
                "ObservationContexts": {"draft_candidate": ["a", "b"]},
                "RepairNeighborhood": {"best_plan": ["draft_candidate"]},
            }
        )
        self.assertEqual(
            result.values[diagram.port("output")],
            {"best_plan": ["observe", "optimize"]},
        )

    def test_structured_lm_duality_exposes_left_and_right_kan_ports(self) -> None:
        diagram = structured_lm_duality(
            StructuredLMDualityConfig(
                name="StructuredLM",
                hidden_object="TokenStates",
                relation_object="TokenRelation",
                noisy_block_object="MaskedFuture",
                condition_object="RepairRelation",
            )
        )
        self.assertIn("predict__aggregate_context", diagram.operations)
        self.assertIn("repair__complete_block", diagram.operations)
        self.assertEqual(diagram.port("hidden"), "TokenStates")
        self.assertEqual(diagram.port("relation"), "TokenRelation")
        self.assertEqual(diagram.port("noisy_block"), "MaskedFuture")
        self.assertEqual(diagram.port("condition"), "RepairRelation")
        self.assertEqual(diagram.port_type("context"), "contextualized_messages")
        self.assertEqual(diagram.port_type("completed"), "completed_state")

    def test_registered_adapter_enables_port_coercion(self) -> None:
        parent = Diagram("AdapterDemo")
        parent.register_adapter(
            "context_to_candidates",
            source_type="contextualized_messages",
            target_type="plan_candidates",
            implementation=lambda value: value,
        )
        draft = parent.include(ket_block(), namespace="encoder")
        repair = parent.include(
            rocket_repair_block(),
            namespace="repair",
            object_aliases={
                "CandidateFragments": draft.port_spec("output"),
            },
        )
        self.assertIn("repair__adapt__candidates", parent.operations)
        self.assertEqual(repair.port_type("output"), "plan")
        compiled = compile_to_callable(parent)
        result = compiled.run(
            {
                "encoder__Values": {
                    "draft_a": "observe -> optimize",
                },
                "encoder__Incidence": {
                    "candidate_pool": ["draft_a"],
                },
                "repair__EditNeighborhood": {
                    "best_plan": ["candidate_pool"],
                },
            }
        )
        self.assertEqual(result.values["repair__repair"], {"best_plan": "observe -> optimize"})

    def test_standard_adapter_library_is_packaged(self) -> None:
        library = get_adapter_library("standard")
        self.assertEqual(library.name, STANDARD_ADAPTER_LIBRARY.name)
        self.assertGreaterEqual(len(library.adapters), 2)

    def test_tutorial_library_is_packaged(self) -> None:
        planning = get_tutorial_library("planning")
        self.assertEqual(planning.name, PLANNING_TUTORIAL_LIBRARY.name)
        self.assertIn("basket_rocket_pipeline", planning.macro_names)
        self.assertIn("standard", planning.adapter_library_names)

    def test_tutorial_library_restricts_macro_surface(self) -> None:
        with self.assertRaises(KeyError):
            PLANNING_TUTORIAL_LIBRARY.build_macro("ket")
        foundations_diagram = FOUNDATIONS_TUTORIAL_LIBRARY.build_macro("ket")
        self.assertEqual(foundations_diagram.name, "KETBlock")

    def test_diagram_can_install_standard_adapter_library(self) -> None:
        parent = Diagram("LibraryAdapterDemo")
        parent.use_adapter_library("standard")
        draft = parent.include(ket_block(), namespace="encoder")
        parent.include(
            rocket_repair_block(),
            namespace="repair",
            object_aliases={
                "CandidateFragments": draft.port_spec("output"),
            },
        )
        self.assertIn("repair__adapt__candidates", parent.operations)
        compiled = compile_to_callable(parent)
        result = compiled.run(
            {
                "encoder__Values": {"draft_a": "observe -> optimize"},
                "encoder__Incidence": {"candidate_pool": ["draft_a"]},
                "repair__EditNeighborhood": {"best_plan": ["candidate_pool"]},
            }
        )
        self.assertEqual(result.values["repair__repair"], {"best_plan": "observe -> optimize"})

    def test_diagram_can_install_tutorial_library(self) -> None:
        parent = Diagram("PlanningLibraryDemo")
        tutorial_library = parent.use_tutorial_library("planning")
        self.assertEqual(tutorial_library.name, "planning")
        diagram = tutorial_library.build_macro(
            "basket_rocket_pipeline",
            config=BasketRocketPipelineConfig(name="PlanningStack"),
        )
        parent.include(diagram, namespace="stack")
        self.assertIn(("contextualized_messages", "plan_candidates"), parent.adapters)

    def test_port_type_mismatch_is_rejected(self) -> None:
        parent = Diagram("Mismatch")
        draft = parent.include(ket_block(), namespace="encoder")
        with self.assertRaises(ValueError):
            parent.include(
                rocket_repair_block(),
                namespace="repair",
                object_aliases={
                    "CandidateFragments": draft.port_spec("output"),
                },
            )


if __name__ == "__main__":
    unittest.main()

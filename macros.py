from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Type, TypeVar

from .core import Diagram


ConfigT = TypeVar("ConfigT")


def _resolve_config(
    config: ConfigT | None,
    config_type: Type[ConfigT],
    overrides: dict[str, Any],
) -> ConfigT:
    if config is None:
        return config_type(**overrides)
    if not isinstance(config, config_type):
        raise TypeError(f"Expected config of type {config_type.__name__}")
    if not overrides:
        return config
    return replace(config, **overrides)


@dataclass(frozen=True)
class KETBlockConfig:
    name: str = "KETBlock"
    source_object: str = "Values"
    relation_object: str = "Incidence"
    target_object: str = "ContextualizedValues"
    aggregate_name: str = "aggregate"
    reducer: str | Callable[..., Any] = "sum"


def ket_block(
    config: KETBlockConfig | None = None,
    **overrides: Any,
) -> Diagram:
    """Build a canonical left-Kan aggregation block."""
    cfg = _resolve_config(config, KETBlockConfig, overrides)
    diagram = Diagram(cfg.name)
    diagram.object(cfg.source_object, kind="messages")
    diagram.object(cfg.relation_object, kind="relation")
    diagram.object(cfg.target_object, kind="contextualized_messages")
    diagram.left_kan(
        source=cfg.source_object,
        along=cfg.relation_object,
        name=cfg.aggregate_name,
        target=cfg.target_object,
        reducer=cfg.reducer,
        description="Universal aggregation over an incidence structure.",
        metadata={"macro": "KETBlock"},
    )
    diagram.expose_port("input", cfg.source_object, direction="input", port_type="messages")
    diagram.expose_port("relation", cfg.relation_object, direction="input", port_type="relation")
    diagram.expose_port(
        "output",
        cfg.aggregate_name,
        direction="output",
        port_type="contextualized_messages",
    )
    return diagram


@dataclass(frozen=True)
class DBSquareConfig:
    name: str = "DBSquare"
    state_object: str = "State"
    first_morphism: str = "f"
    second_morphism: str = "g"
    left_path: str = "p1"
    right_path: str = "p2"
    first_impl: Callable[[Any], Any] | None = None
    second_impl: Callable[[Any], Any] | None = None
    comparator: str | Callable[..., Any] = "l2"
    loss_name: str = "obstruction"


def db_square(
    config: DBSquareConfig | None = None,
    **overrides: Any,
) -> Diagram:
    """Build a canonical diagrammatic backpropagation square."""
    cfg = _resolve_config(config, DBSquareConfig, overrides)
    diagram = Diagram(cfg.name)
    state = diagram.object(cfg.state_object, kind="state")
    diagram.morphism(
        cfg.first_morphism,
        state,
        state,
        implementation=cfg.first_impl,
        description="First route in the commutative square.",
    )
    diagram.morphism(
        cfg.second_morphism,
        state,
        state,
        implementation=cfg.second_impl,
        description="Second route in the commutative square.",
    )
    diagram.compose(cfg.first_morphism, cfg.second_morphism, name=cfg.left_path)
    diagram.compose(cfg.second_morphism, cfg.first_morphism, name=cfg.right_path)
    diagram.obstruction_loss(
        paths=[(cfg.left_path, cfg.right_path)],
        name=cfg.loss_name,
        comparator=cfg.comparator,
        description="Measure how far the square is from commuting.",
        metadata={"macro": "DBSquare"},
    )
    diagram.expose_port("input", cfg.state_object, direction="input", port_type="state")
    diagram.expose_port("left_path", cfg.left_path, direction="output", port_type="state")
    diagram.expose_port("right_path", cfg.right_path, direction="output", port_type="state")
    diagram.expose_port(
        "loss",
        cfg.loss_name,
        direction="output",
        kind="loss",
        port_type="loss",
    )
    return diagram


@dataclass(frozen=True)
class GTNeighborhoodConfig:
    name: str = "GTNeighborhoodBlock"
    token_object: str = "Tokens"
    relation_object: str = "NeighborhoodIncidence"
    message_object: str = "Messages"
    target_object: str = "UpdatedTokens"
    aggregate_name: str = "kan_aggregate"
    reducer: str | Callable[..., Any] = "mean"
    lift_name: str = "lift_messages"


def gt_neighborhood_block(
    config: GTNeighborhoodConfig | None = None,
    **overrides: Any,
) -> Diagram:
    """Build a GT-style neighborhood aggregation block."""
    cfg = _resolve_config(config, GTNeighborhoodConfig, overrides)
    diagram = Diagram(cfg.name)
    tokens = diagram.object(cfg.token_object, kind="token_state")
    message_object_ref = diagram.object(cfg.message_object, kind="messages")
    diagram.object(cfg.relation_object, kind="simplicial_relation")
    diagram.object(cfg.target_object, kind="token_state")
    diagram.morphism(
        cfg.lift_name,
        tokens,
        message_object_ref,
        description="Lift token states into edge or simplex messages.",
    )
    diagram.left_kan(
        source=cfg.message_object,
        along=cfg.relation_object,
        name=cfg.aggregate_name,
        target=cfg.target_object,
        reducer=cfg.reducer,
        description="Aggregate neighborhood messages back onto token states.",
        metadata={"macro": "GTNeighborhoodBlock"},
    )
    diagram.expose_port("input", cfg.token_object, direction="input", port_type="token_state")
    diagram.expose_port(
        "relation",
        cfg.relation_object,
        direction="input",
        port_type="neighborhood_relation",
    )
    diagram.expose_port("messages", cfg.message_object, direction="internal", port_type="message_state")
    diagram.expose_port("output", cfg.aggregate_name, direction="output", port_type="token_state")
    return diagram


@dataclass(frozen=True)
class BASKETWorkflowConfig:
    name: str = "BASKETWorkflowBlock"
    fragment_object: str = "PlanFragments"
    observation_object: str = "ObservationContexts"
    target_object: str = "PlanState"
    aggregate_name: str = "draft_plan"
    reducer: str | Callable[..., Any] = "concat"


def basket_workflow_block(
    config: BASKETWorkflowConfig | None = None,
    **overrides: Any,
) -> Diagram:
    """Build a BASKET-style workflow composition block."""
    cfg = _resolve_config(config, BASKETWorkflowConfig, overrides)
    diagram = Diagram(cfg.name)
    diagram.object(cfg.fragment_object, kind="plan_fragments")
    diagram.object(cfg.observation_object, kind="workflow_relation")
    diagram.object(cfg.target_object, kind="plan_state")
    diagram.left_kan(
        source=cfg.fragment_object,
        along=cfg.observation_object,
        name=cfg.aggregate_name,
        target=cfg.target_object,
        reducer=cfg.reducer,
        description="Compose local plan fragments into a workflow draft.",
        metadata={"macro": "BASKETWorkflowBlock"},
    )
    diagram.expose_port(
        "fragments",
        cfg.fragment_object,
        direction="input",
        port_type="plan_fragments",
    )
    diagram.expose_port(
        "context",
        cfg.observation_object,
        direction="input",
        port_type="workflow_relation",
    )
    diagram.expose_port("output", cfg.aggregate_name, direction="output", port_type="plan_candidates")
    return diagram


@dataclass(frozen=True)
class ROCKETRepairConfig:
    name: str = "ROCKETRepairBlock"
    source_object: str = "CandidateFragments"
    relation_object: str = "EditNeighborhood"
    target_object: str = "RepairedPlan"
    repair_name: str = "repair"
    reducer: str | Callable[..., Any] = "first_non_null"


def rocket_repair_block(
    config: ROCKETRepairConfig | None = None,
    **overrides: Any,
) -> Diagram:
    """Build a ROCKET-style local repair block."""
    cfg = _resolve_config(config, ROCKETRepairConfig, overrides)
    diagram = Diagram(cfg.name)
    diagram.object(cfg.source_object, kind="candidate_fragments")
    diagram.object(cfg.relation_object, kind="edit_relation")
    diagram.object(cfg.target_object, kind="plan_state")
    diagram.right_kan(
        source=cfg.source_object,
        along=cfg.relation_object,
        name=cfg.repair_name,
        target=cfg.target_object,
        reducer=cfg.reducer,
        description="Complete a plan from locally edited candidates.",
        metadata={"macro": "ROCKETRepairBlock"},
    )
    diagram.expose_port(
        "candidates",
        cfg.source_object,
        direction="input",
        port_type="plan_candidates",
    )
    diagram.expose_port(
        "relation",
        cfg.relation_object,
        direction="input",
        port_type="edit_relation",
    )
    diagram.expose_port("output", cfg.repair_name, direction="output", port_type="plan")
    return diagram


@dataclass(frozen=True)
class CompletionBlockConfig:
    name: str = "CompletionBlock"
    source_object: str = "PartialState"
    relation_object: str = "CompatibilityRelation"
    target_object: str = "CompletedState"
    completion_name: str = "complete"
    reducer: str | Callable[..., Any] = "first_non_null"


def completion_block(
    config: CompletionBlockConfig | None = None,
    **overrides: Any,
) -> Diagram:
    """Build a generic right-Kan completion block."""
    cfg = _resolve_config(config, CompletionBlockConfig, overrides)
    diagram = Diagram(cfg.name)
    diagram.object(cfg.source_object, kind="partial_state")
    diagram.object(cfg.relation_object, kind="compatibility_relation")
    diagram.object(cfg.target_object, kind="completed_state")
    diagram.right_kan(
        source=cfg.source_object,
        along=cfg.relation_object,
        name=cfg.completion_name,
        target=cfg.target_object,
        reducer=cfg.reducer,
        description="Complete a partial state under an explicit compatibility relation.",
        metadata={"macro": "CompletionBlock"},
    )
    diagram.expose_port("input", cfg.source_object, direction="input", port_type="partial_state")
    diagram.expose_port(
        "relation",
        cfg.relation_object,
        direction="input",
        port_type="compatibility_relation",
    )
    diagram.expose_port("output", cfg.completion_name, direction="output", port_type="completed_state")
    return diagram


@dataclass(frozen=True)
class StructuredLMDualityConfig:
    name: str = "StructuredLMDuality"
    hidden_object: str = "HiddenStates"
    relation_object: str = "CausalRelation"
    context_object: str = "ContextualizedStates"
    noisy_block_object: str = "NoisyBlock"
    condition_object: str = "DenoiseCondition"
    completed_object: str = "CompletedBlock"
    predict_namespace: str = "predict"
    repair_namespace: str = "repair"
    attention_reducer: str | Callable[..., Any] = "ket_attention"
    completion_reducer: str | Callable[..., Any] = "first_non_null"


def structured_lm_duality(
    config: StructuredLMDualityConfig | None = None,
    **overrides: Any,
) -> Diagram:
    """Compose left-Kan attention and right-Kan completion for structured LM."""
    cfg = _resolve_config(config, StructuredLMDualityConfig, overrides)
    diagram = Diagram(cfg.name)
    predict = diagram.include(
        ket_block(
            KETBlockConfig(
                name="StructuredPredictor",
                source_object="HiddenStates",
                relation_object="CausalRelation",
                target_object="ContextualizedStates",
                aggregate_name="aggregate_context",
                reducer=cfg.attention_reducer,
            )
        ),
        namespace=cfg.predict_namespace,
        object_aliases={
            "HiddenStates": cfg.hidden_object,
            "CausalRelation": cfg.relation_object,
            "ContextualizedStates": cfg.context_object,
        },
    )
    repair = diagram.include(
        completion_block(
            CompletionBlockConfig(
                name="StructuredCompletion",
                source_object="NoisyBlock",
                relation_object="DenoiseCondition",
                target_object="CompletedBlock",
                completion_name="complete_block",
                reducer=cfg.completion_reducer,
            )
        ),
        namespace=cfg.repair_namespace,
        object_aliases={
            "NoisyBlock": cfg.noisy_block_object,
            "DenoiseCondition": cfg.condition_object,
            "CompletedBlock": cfg.completed_object,
        },
    )
    diagram.expose_port("hidden", cfg.hidden_object, direction="input", port_type="messages")
    diagram.expose_port(
        "relation",
        cfg.relation_object,
        direction="input",
        port_type="relation",
    )
    diagram.expose_port(
        "context",
        predict.port_spec("output"),
        direction="output",
        port_type=predict.port_type("output"),
    )
    diagram.expose_port(
        "noisy_block",
        cfg.noisy_block_object,
        direction="input",
        port_type="partial_state",
    )
    diagram.expose_port(
        "condition",
        cfg.condition_object,
        direction="input",
        port_type="compatibility_relation",
    )
    diagram.expose_port(
        "completed",
        repair.port_spec("output"),
        direction="output",
        port_type=repair.port_type("output"),
    )
    return diagram


@dataclass(frozen=True)
class DemocritusGluingConfig:
    name: str = "DemocritusGluingBlock"
    source_object: str = "LocalClaims"
    relation_object: str = "OverlapRegions"
    target_object: str = "GlobalManifold"
    gluing_name: str = "glue"
    reducer: str | Callable[..., Any] = "set_union"


def democritus_gluing_block(
    config: DemocritusGluingConfig | None = None,
    **overrides: Any,
) -> Diagram:
    """Build a Democritus-style local-to-global gluing block."""
    cfg = _resolve_config(config, DemocritusGluingConfig, overrides)
    diagram = Diagram(cfg.name)
    diagram.object(cfg.source_object, kind="local_sections")
    diagram.object(cfg.relation_object, kind="overlap_relation")
    diagram.object(cfg.target_object, kind="global_state")
    diagram.right_kan(
        source=cfg.source_object,
        along=cfg.relation_object,
        name=cfg.gluing_name,
        target=cfg.target_object,
        reducer=cfg.reducer,
        description="Glue compatible local claims into a shared global state.",
        metadata={"macro": "DemocritusGluingBlock"},
    )
    diagram.expose_port("locals", cfg.source_object, direction="input", port_type="local_sections")
    diagram.expose_port(
        "relation",
        cfg.relation_object,
        direction="input",
        port_type="overlap_relation",
    )
    diagram.expose_port("output", cfg.gluing_name, direction="output", port_type="global_state")
    return diagram


@dataclass(frozen=True)
class BasketRocketPipelineConfig:
    name: str = "BASKETROCKETPipeline"
    fragments_object: str = "PlanFragments"
    observation_object: str = "ObservationContexts"
    edit_relation_object: str = "EditNeighborhood"
    repaired_plan_object: str = "RepairedPlan"
    draft_namespace: str = "draft"
    repair_namespace: str = "repair"
    draft_reducer: str | Callable[..., Any] = "concat"
    repair_reducer: str | Callable[..., Any] = "first_non_null"


def basket_rocket_pipeline(
    config: BasketRocketPipelineConfig | None = None,
    **overrides: Any,
) -> Diagram:
    """Compose BASKET drafting with ROCKET repair as a namespaced pipeline."""
    cfg = _resolve_config(config, BasketRocketPipelineConfig, overrides)
    diagram = Diagram(cfg.name)

    draft = diagram.include(
        basket_workflow_block(
            BASKETWorkflowConfig(reducer=cfg.draft_reducer)
        ),
        namespace=cfg.draft_namespace,
        object_aliases={
            "PlanFragments": cfg.fragments_object,
            "ObservationContexts": cfg.observation_object,
        },
    )
    repair = diagram.include(
        rocket_repair_block(
            ROCKETRepairConfig(
                relation_object="EditNeighborhood",
                target_object="RepairedPlan",
                reducer=cfg.repair_reducer,
            )
        ),
        namespace=cfg.repair_namespace,
        object_aliases={
            "CandidateFragments": draft.port_spec("output"),
            "EditNeighborhood": cfg.edit_relation_object,
            "RepairedPlan": cfg.repaired_plan_object,
        },
    )
    diagram.expose_port("fragments", cfg.fragments_object, direction="input", port_type="plan_fragments")
    diagram.expose_port(
        "context",
        cfg.observation_object,
        direction="input",
        port_type="workflow_relation",
    )
    diagram.expose_port(
        "repair_relation",
        cfg.edit_relation_object,
        direction="input",
        port_type="edit_relation",
    )
    diagram.expose_port(
        "draft_output",
        draft.port_spec("output"),
        direction="internal",
        port_type=draft.port_type("output"),
    )
    diagram.expose_port(
        "output",
        repair.port_spec("output"),
        direction="output",
        port_type=repair.port_type("output"),
    )
    return diagram


MACRO_LIBRARY: Dict[str, Callable[..., Diagram]] = {
    "ket": ket_block,
    "completion": completion_block,
    "structured_lm_duality": structured_lm_duality,
    "db_square": db_square,
    "gt_neighborhood": gt_neighborhood_block,
    "basket_workflow": basket_workflow_block,
    "rocket_repair": rocket_repair_block,
    "democritus_gluing": democritus_gluing_block,
    "basket_rocket_pipeline": basket_rocket_pipeline,
}


def build_macro(macro_name: str, **kwargs: Any) -> Diagram:
    try:
        factory = MACRO_LIBRARY[macro_name]
    except KeyError as exc:
        available = ", ".join(sorted(MACRO_LIBRARY))
        raise KeyError(
            f"Unknown FunctorFlow macro '{macro_name}'. Available: {available}"
        ) from exc
    return factory(**kwargs)

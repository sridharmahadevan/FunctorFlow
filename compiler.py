from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .core import Composition, Diagram, KanExtension, Morphism, ObstructionLoss


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_group_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(_is_sequence(v) for v in value.values())


def _normalize_relation(relation: Any) -> dict[Any, list[Any]]:
    if isinstance(relation, Mapping):
        if not relation:
            return {}
        if _is_group_mapping(relation):
            return {target: list(source_keys) for target, source_keys in relation.items()}
        inverted: dict[Any, list[Any]] = {}
        for source_key, target_key in relation.items():
            inverted.setdefault(target_key, []).append(source_key)
        return inverted
    if _is_sequence(relation):
        inverted: dict[Any, list[Any]] = {}
        for pair in relation:
            if not (_is_sequence(pair) and len(pair) == 2):
                raise TypeError("Relation pairs must be length-2 sequences")
            source_key, target_key = pair
            inverted.setdefault(target_key, []).append(source_key)
        return inverted
    raise TypeError(
        "Kan relations must be mappings or source-target pair sequences in the v0 compiler"
    )


def _lookup_source_value(source_value: Any, source_key: Any) -> Any:
    if isinstance(source_value, Mapping):
        return source_value[source_key]
    if _is_sequence(source_value) and isinstance(source_key, int):
        return source_value[source_key]
    raise KeyError(source_key)


def _group_values(source_value: Any, relation: Any) -> dict[Any, list[Any]]:
    normalized = _normalize_relation(relation)
    grouped: dict[Any, list[Any]] = {}
    for target_key, source_keys in normalized.items():
        values: list[Any] = []
        for source_key in source_keys:
            try:
                values.append(_lookup_source_value(source_value, source_key))
            except KeyError:
                continue
        grouped[target_key] = values
    return grouped


def _sum_values(values: list[Any]) -> Any:
    if not values:
        return 0
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def _mean_values(values: list[Any]) -> Any:
    if not values:
        return 0.0
    return _sum_values(values) / len(values)


def _first_non_null(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _concat_values(values: list[Any]) -> Any:
    if not values:
        return []
    if all(isinstance(value, str) for value in values):
        return "".join(values)
    result: list[Any] = []
    for value in values:
        if _is_sequence(value):
            result.extend(value)
        else:
            result.append(value)
    return result


def _majority_value(values: list[Any]) -> Any:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _set_union(values: list[Any]) -> Any:
    result: set[Any] = set()
    for value in values:
        if isinstance(value, set):
            result |= value
        elif _is_sequence(value):
            result |= set(value)
        else:
            result.add(value)
    return result


def _apply_group_reducer(
    source_value: Any,
    relation: Any,
    reducer: Callable[[list[Any]], Any],
) -> dict[Any, Any]:
    grouped = _group_values(source_value, relation)
    return {target_key: reducer(values) for target_key, values in grouped.items()}


def _flatten_numeric(value: Any) -> list[float]:
    if value is None:
        return [0.0]
    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Mapping):
        flattened: list[float] = []
        for key in sorted(value):
            flattened.extend(_flatten_numeric(value[key]))
        return flattened
    if _is_sequence(value):
        flattened = []
        for item in value:
            flattened.extend(_flatten_numeric(item))
        return flattened
    raise TypeError(f"Cannot compute numeric norm for value of type {type(value)!r}")


def _l2_distance(left: Any, right: Any) -> float:
    left_values = _flatten_numeric(left)
    right_values = _flatten_numeric(right)
    if len(left_values) != len(right_values):
        raise ValueError("Obstruction comparisons require equally sized numeric structures")
    total = 0.0
    for left_value, right_value in zip(left_values, right_values):
        diff = left_value - right_value
        total += diff * diff
    return total ** 0.5


def _l1_distance(left: Any, right: Any) -> float:
    left_values = _flatten_numeric(left)
    right_values = _flatten_numeric(right)
    if len(left_values) != len(right_values):
        raise ValueError("Obstruction comparisons require equally sized numeric structures")
    total = 0.0
    for left_value, right_value in zip(left_values, right_values):
        total += abs(left_value - right_value)
    return total


BUILTIN_REDUCERS: dict[str, Callable[..., Any]] = {
    "sum": lambda source, relation, metadata: _apply_group_reducer(source, relation, _sum_values),
    "mean": lambda source, relation, metadata: _apply_group_reducer(source, relation, _mean_values),
    "first_non_null": lambda source, relation, metadata: _apply_group_reducer(
        source, relation, _first_non_null
    ),
    "concat": lambda source, relation, metadata: _apply_group_reducer(
        source, relation, _concat_values
    ),
    "majority": lambda source, relation, metadata: _apply_group_reducer(
        source, relation, _majority_value
    ),
    "set_union": lambda source, relation, metadata: _apply_group_reducer(
        source, relation, _set_union
    ),
    "tuple": lambda source, relation, metadata: _apply_group_reducer(source, relation, tuple),
}

BUILTIN_COMPARATORS: dict[str, Callable[[Any, Any], float]] = {
    "l2": _l2_distance,
    "l1": _l1_distance,
}


@dataclass
class ExecutionResult:
    values: dict[str, Any]
    losses: dict[str, float]


class CompiledDiagram:
    """Backend-neutral executable form of a FunctorFlow diagram."""

    def __init__(
        self,
        diagram: Diagram,
        *,
        morphisms: Mapping[str, Callable[..., Any]] | None = None,
        reducers: Mapping[str, Callable[..., Any]] | None = None,
        comparators: Mapping[str, Callable[..., Any]] | None = None,
    ):
        self.diagram = diagram
        self.morphisms = dict(diagram.implementations)
        self.reducers = {**BUILTIN_REDUCERS, **dict(diagram.reducers)}
        self.comparators = {**BUILTIN_COMPARATORS, **dict(diagram.comparators)}
        if morphisms:
            self.morphisms.update(morphisms)
        if reducers:
            self.reducers.update(reducers)
        if comparators:
            self.comparators.update(comparators)

    def run(
        self,
        inputs: Mapping[str, Any],
        *,
        morphisms: Mapping[str, Callable[..., Any]] | None = None,
        reducers: Mapping[str, Callable[..., Any]] | None = None,
        comparators: Mapping[str, Callable[..., Any]] | None = None,
    ) -> ExecutionResult:
        env: dict[str, Any] = dict(inputs)
        morphism_registry = dict(self.morphisms)
        reducer_registry = dict(self.reducers)
        comparator_registry = dict(self.comparators)
        if morphisms:
            morphism_registry.update(morphisms)
        if reducers:
            reducer_registry.update(reducers)
        if comparators:
            comparator_registry.update(comparators)

        for operation in self.diagram.operations.values():
            if isinstance(operation, Morphism):
                env[operation.name] = self._execute_morphism(operation, env, morphism_registry)
            elif isinstance(operation, Composition):
                env[operation.name] = self._execute_composition(operation, env, morphism_registry)
            elif isinstance(operation, KanExtension):
                env[operation.name] = self._execute_kan(operation, env, reducer_registry)
            else:
                raise TypeError(f"Unsupported operation type {type(operation)!r}")

        losses = {
            loss.name: self._execute_loss(loss, env, comparator_registry)
            for loss in self.diagram.losses.values()
        }
        return ExecutionResult(values=env, losses=losses)

    def _execute_morphism(
        self,
        morphism: Morphism,
        env: Mapping[str, Any],
        registry: Mapping[str, Callable[..., Any]],
    ) -> Any:
        if morphism.source not in env:
            raise KeyError(f"Missing source value '{morphism.source}' for morphism '{morphism.name}'")
        fn = registry.get(morphism.name)
        if fn is None and morphism.implementation_key is not None:
            fn = registry.get(morphism.implementation_key)
        if fn is None:
            raise KeyError(f"No implementation bound for morphism '{morphism.name}'")
        return fn(env[morphism.source])

    def _execute_composition(
        self,
        composition: Composition,
        env: Mapping[str, Any],
        registry: Mapping[str, Callable[..., Any]],
    ) -> Any:
        if composition.source is None:
            raise ValueError(f"Composition '{composition.name}' is missing a source object")
        if composition.source not in env:
            raise KeyError(
                f"Missing source value '{composition.source}' for composition '{composition.name}'"
            )
        current = env[composition.source]
        for morphism_name in composition.chain:
            fn = registry.get(morphism_name)
            if fn is None:
                raise KeyError(
                    f"No implementation bound for morphism '{morphism_name}' in composition '{composition.name}'"
                )
            current = fn(current)
        return current

    def _execute_kan(
        self,
        operation: KanExtension,
        env: Mapping[str, Any],
        registry: Mapping[str, Callable[..., Any]],
    ) -> Any:
        if operation.source not in env:
            raise KeyError(f"Missing source value '{operation.source}' for Kan extension '{operation.name}'")
        if operation.along not in env:
            raise KeyError(f"Missing relation '{operation.along}' for Kan extension '{operation.name}'")
        reducer = registry.get(operation.reducer)
        if reducer is None:
            raise KeyError(
                f"No reducer bound for Kan extension '{operation.name}' with reducer '{operation.reducer}'"
            )
        metadata = {"direction": operation.direction, **operation.metadata}
        return reducer(env[operation.source], env[operation.along], metadata)

    def _execute_loss(
        self,
        loss: ObstructionLoss,
        env: Mapping[str, Any],
        registry: Mapping[str, Callable[..., Any]],
    ) -> float:
        comparator = registry.get(loss.comparator)
        if comparator is None:
            raise KeyError(
                f"No comparator bound for obstruction loss '{loss.name}' with comparator '{loss.comparator}'"
            )
        total = 0.0
        for left_name, right_name in loss.paths:
            if left_name not in env or right_name not in env:
                raise KeyError(
                    f"Obstruction loss '{loss.name}' requires values for '{left_name}' and '{right_name}'"
                )
            total += float(comparator(env[left_name], env[right_name]))
        return loss.weight * total


def compile_to_callable(
    diagram: Diagram,
    *,
    morphisms: Mapping[str, Callable[..., Any]] | None = None,
    reducers: Mapping[str, Callable[..., Any]] | None = None,
    comparators: Mapping[str, Callable[..., Any]] | None = None,
) -> CompiledDiagram:
    return CompiledDiagram(
        diagram,
        morphisms=morphisms,
        reducers=reducers,
        comparators=comparators,
    )


def compile_to_torch(
    diagram: Diagram,
    *,
    morphisms: Mapping[str, Callable[..., Any]] | None = None,
    reducers: Mapping[str, Callable[..., Any]] | None = None,
    comparators: Mapping[str, Callable[..., Any]] | None = None,
):
    try:
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Torch is not installed in this workspace. Install torch to lower FunctorFlow diagrams "
            "to nn.Module, or use compile_to_callable() for the backend-neutral runtime."
        ) from exc

    compiled = compile_to_callable(
        diagram,
        morphisms=morphisms,
        reducers=reducers,
        comparators=comparators,
    )

    class TorchCompiledDiagram(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.diagram = diagram
            morphism_modules = {
                name: impl for name, impl in compiled.morphisms.items() if isinstance(impl, nn.Module)
            }
            reducer_modules = {
                name: reducer for name, reducer in compiled.reducers.items() if isinstance(reducer, nn.Module)
            }
            comparator_modules = {
                name: comparator
                for name, comparator in compiled.comparators.items()
                if isinstance(comparator, nn.Module)
            }
            self.lowered_morphisms = nn.ModuleDict(morphism_modules)
            self.lowered_reducers = nn.ModuleDict(reducer_modules)
            self.lowered_comparators = nn.ModuleDict(comparator_modules)

        def forward(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
            local_morphisms = dict(compiled.morphisms)
            local_morphisms.update(self.lowered_morphisms)
            local_reducers = dict(compiled.reducers)
            local_reducers.update(self.lowered_reducers)
            local_comparators = dict(compiled.comparators)
            local_comparators.update(self.lowered_comparators)
            return compiled.run(
                inputs,
                morphisms=local_morphisms,
                reducers=local_reducers,
                comparators=local_comparators,
            ).values

    return TorchCompiledDiagram()

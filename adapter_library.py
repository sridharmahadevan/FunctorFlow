from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    source_type: str
    target_type: str
    implementation: Callable[..., Any]
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterLibrary:
    name: str
    adapters: tuple[AdapterSpec, ...]
    description: str = ""


def _identity(value: Any) -> Any:
    return value


def _string_plan_to_steps(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _string_plan_to_steps(inner_value) for key, inner_value in value.items()
        }
    if isinstance(value, str):
        return [part.strip() for part in value.split("->")]
    return value


STANDARD_ADAPTER_LIBRARY = AdapterLibrary(
    name="standard",
    description="Default FunctorFlow adapter pack for early tutorial workflows.",
    adapters=(
        AdapterSpec(
            name="context_to_candidates",
            source_type="contextualized_messages",
            target_type="plan_candidates",
            implementation=_identity,
            description="Treat contextualized message collections as candidate plan pools.",
            metadata={"library": "standard"},
        ),
        AdapterSpec(
            name="plan_candidates_to_plan",
            source_type="plan_candidates",
            target_type="plan",
            implementation=_identity,
            description="Collapse a selected candidate pool into a concrete plan representation.",
            metadata={"library": "standard"},
        ),
        AdapterSpec(
            name="string_plan_to_plan_steps",
            source_type="plan",
            target_type="plan_steps",
            implementation=_string_plan_to_steps,
            description="Convert string-serialized plans into tokenized step lists.",
            metadata={"library": "standard"},
        ),
    ),
)


ADAPTER_LIBRARIES: Dict[str, AdapterLibrary] = {
    STANDARD_ADAPTER_LIBRARY.name: STANDARD_ADAPTER_LIBRARY,
}


def get_adapter_library(name: str) -> AdapterLibrary:
    try:
        return ADAPTER_LIBRARIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(ADAPTER_LIBRARIES))
        raise KeyError(f"Unknown FunctorFlow adapter library '{name}'. Available: {available}") from exc


def iter_adapter_specs(library: str | AdapterLibrary) -> Iterable[AdapterSpec]:
    if isinstance(library, str):
        return get_adapter_library(library).adapters
    return library.adapters

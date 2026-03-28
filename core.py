from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

from .adapter_library import AdapterLibrary, AdapterSpec, get_adapter_library, iter_adapter_specs


def _ref_name(ref: Any) -> str:
    if isinstance(ref, str):
        return ref
    if hasattr(ref, "name"):
        return str(ref.name)
    raise TypeError(f"Expected a named reference or string, received {type(ref)!r}")


def _metadata_dict(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


@dataclass
class Object:
    name: str
    kind: str = "object"
    shape: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Morphism:
    name: str
    source: str | Object
    target: str | Object
    description: str = ""
    implementation_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source = _ref_name(self.source)
        self.target = _ref_name(self.target)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Composition:
    name: str
    chain: Sequence[str | Morphism]
    source: str | None = None
    target: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chain = tuple(_ref_name(part) for part in self.chain)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KanExtension:
    name: str
    direction: str
    source: str | Object | Morphism | Composition
    along: str | Object | Morphism | Composition
    target: str | Object | None = None
    reducer: str = "sum"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in {"left", "right"}:
            raise ValueError("KanExtension.direction must be 'left' or 'right'")
        self.source = _ref_name(self.source)
        self.along = _ref_name(self.along)
        if self.target is not None:
            self.target = _ref_name(self.target)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObstructionLoss:
    name: str
    paths: Sequence[tuple[str | Morphism | Composition, str | Morphism | Composition]]
    comparator: str = "l2"
    weight: float = 1.0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.paths = tuple((_ref_name(left), _ref_name(right)) for left, right in self.paths)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Port:
    name: str
    ref: str
    kind: str
    port_type: str
    direction: str = "internal"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Adapter:
    name: str
    source_type: str
    target_type: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Operation = Union[Morphism, Composition, KanExtension]


@dataclass
class DiagramIR:
    name: str
    objects: list[Object]
    operations: list[dict[str, Any]]
    losses: list[ObstructionLoss]
    ports: list[Port]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objects": [obj.to_dict() for obj in self.objects],
            "operations": list(self.operations),
            "losses": [loss.to_dict() for loss in self.losses],
            "ports": [port.to_dict() for port in self.ports],
        }


@dataclass
class IncludedDiagram:
    namespace: str
    diagram_name: str
    object_map: dict[str, str]
    operation_map: dict[str, str]
    loss_map: dict[str, str]
    port_specs: dict[str, Port]

    def object(self, name: str) -> str:
        return self.object_map[name]

    def operation(self, name: str) -> str:
        return self.operation_map[name]

    def loss(self, name: str) -> str:
        return self.loss_map[name]

    def port(self, name: str) -> str:
        return self.port_specs[name].ref

    def port_type(self, name: str) -> str:
        return self.port_specs[name].port_type

    def port_spec(self, name: str) -> Port:
        return self.port_specs[name]


class Diagram:
    """A small DSL for categorical model sketches and structural constraints."""

    def __init__(self, name: str):
        self.name = name
        self.objects: "OrderedDict[str, Object]" = OrderedDict()
        self.operations: "OrderedDict[str, Operation]" = OrderedDict()
        self.losses: "OrderedDict[str, ObstructionLoss]" = OrderedDict()
        self.ports: "OrderedDict[str, Port]" = OrderedDict()
        self._implementations: dict[str, Callable[..., Any]] = {}
        self._reducers: dict[str, Callable[..., Any]] = {}
        self._comparators: dict[str, Callable[..., Any]] = {}
        self._adapters: dict[tuple[str, str], Adapter] = {}
        self._adapter_implementations: dict[str, Callable[..., Any]] = {}

    def object(
        self,
        name: str,
        *,
        kind: str = "object",
        shape: str | None = None,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Object:
        obj = Object(
            name=name,
            kind=kind,
            shape=shape,
            description=description,
            metadata=_metadata_dict(metadata),
        )
        self.add(obj)
        return obj

    def morphism(
        self,
        name: str,
        source: str | Object,
        target: str | Object,
        *,
        implementation: Callable[..., Any] | None = None,
        implementation_key: str | None = None,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Morphism:
        morphism = Morphism(
            name=name,
            source=source,
            target=target,
            description=description,
            implementation_key=implementation_key,
            metadata=_metadata_dict(metadata),
        )
        self.add(morphism)
        if implementation is not None:
            self.bind_morphism(name, implementation)
        return morphism

    def compose(
        self,
        *chain: str | Morphism,
        name: str,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Composition:
        if not chain:
            raise ValueError("compose() requires at least one morphism")
        chain_names = tuple(_ref_name(part) for part in chain)
        source, target = self._infer_chain_endpoints(chain_names)
        composition = Composition(
            name=name,
            chain=chain_names,
            source=source,
            target=target,
            description=description,
            metadata=_metadata_dict(metadata),
        )
        self.add(composition)
        return composition

    def left_kan(
        self,
        *,
        source: str | Object | Morphism | Composition,
        along: str | Object | Morphism | Composition,
        name: str | None = None,
        target: str | Object | None = None,
        reducer: str | Callable[..., Any] = "sum",
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> KanExtension:
        return self._kan(
            direction="left",
            source=source,
            along=along,
            name=name,
            target=target,
            reducer=reducer,
            description=description,
            metadata=metadata,
        )

    def right_kan(
        self,
        *,
        source: str | Object | Morphism | Composition,
        along: str | Object | Morphism | Composition,
        name: str | None = None,
        target: str | Object | None = None,
        reducer: str | Callable[..., Any] = "first_non_null",
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> KanExtension:
        return self._kan(
            direction="right",
            source=source,
            along=along,
            name=name,
            target=target,
            reducer=reducer,
            description=description,
            metadata=metadata,
        )

    def obstruction_loss(
        self,
        *,
        paths: Sequence[tuple[str | Morphism | Composition, str | Morphism | Composition]],
        name: str | None = None,
        comparator: str | Callable[..., Any] = "l2",
        weight: float = 1.0,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ObstructionLoss:
        loss_name = name or f"obstruction_{len(self.losses)}"
        comparator_key = comparator if isinstance(comparator, str) else loss_name
        if callable(comparator):
            self.bind_comparator(comparator_key, comparator)
        loss = ObstructionLoss(
            name=loss_name,
            paths=paths,
            comparator=str(comparator_key),
            weight=weight,
            description=description,
            metadata=_metadata_dict(metadata),
        )
        self._ensure_unique(loss.name)
        self.losses[loss.name] = loss
        return loss

    def add(self, *items: Object | Operation | ObstructionLoss) -> None:
        for item in items:
            if isinstance(item, Object):
                self._register_object(item)
                continue
            if isinstance(item, Morphism):
                self._register_endpoint_placeholder(item.source)
                self._register_endpoint_placeholder(item.target)
                self._ensure_unique(item.name)
                self.operations[item.name] = item
                continue
            if isinstance(item, Composition):
                self._infer_chain_endpoints(item.chain)
                self._ensure_unique(item.name)
                if item.source is None or item.target is None:
                    item.source, item.target = self._infer_chain_endpoints(item.chain)
                self.operations[item.name] = item
                continue
            if isinstance(item, KanExtension):
                if item.target is not None:
                    self._register_endpoint_placeholder(item.target)
                self._ensure_unique(item.name)
                self.operations[item.name] = item
                continue
            if isinstance(item, ObstructionLoss):
                self._ensure_unique(item.name)
                self.losses[item.name] = item
                continue
            raise TypeError(f"Unsupported diagram item: {type(item)!r}")

    def bind_morphism(self, name: str, implementation: Callable[..., Any]) -> None:
        self._implementations[name] = implementation

    def bind_reducer(self, name: str, reducer: Callable[..., Any]) -> None:
        self._reducers[name] = reducer

    def bind_comparator(self, name: str, comparator: Callable[..., Any]) -> None:
        self._comparators[name] = comparator

    def register_adapter(
        self,
        name: str,
        *,
        source_type: str,
        target_type: str,
        implementation: Callable[..., Any] | None = None,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Adapter:
        key = (source_type, target_type)
        if key in self._adapters and self._adapters[key].name != name:
            raise ValueError(
                f"Adapter for {source_type} -> {target_type} already registered as "
                f"'{self._adapters[key].name}'"
            )
        adapter = Adapter(
            name=name,
            source_type=source_type,
            target_type=target_type,
            description=description,
            metadata=_metadata_dict(metadata),
        )
        self._adapters[key] = adapter
        if implementation is not None:
            self._adapter_implementations[name] = implementation
            self.bind_morphism(name, implementation)
        return adapter

    def use_adapter_library(self, library: str | AdapterLibrary) -> None:
        for spec in iter_adapter_specs(library):
            self.register_adapter(
                spec.name,
                source_type=spec.source_type,
                target_type=spec.target_type,
                implementation=spec.implementation,
                description=spec.description,
                metadata=spec.metadata,
            )

    def use_tutorial_library(self, library) -> Any:
        from .tutorial_library import install_tutorial_library

        return install_tutorial_library(self, library)

    def coerce(
        self,
        ref: str | Object | Morphism | Composition | KanExtension | ObstructionLoss | Port,
        *,
        to_type: str,
        from_type: str | None = None,
        name: str | None = None,
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        ref_name = self._resolve_ref(ref)
        actual_type = from_type
        if actual_type is None:
            if isinstance(ref, Port):
                actual_type = ref.port_type
            else:
                actual_type = self._infer_ref_port_type(ref_name)
        if self._port_types_compatible(actual_type, to_type):
            return ref_name
        adapter = self._adapters.get((actual_type, to_type))
        if adapter is None:
            raise ValueError(f"No adapter registered for {actual_type} -> {to_type}")

        op_name = name or f"adapt_{len(self.operations)}"
        target_name = f"{op_name}__out"
        if target_name not in self.objects:
            self.object(
                target_name,
                kind=to_type,
                description=f"Adapter output for '{op_name}'",
            )
        self.morphism(
            op_name,
            ref_name,
            target_name,
            implementation_key=adapter.name,
            description=description or adapter.description or f"Adapt {actual_type} -> {to_type}",
            metadata={
                **dict(metadata or {}),
                "adapter": adapter.name,
                "source_type": actual_type,
                "target_type": to_type,
            },
        )
        if adapter.name in self._adapter_implementations:
            self.bind_morphism(adapter.name, self._adapter_implementations[adapter.name])
        return op_name

    def expose_port(
        self,
        name: str,
        ref: str | Object | Morphism | Composition | KanExtension | ObstructionLoss | Port,
        *,
        kind: str | None = None,
        port_type: str | None = None,
        direction: str = "internal",
        description: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Port:
        ref_name = self._resolve_ref(ref)
        resolved_kind = kind or self._infer_ref_kind(ref_name)
        resolved_port_type = port_type or self._infer_ref_port_type(ref_name)
        if name in self.ports:
            raise ValueError(f"Duplicate FunctorFlow port '{name}'")
        port = Port(
            name=name,
            ref=ref_name,
            kind=resolved_kind,
            port_type=resolved_port_type,
            direction=direction,
            description=description,
            metadata=_metadata_dict(metadata),
        )
        self.ports[name] = port
        return port

    def port(self, name: str) -> str:
        return self.ports[name].ref

    def port_type(self, name: str) -> str:
        return self.ports[name].port_type

    def port_spec(self, name: str) -> Port:
        return self.ports[name]

    def include(
        self,
        diagram: "Diagram",
        *,
        namespace: str,
        object_aliases: Mapping[str, str] | None = None,
    ) -> IncludedDiagram:
        object_aliases = dict(object_aliases or {})
        def resolve_alias(alias: Any) -> tuple[str, str | None]:
            if isinstance(alias, Port):
                return alias.ref, alias.port_type
            resolved_ref = self._resolve_ref(alias)
            actual_type = None
            if (
                resolved_ref in self.objects
                or resolved_ref in self.operations
                or resolved_ref in self.losses
            ):
                actual_type = self._infer_ref_port_type(resolved_ref)
            return resolved_ref, actual_type

        resolved_aliases = {
            name: resolve_alias(alias_value) for name, alias_value in object_aliases.items()
        }
        object_map = {
            name: resolved_aliases.get(name, (f"{namespace}__{name}", None))[0]
            for name in diagram.objects
        }
        operation_map = {
            name: f"{namespace}__{name}" for name in diagram.operations
        }
        loss_map = {
            name: f"{namespace}__{name}" for name in diagram.losses
        }
        reducer_map = {
            name: f"{namespace}__{name}" for name in diagram.reducers
        }
        comparator_map = {
            name: f"{namespace}__{name}" for name in diagram.comparators
        }
        implementation_map = {
            name: f"{namespace}__{name}" for name in diagram.implementations
        }

        def map_ref(name: str | None) -> str | None:
            if name is None:
                return None
            if name in operation_map:
                return operation_map[name]
            if name in object_map:
                return object_map[name]
            return name

        def merged_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
            return {
                **dict(metadata),
                "namespace": namespace,
                "included_from": diagram.name,
            }

        for port in diagram.ports.values():
            if port.direction != "input":
                continue
            if port.ref not in object_aliases:
                continue
            actual_ref, actual_type = resolved_aliases[port.ref]
            if actual_type is None:
                continue
            if not self._port_types_compatible(actual_type, port.port_type):
                try:
                    coerced_ref = self.coerce(
                        actual_ref,
                        from_type=actual_type,
                        to_type=port.port_type,
                        name=f"{namespace}__adapt__{port.name}",
                        description=(
                            f"Auto-inserted adapter for {actual_type} -> {port.port_type} "
                            f"while including '{diagram.name}'"
                        ),
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Cannot wire '{actual_ref}' ({actual_type}) into "
                        f"'{diagram.name}.{port.name}' ({port.port_type})"
                    ) from exc
                resolved_aliases[port.ref] = (coerced_ref, port.port_type)

        for obj in diagram.objects.values():
            mapped_name = object_map[obj.name]
            if mapped_name in self.operations or mapped_name in self.losses:
                continue
            self._register_object(
                Object(
                    name=mapped_name,
                    kind=obj.kind,
                    shape=obj.shape,
                    description=obj.description,
                    metadata=merged_metadata(obj.metadata),
                )
            )

        for name, op in diagram.operations.items():
            mapped_name = operation_map[name]
            if isinstance(op, Morphism):
                implementation_key = op.implementation_key
                if implementation_key in implementation_map:
                    implementation_key = implementation_map[implementation_key]
                cloned = Morphism(
                    name=mapped_name,
                    source=map_ref(op.source),
                    target=map_ref(op.target),
                    description=op.description,
                    implementation_key=implementation_key,
                    metadata=merged_metadata(op.metadata),
                )
                self.add(cloned)
                if name in diagram.implementations:
                    self.bind_morphism(mapped_name, diagram.implementations[name])
                if (
                    op.implementation_key is not None
                    and op.implementation_key in diagram.implementations
                    and implementation_key is not None
                ):
                    self.bind_morphism(implementation_key, diagram.implementations[op.implementation_key])
                continue

            if isinstance(op, Composition):
                cloned = Composition(
                    name=mapped_name,
                    chain=tuple(operation_map[part] for part in op.chain),
                    source=map_ref(op.source),
                    target=map_ref(op.target),
                    description=op.description,
                    metadata=merged_metadata(op.metadata),
                )
                self.add(cloned)
                continue

            if isinstance(op, KanExtension):
                reducer = reducer_map.get(op.reducer, op.reducer)
                cloned = KanExtension(
                    name=mapped_name,
                    direction=op.direction,
                    source=map_ref(op.source),
                    along=map_ref(op.along),
                    target=map_ref(op.target),
                    reducer=reducer,
                    description=op.description,
                    metadata=merged_metadata(op.metadata),
                )
                self.add(cloned)
                if op.reducer in diagram.reducers:
                    self.bind_reducer(reducer, diagram.reducers[op.reducer])
                continue

            raise TypeError(f"Unsupported included operation type: {type(op)!r}")

        for name, loss in diagram.losses.items():
            comparator = comparator_map.get(loss.comparator, loss.comparator)
            cloned = ObstructionLoss(
                name=loss_map[name],
                paths=[
                    (
                        map_ref(left_name) or left_name,
                        map_ref(right_name) or right_name,
                    )
                    for left_name, right_name in loss.paths
                ],
                comparator=comparator,
                weight=loss.weight,
                description=loss.description,
                metadata=merged_metadata(loss.metadata),
            )
            self.add(cloned)
            if loss.comparator in diagram.comparators:
                self.bind_comparator(comparator, diagram.comparators[loss.comparator])

        for name, port in diagram.ports.items():
            mapped_ref = map_ref(port.ref)
            if mapped_ref is None:
                raise ValueError(f"Cannot map port '{name}' from included diagram '{diagram.name}'")
            included_port_name = f"{namespace}__{name}"
            cloned = Port(
                name=included_port_name,
                ref=mapped_ref,
                kind=port.kind,
                port_type=port.port_type,
                direction=port.direction,
                description=port.description,
                metadata=merged_metadata(port.metadata),
            )
            if included_port_name in self.ports:
                raise ValueError(f"Duplicate included port '{included_port_name}'")
            self.ports[included_port_name] = cloned

        return IncludedDiagram(
            namespace=namespace,
            diagram_name=diagram.name,
            object_map=object_map,
            operation_map=operation_map,
            loss_map=loss_map,
            port_specs={
                name: self.ports[f"{namespace}__{name}"]
                for name in diagram.ports
            },
        )

    def to_ir(self) -> DiagramIR:
        return DiagramIR(
            name=self.name,
            objects=list(self.objects.values()),
            operations=[self._operation_dict(op) for op in self.operations.values()],
            losses=list(self.losses.values()),
            ports=list(self.ports.values()),
        )

    def summary(self) -> str:
        lines = [f"Diagram({self.name})"]
        lines.append(f"  Objects: {', '.join(self.objects) or '<none>'}")
        lines.append(f"  Operations: {', '.join(self.operations) or '<none>'}")
        lines.append(f"  Losses: {', '.join(self.losses) or '<none>'}")
        lines.append(f"  Ports: {', '.join(self.ports) or '<none>'}")
        return "\n".join(lines)

    def _kan(
        self,
        *,
        direction: str,
        source: str | Object | Morphism | Composition,
        along: str | Object | Morphism | Composition,
        name: str | None,
        target: str | Object | None,
        reducer: str | Callable[..., Any],
        description: str,
        metadata: Mapping[str, Any] | None,
    ) -> KanExtension:
        op_name = name or f"{direction}_kan_{len(self.operations)}"
        reducer_key = reducer if isinstance(reducer, str) else op_name
        if callable(reducer):
            self.bind_reducer(str(reducer_key), reducer)
        kan = KanExtension(
            name=op_name,
            direction=direction,
            source=source,
            along=along,
            target=target,
            reducer=str(reducer_key),
            description=description,
            metadata=_metadata_dict(metadata),
        )
        self.add(kan)
        return kan

    def _register_object(self, obj: Object) -> None:
        if obj.name in self.operations or obj.name in self.losses:
            raise ValueError(f"Object '{obj.name}' conflicts with an existing operation or loss")
        existing = self.objects.get(obj.name)
        if existing is not None:
            if existing == obj:
                self.objects[obj.name] = obj
                return
            if self._is_placeholder_object(existing):
                self.objects[obj.name] = obj
                return
            raise ValueError(f"Object '{obj.name}' already exists with different metadata")
        self.objects[obj.name] = obj

    def _register_endpoint_placeholder(self, name: str) -> None:
        if name not in self.objects and name not in self.operations and name not in self.losses:
            self.objects[name] = Object(name=name)

    def _ensure_unique(self, name: str) -> None:
        if name in self.objects or name in self.operations or name in self.losses:
            raise ValueError(f"Duplicate FunctorFlow name '{name}'")

    def _infer_chain_endpoints(self, chain: Iterable[str]) -> tuple[str, str]:
        names = list(chain)
        if not names:
            raise ValueError("Empty composition chain")
        morphisms = [self._require_morphism(name) for name in names]
        for left, right in zip(morphisms, morphisms[1:]):
            if left.target != right.source:
                raise ValueError(
                    f"Cannot compose '{left.name}' ({left.target}) with '{right.name}' ({right.source})"
                )
        return morphisms[0].source, morphisms[-1].target

    def _require_morphism(self, name: str) -> Morphism:
        operation = self.operations.get(name)
        if not isinstance(operation, Morphism):
            raise ValueError(f"Composition requires morphism '{name}', found {type(operation)!r}")
        return operation

    def _operation_dict(self, op: Operation) -> dict[str, Any]:
        payload = op.to_dict()
        payload["kind"] = op.__class__.__name__.lower()
        return payload

    def _infer_ref_kind(self, name: str) -> str:
        if name in self.objects:
            return "object"
        if name in self.operations:
            return "operation"
        if name in self.losses:
            return "loss"
        raise KeyError(f"Cannot expose port for unknown FunctorFlow reference '{name}'")

    def _infer_ref_port_type(self, name: str) -> str:
        if name in self.objects:
            return self.objects[name].kind
        if name in self.losses:
            return "loss"
        if name in self.operations:
            operation = self.operations[name]
            if isinstance(operation, Morphism):
                return self.objects.get(operation.target, Object(operation.target)).kind
            if isinstance(operation, Composition):
                if operation.target is not None:
                    return self.objects.get(operation.target, Object(operation.target)).kind
                return "composition"
            if isinstance(operation, KanExtension):
                if operation.target is not None:
                    return self.objects.get(operation.target, Object(operation.target)).kind
                return f"{operation.direction}_kan"
        raise KeyError(f"Cannot infer port type for unknown FunctorFlow reference '{name}'")

    def _resolve_ref(
        self,
        ref: str | Object | Morphism | Composition | KanExtension | ObstructionLoss | Port,
    ) -> str:
        if isinstance(ref, Port):
            return ref.ref
        return _ref_name(ref)

    def _port_types_compatible(self, actual: str, expected: str) -> bool:
        return actual == expected or actual == "any" or expected == "any"

    def _is_placeholder_object(self, obj: Object) -> bool:
        return (
            obj.kind == "object"
            and obj.shape is None
            and obj.description == ""
            and obj.metadata == {}
        )

    @property
    def implementations(self) -> Mapping[str, Callable[..., Any]]:
        return self._implementations

    @property
    def reducers(self) -> Mapping[str, Callable[..., Any]]:
        return self._reducers

    @property
    def comparators(self) -> Mapping[str, Callable[..., Any]]:
        return self._comparators

    @property
    def adapters(self) -> Mapping[tuple[str, str], Adapter]:
        return self._adapters

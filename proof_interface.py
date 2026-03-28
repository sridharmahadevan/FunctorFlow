from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .core import Diagram


PROOFS_ROOT = Path(__file__).resolve().parent / "proofs"
GENERATED_ROOT = PROOFS_ROOT / "FunctorFlowProofs" / "Generated"


def _lean_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _lean_list(items: list[str]) -> str:
    return "[" + ", ".join(items) + "]"


def _sanitize_module_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "GeneratedCertificate"
    if cleaned[0].isdigit():
        cleaned = f"Diagram_{cleaned}"
    return "".join(part[:1].upper() + part[1:] for part in cleaned.split("_"))


def _operation_kind(kind: str) -> str:
    mapping = {
        "morphism": "OperationKind.morphism",
        "composition": "OperationKind.composition",
        "kanextension": None,
    }
    if kind == "composition":
        return mapping[kind]
    if kind == "morphism":
        return mapping[kind]
    raise ValueError(f"Unsupported direct operation kind '{kind}'")


def _operation_decl(operation: dict[str, Any]) -> tuple[str, list[str]]:
    kind = operation["kind"]
    if kind == "morphism":
        refs = [operation["source"], operation["target"]]
        return "OperationKind.morphism", refs
    if kind == "composition":
        refs = list(operation["chain"])
        if operation.get("source") is not None:
            refs.append(operation["source"])
        if operation.get("target") is not None:
            refs.append(operation["target"])
        return "OperationKind.composition", refs
    if kind == "kanextension":
        refs = [operation["source"], operation["along"]]
        if operation.get("target") is not None:
            refs.append(operation["target"])
        direction = operation["direction"]
        if direction == "left":
            return "OperationKind.leftKan", refs
        if direction == "right":
            return "OperationKind.rightKan", refs
        raise ValueError(f"Unsupported Kan direction '{direction}'")
    raise ValueError(f"Unsupported operation kind '{kind}'")


def diagram_certificate_payload(diagram: Diagram) -> dict[str, Any]:
    ir = diagram.to_ir().as_dict()
    return {
        "diagram_name": ir["name"],
        "objects": [obj["name"] for obj in ir["objects"]],
        "operations": [
            {
                "name": operation["name"],
                "kind": _operation_decl(operation)[0],
                "refs": _operation_decl(operation)[1],
            }
            for operation in ir["operations"]
        ],
        "ports": [
            {
                "name": port["name"],
                "ref": port["ref"],
            }
            for port in ir["ports"]
        ],
        "lowered_ops": [operation["name"] for operation in ir["operations"]],
    }


def render_lean_certificate(diagram: Diagram, *, module_name: str | None = None) -> str:
    payload = diagram_certificate_payload(diagram)
    module_base = _sanitize_module_name(module_name or diagram.name)
    lines = [
        "import FunctorFlowProofs.Compiler",
        "",
        "open FunctorFlowProofs",
        "",
        f"namespace FunctorFlowProofs.Generated.{module_base}",
        "",
        "def exportedDiagram : DiagramDecl := {",
        f"  name := {_lean_string(payload['diagram_name'])}",
        f"  objects := {_lean_list([_lean_string(name) for name in payload['objects']])}",
        "  operations := [",
    ]
    for operation in payload["operations"]:
        lines.extend(
            [
                "    {",
                f"      name := {_lean_string(operation['name'])}",
                f"      kind := {operation['kind']}",
                f"      refs := {_lean_list([_lean_string(ref) for ref in operation['refs']])}",
                "    },",
            ]
        )
    lines.extend(
        [
            "  ]",
            "  ports := [",
        ]
    )
    for port in payload["ports"]:
        lines.extend(
            [
                "    {",
                f"      name := {_lean_string(port['name'])}",
                f"      ref := {_lean_string(port['ref'])}",
                "    },",
            ]
        )
    lines.extend(
        [
            "  ]",
            "}",
            "",
            "def exportedArtifact : LoweringArtifact := {",
            "  diagram := exportedDiagram",
            f"  loweredOps := {_lean_list([_lean_string(name) for name in payload['lowered_ops']])}",
            "}",
            "",
            "theorem exportedArtifact_checks : exportedArtifact.check = true := rfl",
            "",
            "theorem exportedArtifact_sound : exportedArtifact.Sound :=",
            "  LoweringArtifact.sound_of_check_eq_true exportedArtifact_checks",
            "",
            f"end FunctorFlowProofs.Generated.{module_base}",
            "",
        ]
    )
    return "\n".join(lines)


def write_lean_certificate(
    diagram: Diagram,
    *,
    module_name: str | None = None,
    output_dir: str | Path = GENERATED_ROOT,
) -> Path:
    module_base = _sanitize_module_name(module_name or diagram.name)
    output_path = Path(output_dir) / f"{module_base}.lean"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_lean_certificate(diagram, module_name=module_base),
        encoding="utf-8",
    )
    return output_path

from __future__ import annotations

from FunctorFlow import KETBlockConfig, ket_block, write_lean_certificate


def main() -> None:
    diagram = ket_block(
        KETBlockConfig(
            name="ProofCertificateKET",
            source_object="HiddenStates",
            relation_object="CausalRelation",
            target_object="ContextualizedStates",
        )
    )
    output_path = write_lean_certificate(diagram)
    print(output_path)


if __name__ == "__main__":
    main()

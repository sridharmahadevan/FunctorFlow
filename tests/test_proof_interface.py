from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from FunctorFlow import KETBlockConfig, ket_block
from FunctorFlow.proof_interface import (
    GENERATED_ROOT,
    PROOFS_ROOT,
    diagram_certificate_payload,
    render_lean_certificate,
    write_lean_certificate,
)


class FunctorFlowProofInterfaceTests(unittest.TestCase):
    def test_diagram_certificate_payload_contains_declared_operations(self) -> None:
        diagram = ket_block(
            KETBlockConfig(
                name="ProofKET",
                source_object="HiddenStates",
                relation_object="CausalRelation",
                target_object="ContextualizedStates",
            )
        )
        payload = diagram_certificate_payload(diagram)
        self.assertEqual(payload["diagram_name"], "ProofKET")
        self.assertIn("HiddenStates", payload["objects"])
        self.assertEqual(payload["lowered_ops"], ["aggregate"])
        self.assertEqual(payload["ports"][0]["name"], "input")

    def test_render_lean_certificate_includes_soundness_theorem(self) -> None:
        diagram = ket_block(KETBlockConfig(name="LeanProofKET"))
        rendered = render_lean_certificate(diagram)
        self.assertIn("theorem exportedArtifact_checks", rendered)
        self.assertIn("theorem exportedArtifact_sound", rendered)
        self.assertIn("OperationKind.leftKan", rendered)

    def test_written_certificate_typechecks_with_lean(self) -> None:
        diagram = ket_block(KETBlockConfig(name="TypedCertificateKET"))
        with tempfile.TemporaryDirectory() as tmpdir:
            module_name = "TypedCertificateKET"
            cert_path = write_lean_certificate(
                diagram,
                module_name=module_name,
                output_dir=Path(tmpdir),
            )
            generated_path = GENERATED_ROOT / cert_path.name
            generated_path.parent.mkdir(parents=True, exist_ok=True)
            generated_path.write_text(cert_path.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                result = subprocess.run(
                    ["lake", "-R", "env", "lean", str(generated_path)],
                    cwd=PROOFS_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            finally:
                if generated_path.exists():
                    generated_path.unlink()
            if result.returncode != 0:
                self.fail(result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()

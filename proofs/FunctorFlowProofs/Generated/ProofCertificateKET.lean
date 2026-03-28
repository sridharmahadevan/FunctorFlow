import FunctorFlowProofs.Compiler

open FunctorFlowProofs

namespace FunctorFlowProofs.Generated.ProofCertificateKET

def exportedDiagram : DiagramDecl := {
  name := "ProofCertificateKET"
  objects := ["HiddenStates", "CausalRelation", "ContextualizedStates"]
  operations := [
    {
      name := "aggregate"
      kind := OperationKind.leftKan
      refs := ["HiddenStates", "CausalRelation", "ContextualizedStates"]
    },
  ]
  ports := [
    {
      name := "input"
      ref := "HiddenStates"
    },
    {
      name := "relation"
      ref := "CausalRelation"
    },
    {
      name := "output"
      ref := "aggregate"
    },
  ]
}

def exportedArtifact : LoweringArtifact := {
  diagram := exportedDiagram
  loweredOps := ["aggregate"]
}

theorem exportedArtifact_checks : exportedArtifact.check = true := rfl

theorem exportedArtifact_sound : exportedArtifact.Sound :=
  LoweringArtifact.sound_of_check_eq_true exportedArtifact_checks

end FunctorFlowProofs.Generated.ProofCertificateKET

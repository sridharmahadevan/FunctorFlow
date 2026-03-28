import FunctorFlowProofs.IR

namespace FunctorFlowProofs

structure LoweringArtifact where
  diagram : DiagramDecl
  loweredOps : List String
deriving Repr, DecidableEq

def LoweringArtifact.diagramOpNames (artifact : LoweringArtifact) : List String :=
  artifact.diagram.operations.map (·.name)

def LoweringArtifact.allLoweredAreDeclared (artifact : LoweringArtifact) : Bool :=
  listAllMembers artifact.loweredOps artifact.diagramOpNames

def LoweringArtifact.allDeclaredAreLowered (artifact : LoweringArtifact) : Bool :=
  listAllMembers artifact.diagramOpNames artifact.loweredOps

def LoweringArtifact.check (artifact : LoweringArtifact) : Bool :=
  artifact.diagram.operationRefsDeclared &&
  artifact.diagram.portRefsDeclared &&
  artifact.allLoweredAreDeclared &&
  artifact.allDeclaredAreLowered

def LoweringArtifact.Sound (artifact : LoweringArtifact) : Prop :=
  artifact.diagram.WellFormed ∧
  (∀ op, op ∈ artifact.loweredOps -> op ∈ artifact.diagramOpNames) ∧
  (∀ op, op ∈ artifact.diagramOpNames -> op ∈ artifact.loweredOps)

theorem LoweringArtifact.sound_of_check_eq_true {artifact : LoweringArtifact}
    (h : artifact.check = true) :
    artifact.Sound := by
  have hChecks :
      ((artifact.diagram.operationRefsDeclared = true ∧
        artifact.diagram.portRefsDeclared = true) ∧
        artifact.allLoweredAreDeclared = true) ∧
      artifact.allDeclaredAreLowered = true := by
    simpa [LoweringArtifact.check, Bool.and_eq_true] using h
  have hOps := hChecks.1.1.1
  have hPorts := hChecks.1.1.2
  have hLowered := hChecks.1.2
  have hDeclared := hChecks.2
  refine ⟨?_, ?_, ?_⟩
  · exact artifact.diagram.wellFormed_of_checks hOps hPorts
  · exact listAllMembers_sound hLowered
  · exact listAllMembers_sound hDeclared

end FunctorFlowProofs

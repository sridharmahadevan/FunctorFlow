namespace FunctorFlowProofs

inductive OperationKind where
  | morphism
  | composition
  | leftKan
  | rightKan
deriving Repr, DecidableEq

structure OperationDecl where
  name : String
  kind : OperationKind
  refs : List String
deriving Repr, DecidableEq

structure PortDecl where
  name : String
  ref : String
deriving Repr, DecidableEq

structure DiagramDecl where
  name : String
  objects : List String
  operations : List OperationDecl
  ports : List PortDecl
deriving Repr, DecidableEq

def listAllMembers (refs declared : List String) : Bool :=
  refs.all fun ref => ref ∈ declared

def operationListRefsDeclared (ops : List OperationDecl) (declared : List String) : Bool :=
  ops.all fun op => listAllMembers op.refs declared

def portListRefsDeclared (ports : List PortDecl) (declared : List String) : Bool :=
  ports.all fun port => port.ref ∈ declared

def DiagramDecl.declaredRefs (diagram : DiagramDecl) : List String :=
  diagram.objects ++ diagram.operations.map (·.name)

def DiagramDecl.operationRefsDeclared (diagram : DiagramDecl) : Bool :=
  operationListRefsDeclared diagram.operations diagram.declaredRefs

def DiagramDecl.portRefsDeclared (diagram : DiagramDecl) : Bool :=
  portListRefsDeclared diagram.ports diagram.declaredRefs

def DiagramDecl.WellFormed (diagram : DiagramDecl) : Prop :=
  (∀ op, op ∈ diagram.operations -> ∀ ref, ref ∈ op.refs -> ref ∈ diagram.declaredRefs) ∧
  (∀ port, port ∈ diagram.ports -> port.ref ∈ diagram.declaredRefs)

theorem listAllMembers_sound {refs declared : List String}
    (h : listAllMembers refs declared = true) :
    ∀ ref, ref ∈ refs -> ref ∈ declared := by
  rw [listAllMembers, List.all_eq_true] at h
  intro ref href
  exact of_decide_eq_true (h ref href)

theorem operationListRefsDeclared_sound {ops : List OperationDecl} {declared : List String}
    (h : operationListRefsDeclared ops declared = true) :
    ∀ op, op ∈ ops -> ∀ ref, ref ∈ op.refs -> ref ∈ declared := by
  rw [operationListRefsDeclared, List.all_eq_true] at h
  intro op hop ref href
  exact listAllMembers_sound (h op hop) ref href

theorem portListRefsDeclared_sound {ports : List PortDecl} {declared : List String}
    (h : portListRefsDeclared ports declared = true) :
    ∀ port, port ∈ ports -> port.ref ∈ declared := by
  rw [portListRefsDeclared, List.all_eq_true] at h
  intro port hport
  exact of_decide_eq_true (h port hport)

theorem DiagramDecl.wellFormed_of_checks {diagram : DiagramDecl}
    (hOps : diagram.operationRefsDeclared = true)
    (hPorts : diagram.portRefsDeclared = true) :
    diagram.WellFormed := by
  constructor
  · exact operationListRefsDeclared_sound hOps
  · exact portListRefsDeclared_sound hPorts

end FunctorFlowProofs

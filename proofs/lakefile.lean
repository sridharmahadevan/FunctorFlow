import Lake
open Lake DSL

package "FunctorFlowProofs"

@[default_target]
lean_lib «FunctorFlowProofs» where
  globs := #[.submodules `FunctorFlowProofs]

# FunctorFlow GitHub Release Draft

## Short GitHub Summary

FunctorFlow is a categorical deep learning implementation unifying attention and diffusion using Kan Extension Transformers. 

## Suggested Repository Description

FunctorFlow: a categorical DSL and executable IR for diagrammatic AI systems,
grounded in *Categories for AGI* and large causal models from LLMs.

## Suggested Release Title

`FunctorFlow v0.1.0 - Initial public release snapshot`

## Suggested Release Description

FunctorFlow is the first public software snapshot of a small categorical
intermediate representation for building AI systems from diagrams, morphisms,
Kan extensions, and obstruction losses.

This release includes:

- the core FunctorFlow DSL and compiler
- executable v0 systems for KET, GT, DB, and Democritus
- forward-looking categorical vocabulary for later BASKET / ROCKET support
- torch-backed structured language-model demos
- generated notebooks and tutorial artifacts
- a proof-of-concept Lean certificate interface under `proofs/`
- PDF documentation in `docs/`, including the user manual and tutorial deck

The release is grounded in:

- Sridhar Mahadevan, *Large Causal Models from Large Language Models*,
  arXiv:2512.07796
- Sridhar Mahadevan, *Categories for AGI*
- the Lean 4 formalization repository: <https://github.com/sridharmahadevan/catagi>

Notes for this public snapshot:

- some data directories are intentionally left as placeholders or metadata-only
  to keep the repository publishable and to avoid unclear redistribution status
- `data/democritus/` is intended as a user-supplied PDF input directory
- BASKET and ROCKET are part of the longer-term FunctorFlow roadmap, but are
  not yet fully implemented in this v0 public release
- FunctorFlow v0 is a research release: small, runnable, and tutorial-oriented,
  rather than a finished production framework

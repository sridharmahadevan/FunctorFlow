# FunctorFlow v1 Roadmap

## Status

FunctorFlow v0 is now frozen as the public baseline release.

The purpose of v1 is not to keep extending the current v0 implementation
incrementally, but to build a new design pass that makes good on the
categorical promise of FunctorFlow more directly. We should freely reuse
architectural components from v0 where they help, but v1 should be judged by
whether categorical structure becomes operational rather than merely
descriptive.

## Core Design Goal

FunctorFlow v1 should let us build AI systems by categorical construction.

That means:

- KET, GT, DB, Democritus, and later systems should be representable as
  first-class objects in a compositional categorical language
- larger architectures should be assembled through categorical operations such
  as products, coproducts, pullbacks, pushouts, equalizers, coequalizers, and
  subobject-style constructions
- the runtime and proof interfaces should preserve those constructions rather
  than flattening them into ad hoc glue code too early

In short:

`categorical model objects -> universal constructions -> executable systems`

## What v0 Gives Us

v0 already provides useful material to build on:

- a surface DSL with typed objects, morphisms, diagrams, Kan operators, and
  obstruction losses
- an initial normalized IR and backend lowering path
- runnable KET, DB, GT, and Democritus-oriented examples
- ports, adapters, and composition scaffolding
- notebook generation, tutorial packaging, and documentation structure
- an initial Lean proof interface for compilation artifacts

These are valuable assets, but they do not yet constitute the full
compositional categorical semantics we want for v1.

## What v1 Must Add

### 1. First-class categorical model objects

We should be able to represent a model family such as a KET module as a
first-class object with:

- an ambient category
- typed interfaces and boundary maps
- declared morphisms to related model objects
- reusable semantic laws for composition

The key change is that a model is no longer just a macro that expands into a
diagram. It becomes an object that can itself participate in higher-level
categorical constructions.

### 2. Universal construction as architecture assembly

FunctorFlow v1 should make universal constructions executable at the language
level. At minimum, the design should target:

- products and coproducts
- pullbacks and pushouts
- equalizers and coequalizers
- subobjects and quotient-style constructions
- subobject classifiers where appropriate

This is the point where compositionality becomes real: not just composition of
named operations, but composition through universal properties.

### 3. Functors and natural transformations as build primitives

FunctorFlow v1 should support:

- functors between model categories
- natural transformations between architectural choices
- transport of model structure across categories
- reusable mappings from abstract categorical design to executable backends

This matters because many AI architectures are best thought of not as isolated
diagrams, but as functorial mappings between representational regimes.

### 4. Proof-aware composition

The Lean-facing part of FunctorFlow should evolve from certificate stubs toward
proof-aware composition claims, such as:

- a claimed pullback really satisfies the pullback interface
- a claimed pushout really satisfies the pushout interface
- a subobject inclusion is well-typed and compositional
- model composition preserves declared interface constraints

This does not require full verification of training behavior. It does require
verification of the categorical structure that the runtime claims to realize.

## Canonical v1 Example

The canonical v1 demonstration should look something like this:

1. Define two KET models as categorical objects.
2. Define interface morphisms from each KET model into a shared context object.
3. Build a pullback object representing the joint constraint-compatible model.
4. Compile that construction into an executable system.
5. Emit a proof artifact that the resulting assembly satisfies the declared
   pullback shape.

Equivalent examples using pushouts, subobjects, or classifier-mediated
selection should follow naturally from the same framework.

If v1 cannot do this kind of example cleanly, then it has not yet met the main
design goal.

## Proposed Workstreams

### Workstream A. Semantic kernel

Design the true v1 core abstractions:

- categories
- objects
- morphisms
- functors
- natural transformations
- cones, cocones, and universal constructions

This kernel should be explicit enough that the language can distinguish
ordinary composition from composition by universal construction.

### Workstream B. Compositional IR

Replace or extend the v0 IR so it can represent:

- model objects as first-class values
- construction provenance for pullbacks, pushouts, and related objects
- interface constraints and commuting squares
- proof-relevant metadata for later verification

### Workstream C. Runtime lowering

Build an execution model that lowers categorical constructions without losing
their declared meaning too early.

Examples:

- a pullback lowers to a constraint-compatible composite model
- a pushout lowers to a merged interface-preserving model
- a subobject lowers to a restricted model family or admissible slice

### Workstream D. Reusable model libraries

Rebuild the v0 systems as v1-native components:

- KET as a first-class compositional model object
- DB as an obstruction-aware categorical component
- GT as a geometry-aware categorical component
- Democritus as a local-to-global gluing construction

BASKET and ROCKET should be revisited only after the compositional kernel is
strong enough to host them properly.

### Workstream E. Lean integration

Upgrade the proof layer from certificate output to categorical construction
checking. The immediate aim is structural verification, not numerical theorem
proving about optimization.

## Suggested Development Phases

### Phase 1. Freeze and extract

- keep v0 stable as the public reference implementation
- identify v0 components worth reusing directly
- isolate which v0 abstractions are semantic assets versus convenience wrappers

### Phase 2. Build the v1 semantic core

- define first-class category, functor, and natural-transformation objects
- define universal construction interfaces
- create a minimal compositional IR for those constructions

### Phase 3. Rebuild KET as the first v1-native model family

- treat KET modules as compositional objects
- support pullback / pushout assembly for KET variants
- verify the construction interface through Lean-side checks

### Phase 4. Lift DB, GT, and Democritus into the same framework

- rebuild DB, GT, and Democritus using the same compositional semantics
- ensure cross-family composition is possible through shared categorical
  structure rather than one-off adapters

### Phase 5. Revisit planning systems

- return to BASKET / ROCKET once the compositional categorical core is mature
- represent planning and repair through the same universal-construction
  language, rather than as bespoke workflow macros

## Non-Goals for v1

The following should not drive the design:

- adding more v0-style macros without improving the semantic core
- maximizing benchmark breadth before the compositional framework is real
- treating proof support as a purely cosmetic export layer
- rebuilding every prior system at once before the KET-centered categorical
  assembly story works

## Success Criteria

FunctorFlow v1 will be on the right track if it can do all of the following:

- represent KET modules as first-class compositional objects
- build larger models from those objects using pullbacks, pushouts, and related
  categorical constructions
- preserve semantic interface information through compilation
- emit proof artifacts that reflect those constructions faithfully
- provide at least one clean end-to-end demonstration that would have been
  awkward or artificial in v0

## Working Principle

FunctorFlow v0 showed that a categorical DSL can be runnable.

FunctorFlow v1 should show that categorical structure can be the actual method
of assembly.

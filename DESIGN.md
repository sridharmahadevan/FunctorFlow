# FunctorFlow Design v0

## Thesis

FunctorFlow should be the language in which the repo's systems become visibly
connected:

- KET as universal aggregation
- DB as explicit obstruction measurement
- GT as geometry-aware diagram execution
- BASKET and ROCKET as workflow diagrams plus repair
- Democritus as gluing local sections into a global state

The right first implementation target is therefore:

`surface DSL -> normalized categorical IR -> backend lowering`

not a monolithic end-to-end training stack.

## Core Semantic Commitments

### 1. Objects are typed interfaces

Objects name the kinds of states a diagram manipulates. In practice they can
stand for:

- token collections
- neighborhood indices
- local fragments
- global latent states
- plan states
- observation spaces

An object is not yet a tensor. It is the semantic slot a backend will later
realize.

### 2. Morphisms are typed computational arrows

Morphisms are transformations between objects:

- neural layers
- geometric lifts
- projection maps
- repair operators
- workflow transitions

In v0, a morphism can be abstract or can be bound to a Python callable.

### 3. Diagrams are architectures

A FunctorFlow `Diagram` is the primary user-facing artifact. It is the place
where we declare:

- objects
- morphisms
- composed paths
- Kan operators
- structural losses

This is the level at which tutorial users should think.

### 4. Left Kan is the native aggregation primitive

`left_kan(source, along)` means:

- there is local information in `source`
- there is an incidence or indexing map `along`
- we want a structured pushforward / aggregation onto a target context

This covers the recurring role played by:

- attention
- pooling
- neighborhood message passing
- context fusion
- plan-fragment integration

In v0, the compiler realizes this through finite incidence maps and pluggable
reducers such as `sum`, `mean`, `concat`, or custom callables.

### 5. Right Kan is the native completion primitive

`right_kan(source, along)` means:

- we have partial or observed information
- there is a projection / compatibility map
- we want a completion consistent with that structure

This is the design slot for:

- denoising
- masked completion
- plan repair
- partial-view reconciliation

In v0, right Kan uses the same finite relation representation but allows
different reducers or repair operators.

### 6. Obstruction is a first-class loss

`obstruction_loss(paths=[("p1", "p2")])` means:

- the named paths should agree
- disagreement is measured explicitly
- inconsistency becomes part of the executable architecture

This gives DB a native home in the language.

## Surface DSL

There are two intended styles.

### Detached declaration style

```python
from FunctorFlow import Diagram, Object, Morphism

Tokens = Object("Tokens")
Nbrs = Object("Nbrs")

inc = Morphism("incidence", Tokens, Nbrs)

D = Diagram("KET")
D.add(inc)
```

### Diagram-first style

```python
from FunctorFlow import Diagram

D = Diagram("ConsistencyAwareLM")
tokens = D.object("Tokens", kind="sequence")
latent = D.object("LatentState", kind="representation")

encode = D.morphism("encode", tokens, latent, implementation=my_encoder)
```

Both are supported because the slide deck already gestures at both.

## IR Shape

The normalized IR records:

- objects
- operations
- losses

Operations are one of:

- `morphism`
- `composition`
- `kanextension`

This is enough to represent the initial FunctorFlow claims without prematurely
committing to one backend.

## Compiler Pipeline

### Stage 1. Surface diagram

The user writes a diagram using `Object`, `Morphism`, `Diagram.left_kan`,
`Diagram.right_kan`, and `Diagram.obstruction_loss`.

### Stage 2. Normalized categorical IR

The diagram is normalized into named objects, operations, and losses with
validated composition endpoints.

### Stage 3. Backend lowering

There are two backends in v0:

- `compile_to_callable`: the real executable target in this workspace
- `compile_to_torch`: an optional wrapper that becomes available when Torch is
  installed

The backend-neutral runtime matters because it lets us test the semantics in a
plain standard-library environment.

## How Repo Systems Map In

### KET

- Objects: tokens, neighborhoods, contextual states
- Morphisms: incidence, value lift, output projection
- Primitive: `left_kan`

### DB

- Objects: intermediate latent states
- Morphisms: competing transformation paths
- Primitive: `obstruction_loss`

### GT

- Objects: token simplices, neighborhood complexes, updated states
- Morphisms: incidence maps, simplicial lifts
- Primitive: `left_kan` plus geometric metadata

### BASKET

- Objects: observations, plan fragments, plan states
- Morphisms: action transitions, workflow lifts
- Primitive: diagram composition

### ROCKET

- Objects: candidate plans, edit neighborhoods, repaired plans
- Morphisms: insert, delete, merge, patch
- Primitive: `right_kan` plus search metadata

### Democritus

- Objects: local claim slices, overlap regions, global relational state
- Morphisms: restriction maps, compatibility projections
- Primitive: right-Kan-flavored completion now, sheaf gluing later

## What v0 Deliberately Does Not Claim

- full categorical semantics for arbitrary small categories
- formal verification of learned neural systems
- a complete optimizer/trainer stack
- exact equivalence between all repo systems

Those would be overclaims for a first pass.

## What v0 Is Good For

- giving the ICML tutorial a concrete language surface
- making the shared abstractions explicit
- prototyping design ideas without committing to tensor plumbing
- creating a home for future pullbacks, sheaves, string diagrams, and macro
  libraries

## Named Macro Library

The package now includes a first named-block layer on top of the base DSL:

- `ket_block`
- `db_square`
- `gt_neighborhood_block`
- `basket_workflow_block`
- `rocket_repair_block`
- `democritus_gluing_block`

This matters because the tutorial should be able to move fluidly between:

- theory language
- system names already used in the repo
- executable FunctorFlow code

The macros are intentionally thin wrappers over the core primitives. They are
not a second semantics; they are a reusable vocabulary layer.

## Typed Parameters and Composition

The package now has two additional language layers:

- typed config dataclasses for named blocks
- namespaced subdiagram composition through `Diagram.include(...)`

This makes it possible to move from:

- isolated primitives

to:

- reusable, typed architectural templates

to:

- larger systems assembled from those templates without name collisions

That is the right direction for FunctorFlow because the tutorial needs to show
how KET, DB, GT, BASKET, ROCKET, and Democritus can become a family of
composable blocks rather than a flat catalog of one-off constructions.

## Ports As Semantic Contracts

The new port layer adds one more important abstraction:

- diagrams expose semantic interfaces such as `input`, `relation`, `output`,
  and `loss`
- included subdiagrams preserve those interfaces under namespace translation
- composite macros wire against those interfaces rather than internal names

This is the right move because FunctorFlow should eventually feel closer to a
typed architectural language than to a bag of string names.

## Adapters and Coercions

Port typing should be strict by default. When two blocks need a legal bridge,
FunctorFlow now supports explicit adapters from one port type to another.

That gives us a principled middle ground:

- mismatches are visible and rejected by default
- intentional bridges are declared as named adapters
- inclusion can auto-insert those adapters instead of relying on silent
  string-level wiring

Those adapters can now be bundled into reusable adapter libraries, which gives
FunctorFlow a standard-basis story parallel to its macro library story.

## Tutorial Libraries

FunctorFlow now also has a tutorial-library layer that bundles:

- a curated subset of named macros
- the adapter libraries expected by that tutorial path

This is useful for the ICML setting because different slices of the story can
now be packaged intentionally:

- `foundations` for KET / DB / GT
- `planning` for BASKET / ROCKET
- `unified` for the whole current stack

## Next Pass Roadmap

1. Add richer typechecking and named object capabilities.
2. Add pullbacks, pushouts, and gluing constraints as native operators.
3. Add a small macro library:
   KET block, DB square, GT neighborhood block, BASKET workflow block,
   Democritus local-to-global block.
4. Add graph / tensor lowerings with explicit backend adapters.
5. Add visualization so diagrams can render directly into tutorial figures.

The key win of this first pass is that FunctorFlow is no longer just a slide
idea. It is now a real package boundary with a real semantics we can keep
tightening.

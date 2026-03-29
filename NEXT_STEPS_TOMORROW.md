# Next Steps for Tomorrow

## Primary Goal

Define the semantic kernel for `FunctorFlow_v1` before writing substantial new
implementation code.

The central principle is:

Categorical structure should become the actual method of assembly, not merely a
descriptive layer over hand-built systems.

## Checklist

- [ ] Create a new `FunctorFlow_v1/` workspace so v0 stays frozen and untouched.
- [ ] Add `FunctorFlow_v1/DESIGN_v1.md` with the core semantic goal:
      categorical structure should be the actual method of assembly.
- [ ] Write down the first v1 core abstractions:
      `Category`, `ModelObject`, `Morphism`, `Functor`,
      `NaturalTransformation`, `Pullback`, `Pushout`.
- [ ] Specify what a first-class KET object is in v1:
      its interfaces, boundary maps, ambient category, and composition laws.
- [ ] Define the first universal construction target precisely:
      pullback of two KET objects over a shared context/interface object.
- [ ] Write the RN-Kan causal interpretation explicitly:
      right Kan as conditioning, left Kan as intervention.
- [ ] Record what v1 will and will not attempt yet for internal topos language,
      especially subobject classifiers and internal predicates.
- [ ] Decide which v0 components are reusable as-is, which are partial
      scaffolding, and which should not be carried forward.
- [ ] Draft one canonical end-to-end example in prose:
      two KET objects composed by pullback, then lowered to an executable
      system.
- [ ] End the session with a short implementation plan listing the first 3
      Python files to build after the design is stable.

## Deliverables

- [ ] `FunctorFlow_v1/`
- [ ] `FunctorFlow_v1/DESIGN_v1.md`
- [ ] `FunctorFlow_v1/IMPLEMENTATION_PLAN.md`
- [ ] One pullback-based KET example written clearly enough to guide coding.

## Stop Condition

- [ ] Do not start broad implementation until the semantic kernel, pullback
      example, and RN-Kan interpretation are written down cleanly.

# FunctorFlow v0

FunctorFlow is a categorical framework for building composable deep learning systems using diagrams. It is intended as a software companion to the textbook   [*Categories for AGI*](https://people.cs.umass.edu/~mahadeva/papers/catagi.pdf).
The guiding pattern is:

`Diagram / Spec -> Categorical IR -> Neural Architecture`

This version is intentionally small. It gives us a concrete design center for
the ICML 2026 tutorial and a lightweight executable kernel for experimentation,
without pretending we already have a full categorical deep learning compiler.

One practical constraint now guides the implementation: code intended for the
public FunctorFlow release should live inside `FunctorFlow/`, so the directory
can be cleanly lifted into a standalone GitHub repo ahead of the July 2026
ICML tutorial.

## User's manual

FunctorFlow carries release-ready reference documents under `FunctorFlow/docs/`:

- `FunctorFlow/docs/functorflow_user_manual.pdf`
- `FunctorFlow/docs/functorflow_v0_tutorial.pdf`

These documents are meant to capture the language surface more carefully than
either the slide deck or the class definitions alone, and should be updated
whenever the public API changes materially.

## Installation

FunctorFlow can now be installed directly with `pip`:

```bash
python -m pip install .
```

If you want the Torch-backed demos as well:

```bash
python -m pip install ".[torch]"
```

For development from a checkout, `pip install -e .` may require a recent
`pip` / `setuptools`. The safest cross-machine path is still:

```bash
python -m pip install .
```

After installation, the recommended way to run examples is through the package:

```bash
python -m FunctorFlow.examples.tutorial_v0
python -m FunctorFlow.examples.predict_detach_regime_demo
```

Running examples directly from a source checkout with `python -m examples...`
is also supported for convenience.

## Relation to Categories for AGI

FunctorFlow is the software companion to *Categories for AGI*. In particular,
this package operationalizes the book's emphasis on:

- diagrams as first-class architectural objects
- left and right Kan structure as aggregation and completion primitives
- GT, KET, DB, Democritus, and the broader BASKET / ROCKET agenda as a shared
  categorical family
- structured language-model duality between prediction and completion
- local-to-global document reasoning in the Democritus pipeline

The goal of this repository is not to reproduce the textbook verbatim, but to
turn part of that categorical design language into runnable Python artifacts,
notebooks, and small demonstration systems.

## Foundational References

This repository is best read together with the following references:

- Sridhar Mahadevan, [*Large Causal Models from Large Language Models*](https://arxiv.org/abs/2512.07796)
  (arXiv:2512.07796), which introduces the DEMOCRITUS system and the systems
  motivation for large causal models built from LLMs
- Sridhar Mahadevan, [*Categories for AGI*](https://people.cs.umass.edu/~mahadeva/papers/catagi.pdf),
  which provides the broader categorical foundations behind FunctorFlow
- [`catagi`](https://github.com/sridharmahadevan/catagi), the Lean 4
  formalization of the theoretical results from *Categories for AGI*

## What v0 includes

- `Object`: typed interfaces such as token spaces, neighborhoods, plan states,
  fragments, observations, or latent manifolds
- `Morphism`: typed transformations between objects
- `Diagram`: a named architecture assembled from objects, morphisms, Kan
  extensions, and obstruction losses
- `left_kan`: a universal aggregation primitive
- `right_kan`: a universal completion / repair primitive
- `obstruction_loss`: a native way to express non-commuting paths
- `compile_to_callable`: a backend-neutral executor using plain Python data
- `compile_to_torch`: an optional `torch.nn.Module` lowering when Torch is
  installed
- executable v0 paths for KET, DB, GT, and Democritus
- categorical planning vocabulary that points toward later BASKET / ROCKET
  support, without claiming a full v0 implementation of those systems
- a generic right-Kan `completion_block()` and a paired `structured_lm_duality()`
  macro for attention/completion diagrams
- typed config dataclasses for macro parameters
- namespaced subdiagram composition via `Diagram.include(...)`
- first-class semantic ports via `Diagram.expose_port(...)`
- explicit port coercions via registered adapters
- packaged reusable adapter libraries
- packaged higher-level tutorial libraries
- notebook rendering for tutorial libraries and KET demos
- a port of the `ket_experiments` block-vs-denoise core for left-Kan / right-Kan demos
- a FunctorFlow mini-Sudoku demo showing explicit row/column/block constraints
- a first FunctorFlow-backed KET language-model path for PTB, Wiki-2, and Wiki-103
- a Democritus document pipeline for user-supplied text or PDF inputs
- package-local `FunctorFlow/data` storage for PTB, Wiki-2, Wiki-103, and Democritus inputs

## Why this shape

The slide deck already identifies the irreducible FunctorFlow core:

- KET: left Kan aggregation
- DB: obstruction / commutativity control
- GT: structured message passing over incidence geometry
- BASKET / ROCKET: plan diagrams plus repair in diagram space, targeted for a
  later v1 release rather than fully implemented in v0
- Democritus: gluing local slices into a coherent whole

So the right first implementation is a *small categorical IR* rather than a
large framework. We want the language of construction first, then richer
backends.

## Macro Library

FunctorFlow now includes named block builders. The executable v0 focus is on
KET, DB, GT, and Democritus; planning-oriented BASKET / ROCKET builders are
currently best understood as forward-looking language scaffolding rather than a
full implementation:

- `ket_block()`
- `completion_block()`
- `structured_lm_duality()`
- `db_square()`
- `gt_neighborhood_block()`
- `basket_workflow_block()`
- `rocket_repair_block()`
- `democritus_gluing_block()`
- `basket_rocket_pipeline()`

You can also build them through the registry:

```python
from FunctorFlow import build_macro

diagram = build_macro("ket", name="TutorialKET")
```

This gives the tutorial a vocabulary that matches the repo's systems directly,
instead of forcing everything through bare low-level DSL calls.

FunctorFlow also now ships curated tutorial libraries such as:

- `foundations`
- `planning`
- `unified`

For clarity, the `planning` and `unified` libraries can expose BASKET / ROCKET
surface vocabulary, but the current v0 release does not yet claim a full
executable BASKET / ROCKET system. That is planned for a later v1 release
before ICML.

## Typed Macro Configs

Each block now has a config dataclass so the parameters are explicit and
documented in code:

```python
from FunctorFlow import KETBlockConfig, ket_block

diagram = ket_block(
    KETBlockConfig(
        name="EdgeAggregator",
        source_object="EdgeValues",
        relation_object="TokenIncidence",
        target_object="TokenStates",
        aggregate_name="gather",
        reducer="mean",
    )
)
```

This is a much better fit for tutorial examples than long untyped keyword lists.

## Subdiagram Composition

FunctorFlow diagrams can now include other diagrams under a namespace:

```python
from FunctorFlow import Diagram, ket_block

parent = Diagram("Parent")
included = parent.include(ket_block(), namespace="encoder")
print(included.operation("aggregate"))  # encoder__aggregate
```

That gives us a clean path from primitive blocks to larger composed systems.

## Ports

Diagrams can now expose semantic interfaces that survive composition:

```python
from FunctorFlow import ket_block

diagram = ket_block()
print(diagram.port("input"))   # Values
print(diagram.port("output"))  # aggregate
```

Included diagrams preserve those interfaces:

```python
from FunctorFlow import Diagram, ket_block

parent = Diagram("Parent")
child = parent.include(ket_block(), namespace="encoder")
print(child.port("output"))  # encoder__aggregate
```

This is the key step that lets composite macros wire blocks together using
stable semantic contracts rather than raw internal operation names.

## Adapters

When two port types are intentionally different, you can register an adapter
instead of weakening the type system:

```python
from FunctorFlow import Diagram

diagram = Diagram("AdapterDemo")
diagram.register_adapter(
    "context_to_candidates",
    source_type="contextualized_messages",
    target_type="plan_candidates",
    implementation=lambda value: value,
)
```

After that, `include(...)` can auto-insert the adapter when a typed port
connection would otherwise be rejected.

FunctorFlow now also ships packaged adapter libraries:

```python
from FunctorFlow import Diagram

diagram = Diagram("AdapterDemo")
diagram.use_adapter_library("standard")
```

That installs the repo's standard bridge vocabulary in one line.

## Tutorial Libraries

Tutorial libraries bundle curated macro subsets and the adapter packs they
expect:

```python
from FunctorFlow import Diagram, get_tutorial_library

diagram = Diagram("Tutorial")
planning = diagram.use_tutorial_library("planning")
stack = planning.build_macro("basket_rocket_pipeline")
```

This gives the tutorial a package-level story that sits one layer above raw
macros and one layer below a full application framework.

## Real KET LM Path

FunctorFlow now includes a first torch-backed KET language-model integration in
`FunctorFlow/ket_lm.py`. It gives us:

- local PTB, Wiki-2, and Wiki-103 loaders aligned with the repo's existing
  word-level setup
- package-local dataset storage under `FunctorFlow/data`
- a learned FunctorFlow KET reducer lowered through `compile_to_torch(...)`
- a small KET language model stack that uses the FunctorFlow diagram as its head

This is the current best starting point for the first real demonstration model,
because it exercises the actual FunctorFlow execution path on a canonical KET
task without requiring the heavier Democritus pipeline or future planning-
system integrations first.

FunctorFlow now also includes a unified structured-language-model experiment
path in `FunctorFlow/structured_lm.py`. It uses the same FunctorFlow language
surface to pair:

- left-Kan block prediction as the attention-style aggregation side
- right-Kan denoising completion as the diffusion-style repair side

This gives the package a closer correspondence to the `ket_experiments`
block-versus-denoise setup and the book chapter's structured-completion story.

## Bundled Assets and Release Notes

`FunctorFlow/data/` currently contains the following asset groups:

- `ptb/`: Penn Treebank word-level files used by the FunctorFlow language-model
  demos
- `wikitext-2/`: WikiText-2 token files used by the FunctorFlow language-model
  demos
- `wikitext-103/`: WikiText-103 token files together with upstream
  `README.txt` and `LICENSE.txt`
- `democritus/`: sample PDF inputs used for local Democritus document-graph
  experiments

The Democritus folder in this workspace snapshot contains a handful of example
PDFs, including newspaper-style articles and short research-style documents,
that were useful for local experimentation. For a public GitHub release,
however, the safest policy is conservative:

- keep only assets whose redistribution terms are clear and whose required
  notices travel with the release
- if PTB or WikiText-2 redistribution has not been independently verified,
  replace the bundled files with download instructions or placeholder
  directories
- if WikiText-103 remains bundled, keep the upstream `README.txt` and
  `LICENSE.txt` with it
- treat `FunctorFlow/data/democritus/` as a user-supplied input directory; if
  permissions for the sample PDFs are uncertain, ship that directory empty and
  ask users to add their own PDF documents

So this README describes the current research snapshot, while the final public
release may intentionally swap some bundled assets for placeholders.

## Notebook Rendering

FunctorFlow can now render tutorial libraries into plain `.ipynb` notebooks
without taking on a runtime dependency on `nbformat` or Jupyter:

```python
from FunctorFlow import render_ket_demo_notebook, write_notebook

notebook = render_ket_demo_notebook()
write_notebook(notebook, "FunctorFlow/notebooks/ket_demo_functorflow.ipynb")
```

For convenience, the package can also write the whole default notebook set:

```python
from FunctorFlow import write_default_notebooks

write_default_notebooks("FunctorFlow/notebooks")
```

That now produces:

- `foundations_tutorial.ipynb`
- `planning_tutorial.ipynb`
- `unified_tutorial.ipynb`
- `ket_demo_functorflow.ipynb`
- `ptb_ket_language_model.ipynb`
- `wiki2_ket_language_model.ipynb`
- `ptb_model_comparison.ipynb`
- `wiki2_model_comparison.ipynb`
- `predict_detach_regime_demo.ipynb`

Each generated notebook now bootstraps the local repo automatically, so opening
it from `FunctorFlow/notebooks/` does not require a separate package install.

This gives us an actual tutorial artifact layer, not just code APIs, and it is
the bridge we can now start using to develop a real demonstration KET model.

## Model Comparisons

FunctorFlow can now run a shared PTB, Wiki-2, or Wiki-103 comparison path
across three model families that all stay inside the package boundary:

- a baseline causal Transformer
- a course-ported GT-Lite geometric language model
- a course-ported FunctorFlow KET attention-as-Kan language model

The comparison harness lives in `FunctorFlow/lm_compare.py`, and a small smoke
demo is available with:

```bash
python -m FunctorFlow.examples.compare_lm_models_demo ptb
python -m FunctorFlow.examples.compare_lm_models_demo wiki-2
python -m FunctorFlow.examples.compare_lm_models_demo wiki-103
```

The generated comparison notebooks use the same harness, so PTB and Wiki-2 now
have both single-model KET notebooks and multi-model comparison notebooks.

FunctorFlow also now carries a synthetic predict-detach regime demo inside the
package, so the tutorial can show why detached predictive bases matter without
depending on the original course notebook:

```bash
python -m FunctorFlow.examples.predict_detach_regime_demo
```

## Lean Proof Interface

FunctorFlow now also carries a small proof-of-concept Lean interface under
`FunctorFlow/proofs/`. The goal is not to verify training quality, but to
verify compilation artifacts such as declared references, port resolution, and
basic lowering correctness.

The proof folder stores the Lean-side artifacts for that interface:

- `FunctorFlow/proofs/FunctorFlowProofs.lean`: root Lean module for the
  FunctorFlow proof stub
- `FunctorFlow/proofs/lakefile.lean`: Lean project configuration
- `FunctorFlow/proofs/lake-manifest.json`: resolved Lake dependency manifest
- `FunctorFlow/proofs/lean-toolchain`: pinned Lean toolchain for reproducible
  builds

You can emit a Lean certificate for a diagram with:

```bash
python -m FunctorFlow.examples.proof_certificate_demo
```

## Minimal example

```python
from FunctorFlow import Diagram, Object, Morphism, compile_to_callable

tokens = Object("Tokens", kind="sequence")
nbrs = Object("Neighborhoods", kind="index")

incidence = Morphism("incidence", tokens, nbrs)

D = Diagram("KETBlock")
D.add(incidence)

values = {
    0: 1.0,
    1: 2.0,
    2: 4.0,
}
relation = {
    "left_context": [0, 1],
    "right_context": [1, 2],
}

agg = D.left_kan(source="Values", along="incidence", name="aggregate")
compiled = compile_to_callable(D)
result = compiled.run({"Values": values, "incidence": relation})
print(result.values["aggregate"])
```

That prints:

```python
{"left_context": 3.0, "right_context": 6.0}
```

## Files

- `pyproject.toml` defines the pip-installable project metadata for modern
  Python packaging
- `setup.py` provides a compatibility packaging path for older Python /
  setuptools environments
- `FunctorFlow/core.py` defines the DSL, IR, and subdiagram inclusion
- `FunctorFlow/compiler.py` defines the backend-neutral executor and optional
  Torch wrapper
- `FunctorFlow/DESIGN.md` captures the language intent and roadmap
- `FunctorFlow/macros.py` defines the named block library, typed configs, and
  composite macros, including forward-looking planning vocabulary
- `FunctorFlow/adapter_library.py` defines packaged adapter libraries and the
  standard adapter pack
- `FunctorFlow/tutorial_library.py` defines packaged tutorial libraries over
  macro and adapter sets
- `FunctorFlow/notebook_renderer.py` renders tutorial libraries and KET demos
  into notebook artifacts
- `FunctorFlow/ket_lm.py` defines the PTB / Wiki-2 / Wiki-103
  FunctorFlow-backed KET language-model path
- `FunctorFlow/structured_lm.py` defines the structured prediction-versus-
  completion path
- `FunctorFlow/democritus.py` defines the Democritus document pipeline over
  text and PDF inputs
- `FunctorFlow/lm_compare.py` defines the shared Transformer-vs-GT-vs-KET
  comparison harness
- `FunctorFlow/proofs/` contains the Lean project files for the proof-of-
  concept certificate interface
- `FunctorFlow/data/` contains the package-local corpora, sample document
  inputs, and data layout notes
- `FunctorFlow/docs/` contains the user manual and tutorial PDFs
- `FunctorFlow/notebooks/` contains generated tutorial and demo notebooks
- `FunctorFlow/examples/tutorial_v0.py` gives a runnable example via
  `python -m FunctorFlow.examples.tutorial_v0`
- `FunctorFlow/examples/macro_library_demo.py` shows the macro library in use
- `FunctorFlow/examples/ptb_ket_lm_demo.py` runs a tiny PTB FunctorFlow-KET
  training smoke test
- `FunctorFlow/examples/wiki2_ket_lm_demo.py` runs a tiny Wiki-2
  FunctorFlow-KET training smoke test
- `FunctorFlow/examples/compare_lm_models_demo.py` runs a tiny PTB, Wiki-2, or
  Wiki-103 Transformer-vs-GT-vs-KET comparison
- `FunctorFlow/examples/democritus_demo.py` runs the Democritus pipeline on a
  PDF, a directory of PDFs, or a text file
- `FunctorFlow/tests/test_notebooks.py` covers notebook rendering and notebook
  file generation
- `FunctorFlow/tests/test_ket_lm.py` covers the PTB / Wiki-2 / Wiki-103 loader,
  FunctorFlow KET head, and LM smoke path
- `FunctorFlow/tests/test_lm_compare.py` covers the shared comparison harness
  and comparison smoke run
- `FunctorFlow/tests/test_democritus.py` covers the Democritus pipeline
- `FunctorFlow/tests/test_functorflow.py` covers the initial semantics
- `FunctorFlow/tests/test_macros.py` covers the macro builders
- `FunctorFlow/tests/test_torch_lowering.py` covers Torch lowering when run in
  an interpreter with Torch installed

## Current limits

This is a design-oriented v0, not a finished theorem:

- Kan semantics are represented as pluggable finite combinators over explicit
  incidence relations, not full universal constructions over arbitrary
  categories
- the compiler currently targets a Python execution plan first, with Torch as
  an optional wrapper
- sheaf gluing, pullbacks, macro-skill search, and richer typechecking still
  belong to the next pass

That is still valuable: it gives the tutorial a real language surface and gives
the repo one place where KET, DB, GT, and Democritus already live together,
while leaving room for fuller BASKET / ROCKET support in a later v1 release.

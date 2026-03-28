from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from .tutorial_library import TUTORIAL_LIBRARIES, TutorialLibrary, get_tutorial_library


def _source_lines(text: str) -> list[str]:
    return [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()]


def markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(text.strip()),
    }


def code_cell(code: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _source_lines(code.strip()),
    }


def bootstrap_cell() -> dict[str, Any]:
    return code_cell(
        """
from pathlib import Path
import sys

def _bootstrap_functorflow() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    for candidate in candidates:
        package_root = candidate / "FunctorFlow"
        if (package_root / "__init__.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError(
        "Could not find the FunctorFlow package. Run this notebook from the repo "
        "root or from FunctorFlow/notebooks."
    )

REPO_ROOT = _bootstrap_functorflow()
print(f"FunctorFlow repo root: {REPO_ROOT}")
"""
    )


def make_notebook(cells: Iterable[dict[str, Any]], *, title: str) -> dict[str, Any]:
    return {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9",
            },
            "functorflow": {
                "title": title,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_tutorial_library_notebook(library: str | TutorialLibrary) -> dict[str, Any]:
    tutorial_library = get_tutorial_library(library) if isinstance(library, str) else library
    title = f"FunctorFlow Tutorial Library: {tutorial_library.name}"
    cells = [
        markdown_cell(
            f"""
# {title}

This notebook was generated from the FunctorFlow tutorial-library layer.

- Library: `{tutorial_library.name}`
- Macros: `{", ".join(tutorial_library.macro_names)}`
- Adapter libraries: `{", ".join(tutorial_library.adapter_library_names) or "none"}`

The goal is to give the ICML tutorial a directly runnable path from packaged
library concepts to actual FunctorFlow diagrams.
"""
        ),
        bootstrap_cell(),
        code_cell(
            f"""
from FunctorFlow import get_tutorial_library

tutorial = get_tutorial_library("{tutorial_library.name}")
tutorial.name, tutorial.macro_names, tutorial.adapter_library_names
"""
        ),
    ]

    for macro_name in tutorial_library.macro_names:
        cells.append(
            markdown_cell(
                f"""
## Macro: `{macro_name}`

Build the diagram from the tutorial library and inspect the exposed ports.
"""
            )
        )
        cells.append(
            code_cell(
                f"""
diagram = tutorial.build_macro("{macro_name}")
print(diagram.summary())
{{name: (port.ref, port.port_type, port.direction) for name, port in diagram.ports.items()}}
"""
            )
        )

    cells.append(
        markdown_cell(
            """
## Next Step

Use one of these blocks as the seed for a larger composed tutorial path, or
adapt the generated notebook into a worked ICML example.
"""
        )
    )
    return make_notebook(cells, title=title)


def render_ket_demo_notebook() -> dict[str, Any]:
    title = "FunctorFlow KET Demo"
    cells = [
        markdown_cell(
            """
# FunctorFlow KET Demo

This notebook is the first concrete bridge from the tutorial-library layer to a
demonstration KET build path. It stays deliberately small: we build a typed KET
block, inspect its ports and IR, and run a toy aggregation example end to end.
"""
        ),
        bootstrap_cell(),
        code_cell(
            """
from FunctorFlow import KETBlockConfig, compile_to_callable, get_tutorial_library, ket_block

foundations = get_tutorial_library("foundations")
foundations.name, foundations.macro_names
"""
        ),
        markdown_cell(
            """
## Build A KET Block

We parameterize the block explicitly so the notebook mirrors the FunctorFlow
language surface, not an opaque helper API.
"""
        ),
        code_cell(
            """
ket = ket_block(
    KETBlockConfig(
        name="TutorialKET",
        source_object="EdgeValues",
        relation_object="TokenIncidence",
        target_object="TokenStates",
        aggregate_name="gather",
        reducer="sum",
    )
)

print(ket.summary())
{name: (port.ref, port.port_type, port.direction) for name, port in ket.ports.items()}
"""
        ),
        markdown_cell(
            """
## Run The FunctorFlow Diagram

This is the smallest executable KET-style demonstration we currently have in
FunctorFlow itself.
"""
        ),
        code_cell(
            """
compiled = compile_to_callable(ket)
result = compiled.run(
    {
        "EdgeValues": {
            0: 1.0,
            1: 3.0,
            2: 5.0,
        },
        "TokenIncidence": {
            "ctx_a": [0, 1],
            "ctx_b": [1, 2],
        },
    }
)

result.values[ket.port("output")]
"""
        ),
        markdown_cell(
            """
## Toward A Real Demonstration Model

The next step after this notebook is to replace the toy Python dictionaries with
a real tensor-backed value flow and start lowering a small KET model through the
FunctorFlow compiler surface.
"""
        ),
    ]
    return make_notebook(cells, title=title)


def render_ptb_ket_language_model_notebook() -> dict[str, Any]:
    title = "FunctorFlow PTB KET Language Model"
    cells = [
        markdown_cell(
            """
# FunctorFlow PTB KET Language Model

This notebook is the first real-data FunctorFlow KET path in the repo. It uses
the local Penn Treebank corpus, instantiates a FunctorFlow-backed KET language
model, runs a short training loop, and evaluates validation perplexity.
"""
        ),
        bootstrap_cell(),
        code_cell(
            """
from FunctorFlow.ket_lm import (
    FunctorFlowKETLanguageModel,
    KETLanguageModelConfig,
    estimate_perplexity,
    load_word_language_modeling_corpus,
    pick_device,
    train_language_model,
)

device = pick_device("cuda")
corpus = load_word_language_modeling_corpus("ptb")
config = KETLanguageModelConfig.historical_ptb_smoke()
corpus.name, corpus.vocab_size, len(corpus.train_ids), len(corpus.valid_ids), config
"""
        ),
        markdown_cell(
            """
## Build The Model

We keep the first pass intentionally small so the notebook stays runnable on a
single workstation while still exercising the real FunctorFlow KET head.
"""
        ),
        code_cell(
            """
model = FunctorFlowKETLanguageModel(
    corpus.vocab_size,
    config=config,
)
model
"""
        ),
        markdown_cell(
            """
## Train Briefly On PTB

This is a smoke-scale run for tutorial development. The next pass can lengthen
training, swap to `KETLanguageModelConfig.historical_ptb_reference()`, or move
the same model family over to Wiki-2 once the FunctorFlow execution path is
stable.
"""
        ),
        code_cell(
            """
history = train_language_model(
    model,
    corpus,
    steps=50,
    block_size=128,
    batch_size=16,
    lr=2e-3,
    device=device,
)
history
"""
        ),
        code_cell(
            """
valid_ppl = estimate_perplexity(
    model,
    corpus.valid_ids,
    block_size=128,
    batch_size=16,
    device=device,
)
valid_ppl
"""
        ),
        markdown_cell(
            """
## Next Step

Once this PTB path looks healthy, the immediate follow-on is to add a matching
Wiki-2 notebook and then start swapping in one of the larger historical KET
language-model variants already present elsewhere in the repo.
"""
        ),
    ]
    return make_notebook(cells, title=title)


def render_wiki2_ket_language_model_notebook() -> dict[str, Any]:
    title = "FunctorFlow Wiki-2 KET Language Model"
    cells = [
        markdown_cell(
            """
# FunctorFlow Wiki-2 KET Language Model

This notebook mirrors the PTB path, but switches to Wiki-2 so the same
FunctorFlow-backed KET family can be exercised on a second standard language
modeling benchmark using code that stays entirely inside `FunctorFlow/`.
"""
        ),
        bootstrap_cell(),
        code_cell(
            """
from FunctorFlow.ket_lm import (
    FunctorFlowKETLanguageModel,
    KETLanguageModelConfig,
    estimate_perplexity,
    load_word_language_modeling_corpus,
    pick_device,
    train_language_model,
)

device = pick_device("cuda")
corpus = load_word_language_modeling_corpus("wiki-2")
config = KETLanguageModelConfig.historical_wiki2_reference()
corpus.name, corpus.vocab_size, len(corpus.train_ids), len(corpus.valid_ids), config
"""
        ),
        markdown_cell(
            """
## Build The Model

This uses the Wiki-2 historical-reference preset so PTB and Wiki-2 are now
driven by one shared FunctorFlow model surface.
"""
        ),
        code_cell(
            """
model = FunctorFlowKETLanguageModel(
    corpus.vocab_size,
    config=config,
)
model
"""
        ),
        markdown_cell(
            """
## Train Briefly On Wiki-2

For tutorial iteration we keep the run short. The main point here is to verify
that the FunctorFlow KET code path transfers cleanly across benchmarks without
leaving the `FunctorFlow/` package boundary.
"""
        ),
        code_cell(
            """
history = train_language_model(
    model,
    corpus,
    steps=50,
    block_size=128,
    batch_size=16,
    lr=2e-3,
    device=device,
)
history
"""
        ),
        code_cell(
            """
valid_ppl = estimate_perplexity(
    model,
    corpus.valid_ids,
    block_size=128,
    batch_size=16,
    device=device,
)
valid_ppl
"""
        ),
        markdown_cell(
            """
## Next Step

With PTB and Wiki-2 both rendered through FunctorFlow, the next move is to tune
one shared KET family more seriously and package the `FunctorFlow/` directory as
the standalone GitHub tutorial repo.
"""
        ),
    ]
    return make_notebook(cells, title=title)


def render_ptb_model_comparison_notebook() -> dict[str, Any]:
    title = "FunctorFlow PTB Model Comparison"
    cells = [
        markdown_cell(
            """
# FunctorFlow PTB Model Comparison

This notebook compares three language-model families on the local Penn
Treebank corpus using one shared `FunctorFlow/` data and training path:

- a baseline causal Transformer
- a GT-style topology-aware model
- the FunctorFlow-backed KET model
"""
        ),
        bootstrap_cell(),
        code_cell(
            """
from FunctorFlow.ket_lm import pick_device
from FunctorFlow.lm_compare import LMComparisonConfig, compare_language_models

device = pick_device("cpu")
config = LMComparisonConfig(
    steps=2,
    block_size=32,
    batch_size=4,
    lr=2e-3,
    model_profile="smoke",
    train_tokens=2048,
    valid_tokens=512,
    test_tokens=512,
    seed=0,
)
comparison = compare_language_models("ptb", comparison_config=config, device=device)
comparison["corpus"], comparison["model_profile"], comparison["train_tokens"], comparison["valid_tokens"]
"""
        ),
        markdown_cell(
            """
## Validation Perplexity

This is still a smoke-scale run, but it gives the tutorial a real executable
comparison surface for Transformer, GT, and KET on one benchmark.
"""
        ),
        code_cell(
            """
{
    model_name: round(model_result["valid_ppl"], 2)
    for model_name, model_result in comparison["models"].items()
}
"""
        ),
        code_cell(
            """
{
    model_name: model_result["history"]["train_loss"]
    for model_name, model_result in comparison["models"].items()
}
"""
        ),
    ]
    return make_notebook(cells, title=title)


def render_wiki2_model_comparison_notebook() -> dict[str, Any]:
    title = "FunctorFlow Wiki-2 Model Comparison"
    cells = [
        markdown_cell(
            """
# FunctorFlow Wiki-2 Model Comparison

This notebook mirrors the PTB comparison path while switching to Wiki-2, so the
same three model families can be compared without leaving the `FunctorFlow/`
package boundary.

- a baseline causal Transformer
- a GT-style topology-aware model
- the FunctorFlow-backed KET model
"""
        ),
        bootstrap_cell(),
        code_cell(
            """
from FunctorFlow.ket_lm import pick_device
from FunctorFlow.lm_compare import LMComparisonConfig, compare_language_models

device = pick_device("cpu")
config = LMComparisonConfig(
    steps=2,
    block_size=32,
    batch_size=4,
    lr=2e-3,
    model_profile="smoke",
    train_tokens=2048,
    valid_tokens=512,
    test_tokens=512,
    seed=0,
)
comparison = compare_language_models("wiki-2", comparison_config=config, device=device)
comparison["corpus"], comparison["model_profile"], comparison["train_tokens"], comparison["valid_tokens"]
"""
        ),
        markdown_cell(
            """
## Validation Perplexity

This keeps the run intentionally short so the comparison notebook remains a
practical tutorial artifact on a single workstation.
"""
        ),
        code_cell(
            """
{
    model_name: round(model_result["valid_ppl"], 2)
    for model_name, model_result in comparison["models"].items()
}
"""
        ),
        code_cell(
            """
{
    model_name: model_result["history"]["train_loss"]
    for model_name, model_result in comparison["models"].items()
}
"""
        ),
    ]
    return make_notebook(cells, title=title)


def render_predict_detach_regime_notebook() -> dict[str, Any]:
    title = "FunctorFlow Predict-Detach Regime Demo"
    cells = [
        markdown_cell(
            """
# FunctorFlow Predict-Detach Regime Demo

This notebook ports the course-era synthetic leakage experiment into the
`FunctorFlow/` package. It compares three KET-style regimes on a toy sequence
task:

- `causal`: strict causal aggregation
- `leaky_noncausal`: leaky noncausal aggregation
- `predict_detach`: noncausal aggregation with detached predictive bases
"""
        ),
        bootstrap_cell(),
        code_cell(
            """
from FunctorFlow.predict_detach_demo import PredictDetachDemoConfig, run_predict_detach_regime_demo

config = PredictDetachDemoConfig(
    steps=25,
    batch_size=32,
    train_sequences=512,
    test_sequences=128,
    embed_dim=32,
    seed=0,
)
result = run_predict_detach_regime_demo(config)
result["config"]
"""
        ),
        markdown_cell(
            """
## Final Losses

This is a smoke-scale run, but it is enough to show how the regimes separate in
a directly runnable FunctorFlow-native artifact.
"""
        ),
        code_cell(
            """
{
    regime: {
        "final_train_loss": round(payload["final_train_loss"], 4),
        "eval_loss": round(payload["eval_loss"], 4),
    }
    for regime, payload in result["regimes"].items()
}
"""
        ),
        code_cell(
            """
{
    regime: payload["losses"][:5]
    for regime, payload in result["regimes"].items()
}
"""
        ),
    ]
    return make_notebook(cells, title=title)


def render_sudoku_demo_notebook() -> dict[str, Any]:
    title = "FunctorFlow Mini-Sudoku Demo"
    cells = [
        markdown_cell(
            """
# FunctorFlow Mini-Sudoku Demo

This notebook ports the course mini-Sudoku example into `FunctorFlow/` as a
small structural-constraint demo. It uses a 4x4 Sudoku board with 2x2 blocks so
the notebook stays lightweight while still showing explicit row, column, and
block constraints.
"""
        ),
        bootstrap_cell(),
        code_cell(
            """
import torch

from FunctorFlow.sudoku_demo import (
    SudokuDemoConfig,
    analyze_sudoku_board,
    base_solution_matrix,
    build_sudoku_constraint_diagram,
    format_grid,
    run_sudoku_demo,
)

diagram = build_sudoku_constraint_diagram()
print(diagram.summary())
"""
        ),
        markdown_cell(
            """
## Constraint Diagram

FunctorFlow expresses the Sudoku structure as a diagram with:

- cell digits as the source object
- row, column, and block incidence relations
- left-Kan aggregations into unit-level digit histograms
- duplicate-count morphisms as explicit constraint checks
"""
        ),
        code_cell(
            """
solved = base_solution_matrix().reshape(-1)
invalid = solved.clone()
invalid[1] = invalid[0]

solved_report = analyze_sudoku_board(solved)
invalid_report = analyze_sudoku_board(invalid)

{
    "solved_duplicates": solved_report["total_duplicates"],
    "invalid_duplicates": invalid_report["total_duplicates"],
}
"""
        ),
        code_cell(
            """
print("Solved board:")
print(format_grid(solved))
print()
print("Invalid board:")
print(format_grid(invalid))
"""
        ),
        markdown_cell(
            """
## Smoke-Scale Training Comparison

The course notebook compared a plain Transformer with a GT-style model. This
FunctorFlow port keeps that spirit but uses a very short run so the notebook
remains a practical tutorial artifact.
"""
        ),
        code_cell(
            """
config = SudokuDemoConfig(
    train_samples=64,
    val_samples=24,
    batch_size=16,
    epochs=1,
    d_model=32,
    n_heads=4,
    num_layers=1,
    lambda_db=0.02,
    seed=0,
)
result = run_sudoku_demo(config, device=torch.device("cpu"))
{
    name: {
        "val_cell_acc": round(payload["final_val_cell_acc"], 3),
        "val_puzzle_acc": round(payload["final_val_puzzle_acc"], 3),
    }
    for name, payload in result["models"].items()
}
"""
        ),
        code_cell(
            """
sample = result["sample"]
print("Puzzle (-1 = blank):")
print(format_grid(sample["puzzle"]))
print()
print("Prediction:")
print(format_grid(sample["prediction"]))
print()
print("Solution:")
print(format_grid(sample["solution"]))
sample["prediction_report"]["total_duplicates"]
"""
        ),
    ]
    return make_notebook(cells, title=title)


def render_ket_block_duality_notebook() -> dict[str, Any]:
    title = "FunctorFlow KET Block Duality Demo"
    cells = [
        markdown_cell(
            """
# FunctorFlow KET Block Duality Demo

This notebook ports the smallest reusable core of `scripts/ket_experiments`
into `FunctorFlow/` so the left-Kan/right-Kan story becomes concrete on one
shared language-modeling task.

- `left_kan`: future block prediction from causal context
- `right_kan`: denoising / completion of a corrupted future block
- a real Transformer / GT / KET language-model comparison on PTB, Wiki-2, or Wiki-103
"""
        ),
        bootstrap_cell(),
        code_cell(
            """
import torch

from FunctorFlow.ket_block_duality import (
    KETBlockDualityConfig,
    build_left_kan_block_diagram,
    build_right_kan_denoise_diagram,
    run_ket_block_duality_demo,
)

left_diagram = build_left_kan_block_diagram()
right_diagram = build_right_kan_denoise_diagram()
print(left_diagram.summary())
print()
print(right_diagram.summary())
"""
        ),
        markdown_cell(
            """
## Why This Matters

Both tasks use the same KET-style backbone, but the output semantics differ:

- left Kan predicts forward block offsets from contextualized hidden states
- right Kan completes a noisy future block under a denoising condition

The second half of the notebook then switches from this architectural duality
demo to a real corpus-level model comparison, so the same tutorial artifact can
connect the categorical picture to executable LM behavior.
"""
        ),
        code_cell(
            """
from FunctorFlow.ket_lm import pick_device

config = KETBlockDualityConfig(
    corpus_name="ptb",
    seq_len=32,
    batch_size=4,
    steps=2,
    block_size=4,
    num_denoise_steps=6,
    seed=0,
)
device = pick_device("cuda")
result = run_ket_block_duality_demo(config, device=device)
result["corpus"], result["config"]
"""
        ),
        code_cell(
            """
{
    "left_kan_block_ppl": round(result["left_kan"]["eval"]["block_ppl"], 2),
    "left_kan_offset@1": round(result["left_kan"]["eval"]["offset_accuracy"][1], 3),
    "right_kan_block_ppl": round(result["right_kan"]["eval"]["block_ppl"], 2),
    "right_kan_offset@1": round(result["right_kan"]["eval"]["offset_accuracy"][1], 3),
}
"""
        ),
        markdown_cell(
            """
## Real Language-Model Comparison

This section now mirrors the structured language-model comparison reported in
the book, rather than the generic autoregressive LM comparison.

Set `corpus_name` to `"ptb"`, `"wiki-2"`, or `"wiki-103"`.

- use `model_profile="smoke"` for a quick validation run
- use `model_profile="reference"` for the book-style `B=4`, context `=128`
  structured-LM setup
- on a CUDA machine, `pick_device("cuda")` will automatically target the GPU
"""
        ),
        code_cell(
            """
import matplotlib.pyplot as plt

from FunctorFlow.structured_lm import (
    StructuredLMComparisonConfig,
    compare_structured_language_models,
)

corpus_name = "ptb"
model_profile = "smoke"  # switch to "reference" for longer benchmark-style runs
comparison_config = (
    StructuredLMComparisonConfig.historical_smoke(corpus_name)
    if model_profile == "smoke"
    else StructuredLMComparisonConfig.historical_reference(corpus_name)
)
comparison = compare_structured_language_models(comparison_config, device=device)
comparison["corpus"], comparison["config"]
"""
        ),
        code_cell(
            """
{
    model_name: {
        "first_ppl": round(model_result["eval"]["first_offset_ppl"], 2),
        "block_ppl": round(model_result["eval"]["block_ppl"], 2),
    }
    for model_name, model_result in comparison["models"].items()
}
"""
        ),
        markdown_cell(
            """
## Book Reference Table

The next cell hard-codes the structured language-model results reported in the
book chapter, so the current run can be compared directly against the published
numbers.
"""
        ),
        code_cell(
            """
book_results = {
    "ptb": {
        "TF-Block-4": {"first_ppl": 61.13, "block_ppl": 201.49},
        "KET-Block-4": {"first_ppl": 59.52, "block_ppl": 198.92},
        "TF-Denoise-4": {"first_ppl": 3.14, "block_ppl": 3.95},
        "KET-Denoise-4": {"first_ppl": 3.12, "block_ppl": 3.94},
    },
    "wiki-2": {
        "TF-Block-4": {"first_ppl": 112.44, "block_ppl": 266.96},
        "KET-Block-4": {"first_ppl": 109.86, "block_ppl": 262.08},
        "TF-Denoise-4": {"first_ppl": 3.44, "block_ppl": 4.11},
        "KET-Denoise-4": {"first_ppl": 3.39, "block_ppl": 4.08},
    },
    "wiki-103": {
        "TF-Block-4": {"first_ppl": 455.83, "block_ppl": 782.74},
        "KET-Block-4": {"first_ppl": 416.00, "block_ppl": 748.04},
        "TF-Denoise-4": {"first_ppl": 5.44, "block_ppl": 5.75},
        "KET-Denoise-4": {"first_ppl": 5.37, "block_ppl": 5.70},
    },
}

merged_rows = []

reference_for_corpus = book_results[comparison["corpus"]]
for model_name in comparison["models"]:
    measured = comparison["models"][model_name]["eval"]
    reference = reference_for_corpus[model_name]
    delta_first = measured["first_offset_ppl"] - reference["first_ppl"]
    delta_block = measured["block_ppl"] - reference["block_ppl"]
    merged_rows.append(
        {
            "model": model_name,
            "book_first_ppl": round(reference["first_ppl"], 2),
            "book_block_ppl": round(reference["block_ppl"], 2),
            "run_first_ppl": round(measured["first_offset_ppl"], 2),
            "run_block_ppl": round(measured["block_ppl"], 2),
            "delta_first_ppl": round(delta_first, 2),
            "delta_block_ppl": round(delta_block, 2),
            "pct_error_first": round(100.0 * delta_first / reference["first_ppl"], 2),
            "pct_error_block": round(100.0 * delta_block / reference["block_ppl"], 2),
        }
    )

try:
    import pandas as pd

    comparison_df = pd.DataFrame(merged_rows)

    def highlight_delta(value):
        if value < 0:
            return "background-color: #d8f5d0"
        if value > 0:
            return "background-color: #f8d7da"
        return ""

    styler = comparison_df.style.format(
        {
            "book_first_ppl": "{:.2f}",
            "book_block_ppl": "{:.2f}",
            "run_first_ppl": "{:.2f}",
            "run_block_ppl": "{:.2f}",
            "delta_first_ppl": "{:+.2f}",
            "delta_block_ppl": "{:+.2f}",
            "pct_error_first": "{:+.2f}%",
            "pct_error_block": "{:+.2f}%",
        }
    )
    if hasattr(styler, "map"):
        styled = styler.map(highlight_delta, subset=["delta_first_ppl", "delta_block_ppl"])
    else:
        styled = styler.applymap(highlight_delta, subset=["delta_first_ppl", "delta_block_ppl"])

    styled
except ModuleNotFoundError:
    merged_rows
"""
        ),
        code_cell(
            """
model_names = list(comparison["models"])
first_ppls = [comparison["models"][name]["eval"]["first_offset_ppl"] for name in model_names]
block_ppls = [comparison["models"][name]["eval"]["block_ppl"] for name in model_names]
train_histories = [comparison["models"][name]["history"]["train_loss"] for name in model_names]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = range(len(model_names))
bar_width = 0.38

axes[0].bar([i - bar_width / 2 for i in x], first_ppls, width=bar_width, label="first-PPL", color="#4c78a8")
axes[0].bar([i + bar_width / 2 for i in x], block_ppls, width=bar_width, label="block-PPL", color="#f58518")
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(model_names, rotation=20, ha="right")
axes[0].set_title(f"{comparison['corpus']} structured LM metrics")
axes[0].set_ylabel("Perplexity")
axes[0].set_yscale("log")
axes[0].legend()

for model_name, history in zip(model_names, train_histories):
    axes[1].plot(range(1, len(history) + 1), history, marker="o", label=model_name)
axes[1].set_title("Training loss")
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Cross-entropy")
axes[1].legend()

plt.tight_layout()
plt.show()
"""
        ),
    ]
    return make_notebook(cells, title=title)


def write_notebook(notebook: dict[str, Any], path: str | Path) -> Path:
    notebook_path = Path(path)
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return notebook_path


def write_default_notebooks(output_dir: str | Path) -> list[Path]:
    outdir = Path(output_dir)
    paths: list[Path] = []
    for library_name in sorted(TUTORIAL_LIBRARIES):
        notebook = render_tutorial_library_notebook(library_name)
        paths.append(write_notebook(notebook, outdir / f"{library_name}_tutorial.ipynb"))
    paths.append(write_notebook(render_ket_demo_notebook(), outdir / "ket_demo_functorflow.ipynb"))
    paths.append(
        write_notebook(
            render_ptb_ket_language_model_notebook(),
            outdir / "ptb_ket_language_model.ipynb",
        )
    )
    paths.append(
        write_notebook(
            render_wiki2_ket_language_model_notebook(),
            outdir / "wiki2_ket_language_model.ipynb",
        )
    )
    paths.append(
        write_notebook(
            render_ptb_model_comparison_notebook(),
            outdir / "ptb_model_comparison.ipynb",
        )
    )
    paths.append(
        write_notebook(
            render_wiki2_model_comparison_notebook(),
            outdir / "wiki2_model_comparison.ipynb",
        )
    )
    paths.append(
        write_notebook(
            render_predict_detach_regime_notebook(),
            outdir / "predict_detach_regime_demo.ipynb",
        )
    )
    paths.append(write_notebook(render_sudoku_demo_notebook(), outdir / "mini_sudoku_demo.ipynb"))
    paths.append(
        write_notebook(
            render_ket_block_duality_notebook(),
            outdir / "ket_block_duality_demo.ipynb",
        )
    )
    return paths


def main() -> None:
    default_dir = Path(__file__).resolve().parent / "notebooks"
    paths = write_default_notebooks(default_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

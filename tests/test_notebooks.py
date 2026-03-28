from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from FunctorFlow import (
    render_ket_demo_notebook,
    render_ket_block_duality_notebook,
    render_predict_detach_regime_notebook,
    render_sudoku_demo_notebook,
    render_ptb_model_comparison_notebook,
    render_ptb_ket_language_model_notebook,
    render_tutorial_library_notebook,
    render_wiki2_model_comparison_notebook,
    render_wiki2_ket_language_model_notebook,
    write_default_notebooks,
)


def run_notebook_code_cells(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict = {}
    previous_cwd = Path.cwd()
    try:
        os.chdir(path.parent)
        for cell in payload["cells"]:
            if cell["cell_type"] != "code":
                continue
            exec("".join(cell["source"]), namespace)
    finally:
        os.chdir(previous_cwd)
    return namespace


class FunctorFlowNotebookTests(unittest.TestCase):
    def test_render_tutorial_library_notebook(self) -> None:
        notebook = render_tutorial_library_notebook("planning")
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreater(len(notebook["cells"]), 2)
        first_cell = "".join(notebook["cells"][0]["source"])
        self.assertIn("planning", first_cell)
        bootstrap = "".join(notebook["cells"][1]["source"])
        self.assertIn("_bootstrap_functorflow", bootstrap)

    def test_render_ket_demo_notebook(self) -> None:
        notebook = render_ket_demo_notebook()
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("TutorialKET", sources)
        self.assertIn("compile_to_callable", sources)
        self.assertIn("REPO_ROOT", sources)

    def test_render_ket_block_duality_notebook(self) -> None:
        notebook = render_ket_block_duality_notebook()
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("run_ket_block_duality_demo", sources)
        self.assertIn("build_left_kan_block_diagram", sources)
        self.assertIn("build_right_kan_denoise_diagram", sources)
        self.assertIn("compare_structured_language_models", sources)
        self.assertIn("StructuredLMComparisonConfig", sources)
        self.assertIn("wiki-103", sources)
        self.assertIn("book_results", sources)
        self.assertIn("merged_rows", sources)
        self.assertIn("pct_error_first", sources)
        self.assertIn("highlight_delta", sources)
        self.assertIn("59.52", sources)
        self.assertIn("748.04", sources)
        self.assertIn("matplotlib", sources)

    def test_render_ptb_ket_language_model_notebook(self) -> None:
        notebook = render_ptb_ket_language_model_notebook()
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("load_word_language_modeling_corpus", sources)
        self.assertIn("FunctorFlowKETLanguageModel", sources)
        self.assertIn("ptb", sources)

    def test_render_ptb_model_comparison_notebook(self) -> None:
        notebook = render_ptb_model_comparison_notebook()
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("compare_language_models", sources)
        self.assertIn("Transformer", sources)
        self.assertIn("ptb", sources)

    def test_render_predict_detach_regime_notebook(self) -> None:
        notebook = render_predict_detach_regime_notebook()
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("run_predict_detach_regime_demo", sources)
        self.assertIn("predict_detach", sources)
        self.assertIn("leaky_noncausal", sources)

    def test_render_sudoku_demo_notebook(self) -> None:
        notebook = render_sudoku_demo_notebook()
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("build_sudoku_constraint_diagram", sources)
        self.assertIn("run_sudoku_demo", sources)
        self.assertIn("Mini-Sudoku", sources)

    def test_render_wiki2_ket_language_model_notebook(self) -> None:
        notebook = render_wiki2_ket_language_model_notebook()
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("load_word_language_modeling_corpus", sources)
        self.assertIn("FunctorFlowKETLanguageModel", sources)
        self.assertIn("wiki-2", sources)

    def test_render_wiki2_model_comparison_notebook(self) -> None:
        notebook = render_wiki2_model_comparison_notebook()
        sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("compare_language_models", sources)
        self.assertIn("KET", sources)
        self.assertIn("wiki-2", sources)

    def test_write_default_notebooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            notebook_dir = Path(tmpdir) / "FunctorFlow" / "notebooks"
            paths = write_default_notebooks(notebook_dir)
            names = {path.name for path in paths}
            self.assertIn("foundations_tutorial.ipynb", names)
            self.assertIn("planning_tutorial.ipynb", names)
            self.assertIn("unified_tutorial.ipynb", names)
            self.assertIn("ket_demo_functorflow.ipynb", names)
            self.assertIn("ket_block_duality_demo.ipynb", names)
            self.assertIn("ptb_ket_language_model.ipynb", names)
            self.assertIn("wiki2_ket_language_model.ipynb", names)
            self.assertIn("ptb_model_comparison.ipynb", names)
            self.assertIn("wiki2_model_comparison.ipynb", names)
            self.assertIn("predict_detach_regime_demo.ipynb", names)
            self.assertIn("mini_sudoku_demo.ipynb", names)

            planning_path = notebook_dir / "planning_tutorial.ipynb"
            payload = json.loads(planning_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)

    def test_generated_notebooks_execute_from_notebook_directory(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        notebook_dir = repo_root / "FunctorFlow" / "notebooks"

        try:
            write_default_notebooks(notebook_dir)
            ket_namespace = run_notebook_code_cells(notebook_dir / "ket_demo_functorflow.ipynb")
            self.assertIn("ket", ket_namespace)
            self.assertIn("result", ket_namespace)
            self.assertEqual(ket_namespace["result"].values[ket_namespace["ket"].port("output")]["ctx_a"], 4.0)

            planning_namespace = run_notebook_code_cells(notebook_dir / "planning_tutorial.ipynb")
            self.assertIn("tutorial", planning_namespace)
            self.assertEqual(planning_namespace["tutorial"].name, "planning")
        finally:
            write_default_notebooks(notebook_dir)


if __name__ == "__main__":
    unittest.main()

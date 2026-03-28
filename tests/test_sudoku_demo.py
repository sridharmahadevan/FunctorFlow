from __future__ import annotations

import unittest

try:
    import torch

    from FunctorFlow.sudoku_demo import (
        SudokuDemoConfig,
        analyze_sudoku_board,
        base_solution_matrix,
        run_sudoku_demo,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this interpreter")
class FunctorFlowSudokuDemoTests(unittest.TestCase):
    def test_analyze_sudoku_board_detects_duplicates(self) -> None:
        solved = base_solution_matrix().reshape(-1)
        solved_report = analyze_sudoku_board(solved)
        self.assertTrue(solved_report["is_consistent"])
        self.assertEqual(solved_report["total_duplicates"], 0)

        invalid = solved.clone()
        invalid[1] = invalid[0]
        invalid_report = analyze_sudoku_board(invalid)
        self.assertFalse(invalid_report["is_consistent"])
        self.assertGreater(invalid_report["total_duplicates"], 0)

    def test_run_sudoku_demo_smoke(self) -> None:
        result = run_sudoku_demo(
            SudokuDemoConfig(
                train_samples=32,
                val_samples=16,
                batch_size=8,
                epochs=1,
                d_model=16,
                n_heads=4,
                num_layers=1,
                lambda_db=0.02,
                seed=0,
            ),
            device=torch.device("cpu"),
        )
        self.assertEqual(set(result["models"]), {"transformer", "gt_db"})
        self.assertGreater(len(result["triangles"]), 0)
        self.assertIn("MiniSudokuConstraintDiagram", result["diagram"].summary())
        self.assertIn("prediction_report", result["sample"])


if __name__ == "__main__":
    unittest.main()

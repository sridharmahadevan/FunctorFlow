from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compiler import compile_to_callable
from .core import Diagram


BOARD_SIZE = 4
BLOCK_SIZE = 2
NUM_DIGITS = 4
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
BLANK_DIGIT = -1


@dataclass(frozen=True)
class SudokuInstance:
    puzzle: torch.Tensor
    solution: torch.Tensor


@dataclass(frozen=True)
class SudokuDemoConfig:
    train_samples: int = 128
    val_samples: int = 48
    num_givens: int = 8
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    batch_size: int = 32
    epochs: int = 2
    lr: float = 1e-3
    lambda_db: float = 0.05
    seed: int = 0


def pick_device(preferred: str = "cpu") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def cell_id(row: int, column: int) -> int:
    return row * BOARD_SIZE + column


def base_solution_matrix() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 2, 3],
            [2, 3, 0, 1],
            [1, 0, 3, 2],
            [3, 2, 1, 0],
        ],
        dtype=torch.long,
    )


def random_sudoku_4x4_solution() -> torch.Tensor:
    matrix = base_solution_matrix()
    digit_perm = torch.randperm(NUM_DIGITS)
    matrix = digit_perm[matrix]

    band_rows = [[0, 1], [2, 3]]
    permuted_rows: list[int] = []
    for band in band_rows:
        order = band.copy()
        random.shuffle(order)
        permuted_rows.extend(order)
    matrix = matrix[permuted_rows, :]
    if random.random() < 0.5:
        matrix = torch.cat([matrix[2:4, :], matrix[0:2, :]], dim=0)

    band_cols = [[0, 1], [2, 3]]
    permuted_cols: list[int] = []
    for band in band_cols:
        order = band.copy()
        random.shuffle(order)
        permuted_cols.extend(order)
    matrix = matrix[:, permuted_cols]
    if random.random() < 0.5:
        matrix = torch.cat([matrix[:, 2:4], matrix[:, 0:2]], dim=1)

    return matrix.reshape(-1)


def mask_puzzle(solution: Sequence[int], num_givens: int) -> list[int]:
    indices = list(range(NUM_CELLS))
    random.shuffle(indices)
    given_indices = set(indices[:num_givens])
    return [value if index in given_indices else BLANK_DIGIT for index, value in enumerate(solution)]


def make_sudoku_dataset(num_samples: int, num_givens: int) -> list[SudokuInstance]:
    dataset: list[SudokuInstance] = []
    for _ in range(num_samples):
        solution = random_sudoku_4x4_solution()
        puzzle = mask_puzzle(solution.tolist(), num_givens)
        dataset.append(
            SudokuInstance(
                puzzle=torch.tensor(puzzle, dtype=torch.long),
                solution=solution.clone(),
            )
        )
    return dataset


def build_sudoku_unit_relations() -> dict[str, dict[str, list[int]]]:
    row_relation = {
        f"row_{row}": [cell_id(row, column) for column in range(BOARD_SIZE)]
        for row in range(BOARD_SIZE)
    }
    column_relation = {
        f"column_{column}": [cell_id(row, column) for row in range(BOARD_SIZE)]
        for column in range(BOARD_SIZE)
    }
    block_relation: dict[str, list[int]] = {}
    block_index = 0
    for row in range(0, BOARD_SIZE, BLOCK_SIZE):
        for column in range(0, BOARD_SIZE, BLOCK_SIZE):
            block_relation[f"block_{block_index}"] = [
                cell_id(row + dr, column + dc)
                for dr in range(BLOCK_SIZE)
                for dc in range(BLOCK_SIZE)
            ]
            block_index += 1
    return {
        "rows": row_relation,
        "columns": column_relation,
        "blocks": block_relation,
    }


def build_sudoku_triangles() -> list[tuple[int, int, int]]:
    triangles: list[tuple[int, int, int]] = []
    relations = build_sudoku_unit_relations()
    for relation in relations.values():
        for cells in relation.values():
            triangles.extend(tuple(triangle) for triangle in itertools.combinations(cells, 3))
    return triangles


def _board_mapping(board: Sequence[int] | torch.Tensor) -> dict[int, int]:
    if isinstance(board, torch.Tensor):
        values = board.detach().cpu().tolist()
    else:
        values = list(board)
    if len(values) != NUM_CELLS:
        raise ValueError(f"Expected {NUM_CELLS} board entries, received {len(values)}")
    return {index: int(value) for index, value in enumerate(values)}


def _digit_histogram_reducer(
    source: dict[int, int],
    relation: dict[str, list[int]],
    metadata: dict[str, object],
) -> dict[str, dict[int, int]]:
    del metadata
    summaries: dict[str, dict[int, int]] = {}
    for group_name, cell_indices in relation.items():
        counts = {digit: 0 for digit in range(NUM_DIGITS)}
        for index in cell_indices:
            digit = int(source[index])
            if digit >= 0:
                counts[digit] += 1
        summaries[group_name] = counts
    return summaries


def _duplicate_counts(summary: dict[str, dict[int, int]]) -> dict[str, int]:
    return {
        unit_name: sum(max(0, count - 1) for count in counts.values())
        for unit_name, counts in summary.items()
    }


def build_sudoku_constraint_diagram() -> Diagram:
    diagram = Diagram("MiniSudokuConstraintDiagram")
    diagram.object("CellDigits", kind="cell_digits")
    diagram.object("RowUnits", kind="row_relation")
    diagram.object("ColumnUnits", kind="column_relation")
    diagram.object("BlockUnits", kind="block_relation")
    diagram.object("RowSummaries", kind="digit_histograms")
    diagram.object("ColumnSummaries", kind="digit_histograms")
    diagram.object("BlockSummaries", kind="digit_histograms")
    diagram.object("RowViolations", kind="violation_map")
    diagram.object("ColumnViolations", kind="violation_map")
    diagram.object("BlockViolations", kind="violation_map")

    diagram.left_kan(
        name="row_histograms",
        source="CellDigits",
        along="RowUnits",
        target="RowSummaries",
        reducer=_digit_histogram_reducer,
        description="Aggregate cell digits into row-level digit histograms.",
        metadata={"demo": "sudoku"},
    )
    diagram.left_kan(
        name="column_histograms",
        source="CellDigits",
        along="ColumnUnits",
        target="ColumnSummaries",
        reducer=_digit_histogram_reducer,
        description="Aggregate cell digits into column-level digit histograms.",
        metadata={"demo": "sudoku"},
    )
    diagram.left_kan(
        name="block_histograms",
        source="CellDigits",
        along="BlockUnits",
        target="BlockSummaries",
        reducer=_digit_histogram_reducer,
        description="Aggregate cell digits into block-level digit histograms.",
        metadata={"demo": "sudoku"},
    )
    diagram.morphism(
        "row_duplicates",
        "row_histograms",
        "RowViolations",
        implementation=_duplicate_counts,
        description="Count repeated digits per row.",
    )
    diagram.morphism(
        "column_duplicates",
        "column_histograms",
        "ColumnViolations",
        implementation=_duplicate_counts,
        description="Count repeated digits per column.",
    )
    diagram.morphism(
        "block_duplicates",
        "block_histograms",
        "BlockViolations",
        implementation=_duplicate_counts,
        description="Count repeated digits per 2x2 block.",
    )

    diagram.expose_port("cells", "CellDigits", direction="input", port_type="cell_digits")
    diagram.expose_port("rows", "RowUnits", direction="input", port_type="row_relation")
    diagram.expose_port("columns", "ColumnUnits", direction="input", port_type="column_relation")
    diagram.expose_port("blocks", "BlockUnits", direction="input", port_type="block_relation")
    diagram.expose_port("row_summary", "row_histograms", direction="output", port_type="digit_histograms")
    diagram.expose_port(
        "column_summary",
        "column_histograms",
        direction="output",
        port_type="digit_histograms",
    )
    diagram.expose_port(
        "block_summary",
        "block_histograms",
        direction="output",
        port_type="digit_histograms",
    )
    diagram.expose_port(
        "row_violations",
        "row_duplicates",
        direction="output",
        port_type="violation_map",
    )
    diagram.expose_port(
        "column_violations",
        "column_duplicates",
        direction="output",
        port_type="violation_map",
    )
    diagram.expose_port(
        "block_violations",
        "block_duplicates",
        direction="output",
        port_type="violation_map",
    )
    return diagram


def analyze_sudoku_board(board: Sequence[int] | torch.Tensor) -> dict[str, object]:
    diagram = build_sudoku_constraint_diagram()
    relations = build_sudoku_unit_relations()
    compiled = compile_to_callable(diagram)
    result = compiled.run(
        {
            "CellDigits": _board_mapping(board),
            "RowUnits": relations["rows"],
            "ColumnUnits": relations["columns"],
            "BlockUnits": relations["blocks"],
        }
    )
    row_violations = result.values[diagram.port("row_violations")]
    column_violations = result.values[diagram.port("column_violations")]
    block_violations = result.values[diagram.port("block_violations")]
    total_duplicates = sum(row_violations.values()) + sum(column_violations.values()) + sum(
        block_violations.values()
    )
    return {
        "diagram": diagram,
        "row_summaries": result.values[diagram.port("row_summary")],
        "column_summaries": result.values[diagram.port("column_summary")],
        "block_summaries": result.values[diagram.port("block_summary")],
        "row_violations": row_violations,
        "column_violations": column_violations,
        "block_violations": block_violations,
        "total_duplicates": int(total_duplicates),
        "is_consistent": int(total_duplicates) == 0,
    }


def format_grid(values: Sequence[int] | torch.Tensor) -> str:
    if isinstance(values, torch.Tensor):
        items = values.detach().cpu().tolist()
    else:
        items = list(values)
    rows = [items[index : index + BOARD_SIZE] for index in range(0, NUM_CELLS, BOARD_SIZE)]
    return "\n".join(" ".join(str(value) for value in row) for row in rows)


class PlainTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + self.dropout(attn_out))
        mlp_out = self.mlp(x)
        return self.mlp_norm(x + self.dropout(mlp_out))


class GeomTransLiteBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, conv_kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
        )
        self.conv_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.attn_norm(x + self.dropout(attn_out))
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv_norm(x + self.dropout(conv_out))
        mlp_out = self.mlp(x)
        return self.mlp_norm(x + self.dropout(mlp_out))


class GTReasoner(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [GeomTransLiteBlock(d_model, n_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class SudokuTransformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int):
        super().__init__()
        self.token_embedding = nn.Embedding(NUM_DIGITS + 1, d_model)
        self.position_embedding = nn.Embedding(NUM_CELLS, d_model)
        self.layers = nn.ModuleList(
            [PlainTransformerBlock(d_model, n_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, NUM_DIGITS)

    def _embed(self, puzzles: torch.Tensor) -> torch.Tensor:
        token_ids = puzzles + 1
        positions = torch.arange(NUM_CELLS, device=puzzles.device)
        return self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]

    def forward(self, puzzles: torch.Tensor, *, return_embeddings: bool = False):
        hidden = self._embed(puzzles)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        logits = self.output_head(hidden)
        if return_embeddings:
            return logits, hidden
        return logits


class SudokuGT(nn.Module):
    def __init__(self, d_model: int, n_heads: int, num_layers: int):
        super().__init__()
        self.token_embedding = nn.Embedding(NUM_DIGITS + 1, d_model)
        self.position_embedding = nn.Embedding(NUM_CELLS, d_model)
        self.reasoner = GTReasoner(d_model, n_heads, num_layers)
        self.output_head = nn.Linear(d_model, NUM_DIGITS)

    def _embed(self, puzzles: torch.Tensor) -> torch.Tensor:
        token_ids = puzzles + 1
        positions = torch.arange(NUM_CELLS, device=puzzles.device)
        return self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]

    def forward(self, puzzles: torch.Tensor, *, return_embeddings: bool = False):
        hidden = self.reasoner(self._embed(puzzles))
        logits = self.output_head(hidden)
        if return_embeddings:
            return logits, hidden
        return logits


def triangle_consistency(hiddens: torch.Tensor, triangles: Sequence[tuple[int, int, int]]) -> torch.Tensor:
    if hiddens.ndim == 2:
        hiddens = hiddens.unsqueeze(0)
    total = torch.zeros((), device=hiddens.device)
    for first, second, third in triangles:
        v1 = hiddens[:, first, :]
        v2 = hiddens[:, second, :]
        v3 = hiddens[:, third, :]
        mean = (v1 + v2 + v3) / 3.0
        tri_loss = (
            (v1 - mean).pow(2).sum(-1)
            + (v2 - mean).pow(2).sum(-1)
            + (v3 - mean).pow(2).sum(-1)
        )
        total = total + tri_loss.mean()
    return total / max(1, len(triangles))


def _iterate_batches(dataset: Sequence[SudokuInstance], batch_size: int):
    indices = list(range(len(dataset)))
    while True:
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            puzzles = torch.stack([dataset[index].puzzle for index in batch_indices], dim=0)
            solutions = torch.stack([dataset[index].solution for index in batch_indices], dim=0)
            yield puzzles, solutions


def train_sudoku_model(
    model: nn.Module,
    train_ds: Sequence[SudokuInstance],
    val_ds: Sequence[SudokuInstance],
    *,
    batch_size: int,
    num_epochs: int,
    lr: float,
    device: torch.device,
    triangles: Sequence[tuple[int, int, int]] | None = None,
    lambda_db: float = 0.0,
) -> dict[str, list[float] | float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_batches = _iterate_batches(train_ds, batch_size)
    history = {
        "epoch": [],
        "train_loss": [],
        "val_cell_acc": [],
        "val_puzzle_acc": [],
    }
    steps_per_epoch = max(1, len(train_ds) // batch_size)
    use_db = triangles is not None and lambda_db > 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            puzzles, solutions = next(train_batches)
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            if use_db:
                logits, embeddings = model(puzzles, return_embeddings=True)
            else:
                logits = model(puzzles)
                embeddings = None

            loss = F.cross_entropy(logits.reshape(-1, NUM_DIGITS), solutions.reshape(-1))
            if use_db and embeddings is not None:
                loss = loss + lambda_db * triangle_consistency(embeddings, triangles)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().item())

        model.eval()
        correct_cells = 0
        total_cells = 0
        full_correct = 0
        with torch.no_grad():
            for instance in val_ds:
                logits = model(instance.puzzle.unsqueeze(0).to(device))
                prediction = logits.argmax(dim=-1).squeeze(0).cpu()
                solution = instance.solution.cpu()
                correct_cells += int((prediction == solution).sum().item())
                total_cells += int(solution.numel())
                full_correct += int(torch.equal(prediction, solution))

        history["epoch"].append(epoch)
        history["train_loss"].append(running_loss / steps_per_epoch)
        history["val_cell_acc"].append(correct_cells / max(1, total_cells))
        history["val_puzzle_acc"].append(full_correct / max(1, len(val_ds)))
    return history


def run_sudoku_demo(
    config: SudokuDemoConfig | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    demo_config = config or SudokuDemoConfig()
    set_seed(demo_config.seed)
    if device is None:
        device = pick_device("cpu")

    train_ds = make_sudoku_dataset(demo_config.train_samples, demo_config.num_givens)
    val_ds = make_sudoku_dataset(demo_config.val_samples, demo_config.num_givens)
    triangles = build_sudoku_triangles()
    diagram = build_sudoku_constraint_diagram()

    transformer = SudokuTransformer(
        demo_config.d_model,
        demo_config.n_heads,
        demo_config.num_layers,
    )
    gt_db = SudokuGT(
        demo_config.d_model,
        demo_config.n_heads,
        demo_config.num_layers,
    )

    transformer_history = train_sudoku_model(
        transformer,
        train_ds,
        val_ds,
        batch_size=demo_config.batch_size,
        num_epochs=demo_config.epochs,
        lr=demo_config.lr,
        device=device,
    )
    gt_db_history = train_sudoku_model(
        gt_db,
        train_ds,
        val_ds,
        batch_size=demo_config.batch_size,
        num_epochs=demo_config.epochs,
        lr=demo_config.lr,
        device=device,
        triangles=triangles,
        lambda_db=demo_config.lambda_db,
    )

    sample = val_ds[0]
    with torch.no_grad():
        gt_prediction = gt_db(sample.puzzle.unsqueeze(0).to(device)).argmax(dim=-1).squeeze(0).cpu()

    puzzle_report = analyze_sudoku_board(sample.puzzle)
    prediction_report = analyze_sudoku_board(gt_prediction)
    solution_report = analyze_sudoku_board(sample.solution)

    return {
        "config": demo_config,
        "diagram": diagram,
        "triangles": triangles,
        "models": {
            "transformer": {
                "history": transformer_history,
                "final_val_cell_acc": transformer_history["val_cell_acc"][-1],
                "final_val_puzzle_acc": transformer_history["val_puzzle_acc"][-1],
            },
            "gt_db": {
                "history": gt_db_history,
                "final_val_cell_acc": gt_db_history["val_cell_acc"][-1],
                "final_val_puzzle_acc": gt_db_history["val_puzzle_acc"][-1],
            },
        },
        "sample": {
            "puzzle": sample.puzzle,
            "prediction": gt_prediction,
            "solution": sample.solution,
            "puzzle_report": puzzle_report,
            "prediction_report": prediction_report,
            "solution_report": solution_report,
        },
    }


def main() -> None:
    result = run_sudoku_demo()
    print("FunctorFlow mini-Sudoku demo")
    print(result["diagram"].summary())
    for name, payload in result["models"].items():
        print(
            f"{name}: "
            f"val_cell_acc={payload['final_val_cell_acc']:.3f} "
            f"val_puzzle_acc={payload['final_val_puzzle_acc']:.3f}"
        )
    sample = result["sample"]
    print("Puzzle (-1 = blank):")
    print(format_grid(sample["puzzle"]))
    print("Prediction:")
    print(format_grid(sample["prediction"]))
    print("Solution:")
    print(format_grid(sample["solution"]))
    print(
        "Prediction duplicate counts:",
        sample["prediction_report"]["total_duplicates"],
    )


if __name__ == "__main__":
    main()

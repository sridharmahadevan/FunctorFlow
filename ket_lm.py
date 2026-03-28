from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compiler import compile_to_torch
from .macros import KETBlockConfig, ket_block


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_ROOT = PACKAGE_ROOT / "data"
EOS_TOKEN = "<eos>"


@dataclass(frozen=True)
class WordLanguageModelingCorpus:
    name: str
    vocab: dict[str, int]
    train_ids: torch.Tensor
    valid_ids: torch.Tensor
    test_ids: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


@dataclass(frozen=True)
class KETHeadConfig:
    variant: str = "course_attention_kan"
    regime: str = "predict_detach"
    window_k: int = 0
    edge_hidden_dim: int | None = None
    temperature: float = 1.0


@dataclass(frozen=True)
class KETLanguageModelConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    window_k: int = 0
    max_positions: int = 4096
    feedforward_multiplier: int = 4
    dropout: float = 0.1
    head: KETHeadConfig = KETHeadConfig()

    @classmethod
    def historical_ptb_reference(cls) -> "KETLanguageModelConfig":
        return cls(
            d_model=256,
            n_layers=2,
            window_k=0,
            max_positions=4096,
            head=KETHeadConfig(
                variant="course_attention_kan",
                regime="predict_detach",
                window_k=0,
                edge_hidden_dim=256,
                temperature=1.0,
            ),
        )

    @classmethod
    def historical_ptb_smoke(cls) -> "KETLanguageModelConfig":
        return cls(
            d_model=128,
            n_layers=2,
            window_k=64,
            max_positions=512,
            head=KETHeadConfig(
                variant="course_attention_kan",
                regime="predict_detach",
                window_k=64,
                edge_hidden_dim=128,
                temperature=1.0,
            ),
        )

    @classmethod
    def historical_wiki2_reference(cls) -> "KETLanguageModelConfig":
        return cls(
            d_model=256,
            n_layers=2,
            window_k=128,
            max_positions=4096,
            head=KETHeadConfig(
                variant="course_attention_kan",
                regime="predict_detach",
                window_k=128,
                edge_hidden_dim=256,
                temperature=1.0,
            ),
        )

    @classmethod
    def historical_wiki2_smoke(cls) -> "KETLanguageModelConfig":
        return cls(
            d_model=128,
            n_layers=2,
            window_k=64,
            max_positions=512,
            head=KETHeadConfig(
                variant="course_attention_kan",
                regime="predict_detach",
                window_k=64,
                edge_hidden_dim=128,
                temperature=1.0,
            ),
        )

    @classmethod
    def historical_wiki103_reference(cls) -> "KETLanguageModelConfig":
        return cls(
            d_model=256,
            n_layers=2,
            window_k=128,
            max_positions=4096,
            head=KETHeadConfig(
                variant="course_attention_kan",
                regime="predict_detach",
                window_k=128,
                edge_hidden_dim=256,
                temperature=1.0,
            ),
        )

    @classmethod
    def historical_wiki103_smoke(cls) -> "KETLanguageModelConfig":
        return cls(
            d_model=128,
            n_layers=2,
            window_k=64,
            max_positions=512,
            head=KETHeadConfig(
                variant="course_attention_kan",
                regime="predict_detach",
                window_k=64,
                edge_hidden_dim=128,
                temperature=1.0,
            ),
        )


def _tokenize_lines(path: Path) -> list[str]:
    tokens: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line_tokens = [token for token in raw_line.strip().split() if token]
        tokens.extend(line_tokens)
        tokens.append(EOS_TOKEN)
    return tokens


def _encode_tokens(path: Path, vocab: dict[str, int]) -> torch.Tensor:
    tokens = _tokenize_lines(path)
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)


def load_word_language_modeling_corpus(
    name: str,
    *,
    root: str | Path = DEFAULT_DATA_ROOT,
) -> WordLanguageModelingCorpus:
    normalized = name.lower().replace("_", "-")
    root_path = Path(root)
    if normalized == "ptb":
        dataset_dir = root_path / "ptb"
        train_path = dataset_dir / "ptb.train.txt"
        valid_path = dataset_dir / "ptb.valid.txt"
        test_path = dataset_dir / "ptb.test.txt"
        corpus_name = "ptb"
    elif normalized in {"wiki-2", "wikitext-2", "wiki2"}:
        dataset_dir = root_path / "wikitext-2"
        train_path = dataset_dir / "wiki.train.tokens"
        valid_path = dataset_dir / "wiki.valid.tokens"
        test_path = dataset_dir / "wiki.test.tokens"
        corpus_name = "wiki-2"
    elif normalized in {"wiki-103", "wikitext-103", "wiki103"}:
        dataset_dir = root_path / "wikitext-103"
        train_path = dataset_dir / "wiki.train.tokens"
        valid_path = dataset_dir / "wiki.valid.tokens"
        test_path = dataset_dir / "wiki.test.tokens"
        corpus_name = "wiki-103"
    else:
        raise ValueError(f"Unsupported corpus '{name}'. Expected 'ptb', 'wiki-2', or 'wiki-103'.")

    if not dataset_dir.exists() and root_path == DEFAULT_DATA_ROOT:
        fallback_root = REPO_ROOT
        if normalized == "ptb":
            dataset_dir = fallback_root / "ptb"
            train_path = dataset_dir / "ptb.train.txt"
            valid_path = dataset_dir / "ptb.valid.txt"
            test_path = dataset_dir / "ptb.test.txt"
        elif normalized in {"wiki-2", "wikitext-2", "wiki2"}:
            dataset_dir = fallback_root / "wikitext-2"
            train_path = dataset_dir / "wiki.train.tokens"
            valid_path = dataset_dir / "wiki.valid.tokens"
            test_path = dataset_dir / "wiki.test.tokens"
        else:
            dataset_dir = fallback_root / "wikitext-103"
            train_path = dataset_dir / "wiki.train.tokens"
            valid_path = dataset_dir / "wiki.valid.tokens"
            test_path = dataset_dir / "wiki.test.tokens"

    for path in (train_path, valid_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing language-modeling corpus file: {path}")

    train_tokens = _tokenize_lines(train_path)
    vocab_tokens = sorted(set(train_tokens))
    vocab = {token: index for index, token in enumerate(vocab_tokens)}
    return WordLanguageModelingCorpus(
        name=corpus_name,
        vocab=vocab,
        train_ids=torch.tensor([vocab[token] for token in train_tokens], dtype=torch.long),
        valid_ids=_encode_tokens(valid_path, vocab),
        test_ids=_encode_tokens(test_path, vocab),
    )


def make_batches(data_ids: torch.Tensor, block_size: int = 128, batch_size: int = 64):
    n_tokens = int(data_ids.numel())
    if n_tokens < 2:
        raise ValueError(f"Not enough data to make batches: n_tokens={n_tokens}")

    effective_block = min(block_size, max(1, n_tokens - 2))
    max_start = n_tokens - effective_block - 1
    if max_start <= 0:
        starts = torch.zeros((batch_size,), dtype=torch.long)
    else:
        starts = torch.randint(0, max_start + 1, (batch_size,))

    inputs = torch.stack([data_ids[start : start + effective_block] for start in starts])
    targets = torch.stack([data_ids[start + 1 : start + effective_block + 1] for start in starts])
    return inputs, targets, effective_block


def causal_relation_mask(
    length: int,
    *,
    device: torch.device | None = None,
    window_k: int = 0,
) -> torch.Tensor:
    target_index = torch.arange(length, device=device)[:, None]
    source_index = torch.arange(length, device=device)[None, :]
    allow = source_index <= target_index
    if window_k and window_k > 0:
        allow = allow & (source_index >= (target_index - (window_k - 1)))
    return allow


def masked_kan_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    relation: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(float(temperature), 1e-6)
    scores = (query_states @ key_states.transpose(-1, -2)) / scale
    scores = scores.masked_fill(~relation, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return weights @ value_states, weights


class FunctorFlowKETReducer(nn.Module):
    """A learned left-Kan reducer for token-state aggregation."""

    def __init__(
        self,
        d_model: int,
        *,
        variant: str = "historical_vectorized",
        regime: str = "predict_detach",
        window_k: int = 0,
        edge_hidden_dim: int | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.variant = str(variant)
        self.regime = str(regime)
        self.window_k = int(window_k)
        self.edge_hidden_dim = int(edge_hidden_dim or d_model)
        self.temperature = float(temperature or 1.0)

        if self.variant not in {"historical_vectorized", "course_attention_kan"}:
            raise ValueError(
                f"Unsupported KET reducer variant '{self.variant}'. "
                "Expected 'historical_vectorized' or 'course_attention_kan'."
            )

        self.query_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.d_model, self.edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.edge_hidden_dim, 1),
        )
        self.value_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.output_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def forward(
        self,
        source: torch.Tensor,
        relation: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, width = source.shape
        if width != self.d_model:
            raise ValueError(
                f"Expected hidden width {self.d_model} for FunctorFlowKETReducer, received {width}"
            )

        if relation.dim() == 2:
            allow = relation.unsqueeze(0).expand(batch_size, -1, -1)
        elif relation.dim() == 3:
            allow = relation
        else:
            raise ValueError("KET relations must be a (T, T) or (B, T, T) boolean tensor")

        allow = allow.to(device=source.device, dtype=torch.bool)
        if allow.shape != (batch_size, sequence_length, sequence_length):
            raise ValueError(
                f"Expected relation shape {(batch_size, sequence_length, sequence_length)}, "
                f"received {tuple(allow.shape)}"
            )

        basis_states = source
        if metadata and "basis_states" in metadata:
            basis_states = metadata["basis_states"]
            if not isinstance(basis_states, torch.Tensor):
                raise ValueError("Reducer metadata['basis_states'] must be a torch.Tensor")
            if tuple(basis_states.shape) != tuple(source.shape):
                raise ValueError(
                    f"Expected basis state shape {tuple(source.shape)}, received {tuple(basis_states.shape)}"
                )
            basis_states = basis_states.to(device=source.device, dtype=source.dtype)

        if self.regime == "predict_detach":
            if not (metadata and "basis_states" in metadata):
                basis_states = self.predictor(source).detach()
        elif self.regime not in {"vanilla", "predict_detach"}:
            raise ValueError(
                f"Unsupported KET regime '{self.regime}'. Expected 'vanilla' or 'predict_detach'."
            )

        if self.variant == "course_attention_kan":
            queries = self.query_proj(source)
            keys = self.key_proj(basis_states)
            values = self.value_proj(basis_states)
            aggregated, _ = masked_kan_attention(
                queries,
                keys,
                values,
                allow,
                temperature=math.sqrt(self.d_model) * self.temperature,
            )
            return self.output_proj(aggregated)

        values = self.value_proj(source)

        target_states = source.unsqueeze(2).expand(batch_size, sequence_length, sequence_length, width)
        source_states = source.unsqueeze(1).expand(batch_size, sequence_length, sequence_length, width)
        edge_inputs = torch.cat([target_states, source_states], dim=-1)
        scores = self.edge_mlp(edge_inputs.reshape(batch_size * sequence_length * sequence_length, 2 * width))
        scores = scores.reshape(batch_size, sequence_length, sequence_length)
        scores = scores.masked_fill(~allow, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        aggregated = weights @ values
        return self.output_proj(aggregated)


def build_ket_language_model_diagram(name: str = "FunctorFlowKETHead"):
    return ket_block(
        KETBlockConfig(
            name=name,
            source_object="HiddenStates",
            relation_object="CausalRelation",
            target_object="ContextualizedStates",
            aggregate_name="ket_context",
            reducer="ket_attention",
        )
    )


class FunctorFlowKETHead(nn.Module):
    def __init__(self, d_model: int, *, config: KETHeadConfig | None = None):
        super().__init__()
        self.config = config or KETHeadConfig()
        self.window_k = int(self.config.window_k)
        self.diagram = build_ket_language_model_diagram()
        self.reducer = FunctorFlowKETReducer(
            d_model,
            variant=self.config.variant,
            regime=self.config.regime,
            window_k=self.config.window_k,
            edge_hidden_dim=self.config.edge_hidden_dim,
            temperature=self.config.temperature,
        )
        self.compiled = compile_to_torch(
            self.diagram,
            reducers={"ket_attention": self.reducer},
        )
        self.output_ref = self.diagram.port("output")
        self.relation_ref = self.diagram.port("relation")
        self.input_ref = self.diagram.port("input")

    def forward(
        self,
        hidden_states: torch.Tensor,
        relation: torch.Tensor | None = None,
        *,
        basis_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if relation is None:
            relation = causal_relation_mask(
                hidden_states.size(1),
                device=hidden_states.device,
                window_k=self.window_k,
            )
        if basis_states is not None:
            return self.reducer(
                hidden_states,
                relation,
                metadata={"basis_states": basis_states},
            )
        outputs = self.compiled(
            {
                self.input_ref: hidden_states,
                self.relation_ref: relation,
            }
        )
        return outputs[self.output_ref]


class CausalTransformerBackboneBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        n_heads: int = 4,
        feedforward_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)
        ff_width = int(feedforward_multiplier) * d_model
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, ff_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_width, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attention_mask,
            need_weights=False,
        )
        hidden_states = self.norm1(hidden_states + self.dropout(attended))
        hidden_states = self.norm2(hidden_states + self.feedforward(hidden_states))
        return hidden_states


class FunctorFlowKETLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        config: KETLanguageModelConfig | None = None,
    ):
        super().__init__()
        lm_config = config or KETLanguageModelConfig()
        self.config = lm_config
        d_model = lm_config.d_model
        n_heads = int(getattr(lm_config, "n_heads", 4))
        dropout = float(getattr(lm_config, "dropout", 0.1))
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(lm_config.max_positions, d_model)
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBackboneBlock(
                    d_model,
                    n_heads=n_heads,
                    feedforward_multiplier=lm_config.feedforward_multiplier,
                    dropout=dropout,
                )
                for _ in range(lm_config.n_layers)
            ]
        )
        self.pre_ket_norm = nn.LayerNorm(d_model)
        self.ket_head = FunctorFlowKETHead(d_model, config=lm_config.head)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def hidden_states(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape
        del batch_size
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        attention_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        base_hidden = self.pre_ket_norm(hidden_states)
        relation = causal_relation_mask(
            sequence_length,
            device=token_ids.device,
            window_k=self.config.window_k,
        )
        basis_states = None
        if self.config.head.regime == "predict_detach":
            with torch.no_grad():
                hint_logits = self.lm_head(base_hidden) / max(1e-6, float(self.config.head.temperature))
                hint_probabilities = torch.softmax(hint_logits, dim=-1)
                basis_states = (hint_probabilities @ self.token_embedding.weight).detach()

        hidden_states = base_hidden + self.ket_head(
            base_hidden,
            relation=relation,
            basis_states=basis_states,
        )
        return self.final_norm(hidden_states)

    def forward(self, token_ids: torch.Tensor, *, return_hidden: bool = False) -> torch.Tensor:
        hidden_states = self.hidden_states(token_ids)
        if return_hidden:
            return hidden_states
        return self.lm_head(hidden_states)


def pick_device(requested: str = "cpu") -> torch.device:
    normalized = requested.lower()
    if normalized == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device("cpu")


def loss_and_perplexity(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, float]:
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return loss, math.exp(float(loss.detach().item()))


def estimate_perplexity(
    model: nn.Module,
    data_ids: torch.Tensor,
    *,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        inputs, targets, _ = make_batches(data_ids, block_size=block_size, batch_size=batch_size)
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, perplexity = loss_and_perplexity(model, inputs, targets)
    return perplexity


def train_language_model(
    model: nn.Module,
    corpus: WordLanguageModelingCorpus,
    *,
    steps: int = 200,
    block_size: int = 128,
    batch_size: int = 32,
    lr: float = 2e-3,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = {"train_loss": [], "valid_ppl": []}
    for step in range(1, steps + 1):
        model.train()
        inputs, targets, _ = make_batches(
            corpus.train_ids,
            block_size=block_size,
            batch_size=batch_size,
        )
        inputs = inputs.to(device)
        targets = targets.to(device)
        loss, _ = loss_and_perplexity(model, inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == steps:
            history["train_loss"].append(float(loss.detach().item()))
            history["valid_ppl"].append(
                estimate_perplexity(
                    model,
                    corpus.valid_ids,
                    block_size=block_size,
                    batch_size=batch_size,
                    device=device,
                )
            )
    return history

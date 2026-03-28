from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ket_lm import (
    DEFAULT_DATA_ROOT,
    FunctorFlowKETLanguageModel,
    KETLanguageModelConfig,
    WordLanguageModelingCorpus,
    estimate_perplexity,
    load_word_language_modeling_corpus,
    train_language_model,
)


@dataclass(frozen=True)
class TransformerLanguageModelConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_positions: int = 4096
    dropout: float = 0.1
    future_hint: bool = False
    hint_scale: float = 0.2

    @classmethod
    def historical_ptb_smoke(cls) -> "TransformerLanguageModelConfig":
        return cls(d_model=128, n_layers=2, n_heads=4, max_positions=512)

    @classmethod
    def historical_ptb_reference(cls) -> "TransformerLanguageModelConfig":
        return cls(d_model=256, n_layers=2, n_heads=4, max_positions=4096)

    @classmethod
    def historical_wiki2_smoke(cls) -> "TransformerLanguageModelConfig":
        return cls(d_model=128, n_layers=2, n_heads=4, max_positions=512)

    @classmethod
    def historical_wiki2_reference(cls) -> "TransformerLanguageModelConfig":
        return cls(d_model=256, n_layers=2, n_heads=4, max_positions=4096)

    @classmethod
    def historical_wiki103_smoke(cls) -> "TransformerLanguageModelConfig":
        return cls(d_model=128, n_layers=2, n_heads=4, max_positions=512)

    @classmethod
    def historical_wiki103_reference(cls) -> "TransformerLanguageModelConfig":
        return cls(d_model=256, n_layers=2, n_heads=4, max_positions=4096)


@dataclass(frozen=True)
class GTGeometryConfig:
    geo_causal: bool = True
    geo_mix_source: str = "pred_prev_causal_detach"
    pred_temp: float = 1.0
    dropout: float = 0.1


@dataclass(frozen=True)
class GTLanguageModelConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    max_positions: int = 4096
    geometry: GTGeometryConfig = GTGeometryConfig()

    @classmethod
    def historical_ptb_smoke(cls) -> "GTLanguageModelConfig":
        return cls(
            d_model=128,
            n_layers=2,
            n_heads=4,
            max_positions=512,
            geometry=GTGeometryConfig(
                geo_causal=True,
                geo_mix_source="pred_prev_causal_detach",
                pred_temp=1.0,
                dropout=0.1,
            ),
        )

    @classmethod
    def historical_ptb_reference(cls) -> "GTLanguageModelConfig":
        return cls(
            d_model=256,
            n_layers=2,
            n_heads=4,
            max_positions=4096,
            geometry=GTGeometryConfig(
                geo_causal=True,
                geo_mix_source="pred_prev_causal_detach",
                pred_temp=1.0,
                dropout=0.1,
            ),
        )

    @classmethod
    def historical_wiki2_smoke(cls) -> "GTLanguageModelConfig":
        return cls(
            d_model=128,
            n_layers=2,
            n_heads=4,
            max_positions=512,
            geometry=GTGeometryConfig(
                geo_causal=True,
                geo_mix_source="pred_prev_causal_detach",
                pred_temp=1.0,
                dropout=0.1,
            ),
        )

    @classmethod
    def historical_wiki2_reference(cls) -> "GTLanguageModelConfig":
        return cls(
            d_model=256,
            n_layers=2,
            n_heads=4,
            max_positions=4096,
            geometry=GTGeometryConfig(
                geo_causal=True,
                geo_mix_source="pred_prev_causal_detach",
                pred_temp=1.0,
                dropout=0.1,
            ),
        )

    @classmethod
    def historical_wiki103_smoke(cls) -> "GTLanguageModelConfig":
        return cls(
            d_model=128,
            n_layers=2,
            n_heads=4,
            max_positions=512,
            geometry=GTGeometryConfig(
                geo_causal=True,
                geo_mix_source="pred_prev_causal_detach",
                pred_temp=1.0,
                dropout=0.1,
            ),
        )

    @classmethod
    def historical_wiki103_reference(cls) -> "GTLanguageModelConfig":
        return cls(
            d_model=256,
            n_layers=2,
            n_heads=4,
            max_positions=4096,
            geometry=GTGeometryConfig(
                geo_causal=True,
                geo_mix_source="pred_prev_causal_detach",
                pred_temp=1.0,
                dropout=0.1,
            ),
        )


@dataclass(frozen=True)
class LMComparisonConfig:
    steps: int = 25
    block_size: int = 64
    batch_size: int = 8
    lr: float = 2e-3
    model_profile: str = "smoke"
    train_tokens: int | None = 4096
    valid_tokens: int | None = 1024
    test_tokens: int | None = 1024
    seed: int = 0


class GeometricTransformerBlock(nn.Module):
    """Course-ported GT-Lite block: causal attention plus a geometric depthwise mixer."""

    def __init__(self, d_model: int, n_heads: int, *, config: GTGeometryConfig | None = None):
        super().__init__()
        geometry = config or GTGeometryConfig()
        self.config = geometry
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            batch_first=True,
            dropout=geometry.dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        padding = 0 if geometry.geo_causal else 1
        self.geo_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=padding,
            groups=d_model,
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_geo = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(geometry.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        geometry_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=attention_mask,
            need_weights=False,
        )
        hidden_states = self.norm_attn(hidden_states + self.dropout(attended))
        hidden_states = self.norm_ffn(hidden_states + self.dropout(self.ffn(hidden_states)))

        base = hidden_states if geometry_source is None else geometry_source
        geometry = base.transpose(1, 2)
        if self.config.geo_causal:
            geometry = F.pad(geometry, (2, 0))
        geometry = self.geo_conv(geometry).transpose(1, 2)
        hidden_states = self.norm_geo(hidden_states + self.dropout(geometry))
        return hidden_states


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, *, config: TransformerLanguageModelConfig | None = None):
        super().__init__()
        lm_config = config or TransformerLanguageModelConfig()
        self.config = lm_config
        self.token_embedding = nn.Embedding(vocab_size, lm_config.d_model)
        self.position_embedding = nn.Embedding(lm_config.max_positions, lm_config.d_model)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=lm_config.d_model,
                    nhead=lm_config.n_heads,
                    dim_feedforward=4 * lm_config.d_model,
                    batch_first=True,
                    dropout=lm_config.dropout,
                    activation="gelu",
                )
                for _ in range(lm_config.n_layers)
            ]
        )
        self.output_head = nn.Linear(lm_config.d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)
        attention_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            hidden_states = block(hidden_states, src_mask=attention_mask)

        if self.config.future_hint and sequence_length > 1:
            hint = torch.zeros_like(hidden_states)
            hint[:, :-1, :] = hidden_states[:, 1:, :]
            hidden_states = hidden_states + (self.config.hint_scale * hint)

        return self.output_head(hidden_states)


class GTLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, *, config: GTLanguageModelConfig | None = None):
        super().__init__()
        lm_config = config or GTLanguageModelConfig()
        self.config = lm_config
        self.token_embedding = nn.Embedding(vocab_size, lm_config.d_model)
        self.position_embedding = nn.Embedding(lm_config.max_positions, lm_config.d_model)
        self.layers = nn.ModuleList(
            [
                GeometricTransformerBlock(
                    lm_config.d_model,
                    lm_config.n_heads,
                    config=lm_config.geometry,
                )
                for _ in range(lm_config.n_layers)
            ]
        )
        self.output_head = nn.Linear(lm_config.d_model, vocab_size)

    def _predictive_embedding(self, hidden_states: torch.Tensor, *, detach: bool = True) -> torch.Tensor:
        logits = self.output_head(hidden_states) / max(1e-6, float(self.config.geometry.pred_temp))
        probabilities = torch.softmax(logits, dim=-1)
        predicted = probabilities @ self.token_embedding.weight
        return predicted.detach() if detach else predicted

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device).unsqueeze(0)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)
        attention_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for layer in self.layers:
            geometry_source = None
            mode = self.config.geometry.geo_mix_source
            if mode == "pred_next_detach":
                geometry_source = self._predictive_embedding(hidden_states, detach=True)
            elif mode == "pred_prev_causal_detach":
                predicted = self._predictive_embedding(hidden_states, detach=True)
                geometry_source = torch.zeros_like(predicted)
                geometry_source[:, 1:, :] = predicted[:, :-1, :]
            elif mode not in {"hidden", ""}:
                raise ValueError(
                    f"Unsupported GT geometry source '{mode}'. "
                    "Expected 'hidden', 'pred_next_detach', or 'pred_prev_causal_detach'."
                )
            hidden_states = layer(
                hidden_states,
                attention_mask,
                geometry_source=geometry_source,
            )

        return self.output_head(hidden_states)


def _truncate_ids(token_ids: torch.Tensor, limit: int | None) -> torch.Tensor:
    if limit is None:
        return token_ids
    return token_ids[: max(2, int(limit))]


def truncate_corpus(
    corpus: WordLanguageModelingCorpus,
    *,
    train_tokens: int | None = None,
    valid_tokens: int | None = None,
    test_tokens: int | None = None,
) -> WordLanguageModelingCorpus:
    return WordLanguageModelingCorpus(
        name=corpus.name,
        vocab=corpus.vocab,
        train_ids=_truncate_ids(corpus.train_ids, train_tokens),
        valid_ids=_truncate_ids(corpus.valid_ids, valid_tokens),
        test_ids=_truncate_ids(corpus.test_ids, test_tokens),
    )


def _resolve_profile_config(corpus_name: str, profile: str) -> tuple[
    TransformerLanguageModelConfig,
    GTLanguageModelConfig,
    KETLanguageModelConfig,
]:
    normalized = corpus_name.lower().replace("_", "-")
    profile_name = profile.lower()
    if profile_name not in {"smoke", "reference"}:
        raise ValueError(f"Unsupported model profile '{profile}'. Expected 'smoke' or 'reference'.")

    if normalized == "ptb":
        if profile_name == "reference":
            return (
                TransformerLanguageModelConfig.historical_ptb_reference(),
                GTLanguageModelConfig.historical_ptb_reference(),
                KETLanguageModelConfig.historical_ptb_reference(),
            )
        return (
            TransformerLanguageModelConfig.historical_ptb_smoke(),
            GTLanguageModelConfig.historical_ptb_smoke(),
            KETLanguageModelConfig.historical_ptb_smoke(),
        )

    if normalized in {"wiki-2", "wikitext-2", "wiki2"}:
        if profile_name == "reference":
            return (
                TransformerLanguageModelConfig.historical_wiki2_reference(),
                GTLanguageModelConfig.historical_wiki2_reference(),
                KETLanguageModelConfig.historical_wiki2_reference(),
            )
        return (
            TransformerLanguageModelConfig.historical_wiki2_smoke(),
            GTLanguageModelConfig.historical_wiki2_smoke(),
            KETLanguageModelConfig.historical_wiki2_smoke(),
        )

    if normalized in {"wiki-103", "wikitext-103", "wiki103"}:
        if profile_name == "reference":
            return (
                TransformerLanguageModelConfig.historical_wiki103_reference(),
                GTLanguageModelConfig.historical_wiki103_reference(),
                KETLanguageModelConfig.historical_wiki103_reference(),
            )
        return (
            TransformerLanguageModelConfig.historical_wiki103_smoke(),
            GTLanguageModelConfig.historical_wiki103_smoke(),
            KETLanguageModelConfig.historical_wiki103_smoke(),
        )

    raise ValueError(f"Unsupported comparison corpus '{corpus_name}'.")


def train_model_for_comparison(
    model: nn.Module,
    corpus: WordLanguageModelingCorpus,
    *,
    steps: int,
    block_size: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict[str, object]:
    history = train_language_model(
        model,
        corpus,
        steps=steps,
        block_size=block_size,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    valid_ppl = estimate_perplexity(
        model,
        corpus.valid_ids,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
    )
    return {"history": history, "valid_ppl": valid_ppl}


def build_default_model_suite(
    corpus_name: str,
    vocab_size: int,
    *,
    profile: str = "smoke",
) -> dict[str, nn.Module]:
    transformer_config, gt_config, ket_config = _resolve_profile_config(corpus_name, profile)
    return {
        "transformer": TransformerLanguageModel(vocab_size, config=transformer_config),
        "gt": GTLanguageModel(vocab_size, config=gt_config),
        "ket": FunctorFlowKETLanguageModel(vocab_size, config=ket_config),
    }


def compare_language_models(
    corpus_name: str,
    *,
    root: str | Path | None = None,
    comparison_config: LMComparisonConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, object]:
    config = comparison_config or LMComparisonConfig()
    torch.manual_seed(config.seed)

    corpus = load_word_language_modeling_corpus(
        corpus_name,
        root=DEFAULT_DATA_ROOT if root is None else root,
    )
    corpus = truncate_corpus(
        corpus,
        train_tokens=config.train_tokens,
        valid_tokens=config.valid_tokens,
        test_tokens=config.test_tokens,
    )

    if device is None:
        device = torch.device("cpu")

    models = build_default_model_suite(
        corpus.name,
        corpus.vocab_size,
        profile=config.model_profile,
    )
    results: dict[str, object] = {
        "corpus": corpus.name,
        "vocab_size": corpus.vocab_size,
        "steps": config.steps,
        "block_size": config.block_size,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "model_profile": config.model_profile,
        "train_tokens": int(corpus.train_ids.numel()),
        "valid_tokens": int(corpus.valid_ids.numel()),
        "test_tokens": int(corpus.test_ids.numel()),
        "models": {},
    }
    for model_name, model in models.items():
        run_result = train_model_for_comparison(
            model,
            corpus,
            steps=config.steps,
            block_size=config.block_size,
            batch_size=config.batch_size,
            lr=config.lr,
            device=device,
        )
        results["models"][model_name] = run_result
    return results

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .ket_block_duality import (
    PAD_ID,
    DenoisingBlockHead,
    FunctorFlowKETBlockPredictor,
    FunctorFlowKETDenoiser,
    OffsetBlockLMHead,
    corrupt_block_targets,
    evaluate_block_predictor,
    evaluate_denoiser,
    make_block_targets,
    make_token_blocks,
    set_seed,
    train_block_predictor,
    train_denoiser,
)
from .ket_lm import (
    KETLanguageModelConfig,
    WordLanguageModelingCorpus,
    load_word_language_modeling_corpus,
    pick_device,
)
from .macros import StructuredLMDualityConfig, structured_lm_duality


def _normalize_corpus_name(name: str) -> str:
    normalized = name.lower().replace("_", "-")
    if normalized == "wiki2":
        return "wiki-2"
    if normalized == "wiki103":
        return "wiki-103"
    return normalized


def _historical_lm_config(corpus_name: str, *, smoke: bool) -> KETLanguageModelConfig:
    normalized = _normalize_corpus_name(corpus_name)
    if normalized == "ptb":
        return (
            KETLanguageModelConfig.historical_ptb_smoke()
            if smoke
            else KETLanguageModelConfig.historical_ptb_reference()
        )
    if normalized in {"wiki-2", "wikitext-2"}:
        return (
            KETLanguageModelConfig.historical_wiki2_smoke()
            if smoke
            else KETLanguageModelConfig.historical_wiki2_reference()
        )
    if normalized in {"wiki-103", "wikitext-103"}:
        return (
            KETLanguageModelConfig.historical_wiki103_smoke()
            if smoke
            else KETLanguageModelConfig.historical_wiki103_reference()
        )
    raise ValueError(f"Unsupported structured-LM corpus '{corpus_name}'")


@dataclass(frozen=True)
class KETStructuredLanguageModelConfig:
    corpus_name: str = "ptb"
    task: str = "block"
    seq_len: int = 64
    batch_size: int = 8
    steps: int = 12
    lr: float = 2e-3
    block_size: int = 4
    num_denoise_steps: int = 8
    eval_batches: int = 4
    seed: int = 0
    lm_config: KETLanguageModelConfig = field(
        default_factory=KETLanguageModelConfig.historical_ptb_smoke
    )

    @classmethod
    def historical_smoke(
        cls,
        corpus_name: str = "ptb",
        *,
        task: str = "block",
    ) -> "KETStructuredLanguageModelConfig":
        return cls(
            corpus_name=_normalize_corpus_name(corpus_name),
            task=task,
            lm_config=_historical_lm_config(corpus_name, smoke=True),
        )

    @classmethod
    def historical_reference(
        cls,
        corpus_name: str = "ptb",
        *,
        task: str = "block",
    ) -> "KETStructuredLanguageModelConfig":
        return cls(
            corpus_name=_normalize_corpus_name(corpus_name),
            task=task,
            lm_config=_historical_lm_config(corpus_name, smoke=False),
        )


@dataclass(frozen=True)
class StructuredLMComparisonConfig:
    corpus_name: str = "ptb"
    seq_len: int = 128
    batch_size: int = 16
    steps: int = 1000
    lr: float = 3e-4
    block_size: int = 4
    num_denoise_steps: int = 8
    eval_batches: int = 16
    seed: int = 0
    lm_config: KETLanguageModelConfig = field(
        default_factory=KETLanguageModelConfig.historical_ptb_reference
    )

    @classmethod
    def historical_smoke(
        cls,
        corpus_name: str = "ptb",
    ) -> "StructuredLMComparisonConfig":
        return cls(
            corpus_name=_normalize_corpus_name(corpus_name),
            seq_len=64,
            batch_size=4,
            steps=2,
            lr=2e-3,
            block_size=4,
            num_denoise_steps=6,
            eval_batches=2,
            lm_config=_historical_lm_config(corpus_name, smoke=True),
        )

    @classmethod
    def historical_reference(
        cls,
        corpus_name: str = "ptb",
    ) -> "StructuredLMComparisonConfig":
        return cls(
            corpus_name=_normalize_corpus_name(corpus_name),
            seq_len=128,
            batch_size=16,
            steps=1000,
            lr=3e-4,
            block_size=4,
            num_denoise_steps=8,
            eval_batches=16,
            lm_config=_historical_lm_config(corpus_name, smoke=False),
        )


def build_structured_language_model_diagram(name: str = "FunctorFlowStructuredLM"):
    return structured_lm_duality(
        StructuredLMDualityConfig(
            name=name,
            hidden_object="HiddenStates",
            relation_object="CausalRelation",
            context_object="ContextualizedStates",
            noisy_block_object="NoisyFutureBlock",
            condition_object="DenoiseCondition",
            completed_object="CompletedFutureBlock",
        )
    )


class _StructuredTransformerBackboneBlock(nn.Module):
    def __init__(self, d_model: int, *, n_heads: int, feedforward_multiplier: int, dropout: float):
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


class StructuredTransformerBackbone(nn.Module):
    def __init__(self, vocab_size: int, *, config: KETLanguageModelConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        n_heads = int(getattr(config, "n_heads", 4))
        dropout = float(getattr(config, "dropout", 0.1))
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(config.max_positions, d_model)
        self.blocks = nn.ModuleList(
            [
                _StructuredTransformerBackboneBlock(
                    d_model,
                    n_heads=n_heads,
                    feedforward_multiplier=config.feedforward_multiplier,
                    dropout=dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, *, return_hidden: bool = False) -> torch.Tensor:
        _, sequence_length = token_ids.shape
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        attention_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.final_norm(hidden_states)
        return hidden_states


class StructuredTransformerBlockPredictor(nn.Module):
    def __init__(self, vocab_size: int, *, lm_config: KETLanguageModelConfig, block_size: int):
        super().__init__()
        self.backbone = StructuredTransformerBackbone(vocab_size, config=lm_config)
        self.head = OffsetBlockLMHead(lm_config.d_model, vocab_size, block_size)
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(tokens, return_hidden=True))


class StructuredTransformerDenoiser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        lm_config: KETLanguageModelConfig,
        block_size: int,
        num_denoise_steps: int,
    ):
        super().__init__()
        self.backbone = StructuredTransformerBackbone(vocab_size, config=lm_config)
        self.head = DenoisingBlockHead(
            lm_config.d_model,
            vocab_size,
            block_size,
            num_denoise_steps,
        )
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.num_denoise_steps = int(num_denoise_steps)

    def forward(self, tokens: torch.Tensor, noisy_block: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(tokens, return_hidden=True), noisy_block, timestep)


def _first_offset_perplexity(
    model: nn.Module,
    corpus: WordLanguageModelingCorpus,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    n_batches: int,
    task: str,
) -> float:
    model.eval()
    total_nll = 0.0
    total_count = 0
    with torch.no_grad():
        for _ in range(n_batches):
            tokens = make_token_blocks(corpus.valid_ids, seq_len=seq_len, batch_size=batch_size).to(device)
            batch_size_actual, sequence_length = tokens.shape
            targets3, valid_mask3 = make_block_targets(tokens, model.block_size, pad_id=PAD_ID)

            if task == "block":
                logits = model(tokens)[:, :, 0, :]
            else:
                timestep = torch.full(
                    (batch_size_actual,),
                    max(1, model.num_denoise_steps // 2),
                    dtype=torch.long,
                    device=device,
                )
                noisy_block = corrupt_block_targets(
                    targets=targets3,
                    valid_mask=valid_mask3,
                    vocab_size=model.vocab_size,
                    noise_level=timestep,
                    num_denoise_steps=model.num_denoise_steps,
                    mask_token_id=model.vocab_size - 1,
                )
                logits = model(tokens, noisy_block, timestep)[:, :, 0, :]

            targets = torch.full((batch_size_actual, sequence_length), PAD_ID, dtype=torch.long, device=device)
            valid = torch.zeros((batch_size_actual, sequence_length), dtype=torch.bool, device=device)
            if sequence_length > 1:
                targets[:, :-1] = tokens[:, 1:]
                valid[:, :-1] = True

            nll = torch.nn.functional.cross_entropy(
                logits.reshape(batch_size_actual * sequence_length, logits.size(-1)),
                targets.reshape(batch_size_actual * sequence_length),
                ignore_index=PAD_ID,
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_count += int(valid.sum().item())
    mean_nll = total_nll / max(total_count, 1)
    return math.exp(mean_nll) if mean_nll < 20 else float("inf")


def _denoise_reconstruction_accuracy(
    model: nn.Module,
    corpus: WordLanguageModelingCorpus,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    n_batches: int,
) -> float:
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for _ in range(n_batches):
            tokens = make_token_blocks(corpus.valid_ids, seq_len=seq_len, batch_size=batch_size).to(device)
            targets, valid_mask = make_block_targets(tokens, model.block_size, pad_id=PAD_ID)
            timestep = torch.full(
                (tokens.size(0),),
                max(1, model.num_denoise_steps // 2),
                dtype=torch.long,
                device=device,
            )
            noisy_block = corrupt_block_targets(
                targets=targets,
                valid_mask=valid_mask,
                vocab_size=model.vocab_size,
                noise_level=timestep,
                num_denoise_steps=model.num_denoise_steps,
                mask_token_id=model.vocab_size - 1,
            )
            logits = model(tokens, noisy_block, timestep)
            prediction = logits.argmax(dim=-1)
            total_correct += int(((prediction == targets) & valid_mask).sum().item())
            total_count += int(valid_mask.sum().item())
    return total_correct / max(total_count, 1)


def evaluate_structured_language_model(
    model: nn.Module,
    corpus: WordLanguageModelingCorpus,
    *,
    task: str,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    n_batches: int,
) -> dict[str, object]:
    if task == "block":
        payload = evaluate_block_predictor(
            model,
            corpus,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            n_batches=n_batches,
        )
    elif task == "denoise":
        payload = evaluate_denoiser(
            model,
            corpus,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            n_batches=n_batches,
        )
        payload["reconstruction_accuracy"] = _denoise_reconstruction_accuracy(
            model,
            corpus,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            n_batches=n_batches,
        )
    else:
        raise ValueError(f"Unsupported structured-LM task '{task}'. Expected 'block' or 'denoise'.")

    payload["first_offset_ppl"] = _first_offset_perplexity(
        model,
        corpus,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        n_batches=n_batches,
        task=task,
    )
    return payload


def run_structured_language_model_experiment(
    config: KETStructuredLanguageModelConfig | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    structured_config = config or KETStructuredLanguageModelConfig.historical_smoke()
    task = structured_config.task
    if task not in {"block", "denoise"}:
        raise ValueError(f"Unsupported structured-LM task '{task}'. Expected 'block' or 'denoise'.")

    set_seed(structured_config.seed)
    if device is None:
        device = pick_device("cpu")

    corpus = load_word_language_modeling_corpus(structured_config.corpus_name)
    language_diagram = build_structured_language_model_diagram()

    if task == "block":
        model = FunctorFlowKETBlockPredictor(
            corpus.vocab_size,
            lm_config=structured_config.lm_config,
            block_size=structured_config.block_size,
        )
        history = train_block_predictor(
            model,
            corpus,
            seq_len=structured_config.seq_len,
            batch_size=structured_config.batch_size,
            steps=structured_config.steps,
            lr=structured_config.lr,
            device=device,
        )
    else:
        model = FunctorFlowKETDenoiser(
            corpus.vocab_size,
            lm_config=structured_config.lm_config,
            block_size=structured_config.block_size,
            num_denoise_steps=structured_config.num_denoise_steps,
        )
        history = train_denoiser(
            model,
            corpus,
            seq_len=structured_config.seq_len,
            batch_size=structured_config.batch_size,
            steps=structured_config.steps,
            lr=structured_config.lr,
            device=device,
        )

    evaluation = evaluate_structured_language_model(
        model,
        corpus,
        task=task,
        seq_len=structured_config.seq_len,
        batch_size=structured_config.batch_size,
        device=device,
        n_batches=structured_config.eval_batches,
    )
    return {
        "config": structured_config,
        "corpus": corpus.name,
        "task": task,
        "language_diagram": language_diagram,
        "task_diagram": model.diagram,
        "history": history,
        "eval": evaluation,
    }


def compare_structured_language_models(
    config: StructuredLMComparisonConfig | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    comparison_config = config or StructuredLMComparisonConfig.historical_smoke()
    set_seed(comparison_config.seed)
    if device is None:
        device = pick_device("cpu")

    corpus = load_word_language_modeling_corpus(comparison_config.corpus_name)
    lm_config = comparison_config.lm_config
    vocab_size = corpus.vocab_size

    model_builders = {
        "TF-Block-4": lambda: StructuredTransformerBlockPredictor(
            vocab_size,
            lm_config=lm_config,
            block_size=comparison_config.block_size,
        ),
        "KET-Block-4": lambda: FunctorFlowKETBlockPredictor(
            vocab_size,
            lm_config=lm_config,
            block_size=comparison_config.block_size,
        ),
        "TF-Denoise-4": lambda: StructuredTransformerDenoiser(
            vocab_size,
            lm_config=lm_config,
            block_size=comparison_config.block_size,
            num_denoise_steps=comparison_config.num_denoise_steps,
        ),
        "KET-Denoise-4": lambda: FunctorFlowKETDenoiser(
            vocab_size,
            lm_config=lm_config,
            block_size=comparison_config.block_size,
            num_denoise_steps=comparison_config.num_denoise_steps,
        ),
    }
    task_by_model = {
        "TF-Block-4": "block",
        "KET-Block-4": "block",
        "TF-Denoise-4": "denoise",
        "KET-Denoise-4": "denoise",
    }

    results: dict[str, object] = {
        "config": comparison_config,
        "corpus": corpus.name,
        "models": {},
    }
    for model_name, build_model in model_builders.items():
        model = build_model()
        task = task_by_model[model_name]
        if task == "block":
            history = train_block_predictor(
                model,
                corpus,
                seq_len=comparison_config.seq_len,
                batch_size=comparison_config.batch_size,
                steps=comparison_config.steps,
                lr=comparison_config.lr,
                device=device,
            )
        else:
            history = train_denoiser(
                model,
                corpus,
                seq_len=comparison_config.seq_len,
                batch_size=comparison_config.batch_size,
                steps=comparison_config.steps,
                lr=comparison_config.lr,
                device=device,
            )

        evaluation = evaluate_structured_language_model(
            model,
            corpus,
            task=task,
            seq_len=comparison_config.seq_len,
            batch_size=comparison_config.batch_size,
            device=device,
            n_batches=comparison_config.eval_batches,
        )
        results["models"][model_name] = {
            "task": task,
            "history": history,
            "eval": evaluation,
        }
    return results


def main() -> None:
    result = run_structured_language_model_experiment()
    print(f"FunctorFlow structured LM on {result['corpus']} ({result['task']})")
    print(result["language_diagram"].summary())
    print(result["task_diagram"].summary())
    print(
        f"block_ppl={result['eval']['block_ppl']:.2f} "
        f"first_offset_ppl={result['eval']['first_offset_ppl']:.2f}"
    )
    if "reconstruction_accuracy" in result["eval"]:
        print(f"reconstruction_accuracy={result['eval']['reconstruction_accuracy']:.3f}")


if __name__ == "__main__":
    main()

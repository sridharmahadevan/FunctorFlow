from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import Diagram
from .ket_lm import (
    KETLanguageModelConfig,
    FunctorFlowKETLanguageModel,
    WordLanguageModelingCorpus,
    load_word_language_modeling_corpus,
    pick_device,
)


PAD_ID = -100


@dataclass(frozen=True)
class KETBlockDualityConfig:
    corpus_name: str = "ptb"
    seq_len: int = 64
    batch_size: int = 8
    steps: int = 12
    lr: float = 2e-3
    block_size: int = 4
    num_denoise_steps: int = 8
    seed: int = 0
    lm_config: KETLanguageModelConfig = field(
        default_factory=KETLanguageModelConfig.historical_ptb_smoke
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_token_blocks(data_ids: torch.Tensor, *, seq_len: int, batch_size: int) -> torch.Tensor:
    n_tokens = int(data_ids.numel())
    if n_tokens <= seq_len:
        raise ValueError(f"Not enough tokens ({n_tokens}) for seq_len={seq_len}")
    max_start = n_tokens - seq_len - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    return torch.stack([data_ids[start : start + seq_len] for start in starts], dim=0)


def make_block_targets(tokens: torch.Tensor, block_size: int, pad_id: int = PAD_ID):
    batch_size, seq_len = tokens.shape
    targets = torch.full(
        (batch_size, seq_len, block_size),
        pad_id,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    valid_mask = torch.zeros((batch_size, seq_len, block_size), dtype=torch.bool, device=tokens.device)
    for offset in range(block_size):
        dst_len = seq_len - (offset + 1)
        if dst_len > 0:
            targets[:, :dst_len, offset] = tokens[:, offset + 1 :]
            valid_mask[:, :dst_len, offset] = True
    return targets, valid_mask


def block_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, *, pad_id: int = PAD_ID) -> torch.Tensor:
    batch_size, seq_len, block_size, vocab_size = logits.shape
    return F.cross_entropy(
        logits.reshape(batch_size * seq_len * block_size, vocab_size),
        targets.reshape(batch_size * seq_len * block_size),
        ignore_index=pad_id,
    )


def offset_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[int, float]:
    prediction = logits.argmax(dim=-1)
    result: dict[int, float] = {}
    for offset in range(logits.size(2)):
        offset_mask = valid_mask[:, :, offset]
        correct = ((prediction[:, :, offset] == targets[:, :, offset]) & offset_mask).sum().item()
        total = offset_mask.sum().item()
        result[offset + 1] = correct / max(total, 1)
    return result


def corrupt_block_targets(
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    vocab_size: int,
    noise_level: torch.Tensor,
    num_denoise_steps: int,
    mask_token_id: int | None = None,
) -> torch.Tensor:
    batch_size, seq_len, block_size = targets.shape
    del seq_len, block_size
    probability = 0.05 + 0.45 * (noise_level.float() - 1) / max(num_denoise_steps - 1, 1)
    probability = probability.view(batch_size, 1, 1)
    rand = torch.rand_like(targets.float())
    corrupt_mask = (rand < probability) & valid_mask
    noisy = targets.clone()
    if mask_token_id is not None:
        noisy[corrupt_mask] = mask_token_id
    else:
        noisy[corrupt_mask] = torch.randint(
            0,
            vocab_size,
            (int(corrupt_mask.sum().item()),),
            device=targets.device,
        )
    noisy[~valid_mask] = 0
    return noisy


class OffsetBlockLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, block_size: int):
        super().__init__()
        self.block_size = int(block_size)
        self.offset_emb = nn.Parameter(torch.randn(block_size, d_model) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, vocab_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, _, width = hidden_states.shape
        offset_states = hidden_states.unsqueeze(2) + self.offset_emb.view(1, 1, self.block_size, width)
        return self.mlp(offset_states)


class DenoisingBlockHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, block_size: int, num_denoise_steps: int):
        super().__init__()
        self.block_size = int(block_size)
        self.noisy_token_embedding = nn.Embedding(vocab_size, d_model)
        self.time_embedding = nn.Embedding(num_denoise_steps + 1, d_model)
        self.offset_emb = nn.Parameter(torch.randn(block_size, d_model) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, vocab_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        noisy_block: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, width = hidden_states.shape
        del width
        block_size = self.block_size
        context = hidden_states.unsqueeze(2).expand(batch_size, seq_len, block_size, -1)
        noisy_emb = self.noisy_token_embedding(noisy_block)
        time_emb = self.time_embedding(timestep).view(batch_size, 1, 1, -1).expand_as(context)
        offset_emb = self.offset_emb.view(1, 1, block_size, -1).expand_as(context)
        return self.mlp(torch.cat([context + offset_emb, noisy_emb, time_emb], dim=-1))


def build_left_kan_block_diagram(name: str = "FunctorFlowLeftKanBlock") -> Diagram:
    diagram = Diagram(name)
    diagram.object("HiddenStates", kind="hidden_state")
    diagram.object("CausalRelation", kind="causal_relation")
    diagram.object("ContextualizedStates", kind="contextualized_hidden_state")
    diagram.object("FutureBlockLogits", kind="block_logits")
    diagram.left_kan(
        name="aggregate_future_context",
        source="HiddenStates",
        along="CausalRelation",
        target="ContextualizedStates",
        reducer="ket_attention",
        description="Left-Kan aggregation for context-conditioned block prediction.",
        metadata={"demo": "left_kan_block"},
    )
    diagram.morphism(
        "decode_future_block",
        "aggregate_future_context",
        "FutureBlockLogits",
        description="Decode future block offsets from contextualized states.",
    )
    diagram.expose_port("input", "HiddenStates", direction="input", port_type="hidden_state")
    diagram.expose_port("relation", "CausalRelation", direction="input", port_type="causal_relation")
    diagram.expose_port(
        "context",
        "aggregate_future_context",
        direction="output",
        port_type="contextualized_hidden_state",
    )
    diagram.expose_port("output", "decode_future_block", direction="output", port_type="block_logits")
    return diagram


def build_right_kan_denoise_diagram(name: str = "FunctorFlowRightKanDenoise") -> Diagram:
    diagram = Diagram(name)
    diagram.object("HiddenStates", kind="hidden_state")
    diagram.object("NoisyBlock", kind="noisy_block")
    diagram.object("DenoiseCondition", kind="denoise_condition")
    diagram.object("CompletedBlock", kind="completed_block_state")
    diagram.object("CompletedBlockLogits", kind="block_logits")
    diagram.right_kan(
        name="complete_block",
        source="NoisyBlock",
        along="DenoiseCondition",
        target="CompletedBlock",
        reducer="first_non_null",
        description="Right-Kan completion of a corrupted future block.",
        metadata={"demo": "right_kan_denoise"},
    )
    diagram.morphism(
        "decode_completed_block",
        "complete_block",
        "CompletedBlockLogits",
        description="Decode denoised block logits from the repaired block state.",
    )
    diagram.expose_port("hidden", "HiddenStates", direction="input", port_type="hidden_state")
    diagram.expose_port("noisy_block", "NoisyBlock", direction="input", port_type="noisy_block")
    diagram.expose_port(
        "condition",
        "DenoiseCondition",
        direction="input",
        port_type="denoise_condition",
    )
    diagram.expose_port("completed", "complete_block", direction="output", port_type="completed_block_state")
    diagram.expose_port(
        "output",
        "decode_completed_block",
        direction="output",
        port_type="block_logits",
    )
    return diagram


class FunctorFlowKETBlockPredictor(nn.Module):
    def __init__(self, vocab_size: int, *, lm_config: KETLanguageModelConfig, block_size: int):
        super().__init__()
        self.backbone = FunctorFlowKETLanguageModel(vocab_size, config=lm_config)
        self.head = OffsetBlockLMHead(lm_config.d_model, vocab_size, block_size)
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.diagram = build_left_kan_block_diagram()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(tokens, return_hidden=True)
        return self.head(hidden_states)


class FunctorFlowKETDenoiser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        lm_config: KETLanguageModelConfig,
        block_size: int,
        num_denoise_steps: int,
    ):
        super().__init__()
        self.backbone = FunctorFlowKETLanguageModel(vocab_size, config=lm_config)
        self.head = DenoisingBlockHead(
            lm_config.d_model,
            vocab_size,
            block_size,
            num_denoise_steps,
        )
        self.block_size = int(block_size)
        self.vocab_size = int(vocab_size)
        self.num_denoise_steps = int(num_denoise_steps)
        self.diagram = build_right_kan_denoise_diagram()

    def forward(self, tokens: torch.Tensor, noisy_block: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone(tokens, return_hidden=True)
        return self.head(hidden_states, noisy_block, timestep)


@torch.no_grad()
def evaluate_block_predictor(
    model: FunctorFlowKETBlockPredictor,
    corpus: WordLanguageModelingCorpus,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    n_batches: int = 4,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    acc_sums: dict[int, float] | None = None
    for _ in range(n_batches):
        tokens = make_token_blocks(corpus.valid_ids, seq_len=seq_len, batch_size=batch_size).to(device)
        targets, valid_mask = make_block_targets(tokens, model.block_size, pad_id=PAD_ID)
        logits = model(tokens)
        loss = block_cross_entropy_loss(logits, targets, pad_id=PAD_ID)
        total_loss += float(loss.item())
        total_batches += 1
        acc = offset_accuracy(logits, targets, valid_mask)
        if acc_sums is None:
            acc_sums = {offset: 0.0 for offset in acc}
        for offset, value in acc.items():
            acc_sums[offset] += value
    mean_loss = total_loss / max(total_batches, 1)
    metrics = {offset: value / max(total_batches, 1) for offset, value in (acc_sums or {}).items()}
    return {
        "loss": mean_loss,
        "block_ppl": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
        "offset_accuracy": metrics,
    }


@torch.no_grad()
def evaluate_denoiser(
    model: FunctorFlowKETDenoiser,
    corpus: WordLanguageModelingCorpus,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    n_batches: int = 4,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    acc_sums: dict[int, float] | None = None
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
            targets,
            valid_mask,
            vocab_size=model.vocab_size,
            noise_level=timestep,
            num_denoise_steps=model.num_denoise_steps,
            mask_token_id=model.vocab_size - 1,
        )
        logits = model(tokens, noisy_block, timestep)
        loss = block_cross_entropy_loss(logits, targets, pad_id=PAD_ID)
        total_loss += float(loss.item())
        total_batches += 1
        acc = offset_accuracy(logits, targets, valid_mask)
        if acc_sums is None:
            acc_sums = {offset: 0.0 for offset in acc}
        for offset, value in acc.items():
            acc_sums[offset] += value
    mean_loss = total_loss / max(total_batches, 1)
    metrics = {offset: value / max(total_batches, 1) for offset, value in (acc_sums or {}).items()}
    return {
        "loss": mean_loss,
        "block_ppl": math.exp(mean_loss) if mean_loss < 20 else float("inf"),
        "offset_accuracy": metrics,
    }


def train_block_predictor(
    model: FunctorFlowKETBlockPredictor,
    corpus: WordLanguageModelingCorpus,
    *,
    seq_len: int,
    batch_size: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> dict[str, list[float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = {"train_loss": []}
    for _ in range(steps):
        model.train()
        tokens = make_token_blocks(corpus.train_ids, seq_len=seq_len, batch_size=batch_size).to(device)
        targets, _ = make_block_targets(tokens, model.block_size, pad_id=PAD_ID)
        logits = model(tokens)
        loss = block_cross_entropy_loss(logits, targets, pad_id=PAD_ID)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history["train_loss"].append(float(loss.detach().item()))
    return history


def train_denoiser(
    model: FunctorFlowKETDenoiser,
    corpus: WordLanguageModelingCorpus,
    *,
    seq_len: int,
    batch_size: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> dict[str, list[float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = {"train_loss": []}
    for _ in range(steps):
        model.train()
        tokens = make_token_blocks(corpus.train_ids, seq_len=seq_len, batch_size=batch_size).to(device)
        targets, valid_mask = make_block_targets(tokens, model.block_size, pad_id=PAD_ID)
        timestep = torch.randint(
            1,
            model.num_denoise_steps + 1,
            (tokens.size(0),),
            dtype=torch.long,
            device=device,
        )
        noisy_block = corrupt_block_targets(
            targets,
            valid_mask,
            vocab_size=model.vocab_size,
            noise_level=timestep,
            num_denoise_steps=model.num_denoise_steps,
            mask_token_id=model.vocab_size - 1,
        )
        logits = model(tokens, noisy_block, timestep)
        loss = block_cross_entropy_loss(logits, targets, pad_id=PAD_ID)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history["train_loss"].append(float(loss.detach().item()))
    return history


def run_ket_block_duality_demo(
    config: KETBlockDualityConfig | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    duality_config = config or KETBlockDualityConfig()
    set_seed(duality_config.seed)
    if device is None:
        device = pick_device("cpu")

    corpus = load_word_language_modeling_corpus(duality_config.corpus_name)
    left_model = FunctorFlowKETBlockPredictor(
        corpus.vocab_size,
        lm_config=duality_config.lm_config,
        block_size=duality_config.block_size,
    )
    right_model = FunctorFlowKETDenoiser(
        corpus.vocab_size,
        lm_config=duality_config.lm_config,
        block_size=duality_config.block_size,
        num_denoise_steps=duality_config.num_denoise_steps,
    )

    left_history = train_block_predictor(
        left_model,
        corpus,
        seq_len=duality_config.seq_len,
        batch_size=duality_config.batch_size,
        steps=duality_config.steps,
        lr=duality_config.lr,
        device=device,
    )
    right_history = train_denoiser(
        right_model,
        corpus,
        seq_len=duality_config.seq_len,
        batch_size=duality_config.batch_size,
        steps=duality_config.steps,
        lr=duality_config.lr,
        device=device,
    )

    left_eval = evaluate_block_predictor(
        left_model,
        corpus,
        seq_len=duality_config.seq_len,
        batch_size=duality_config.batch_size,
        device=device,
    )
    right_eval = evaluate_denoiser(
        right_model,
        corpus,
        seq_len=duality_config.seq_len,
        batch_size=duality_config.batch_size,
        device=device,
    )

    return {
        "config": duality_config,
        "corpus": corpus.name,
        "left_kan": {
            "diagram": left_model.diagram,
            "history": left_history,
            "eval": left_eval,
        },
        "right_kan": {
            "diagram": right_model.diagram,
            "history": right_history,
            "eval": right_eval,
        },
    }


def main() -> None:
    result = run_ket_block_duality_demo()
    print(f"FunctorFlow KET block duality demo on {result['corpus']}")
    print(result["left_kan"]["diagram"].summary())
    print(
        "left_kan:",
        f"block_ppl={result['left_kan']['eval']['block_ppl']:.2f}",
        f"offset@1={result['left_kan']['eval']['offset_accuracy'].get(1, 0.0):.3f}",
    )
    print(result["right_kan"]["diagram"].summary())
    print(
        "right_kan:",
        f"block_ppl={result['right_kan']['eval']['block_ppl']:.2f}",
        f"offset@1={result['right_kan']['eval']['offset_accuracy'].get(1, 0.0):.3f}",
    )


if __name__ == "__main__":
    main()

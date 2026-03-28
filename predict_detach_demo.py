from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ket_lm import FunctorFlowKETReducer, causal_relation_mask


@dataclass(frozen=True)
class PredictDetachDemoConfig:
    vocab_size: int = 50
    sequence_length: int = 16
    offset_k: int = 5
    train_sequences: int = 3000
    test_sequences: int = 500
    embed_dim: int = 64
    batch_size: int = 64
    steps: int = 100
    lr: float = 2e-3
    temperature: float = 1.0
    seed: int = 0


def make_synthetic_copy_data(
    n_seq: int = 2000,
    *,
    length: int = 20,
    vocab_size: int = 50,
    offset_k: int = 5,
) -> torch.Tensor:
    sequences = torch.randint(0, vocab_size, (n_seq, length), dtype=torch.long)
    for t in range(length - offset_k - 1):
        sequences[:, t + 1] = sequences[:, t + offset_k]
    return sequences


def noncausal_relation_mask(length: int, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.ones(length, length, dtype=torch.bool, device=device)


class PredictDetachToyModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        embed_dim: int = 64,
        max_positions: int = 64,
        offset_k: int = 5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)
        self.offset_k = int(offset_k)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_positions, embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)
        self.reducer = FunctorFlowKETReducer(
            embed_dim,
            variant="course_attention_kan",
            regime="vanilla",
            temperature=temperature,
        )

    def _position_emb(self, length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(length, device=device)
        return self.position_embedding(positions)[None, :, :]

    def forward(self, token_ids: torch.Tensor, *, regime: str) -> torch.Tensor:
        batch_size, length = token_ids.shape
        del batch_size
        position_emb = self._position_emb(length, token_ids.device)
        hidden_states = self.token_embedding(token_ids) + position_emb

        if regime == "causal":
            relation = causal_relation_mask(length, device=token_ids.device)
            basis_states = hidden_states
        elif regime == "leaky_noncausal":
            relation = noncausal_relation_mask(length, device=token_ids.device)
            basis_states = hidden_states
        elif regime == "predict_detach":
            relation = noncausal_relation_mask(length, device=token_ids.device)
            with torch.no_grad():
                logits0 = self.output_head(hidden_states)
                probabilities = F.softmax(logits0, dim=-1)
                predicted = (probabilities @ self.token_embedding.weight).detach()
            basis_states = predicted + position_emb
        else:
            raise ValueError(
                f"Unsupported predict-detach regime '{regime}'. "
                "Expected 'causal', 'leaky_noncausal', or 'predict_detach'."
            )

        pooled = self.reducer(
            hidden_states,
            relation,
            metadata={"basis_states": basis_states},
        )

        leak = torch.zeros_like(pooled)
        if regime in {"leaky_noncausal", "predict_detach"} and length > self.offset_k:
            leak[:, :-self.offset_k, :] = basis_states[:, self.offset_k :, :]

        outputs = pooled + leak
        return self.output_head(outputs)


def _sample_batch(sequences: torch.Tensor, batch_size: int) -> torch.Tensor:
    indices = torch.randint(0, sequences.size(0), (batch_size,))
    return sequences[indices]


def run_predict_detach_regime_demo(
    config: PredictDetachDemoConfig | None = None,
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    demo_config = config or PredictDetachDemoConfig()
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(demo_config.seed)
    train = make_synthetic_copy_data(
        demo_config.train_sequences,
        length=demo_config.sequence_length,
        vocab_size=demo_config.vocab_size,
        offset_k=demo_config.offset_k,
    )
    test = make_synthetic_copy_data(
        demo_config.test_sequences,
        length=demo_config.sequence_length,
        vocab_size=demo_config.vocab_size,
        offset_k=demo_config.offset_k,
    )

    regimes = ("causal", "leaky_noncausal", "predict_detach")
    results: dict[str, object] = {
        "config": demo_config,
        "regimes": {},
    }

    for regime in regimes:
        model = PredictDetachToyModel(
            demo_config.vocab_size,
            embed_dim=demo_config.embed_dim,
            max_positions=demo_config.sequence_length,
            offset_k=demo_config.offset_k,
            temperature=demo_config.temperature,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=demo_config.lr)
        losses: list[float] = []
        for _ in range(demo_config.steps):
            batch = _sample_batch(train, demo_config.batch_size).to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs, regime=regime)
            loss = F.cross_entropy(logits.reshape(-1, demo_config.vocab_size), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))

        with torch.no_grad():
            eval_batch = test[: demo_config.batch_size].to(device)
            inputs = eval_batch[:, :-1]
            targets = eval_batch[:, 1:]
            logits = model(inputs, regime=regime)
            eval_loss = F.cross_entropy(logits.reshape(-1, demo_config.vocab_size), targets.reshape(-1))

        results["regimes"][regime] = {
            "losses": losses,
            "final_train_loss": losses[-1],
            "eval_loss": float(eval_loss.detach().item()),
        }

    return results


def main() -> None:
    result = run_predict_detach_regime_demo()
    print("FunctorFlow predict_detach regime demo")
    for regime, payload in result["regimes"].items():
        print(
            f"{regime}: "
            f"final_train_loss={payload['final_train_loss']:.4f} "
            f"eval_loss={payload['eval_loss']:.4f}"
        )


if __name__ == "__main__":
    main()

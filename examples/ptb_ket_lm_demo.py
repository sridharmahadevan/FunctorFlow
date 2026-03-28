from __future__ import annotations

from FunctorFlow.ket_lm import (
    FunctorFlowKETLanguageModel,
    KETLanguageModelConfig,
    estimate_perplexity,
    load_word_language_modeling_corpus,
    pick_device,
    train_language_model,
)


def main() -> None:
    device = pick_device("cpu")
    corpus = load_word_language_modeling_corpus("ptb")
    config = KETLanguageModelConfig.historical_ptb_smoke()
    model = FunctorFlowKETLanguageModel(
        corpus.vocab_size,
        config=config,
    )

    history = train_language_model(
        model,
        corpus,
        steps=2,
        block_size=32,
        batch_size=4,
        lr=2e-3,
        device=device,
    )
    valid_ppl = estimate_perplexity(
        model,
        corpus.valid_ids,
        block_size=32,
        batch_size=4,
        device=device,
    )

    print("FunctorFlow PTB KET demo")
    print(f"vocab_size={corpus.vocab_size}")
    print(f"model_config={config}")
    print(f"recorded_train_losses={history['train_loss']}")
    print(f"valid_perplexity={valid_ppl:.2f}")


if __name__ == "__main__":
    main()

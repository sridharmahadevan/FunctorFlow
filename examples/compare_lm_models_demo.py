from __future__ import annotations

import sys

from FunctorFlow.ket_lm import pick_device
from FunctorFlow.lm_compare import LMComparisonConfig, compare_language_models


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    corpus_name = args[0] if args else "ptb"
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
    comparison = compare_language_models(
        corpus_name,
        comparison_config=config,
        device=device,
    )

    print(f"FunctorFlow {comparison['corpus']} model comparison")
    print(f"model_profile={comparison['model_profile']}")
    print(f"train_tokens={comparison['train_tokens']}")
    print(f"valid_tokens={comparison['valid_tokens']}")
    for model_name, model_result in comparison["models"].items():
        print(
            f"{model_name}: "
            f"train_losses={model_result['history']['train_loss']} "
            f"valid_perplexity={model_result['valid_ppl']:.2f}"
        )


if __name__ == "__main__":
    main()

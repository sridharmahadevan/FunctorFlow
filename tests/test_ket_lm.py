from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from FunctorFlow.ket_lm import (
        DEFAULT_DATA_ROOT,
        FunctorFlowKETHead,
        FunctorFlowKETLanguageModel,
        FunctorFlowKETReducer,
        KETHeadConfig,
        KETLanguageModelConfig,
        WordLanguageModelingCorpus,
        causal_relation_mask,
        estimate_perplexity,
        masked_kan_attention,
        load_word_language_modeling_corpus,
        train_language_model,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed in this interpreter")
class FunctorFlowKETLanguageModelTests(unittest.TestCase):
    def test_package_local_data_root_exists(self) -> None:
        self.assertTrue(DEFAULT_DATA_ROOT.exists())
        self.assertTrue((DEFAULT_DATA_ROOT / "ptb").exists())
        self.assertTrue((DEFAULT_DATA_ROOT / "wikitext-2").exists())

    def test_load_ptb_corpus_uses_local_repo_data(self) -> None:
        corpus = load_word_language_modeling_corpus("ptb")
        self.assertEqual(corpus.name, "ptb")
        self.assertGreater(corpus.vocab_size, 1000)
        self.assertGreater(int(corpus.train_ids.numel()), 1000)
        self.assertGreater(int(corpus.valid_ids.numel()), 100)

    def test_ket_head_runs_through_functorflow_diagram(self) -> None:
        head = FunctorFlowKETHead(
            16,
            config=KETHeadConfig(
                variant="course_attention_kan",
                regime="predict_detach",
                window_k=4,
                edge_hidden_dim=16,
            ),
        )
        hidden_states = torch.randn(2, 8, 16)
        relation = causal_relation_mask(8, window_k=4)
        outputs = head(hidden_states, relation=relation)
        self.assertEqual(tuple(outputs.shape), (2, 8, 16))
        self.assertIn("ket_attention", head.compiled.lowered_reducers)

    def test_historical_presets_are_exposed(self) -> None:
        config = KETLanguageModelConfig.historical_ptb_reference()
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.head.variant, "course_attention_kan")
        self.assertEqual(config.head.regime, "predict_detach")
        wiki103 = KETLanguageModelConfig.historical_wiki103_smoke()
        self.assertEqual(wiki103.max_positions, 512)
        self.assertEqual(wiki103.head.window_k, 64)

    def test_load_wiki103_corpus_from_custom_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "wikitext-103"
            dataset_dir.mkdir(parents=True)
            sample = "alpha beta alpha\n"
            (dataset_dir / "wiki.train.tokens").write_text(sample, encoding="utf-8")
            (dataset_dir / "wiki.valid.tokens").write_text(sample, encoding="utf-8")
            (dataset_dir / "wiki.test.tokens").write_text(sample, encoding="utf-8")

            corpus = load_word_language_modeling_corpus("wiki103", root=tmpdir)
            self.assertEqual(corpus.name, "wiki-103")
            self.assertGreater(corpus.vocab_size, 0)
            self.assertGreater(int(corpus.train_ids.numel()), 0)

    def test_course_attention_variant_matches_masked_attention(self) -> None:
        reducer = FunctorFlowKETReducer(
            4,
            variant="course_attention_kan",
            regime="vanilla",
            temperature=1.0,
        )
        with torch.no_grad():
            eye = torch.eye(4)
            reducer.query_proj.weight.copy_(eye)
            reducer.key_proj.weight.copy_(eye)
            reducer.value_proj.weight.copy_(eye)
            reducer.output_proj.weight.copy_(eye)

        source = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0], [0.25, 0.5, 1.0, 0.0]]]
        )
        relation = causal_relation_mask(3)
        outputs = reducer(source, relation)
        expected, weights = masked_kan_attention(
            source,
            source,
            source,
            relation.unsqueeze(0),
            temperature=4 ** 0.5,
        )
        self.assertEqual(tuple(weights.shape), (1, 3, 3))
        self.assertTrue(torch.allclose(outputs, expected, atol=1e-6, rtol=1e-6))

    def test_language_model_forward_shape(self) -> None:
        model = FunctorFlowKETLanguageModel(
            vocab_size=50,
            config=KETLanguageModelConfig(
                d_model=16,
                n_layers=1,
                max_positions=64,
                head=KETHeadConfig(edge_hidden_dim=16),
            ),
        )
        token_ids = torch.randint(0, 50, (4, 12))
        logits = model(token_ids)
        self.assertEqual(tuple(logits.shape), (4, 12, 50))

    def test_train_and_eval_smoke(self) -> None:
        torch.manual_seed(0)
        corpus = load_word_language_modeling_corpus("ptb")
        tiny_corpus = WordLanguageModelingCorpus(
            name=corpus.name,
            vocab=corpus.vocab,
            train_ids=corpus.train_ids[:512],
            valid_ids=corpus.valid_ids[:256],
            test_ids=corpus.test_ids[:256],
        )
        model = FunctorFlowKETLanguageModel(
            tiny_corpus.vocab_size,
            config=KETLanguageModelConfig(
                d_model=24,
                n_layers=1,
                window_k=16,
                max_positions=128,
                head=KETHeadConfig(window_k=16, edge_hidden_dim=24, variant="course_attention_kan"),
            ),
        )
        history = train_language_model(
            model,
            tiny_corpus,
            steps=1,
            block_size=16,
            batch_size=2,
            lr=1e-3,
            device=torch.device("cpu"),
        )
        valid_ppl = estimate_perplexity(
            model,
            tiny_corpus.valid_ids,
            block_size=16,
            batch_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(len(history["train_loss"]), 1)
        self.assertEqual(len(history["valid_ppl"]), 1)
        self.assertGreater(valid_ppl, 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import json

from FunctorFlow.democritus import (
    DemocritusPipelineConfig,
    DemocritusTriple,
    _build_local_causal_graph,
    _build_local_model_spec,
    _extract_local_model_graph,
    _render_sample_local_models,
    _select_local_model_focuses_from_graph,
    _select_local_model_focuses,
    _select_manifold_label_indices,
    _shorten_plot_label,
    build_democritus_pipeline_diagram,
    run_democritus_pipelines_from_pdf_directory,
    run_democritus_pipeline_from_text,
)


class ReplayDemocritusLLM:
    def ask(self, prompt: str) -> str:
        return self.ask_batch([prompt])[0]

    def ask_batch(self, prompts: list[str]) -> list[str]:
        return [self._dispatch(prompt) for prompt in prompts]

    def _dispatch(self, prompt: str) -> str:
        lowered = prompt.lower()
        if "topics:" in lowered and "excerpt:" in lowered:
            return "\n".join(
                [
                    "Monsoon variability",
                    "Agricultural yields",
                    "Trade networks",
                ]
            )
        if lowered.startswith("list ") and "subtopics related to" in lowered:
            if "monsoon variability" in lowered:
                return "\n".join(
                    [
                        "Rainfall shocks",
                        "River flooding",
                        "Crop failures",
                        "Food prices",
                    ]
                )
            if "agricultural yields" in lowered:
                return "\n".join(
                    [
                        "Soil moisture",
                        "Seed quality",
                        "Irrigation reliability",
                    ]
                )
            return "\n".join(
                [
                    "Market access",
                    "Supply disruptions",
                    "Regional exchange",
                ]
            )
        if "distinct causal questions about" in lowered:
            if "monsoon variability" in lowered:
                return "\n".join(
                    [
                        "How does monsoon variability affect crop yields?",
                        "How does monsoon variability influence food prices?",
                    ]
                )
            if "agricultural yields" in lowered:
                return "\n".join(
                    [
                        "How does soil moisture affect agricultural yields?",
                        "How does irrigation reliability increase crop stability?",
                    ]
                )
            return "\n".join(
                [
                    "How do supply disruptions reduce market access?",
                    "How do trade networks influence regional exchange?",
                ]
            )
        if "causal knowledge generator" in lowered:
            if "monsoon variability affect crop yields" in lowered:
                return "\n".join(
                    [
                        "Monsoon variability reduces crop yields.",
                        "Rainfall shocks influence food prices.",
                    ]
                )
            if "monsoon variability influence food prices" in lowered:
                return "\n".join(
                    [
                        "Monsoon variability causes food price volatility.",
                        "Crop failures increase market scarcity.",
                    ]
                )
            if "soil moisture affect agricultural yields" in lowered:
                return "\n".join(
                    [
                        "Soil moisture increases agricultural yields.",
                        "Irrigation reliability affects crop stability.",
                    ]
                )
            if "irrigation reliability increase crop stability" in lowered:
                return "\n".join(
                    [
                        "Irrigation reliability increases crop stability.",
                        "Crop stability influences regional exchange.",
                    ]
                )
            if "supply disruptions reduce market access" in lowered:
                return "\n".join(
                    [
                        "Supply disruptions reduce market access.",
                        "Market access affects regional exchange.",
                    ]
                )
            return "\n".join(
                [
                    "Trade networks influence regional exchange.",
                    "Regional exchange affects food prices.",
                ]
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")


class DemocritusTests(unittest.TestCase):
    def test_run_democritus_pipelines_from_pdf_directory_creates_one_run_per_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_dir = Path(tmpdir) / "pdfs"
            outdir = Path(tmpdir) / "runs"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            first_pdf = pdf_dir / "B doc.pdf"
            second_pdf = pdf_dir / "A doc.pdf"
            first_pdf.write_bytes(b"%PDF-1.4\n%stub\n")
            second_pdf.write_bytes(b"%PDF-1.4\n%stub\n")

            def _fake_run(pdf_path, *, llm=None, config=None, outdir=None):
                return type(
                    "ArtifactsStub",
                    (),
                    {
                        "config": config,
                        "root_topics": (),
                        "topic_graph": (),
                        "causal_questions": (),
                        "causal_statements": (),
                        "relational_triples": (),
                        "manifold": type("ManifoldStub", (), {"entities": (), "db_obstruction": 0.0})(),
                        "output_files": {"manifold": str(Path(outdir) / "manifold.npz")},
                    },
                )()

            with patch("FunctorFlow.democritus.run_democritus_pipeline_from_pdf", side_effect=_fake_run) as mocked:
                results = run_democritus_pipelines_from_pdf_directory(
                    pdf_dir,
                    llm=ReplayDemocritusLLM(),
                    config=DemocritusPipelineConfig.smoke(domain_name="batch_demo"),
                    outdir=outdir,
                )

            self.assertEqual(len(results), 2)
            self.assertEqual(mocked.call_count, 2)
            called_outdirs = [Path(call.kwargs["outdir"]) for call in mocked.call_args_list]
            self.assertEqual(called_outdirs[0].name, "01_a_doc")
            self.assertEqual(called_outdirs[1].name, "02_b_doc")
            self.assertEqual(called_outdirs[0].parent, outdir)
            self.assertEqual(called_outdirs[1].parent, outdir)
            called_domains = [call.kwargs["config"].domain_name for call in mocked.call_args_list]
            self.assertEqual(called_domains, ["batch_demo_a_doc", "batch_demo_b_doc"])

    def test_topic_anchored_focus_selection_prefers_dense_topic_nodes(self) -> None:
        triples = [
            DemocritusTriple("Holocene Climate Variability", ("Holocene Climate Variability",), "q1", "Holocene climate variability influences migration.", "holocene climate variability", "influences", "migration patterns", "Holocene Climate Variability"),
            DemocritusTriple("Holocene Climate Variability", ("Holocene Climate Variability",), "q2", "Holocene climate variability influences sea level.", "holocene climate variability", "influences", "sea-level change", "Holocene Climate Variability"),
            DemocritusTriple("Solar forcing", ("Holocene Climate Variability", "Solar forcing"), "q3", "Solar variability influences rainfall.", "solar variability", "influences", "rainfall patterns", "Holocene Climate Variability"),
        ]
        graph, topic_nodes = _build_local_causal_graph(triples)
        focuses = _select_local_model_focuses_from_graph(
            graph,
            topic_nodes=topic_nodes,
            max_models=2,
            min_focus_edges=2,
        )

        self.assertGreaterEqual(len(focuses), 1)
        self.assertEqual(focuses[0], "Holocene Climate Variability")

    def test_select_local_model_focuses_prefers_high_signal_distinct_entities(self) -> None:
        triples = [
            DemocritusTriple("weather", ("weather",), "q1", "rainfall shocks reduce crop yields.", "rainfall shocks", "reduces", "crop yields", "weather"),
            DemocritusTriple("weather", ("weather",), "q2", "rainfall shocks increase food prices.", "rainfall shocks", "increases", "food prices", "weather"),
            DemocritusTriple("weather", ("weather",), "q3", "crop yields influence food prices.", "crop yields", "influences", "food prices", "weather"),
            DemocritusTriple("trade", ("trade",), "q4", "trade networks influence regional exchange.", "trade networks", "influences", "regional exchange", "trade"),
            DemocritusTriple("trade", ("trade",), "q5", "supply disruptions reduce market access.", "supply disruptions", "reduces", "market access", "trade"),
            DemocritusTriple("trade", ("trade",), "q6", "market access affects regional exchange.", "market access", "affects", "regional exchange", "trade"),
        ]
        focuses = _select_local_model_focuses(triples, max_models=2, min_focus_edges=2)

        self.assertEqual(len(focuses), 2)
        self.assertIn("rainfall shocks", focuses)
        self.assertTrue(
            any(focus in focuses for focus in ("regional exchange", "market access", "trade networks"))
        )

    def test_build_local_model_spec_keeps_focus_center_and_compact_edge_set(self) -> None:
        triples = [
            DemocritusTriple("weather", ("weather",), "q1", "rainfall shocks reduce crop yields.", "rainfall shocks", "reduces", "crop yields", "weather"),
            DemocritusTriple("weather", ("weather",), "q2", "rainfall shocks increase food prices.", "rainfall shocks", "increases", "food prices", "weather"),
            DemocritusTriple("weather", ("weather",), "q3", "crop yields influence food prices.", "crop yields", "influences", "food prices", "weather"),
            DemocritusTriple("weather", ("weather",), "q4", "food prices affect household stress.", "food prices", "affects", "household stress", "weather"),
        ]
        model = _build_local_model_spec(
            triples,
            focus="rainfall shocks",
            radius=2,
            max_nodes=4,
            max_edges=3,
        )

        self.assertIsNotNone(model)
        assert model is not None
        self.assertEqual(model["focus"], "rainfall shocks")
        self.assertIn("rainfall shocks", model["nodes"])
        self.assertLessEqual(model["num_nodes"], 4)
        self.assertLessEqual(model["num_edges"], 3)
        self.assertTrue(any(edge["src"] == "rainfall shocks" for edge in model["edges"]))

    def test_build_local_model_spec_uses_radius_to_pull_in_second_hop_structure(self) -> None:
        triples = [
            DemocritusTriple("hydrology", ("hydrology",), "q1", "river discharge reduces drought severity.", "river discharge", "reduces", "drought severity", "hydrology"),
            DemocritusTriple("hydrology", ("hydrology",), "q2", "drought severity increases crop stress.", "drought severity", "increases", "crop stress", "hydrology"),
            DemocritusTriple("hydrology", ("hydrology",), "q3", "crop stress affects migration pressure.", "crop stress", "affects", "migration pressure", "hydrology"),
        ]
        model = _build_local_model_spec(
            triples,
            focus="river discharge",
            radius=2,
            max_nodes=5,
            max_edges=5,
        )

        self.assertIsNotNone(model)
        assert model is not None
        self.assertIn("crop stress", model["nodes"])
        self.assertTrue(
            any(edge["src"] == "drought severity" and edge["dst"] == "crop stress" for edge in model["edges"])
        )

    def test_extract_local_model_graph_preserves_anchor_and_causal_structure(self) -> None:
        triples = [
            DemocritusTriple("Climate impact", ("Climate impact",), "q1", "Climate change reduces river discharge.", "climate change", "reduces", "river discharge", "Climate impact"),
            DemocritusTriple("Climate impact", ("Climate impact",), "q2", "Reduced river discharge prolongs drought.", "reduced river discharge", "leads_to", "prolonged drought", "Climate impact"),
        ]
        graph, topic_nodes = _build_local_causal_graph(triples)
        local_graph = _extract_local_model_graph(
            graph,
            focus="Climate impact",
            radius=2,
            max_nodes=10,
        )

        self.assertIsNotNone(local_graph)
        assert local_graph is not None
        self.assertIn("Climate impact", local_graph.nodes())
        self.assertIn(("Climate impact", "climate change"), local_graph.edges())
        self.assertIn(("climate change", "river discharge"), local_graph.edges())

    def test_render_sample_local_models_removes_stale_pngs(self) -> None:
        triples = [
            DemocritusTriple("Climate impact", ("Climate impact",), "q1", "Climate change reduces river discharge.", "climate change", "reduces", "river discharge", "Climate impact"),
            DemocritusTriple("Climate impact", ("Climate impact",), "q2", "Reduced river discharge prolongs drought.", "reduced river discharge", "leads_to", "prolonged drought", "Climate impact"),
            DemocritusTriple("Climate impact", ("Climate impact",), "q3", "Climate change affects snowmelt.", "climate change", "affects", "snowmelt", "Climate impact"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "local_causal_models"
            models_dir.mkdir(parents=True, exist_ok=True)
            stale = models_dir / "local_causal_model_99_stale.png"
            stale.write_text("stale", encoding="utf-8")

            rendered = _render_sample_local_models(
                triples,
                outdir=Path(tmpdir),
                max_models=2,
                radius=2,
                min_focus_edges=2,
                max_nodes=10,
                max_edges=14,
            )

            self.assertFalse(stale.exists())
            self.assertGreaterEqual(len(rendered), 1)

    def test_shorten_plot_label_collapses_whitespace_and_truncates(self) -> None:
        shortened = _shorten_plot_label(
            "  a very long   label with extra whitespace that should become shorter for plotting   ",
            max_chars=30,
        )
        self.assertEqual(shortened, "a very long label with extr...")

    def test_select_manifold_label_indices_prefers_separated_extremes(self) -> None:
        points = np.array(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.1, 0.1],
            ],
            dtype=np.float32,
        )
        selected = _select_manifold_label_indices(points, max_labels=4, grid_shape=(3, 3))

        self.assertEqual(len(selected), 4)
        self.assertEqual(set(selected), {0, 1, 2, 3})

    def test_build_democritus_pipeline_diagram_exposes_manifold_surface(self) -> None:
        diagram = build_democritus_pipeline_diagram()
        self.assertIn("topic_sections__aggregate_topics", diagram.operations)
        self.assertIn("gt_refine__refine_claims", diagram.operations)
        self.assertIn("glue__glue_claims", diagram.operations)
        self.assertIn("consistency__gluing_obstruction", diagram.losses)
        self.assertEqual(diagram.port("document"), "DocumentText")
        self.assertEqual(diagram.port("manifold"), "glue__glue_claims")
        self.assertEqual(diagram.port_type("manifold"), "global_state")

    def test_run_democritus_pipeline_from_text_writes_core_artifacts(self) -> None:
        text = (
            "Monsoon variability reshapes agricultural production across regions. "
            "Trade networks and food prices respond when crop failures propagate."
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = run_democritus_pipeline_from_text(
                text,
                llm=ReplayDemocritusLLM(),
                config=DemocritusPipelineConfig.smoke(domain_name="indus_smoke"),
                outdir=tmpdir,
            )

            self.assertGreaterEqual(len(artifacts.root_topics), 3)
            self.assertGreaterEqual(len(artifacts.causal_questions), 3)
            self.assertGreaterEqual(len(artifacts.causal_statements), 3)
            self.assertGreaterEqual(len(artifacts.relational_triples), 6)
            self.assertEqual(artifacts.manifold.embeddings_2d.shape[1], 2)
            self.assertEqual(artifacts.manifold.embeddings_3d.shape[1], 3)

            outdir_path = Path(tmpdir)
            for filename in (
                "root_topics.txt",
                "topic_graph.jsonl",
                "causal_questions.jsonl",
                "causal_statements.jsonl",
                "relational_triples.jsonl",
                "manifold.npz",
                "manifold_summary.json",
                "democritus_diagram.json",
            ):
                self.assertTrue((outdir_path / filename).exists(), filename)

            if "plot" in artifacts.output_files:
                self.assertTrue((outdir_path / "causal_manifold.png").exists())
            if "local_causal_models_manifest" in artifacts.output_files:
                manifest_path = outdir_path / "local_causal_models.json"
                self.assertTrue(manifest_path.exists())
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertGreaterEqual(len(manifest), 1)
                for entry in manifest:
                    self.assertTrue(Path(entry["png"]).exists())

            manifold = np.load(outdir_path / "manifold.npz", allow_pickle=True)
            self.assertGreaterEqual(len(manifold["entities"]), 4)
            self.assertEqual(manifold["embeddings_2d"].shape[1], 2)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
from pathlib import Path

from FunctorFlow.democritus import (
    DemocritusPipelineConfig,
    run_democritus_pipeline_from_pdf,
    run_democritus_pipelines_from_pdf_directory,
    run_democritus_pipeline_from_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the FunctorFlow Democritus pipeline on a PDF, plain-text file, or directory of PDFs."
    )
    parser.add_argument("--pdf-file", type=Path, help="Path to an input PDF document")
    parser.add_argument("--pdf-dir", type=Path, help="Path to a directory of input PDF documents")
    parser.add_argument("--text-file", type=Path, help="Optional plain-text fallback input")
    parser.add_argument("--outdir", type=Path, default=Path("FunctorFlow/democritus_runs"))
    parser.add_argument("--domain-name", default="democritus_demo")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a smaller config for quick pipeline validation",
    )
    args = parser.parse_args()
    provided = sum(
        value is not None for value in (args.pdf_file, args.pdf_dir, args.text_file)
    )
    if provided != 1:
        parser.error("Provide exactly one of --pdf-file, --pdf-dir, or --text-file.")
    return args


def main() -> None:
    args = parse_args()
    config = (
        DemocritusPipelineConfig.smoke(domain_name=args.domain_name)
        if args.smoke
        else DemocritusPipelineConfig(domain_name=args.domain_name)
    )

    if args.pdf_dir is not None:
        batch_results = run_democritus_pipelines_from_pdf_directory(
            args.pdf_dir,
            config=config,
            outdir=args.outdir,
        )
        print(f"Processed {len(batch_results)} PDF documents from {args.pdf_dir}")
        for pdf_path, artifacts in batch_results.items():
            print(f"\nPDF: {pdf_path}")
            print(f"Domain: {artifacts.config.domain_name}")
            print(f"Root topics: {len(artifacts.root_topics)}")
            print(f"Topic graph nodes: {len(artifacts.topic_graph)}")
            print(f"Triples: {len(artifacts.relational_triples)}")
            print(f"Entities: {len(artifacts.manifold.entities)}")
            print(f"DB obstruction: {artifacts.manifold.db_obstruction:.4f}")
            print(f"Output directory: {Path(artifacts.output_files['manifold']).parent}")
            for name, path in sorted(artifacts.output_files.items()):
                print(f"{name}: {path}")
        return
    if args.pdf_file is not None:
        artifacts = run_democritus_pipeline_from_pdf(
            args.pdf_file,
            config=config,
            outdir=args.outdir,
        )
    else:
        text = args.text_file.read_text(encoding="utf-8")
        artifacts = run_democritus_pipeline_from_text(
            text,
            config=config,
            outdir=args.outdir,
        )

    print(f"Domain: {artifacts.config.domain_name}")
    print(f"Root topics: {len(artifacts.root_topics)}")
    print(f"Topic graph nodes: {len(artifacts.topic_graph)}")
    print(f"Triples: {len(artifacts.relational_triples)}")
    print(f"Entities: {len(artifacts.manifold.entities)}")
    print(f"DB obstruction: {artifacts.manifold.db_obstruction:.4f}")
    for name, path in sorted(artifacts.output_files.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

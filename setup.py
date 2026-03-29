from __future__ import annotations

from setuptools import setup


setup(
    name="FunctorFlow",
    version="0.1.0",
    description="A lightweight categorical DSL and executable IR for diagrammatic AI systems.",
    packages=["FunctorFlow", "FunctorFlow.examples"],
    package_dir={
        "FunctorFlow": ".",
        "FunctorFlow.examples": "examples",
    },
    package_data={
        "FunctorFlow": [
            "docs/*.pdf",
            "notebooks/*.ipynb",
            "data/README.md",
            "data/democritus/.gitkeep",
            "data/ptb/.gitkeep",
            "data/wikitext-2/.gitkeep",
            "data/wikitext-103/LICENSE.txt",
            "data/wikitext-103/README.txt",
            "proofs/*.lean",
            "proofs/*.json",
            "proofs/lakefile.lean",
            "proofs/lean-toolchain",
            "proofs/FunctorFlowProofs/*.lean",
            "proofs/FunctorFlowProofs/Generated/*.lean",
        ],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
    },
)

# FunctorFlow Data

This directory is the package-local data root for the standalone FunctorFlow
release.

Current bundled assets:

- `ptb/`
  Contains the Penn Treebank word-level files used by the FunctorFlow KET
  language-model demos:
  - `ptb.train.txt`
  - `ptb.valid.txt`
  - `ptb.test.txt`
- `wikitext-2/`
  Contains the WikiText-2 token files used by the FunctorFlow KET
  language-model demos:
  - `wiki.train.tokens`
  - `wiki.valid.tokens`
  - `wiki.test.tokens`
- `wikitext-103/`
  Contains the WikiText-103 token files used by the larger FunctorFlow
  language-model comparison path, together with the upstream metadata files:
  - `wiki.train.tokens`
  - `wiki.valid.tokens`
  - `wiki.test.tokens`
  - `README.txt`
  - `LICENSE.txt`
- `democritus/`
  Holds sample PDF inputs for local Democritus document-graph experiments.
  For a public release, this directory is best treated as user-supplied input
  unless the redistribution status of each sample PDF has been confirmed.

The loader in `FunctorFlow/ket_lm.py` now prefers this directory by default.
That keeps the public FunctorFlow package self-contained and easier to publish
as a standalone GitHub repository ahead of the ICML 2026 tutorial.

For GitHub release preparation, a conservative publication policy is
recommended:

- keep upstream attribution and license notices with any redistributed corpus
- if PTB or WikiText-2 redistribution is uncertain, replace the files with
  download instructions or empty placeholder directories
- if `democritus/` sample PDFs are not cleared for redistribution, ship the
  directory empty and ask users to add their own documents

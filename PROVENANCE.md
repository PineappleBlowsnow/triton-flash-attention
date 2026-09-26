# Provenance and publication changes

Source classroom repository: https://github.com/llm-2026/flash-attention-PineappleBlowsnow

Upstream teaching repository: https://github.com/dataflowr/gpu_llm_flash-attention

- Initial classroom commit: `4e1ad0154d2f21d5e9e947a317a4767274dd9234`.
- Student implementation commit: `96d7d5f8d8b4c76920af29f21a7dbd974d395b5b`.
- Final classroom commit used here: `01c98c0b6b5c10ebf760f810e9093b25c233a007`.
- Implementation/benchmark commits are attributed to the PineappleBlowsnow account in the original repository.

The attention implementation and benchmark are copied without semantic changes. The tests are supplied course material under the repository's Apache-2.0 license. The adapters are narrowed to attention only; unrelated online-softmax and fused-softmax-matmul adapters are omitted. LICENSE is unchanged. New README, package metadata, NOTICE and validation/provenance files were prepared for this portfolio publication with AI assistance; they do not represent additional model experiments.

`provenance.json` records relative source paths and SHA-256 hashes without disclosing private filesystem locations. CRLF/LF differences are ignored only when comparing against Git's normalized blobs; copy hashes use the actual bytes. The historical XML is summarized to remove machine hostname and timestamps; its original SHA-256 is retained. The complete 43-test history covers more than this package's selected attention tests.

Excluded: `softmax_matmul/softmax_matmul_sol.py`, other softmax exercises, the full class notebooks/slides, grading and submission scripts, images, uncommitted notes, caches, and original Git metadata. No model weights, datasets, credentials or private machine paths are required by this package.

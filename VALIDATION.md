# Publication validation — 2026-09-26

## Completed in the packaging environment

- Parsed every published Python file with Python 3.12's AST parser.
- Parsed `pyproject.toml` and all published JSON files.
- Verified copied file SHA-256 hashes against the original files; compared source copies with final classroom Git blobs, normalizing line endings only for the Git comparison.
- Parsed the original historical JUnit XML: 43 tests, zero failures/errors/skips. Removed hostname/timestamps in the published summary. This total includes separate online-softmax/fused-matmul exercises omitted here.
- Validated the benchmark CSV: 10 records, two implementations, five sequence lengths, all recorded statuses `ok`, one recorded GPU model.
- Checked local documentation links and scanned the selected text for common credential formats, private paths and unexpected large/binary files.

## Not performed

PyTorch, Triton and pytest are not installed in this packaging Python environment. No `nvidia-smi` executable was found. No dependency installation, numerical test, GPU kernel launch or benchmark rerun was performed. Static checks cannot establish numerical correctness or compatibility on a new GPU.

Historical tests and measurements are preserved as historical evidence only. For numerical verification, use the installation/testing instructions in README on a compatible Linux CUDA environment.

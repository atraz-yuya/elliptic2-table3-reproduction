# Table 3 reproduction handoff

Start with [the reconstructed experiment](README_paper.md), then read
[the data and label audit](DATA_AUDIT.md) and [the funding holdout protocol](README_funding_holdout.md).
Run commands from the repository root in Linux or WSL.

## Data and environment

Use the public [Elliptic2 release](https://github.com/MITIBMxGraph/Elliptic2).
Place `connected_components.csv`, `nodes.csv`, `edges.csv`,
`background_nodes.csv` and `background_edges.csv.zip` under
`data/elliptic2_raw/`. Keep the background edge archive zipped. Large data,
predictions, checkpoints, and environments are local artifacts.

Use Python 3.12 and a CUDA-enabled PyTorch installation to reproduce the CUDA runs.
The original environment and dependency versions are recorded in the reference
[manifest](results_paper/experiment.json). Replace `python` below with
`.venv-gnn-v1/bin/python` when using the existing workspace environment.
The pipeline uses NumPy, pandas, scikit-learn, SciPy, PyTorch, and pytest.

On a fresh data directory, generate the processed sources in this order.
Each extractor expects its output paths to be free. On the original workstation,
reuse the complete caches referenced in the manifest.

```bash
python scripts/build_node_boundary_features.py
python scripts/extract_onehop_neighbor_attributes.py
python scripts/build_labeled_internal_nodes.py
```

## Re-run in a fresh checkout

Copy code and the archived reference files by checking out this commit.
The commands below place newly trained results in ignored folders so the
reference manifests and summaries stay available for comparison.

```bash
python -m pytest -q -p no:cacheprovider table3_message_passing/test_smoke.py table3_message_passing/test_funding_holdout.py
python -u table3_message_passing/run.py --device cuda --out-dir table3_message_passing/results/replay
python -u table3_message_passing/paper.py --device cuda --out-dir table3_message_passing/results_paper/replay
```

The holdout, audit, and role-flag scripts read the baseline from the default
`results_paper/` directory. For an end-to-end replay in a separate checkout,
first move the tracked reference snapshots out of the default results locations
in that checkout, then run `paper.py --device cuda` with its default output
and restore a copy of `role_flags_control/run_control.py` under the newly
generated baseline directory. The data audit verifies exact source-file
fingerprints against the generated baseline. See the run-specific READMEs for
the rest of the commands.

## Reference snapshots

- [Public pool, 121,810 graphs](results/summary.csv)
- [Table 2 reconstruction, 97,448 graphs](results_paper/summary.csv)
- [GCN, role indicator sensitivity](results_paper/role_flags_control/summary.csv)
- [All four methods, test funding holdout](results_funding_holdout/summary.csv)

Review Table 2 counts and entity overlap alongside AP. The author split and
baseline implementations remain unverified. The funding holdout keeps the
outer test graphs intact and removes about 70% of external node instances
from training, so its AP difference includes the change in training context.

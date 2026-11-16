"""Paired diagnostic: zero the GCN role indicators on the reconstructed Table 2 input."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
PAPER_OUT = HERE.parent
sys.path.insert(0, str(PAPER_OUT.parent))

from dataset import sha256, write_json
from paper import reconstruct
from run import FOLDS, METRICS, PAPER, STEPS, VAL_INTERVAL, fit_model, read_splits, run_lock, write_csv


def main():
    with run_lock(HERE):
        original = json.loads((PAPER_OUT / "experiment.json").read_text(encoding="utf-8"))
        split_file = PAPER_OUT / "splits.npz"
        if sha256(split_file) != original["splits_sha256"]:
            raise ValueError("The original split file changed")
        for filename, digest in original["code_sha256"].items():
            if sha256(PAPER_OUT.parent / filename) != digest:
                raise ValueError(f"The original code changed: {filename}")

        torch.set_num_threads(2)
        device = torch.device(original["environment"]["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to match the original experiment")
        graphs, provenance, _, _ = reconstruct()
        if provenance != original["sources"]:
            raise ValueError("The original data sources changed")
        splits = read_splits(split_file, graphs)
        if not graphs.roles.any():
            raise ValueError("Expected the original nonzero role indicators")
        # Keep the input shape and initialization identical; only erase the three input values.
        graphs.roles.fill(False)
        manifest = {
            "status": "exploratory paired sensitivity analysis",
            "changed": "Set all three role-indicator input columns to 0 before training and evaluation",
            "model": "gcn", "folds": FOLDS, "baseline_manifest_sha256": sha256(PAPER_OUT / "experiment.json"),
            "split_sha256": sha256(split_file), "script_sha256": sha256(__file__),
            "torch": str(torch.__version__), "device": str(device),
            "command": [sys.executable, *sys.argv],
        }
        manifest_path = HERE / "experiment.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if {**previous, "command": manifest["command"]} != manifest:
                raise ValueError("Diagnostic configuration changed")
        write_json(manifest_path, manifest)

        rows = []
        for fold, parts in splits.items():
            baseline = json.loads((PAPER_OUT / "gcn" / f"fold_{fold:02d}" /
                                   "metrics.json").read_text(encoding="utf-8"))
            folder = HERE / f"fold_{fold:02d}"
            path = folder / "metrics.json"
            if not path.exists():
                fit_model("gcn", fold, graphs, parts, device, folder, STEPS["gcn"], VAL_INTERVAL)
            result = json.loads(path.read_text(encoding="utf-8"))
            assert result["scaler"] == baseline["scaler"]
            assert result["seed"] == baseline["seed"] and result["steps"] == baseline["steps"]
            orig = pd.read_csv(PAPER_OUT / "gcn" / f"fold_{fold:02d}" /
                               "outer_predictions.csv", usecols=["ccId", "y"])
            actual = pd.read_csv(folder / "outer_predictions.csv", usecols=["ccId", "y"])
            pd.testing.assert_frame_equal(actual, orig)
            for metric, (paper_mean, _) in zip(METRICS, PAPER["gcn"]):
                rows.append({"fold": fold, "metric": metric, "paper_mean": paper_mean,
                             "role_flags": baseline[metric], "without_role_flags": result[metric],
                             "paired_delta": result[metric] - baseline[metric]})
            write_csv(HERE / "fold_comparison.csv", pd.DataFrame(rows))

        summary = []
        frame = pd.DataFrame(rows)
        for metric in METRICS:
            group = frame[frame.metric == metric]
            summary.append({
                "metric": metric, "folds": len(group), "paper_mean": group.paper_mean.iloc[0],
                "role_flags_mean": group.role_flags.mean(), "role_flags_sd": group.role_flags.std(ddof=1),
                "without_role_flags_mean": group.without_role_flags.mean(),
                "without_role_flags_sd": group.without_role_flags.std(ddof=1),
                "paired_delta_mean": group.paired_delta.mean(), "paired_delta_sd": group.paired_delta.std(ddof=1),
            })
        write_csv(HERE / "summary.csv", pd.DataFrame(summary))
        print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

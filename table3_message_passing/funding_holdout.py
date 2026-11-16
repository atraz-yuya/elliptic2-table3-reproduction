"""Evaluate on unchanged test graphs with every test funding entity withheld from fit and validation."""
import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd
import torch

from dataset import Graphs, positions, sha256, write_json
from models import MODELS
from paper import reconstruct
from run import METRICS, STEPS, VAL_INTERVAL, fit_model, read_splits, run_lock, write_csv

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "results_paper"
DEFAULT_OUT = HERE / "results_funding_holdout"


def holdout_graphs(graphs, parts):
    fit, validation, test = parts
    test_nodes, _, _ = positions(graphs.node_ptr, test)
    funding_entities = np.unique(graphs.entity_rows[test_nodes[graphs.roles[test_nodes, 1]]])
    fit_val_nodes, _, _ = positions(graphs.node_ptr, np.r_[fit, validation])
    removed = np.zeros(len(graphs.entity_rows), dtype=bool)
    removed[fit_val_nodes] = np.isin(graphs.entity_rows[fit_val_nodes], funding_entities)
    if (removed & graphs.roles[:, 0]).any():
        raise ValueError("A held-out funding entity is an internal member. Keep membership intact.")
    keep_nodes = ~removed
    kept_by_graph = np.add.reduceat(keep_nodes, graphs.node_ptr[:-1])
    new_node_ptr = np.r_[0, kept_by_graph.cumsum()]
    node_graph = np.repeat(np.arange(len(graphs.ids)), np.diff(graphs.node_ptr))
    new_node_indices = np.cumsum(keep_nodes) - 1
    edge_graph = np.repeat(np.arange(len(graphs.ids)), np.diff(graphs.edge_ptr))
    original_endpoints = graphs.edges + graphs.node_ptr[edge_graph]
    keep_edges = keep_nodes[original_endpoints].all(axis=0)
    edge_counts = np.bincount(edge_graph[keep_edges], minlength=len(graphs.ids))
    new_edges = new_node_indices[original_endpoints[:, keep_edges]] - new_node_ptr[edge_graph[keep_edges]]

    held = Graphs(ids=graphs.ids, y=graphs.y, node_ptr=new_node_ptr,
                  edge_ptr=np.r_[0, edge_counts.cumsum()], entity_ids=graphs.entity_ids,
                  entity_rows=graphs.entity_rows[keep_nodes], features=graphs.features,
                  roles=graphs.roles[keep_nodes], edges=new_edges)

    new_nodes, _, _ = positions(held.node_ptr, np.r_[fit, validation])
    assert not np.intersect1d(funding_entities, held.entity_rows[new_nodes]).size
    new_test_nodes, _, _ = positions(held.node_ptr, test)
    old_test_edges, _, _ = positions(graphs.edge_ptr, test)
    new_test_edges, _, _ = positions(held.edge_ptr, test)
    np.testing.assert_array_equal(graphs.entity_rows[test_nodes], held.entity_rows[new_test_nodes])
    np.testing.assert_array_equal(graphs.roles[test_nodes], held.roles[new_test_nodes])
    np.testing.assert_array_equal(graphs.edges[:, old_test_edges], held.edges[:, new_test_edges])
    np.testing.assert_array_equal(np.diff(graphs.node_ptr)[test], np.diff(held.node_ptr)[test])
    np.testing.assert_array_equal(np.diff(graphs.edge_ptr)[test], np.diff(held.edge_ptr)[test])

    audit = {"test_funding_clusters": len(funding_entities),
             "test_funding_clusters_in_fit_or_validation": 0,
             "test_graphs_and_labels_unchanged": True}
    for name, rows in zip(("fit", "validation", "test"), parts):
        affected = np.bincount(node_graph[removed], minlength=len(graphs.ids))[rows]
        audit[name] = {"graphs": len(rows), "suspicious": int(graphs.y[rows].sum()),
                       "graphs_with_removed_nodes": int((affected > 0).sum()),
                       "removed_node_instances": int(affected.sum()),
                       "removed_edges": int((np.diff(graphs.edge_ptr)[rows] - edge_counts[rows]).sum())}
    return held, audit


def summarize(out):
    rows = []
    for model in MODELS:
        for folder in sorted((out / model).glob("fold_*")):
            path = folder / "metrics.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            ref_folder = BASELINE / model / folder.name
            baseline = json.loads((ref_folder / "metrics.json").read_text(encoding="utf-8"))
            columns = ["ccId", "y"]
            pd.testing.assert_frame_equal(pd.read_csv(ref_folder / "outer_predictions.csv", usecols=columns),
                                          pd.read_csv(folder / "outer_predictions.csv", usecols=columns))
            for metric in METRICS:
                rows.append({"model": model, "fold": result["fold"], "metric": metric,
                             "original": baseline[metric], "funding_holdout": result[metric],
                             "delta": result[metric] - baseline[metric]})
    if not rows:
        return
    paired = pd.DataFrame(rows)
    result = paired.groupby(["model", "metric"], sort=False).agg(
        folds=("fold", "size"), original=("original", "mean"), original_sd=("original", "std"),
        funding_holdout=("funding_holdout", "mean"), funding_holdout_sd=("funding_holdout", "std"),
        delta=("delta", "mean"), delta_sd=("delta", "std"),
    ).reset_index()
    write_csv(out / "paired_folds.csv", paired)
    write_csv(out / "summary.csv", result)
    print(result[result.metric == "ap"].to_string(index=False), flush=True)
    print(f"Completed {len(rows) // len(METRICS)}/40 runs. Output: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--fold", type=int, choices=range(1, 11))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out == BASELINE:
        parser.error("Use an output directory separate from the original run.")
    with run_lock(out):
        source_manifest = BASELINE / "experiment.json"
        original = json.loads(source_manifest.read_text(encoding="utf-8"))
        split_path = BASELINE / "splits.npz"
        if sha256(split_path) != original["splits_sha256"]:
            raise ValueError("Original split fingerprint changed")
        if any(sha256(HERE / name) != digest for name, digest in original["code_sha256"].items()):
            raise ValueError("Original run code changed")
        torch.set_num_threads(2)
        device = torch.device(original["environment"]["device"])
        graphs, provenance, _, _ = reconstruct()
        if provenance != original["sources"]:
            raise ValueError("Original data fingerprint changed")
        splits = read_splits(split_path, graphs)
        config = {
            "protocol": "Outer test funding IDs absent from all fit/validation node roles",
            "mask": "Remove held-out external node instances and incident edges in fit/validation only",
            "evaluation": "Same test graph data, IDs and labels as the Table 2 reconstruction",
            "baseline_manifest_sha256": sha256(source_manifest), "split_sha256": sha256(split_path),
            "code_sha256": sha256(__file__), "device": str(device), "torch": str(torch.__version__),
            "training": original["training"],
        }
        manifest = out / "experiment.json"
        if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing output uses a different experiment configuration")
        write_json(manifest, config)
        with (out / "commands.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps([sys.executable, *sys.argv]) + "\n")

        failures = []
        for fold in [args.fold] if args.fold else splits:
            held, audit = holdout_graphs(graphs, splits[fold])
            write_json(out / f"fold_{fold:02d}_audit.json", audit)
            print(f"Fold {fold}: {audit['test_funding_clusters']:,} test funding clusters held out, "
                  f"{audit['fit']['removed_node_instances']:,} fit node instances removed.", flush=True)
            for model in [args.model] if args.model else MODELS:
                folder = out / model / f"fold_{fold:02d}"
                if (folder / "metrics.json").is_file():
                    print(f"Reusing {model} fold {fold}", flush=True)
                    continue
                for attempt in (1, 2):
                    try:
                        fit_model(model, fold, held, splits[fold], device, folder,
                                  STEPS[model], VAL_INTERVAL)
                        break
                    except Exception:
                        error = traceback.format_exc()
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / f"error_attempt_{attempt}.log").write_text(error, encoding="utf-8")
                        print(error, file=sys.stderr, flush=True)
                else:
                    failures.append((model, fold))
            summarize(out)
        if failures:
            raise SystemExit(f"Failed after retry: {failures}")


if __name__ == "__main__":
    main()

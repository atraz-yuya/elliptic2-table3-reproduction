
"""Evaluate the Table 3 models on the reconstructed paper observation graphs."""
import argparse
from contextlib import closing
from importlib import metadata
import json
from pathlib import Path
import platform
import sqlite3
import sys
import traceback

import numpy as np
import pandas as pd
import torch

from dataset import FEATURES, ROOT, describe, pack, prepare, sha256, write_json
from models import MODELS
from run import (
    BATCH_SIZE, FOLDS, METRICS, PAPER, SEED, STEPS, VAL_INTERVAL, fit_model,
    make_splits, read_splits, run_lock, write_csv,
)

HERE = Path(__file__).resolve().parent


def paper_pool(members):
    # Elliptic2 preprocess_glass.py enumerates component IDs in order of first appearance.
    cc_ids = members.ccId.drop_duplicates().to_numpy()
    counter = np.arange(1, len(cc_ids) + 1)
    return cc_ids[counter % 10 <= 7]


def restrict_boundary(boundary, internal):
    incoming = boundary.direction.eq(0)
    outgoing = boundary.direction.eq(1)
    return boundary[(incoming & ~boundary.node_id.isin(internal.dst)) |
                    (outgoing & ~boundary.node_id.isin(internal.src))]


def reconstruct():
    # Read the validated Elliptic2 attribute caches and raw data provenance.
    public, provenance = prepare()
    paths = {name: ROOT / item["path"] for name, item in provenance["sources"].items()}
    members = pd.read_csv(paths["members"], usecols=["ccId", "clId"])
    pool = paper_pool(members)
    members = members[members.ccId.isin(pool)]
    labels = pd.DataFrame({"ccId": public.ids, "y": public.y})
    labels = labels[labels.ccId.isin(pool)]
    internal = pd.read_csv(paths["internal"], usecols=["clId1", "clId2"])
    internal.columns = ["src", "dst"]
    internal = internal[internal.src.isin(members.clId)]
    with closing(sqlite3.connect(paths["boundary"].as_uri() + "?mode=ro", uri=True)) as db:
        boundary = pd.read_sql_query("SELECT node_id, direction, neighbor_id FROM unique_neighbors", db)
    boundary = restrict_boundary(boundary[boundary.node_id.isin(members.clId)], internal)
    attrs = pd.DataFrame(public.features, index=public.entity_ids, columns=FEATURES)
    graphs = pack(labels, members, internal, boundary, attrs)
    inventory, composition = describe(graphs)
    observed = composition.pop("public_pool")
    composition["reconstructed_pool"] = observed
    for label, expected in composition["paper_table2"].items():
        if observed[label]["graphs"] != expected["graphs"]:
            raise ValueError(f"{label} class count differs from Table 2")
        for metric in ("member", "funding", "receiving", "nodes", "edges"):
            if [round(value, 2) for value in observed[label][metric]] != expected[metric]:
                raise ValueError(f"{label} {metric} mean or sample SD differs from Table 2")
    return graphs, provenance, inventory, composition


def setup(out, graphs, provenance, inventory, composition, device):
    split_path = out / "splits.npz"
    manifest_path = out / "experiment.json"
    manifest = {
        "protocol": "Table 3 with observation graphs reconstructed to match Table 2",
        "authors_inputs": "Author IDs, folds, and baseline code unavailable. Table 2 aggregates match to published precision.",
        "pool": "Elliptic2 official GLASS train rule on first appearances in nodes.csv",
        "graph": "Internal raw arcs; direct incoming edges into internal sources; direct outgoing edges out of internal sinks",
        "references": {
            "paper": "https://doi.org/10.3390/math14213924",
            "official_pool_rule": "https://github.com/MITIBMxGraph/Elliptic2/blob/main/preprocess_glass.py#L67-L79",
            "pdf_sha256": "bb5480425721d38ce845bc1d041326e6f5f232b2d750aef0cd5fe10d29df57c0",
            "assumptions": "README_paper.md",
        },
        "sources": provenance,
        "code_sha256": {name: sha256(HERE / name)
                        for name in ["dataset.py", "models.py", "run.py", "paper.py"]},
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda,
            "libraries": {name: metadata.version(name) for name in
                          ["numpy", "pandas", "scikit-learn", "scipy", "torch"]},
        },
        "training": {
            "outer_split": "stratified 10-fold", "inner_validation": "stratified 10% of outer training",
            "seed": SEED, "fold_seed": "seed plus one-based fold", "steps": STEPS,
            "batch_size": BATCH_SIZE, "validation_interval": VAL_INTERVAL,
            "optimizer": "Adam", "lr": 0.001, "loss": "unweighted BCEWithLogitsLoss",
            "sampler": "shuffled epochs without replacement",
            "checkpoint": "maximum validation AP, earliest step on ties",
            "threshold": "maximum validation F1, highest threshold on ties, score >= threshold",
        },
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["splits_sha256"] = sha256(split_path)
        if previous != manifest:
            raise ValueError("Inputs, splits, code or environment changed. Use a new --out-dir.")
        return read_splits(split_path, graphs)

    parts = make_splits(graphs)
    audit = {}
    arrays = {"graph_ids": graphs.ids}
    for fold, split in parts.items():
        counts = {}
        for name, rows in zip(("fit", "validation", "test"), split):
            arrays[f"fold{fold}_{name}"] = rows
            counts[name] = {"graphs": len(rows), "suspicious": int(graphs.y[rows].sum())}
        audit[fold] = {"counts": counts, "outer_fit_entity_overlap": graphs.overlap(split[0], split[2])}
    tmp = out / "splits.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(split_path)
    parts = read_splits(split_path, graphs)
    manifest["splits_sha256"] = sha256(split_path)
    write_csv(out / "graph_inventory.csv", inventory)
    write_json(out / "composition.json", composition)
    write_json(out / "split_audit.json", audit)
    write_json(manifest_path, manifest)
    print(f"Table 2 matched at published precision: {len(graphs.ids):,} graphs, "
          f"{int(graphs.y.sum()):,} Suspicious.", flush=True)
    return parts


def summarize(out):
    metrics = []
    summary = []
    for name in MODELS:
        rows = []
        for fold in range(1, FOLDS + 1):
            folder = out / name / f"fold_{fold:02d}"
            path = folder / "metrics.json"
            if path.is_file():
                for artifact in ("checkpoint.pt", "outer_predictions.csv", "validation_history.csv"):
                    if not (folder / artifact).is_file():
                        raise ValueError(f"Missing {artifact} for {name} fold {fold}")
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        metrics.extend(rows)
        values_for_display = []
        for metric, (paper_mean, paper_sd) in zip(METRICS, PAPER[name]):
            values = np.array([row[metric] for row in rows])
            mean = float(values.mean()) if len(values) else None
            sd = float(values.std(ddof=1)) if len(values) > 1 else None
            summary.append({
                "model": name, "metric": metric, "folds": len(rows), "mean": mean, "sd": sd,
                "paper_mean": paper_mean, "paper_sd": paper_sd,
                "delta": mean - paper_mean if len(values) == FOLDS else None,
                "pool": "Table 2 reconstruction, 97448 graphs; locally generated folds",
            })
            if mean is not None:
                spread = f"{sd:.4f}" if sd is not None else "n/a"
                values_for_display.append(f"{metric.upper()}={mean:.4f} ± {spread}")
        print(f"{name:10s} {len(rows):2d}/{FOLDS}  " + "  ".join(values_for_display), flush=True)
    write_csv(out / "fold_metrics.csv", pd.DataFrame(metrics).drop(columns="scaler", errors="ignore"))
    write_csv(out / "summary.csv", pd.DataFrame(summary))
    print(f"Completed {len(metrics)}/{len(MODELS) * FOLDS} runs. Output: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--fold", type=int, choices=range(1, FOLDS + 1))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results_paper")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    with run_lock(out):
        if args.aggregate_only:
            if not (out / "experiment.json").is_file():
                parser.error("No experiment.json in --out-dir")
            summarize(out)
            return
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA unavailable; specify --device cpu for a CPU experiment")
        torch.set_num_threads(2)
        graphs, provenance, inventory, composition = reconstruct()
        parts = setup(out, graphs, provenance, inventory, composition, device)
        with (out / "commands.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"cwd": str(Path.cwd()), "argv": [sys.executable, *sys.argv]}) + "\n")
        failures = []
        for fold in [args.fold] if args.fold else parts:
            for model in [args.model] if args.model else MODELS:
                folder = out / model / f"fold_{fold:02d}"
                if (folder / "metrics.json").is_file():
                    print(f"Reusing {model} fold {fold}", flush=True)
                    continue
                for attempt in range(1, 3):
                    try:
                        fit_model(model, fold, graphs, parts[fold], device, folder,
                                  STEPS[model], VAL_INTERVAL)
                        break
                    except Exception:
                        folder.mkdir(parents=True, exist_ok=True)
                        error = traceback.format_exc()
                        print(error, file=sys.stderr, flush=True)
                        (folder / f"error_attempt_{attempt}.log").write_text(error, encoding="utf-8")
                else:
                    failures.append((model, fold))
            summarize(out)
        if failures:
            raise SystemExit(f"Failed after retry: {failures}")


if __name__ == "__main__":
    main()

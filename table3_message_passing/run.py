
"""Train and evaluate the four Table 3 reimplementations."""
import argparse
from contextlib import contextmanager
import fcntl
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch import nn

from dataset import describe, prepare, sha256, write_json
from models import GraphClassifier, MODELS

HERE = Path(__file__).resolve().parent
FOLDS = 10
SEED = 42
BATCH_SIZE = 128
VAL_INTERVAL = 400
STEPS = {"gcn": 12000, "graphsage": 6000, "gat": 12000, "gin": 6000}
METRICS = ("ap", "precision", "recall", "f1")
PAPER = {
    "gcn": [(0.6822, 0.0217), (0.6753, 0.0516), (0.5684, 0.0280), (0.6155, 0.0188)],
    "graphsage": [(0.7150, 0.0454), (0.7049, 0.0340), (0.6994, 0.0523), (0.7008, 0.0296)],
    "gat": [(0.6958, 0.0372), (0.6333, 0.0580), (0.6199, 0.0547), (0.6232, 0.0283)],
    "gin": [(0.8194, 0.0171), (0.8026, 0.0521), (0.7171, 0.0473), (0.7551, 0.0202)],
}


def make_splits(graphs):
    partitions = {}
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for fold, (candidates, test) in enumerate(splitter.split(graphs.ids, graphs.y), 1):
        fit, validation = train_test_split(candidates, test_size=0.1, shuffle=True,
                                           stratify=graphs.y[candidates], random_state=SEED + fold)
        partitions[fold] = tuple(np.sort(rows) for rows in (fit, validation, test))
    return partitions


def read_splits(path, graphs):
    partitions = {}
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["graph_ids"], graphs.ids):
            raise ValueError("Saved split graph IDs do not match the current input")
        for fold in range(1, FOLDS + 1):
            rows = tuple(saved[f"fold{fold}_{name}"] for name in ("fit", "validation", "test"))
            if not np.array_equal(np.sort(np.concatenate(rows)), np.arange(len(graphs.ids))):
                raise ValueError(f"Fold {fold} omits or repeats graph IDs")
            if any(len(np.unique(graphs.y[subset])) != 2 for subset in rows):
                raise ValueError(f"Fold {fold} needs both classes in each partition")
            partitions[fold] = rows
    test_rows = np.sort(np.concatenate([parts[2] for parts in partitions.values()]))
    if not np.array_equal(test_rows, np.arange(len(graphs.ids))):
        raise ValueError("Outer test sets must partition the graph pool")
    return partitions


def write_csv(path, frame):
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def setup(out, graphs, provenance, device):
    manifest_path = out / "experiment.json"
    split_path = out / "splits.npz"
    code_hash = {name: sha256(HERE / name) for name in ["dataset.py", "models.py", "run.py"]}
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "device": str(device), "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
           "libraries": {name: metadata.version(name) for name in
                         ["numpy", "pandas", "scikit-learn", "scipy", "torch"]}}
    manifest = {
        "protocol": "Public Elliptic2 pool, Table 3 message-passing reimplementation",
        "sources": provenance, "code_sha256": code_hash, "environment": env,
        "models": {"hidden_width": 32, "role_indicators": ["member", "funding", "receiving"],
                   "attributes": 43, "pool_scope": "all observed nodes",
                   "gcn": {"layers": 2, "pool": "mean", "activation": "ReLU", "relu_head": 32,
                           "degree": "in-degree with one self-loop at each endpoint"},
                   "graphsage": {"layers": 2, "pool": "mean", "activation": "ReLU then L2",
                                 "neighbors": "all incoming non-self"},
                   "gat": {"layers": 2, "pool": "mean", "activation": "ELU", "heads": 4,
                           "width_per_head": 8, "combine": "concat", "self_loops": True,
                           "attention_negative_slope": 0.2},
                   "gin": {"layers": 4, "pool": "sum at input and every layer", "eps": 0,
                           "mlp": "Linear-BatchNorm-ReLU-Linear; post-MLP BatchNorm-ReLU",
                           "readout_dropout": 0.5, "readout": "sum of layer logits"}},
        "preprocessing": {"direction": "sender to recipient", "deduplicate": "ordered pairs",
                          "edge_scope": "public target internal edges and direct boundary edges",
                          "self_edges": "strip raw self arcs before model-specific self treatment",
                          "standardization": "unique fit clusters; float64 ddof=0, zero scale -> 1"},
        "training": {"seed": SEED, "init_seed": "seed + one-based fold", "optimizer": "Adam",
                     "lr": 0.001, "fused_adam": device.type == "cuda",
                     "loss": "unweighted BCEWithLogitsLoss", "steps": STEPS,
                     "batch_size": BATCH_SIZE, "sampler": "shuffled epochs without replacement",
                     "validation_interval": VAL_INTERVAL, "checkpoint_tie": "earliest step",
                     "threshold_tie": "highest score maximizing validation F1",
                     "prediction": "score >= threshold"},
        "references": {"paper": "https://doi.org/10.3390/math14213924",
                       "paper_pdf_sha256": "bb5480425721d38ce845bc1d041326e6f5f232b2d750aef0cd5fe10d29df57c0",
                       "assumptions": "README.md",
                       "canonical_command": ".venv-gnn-v1/bin/python table3_message_passing/run.py --device cuda"},
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not split_path.exists() or existing.get("splits_sha256") != sha256(split_path):
            raise ValueError("Saved split file changed or is missing")
        manifest["splits_sha256"] = existing["splits_sha256"]
        if existing != manifest:
            raise ValueError("Inputs, code or environment changed. Use a new --out-dir.")
        return read_splits(split_path, graphs), existing

    partitions = make_splits(graphs)
    arrays = {"graph_ids": graphs.ids}
    audit = {}
    for fold, rows in partitions.items():
        counts = {}
        for name, subset in zip(("fit", "validation", "test"), rows):
            arrays[f"fold{fold}_{name}"] = subset
            counts[name] = {"graphs": len(subset), "suspicious": int(graphs.y[subset].sum())}
        audit[fold] = {"counts": counts, "outer_fit_entity_overlap": graphs.overlap(rows[0], rows[2])}
    tmp = out / "splits.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(split_path)
    partitions = read_splits(split_path, graphs)
    manifest["splits_sha256"] = sha256(split_path)
    inventory, composition = describe(graphs)
    write_csv(out / "graph_inventory.csv", inventory)
    write_json(out / "composition.json", composition)
    write_json(out / "split_audit.json", audit)
    write_json(manifest_path, manifest)
    print(f"Input graphs: {len(graphs.ids):,}; positive: {int(graphs.y.sum()):,}", flush=True)
    return partitions, manifest


@torch.inference_mode()
def predict(model, graphs, rows, normalized, device):
    model.eval()
    chunks = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = graphs.batch(rows[start:start + BATCH_SIZE], normalized, device)
        chunks.append(model(batch).sigmoid())
    result = torch.cat(chunks).cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite prediction")
    return result


def threshold_for(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best]), float(f1[best])


def evaluate(y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"ap": float(average_precision_score(y, scores)), "precision": precision, "recall": recall,
            "f1": f1, "tn": tn, "fp": fp, "fn": fn, "tp": tp}, pred


def fit_model(name, fold, graphs, parts, device, folder, steps, interval, batch_size=BATCH_SIZE):
    start_time = time.monotonic()
    fit, validation, test = parts
    mean, scale = graphs.scaler(fit)
    normalized = ((graphs.features - mean) / scale).astype(np.float32)
    seed = SEED + fold
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = GraphClassifier(name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, fused=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    best_ap = -1.0
    history = []
    order = rng.permutation(fit)
    cursor = 0
    model.train()
    for step in range(1, steps + 1):
        if cursor == len(order):
            order, cursor = rng.permutation(fit), 0
        batch = graphs.batch(order[cursor:cursor + batch_size], normalized, device)
        cursor = min(cursor + batch_size, len(order))
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        if step % interval == 0 or step == steps:
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            scores = predict(model, graphs, validation, normalized, device)
            ap = float(average_precision_score(graphs.y[validation], scores))
            history.append({"step": step, "ap": ap})
            if ap > best_ap:
                best_ap, best_step, best_scores = ap, step, scores
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            print(f"{name} fold={fold:02d} step={step:5d}/{steps} val_AP={ap:.6f} "
                  f"seconds={time.monotonic() - start_time:.1f}", flush=True)
            model.train()
    threshold, validation_f1 = threshold_for(graphs.y[validation], best_scores)
    model.load_state_dict(best_state)
    scores = predict(model, graphs, test, normalized, device)
    metrics, pred = evaluate(graphs.y[test], scores, threshold)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "checkpoint.pt.tmp"
    torch.save(best_state, tmp)
    tmp.replace(folder / "checkpoint.pt")
    write_csv(folder / "validation_history.csv", pd.DataFrame(history))
    write_csv(folder / "outer_predictions.csv", pd.DataFrame({
        "ccId": graphs.ids[test], "y": graphs.y[test], "score": scores, "prediction": pred.astype(int),
    }))
    result = {"model": name, "fold": fold, "seed": seed, "steps": steps, "best_step": best_step,
              "validation_ap": best_ap, "validation_f1": validation_f1, "threshold": threshold,
              "scaler": {"mean": mean.tolist(), "scale": scale.tolist(),
                         "unique_fit_clusters": len(graphs.fit_rows(fit))},
              **metrics, "seconds": time.monotonic() - start_time}
    write_json(folder / "metrics.json", result)
    print(f"{name} fold={fold:02d} outer " +
          " ".join(f"{key}={metrics[key]:.6f}" for key in METRICS), flush=True)


def summarize(out):
    results = []
    table = []
    for name in MODELS:
        rows = []
        for fold in range(1, FOLDS + 1):
            folder = out / name / f"fold_{fold:02d}"
            if (folder / "metrics.json").exists():
                required = ("checkpoint.pt", "outer_predictions.csv", "validation_history.csv")
                if not all((folder / artifact).is_file() for artifact in required):
                    raise ValueError(f"Incomplete artifacts for {name} fold {fold}")
                rows.append(json.loads((folder / "metrics.json").read_text(encoding="utf-8")))
        results.extend(rows)
        printed = []
        for key, (paper_mean, paper_sd) in zip(METRICS, PAPER[name]):
            values = np.array([row[key] for row in rows])
            # A partial aggregate is visible, but its delta is withheld until all folds finish.
            mean = float(values.mean()) if len(values) else None
            sd = float(values.std(ddof=1)) if len(values) > 1 else None
            delta = mean - paper_mean if len(values) == FOLDS else None
            table.append({"model": name, "metric": key, "folds": len(rows), "mean": mean, "sd": sd,
                          "paper_mean": paper_mean, "paper_sd": paper_sd, "delta": delta,
                          "pool": "public Elliptic2, 121810 graphs (paper: 97448)"})
            if mean is not None:
                spread = f"{sd:.4f}" if sd is not None else "n/a"
                printed.append(f"{key.upper()}={mean:.4f} ± {spread}")
        print(f"{name:10s} {len(rows):2d}/{FOLDS}  " + "  ".join(printed), flush=True)
    write_csv(out / "fold_metrics.csv", pd.DataFrame(results).drop(columns="scaler", errors="ignore"))
    write_csv(out / "summary.csv", pd.DataFrame(table))
    print(f"Completed {len(results)}/{len(MODELS) * FOLDS} runs. Output: {out}", flush=True)


@contextmanager
def run_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, help="Default: all four models")
    parser.add_argument("--fold", type=int, choices=range(1, FOLDS + 1), help="Default: all ten folds")
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cuda:0 or cpu")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
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
        graphs, provenance = prepare()
        partitions, _ = setup(out, graphs, provenance, device)
        with (out / "commands.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"cwd": str(Path.cwd()), "argv": [sys.executable, *sys.argv]}) + "\n")

        failures = []
        for fold in [args.fold] if args.fold else partitions:
            for name in [args.model] if args.model else MODELS:
                folder = out / name / f"fold_{fold:02d}"
                if (folder / "metrics.json").is_file():
                    print(f"Reusing {name} fold {fold}", flush=True)
                    continue
                for attempt in range(1, 3):
                    try:
                        fit_model(name, fold, graphs, partitions[fold], device, folder,
                                  STEPS[name], VAL_INTERVAL)
                        break
                    except Exception:
                        folder.mkdir(parents=True, exist_ok=True)
                        error = traceback.format_exc()
                        print(error, file=sys.stderr, flush=True)
                        (folder / f"error_attempt_{attempt}.log").write_text(error, encoding="utf-8")
                else:
                    failures.append((name, fold))
            summarize(out)
        if failures:
            raise SystemExit(f"Failed after retry: {failures}")


if __name__ == "__main__":
    main()

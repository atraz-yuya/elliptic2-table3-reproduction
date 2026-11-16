"""Replay all saved Table 3 checkpoints and audit the prediction path's label isolation."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch

from dataset import write_json
from models import MODELS, GraphClassifier
from paper import reconstruct
from run import predict, read_splits, threshold_for

OUT = Path(__file__).resolve().parent / "results_paper"


def main():
    graphs, *_ = reconstruct()
    torch.set_num_threads(2)
    device = torch.device("cuda")
    replay, isolation = [], []
    for fold, (fit, val, test) in read_splits(OUT / "splits.npz", graphs).items():
        mean, scale = graphs.scaler(fit)
        normalized = ((graphs.features - mean) / scale).astype(np.float32)
        sample = np.sort(np.r_[test[graphs.y[test] == 0][:64], test[graphs.y[test] == 1][:64]])
        sample_batch = graphs.batch(sample, normalized, "cpu")
        original = graphs.y
        modified = original.copy()
        modified[sample] = 1 - modified[sample]
        graphs.y = modified
        altered_batch = graphs.batch(sample, normalized, "cpu")
        graphs.y = original
        torch.testing.assert_close(altered_batch.y, 1 - sample_batch.y, atol=0, rtol=0)
        for field in ("x", "edges", "graph", "sizes"):
            torch.testing.assert_close(getattr(altered_batch, field), getattr(sample_batch, field),
                                       atol=0, rtol=0)

        for name in MODELS:
            folder = OUT / name / f"fold_{fold:02d}"
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            model = GraphClassifier(name).eval()
            model.load_state_dict(torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True))
            with torch.inference_mode():
                original_logits = model(sample_batch)
                torch.testing.assert_close(model(altered_batch), original_logits, atol=0, rtol=0)
                torch.testing.assert_close(model(sample_batch._replace(y=None)), original_logits, atol=0, rtol=0)
            isolation.append({"fold": fold, "model": name, "graphs": len(sample), "logits_identical": True})

            model.to(device)
            inner = predict(model, graphs, val, normalized, device)
            threshold, f1 = threshold_for(graphs.y[val], inner)
            outer = predict(model, graphs, test, normalized, device)
            stored = pd.read_csv(folder / "outer_predictions.csv", dtype={"score": np.float32})
            replay.append({
                "fold": fold, "model": name,
                "threshold_absolute_delta": abs(threshold - metrics["threshold"]),
                "validation_f1_absolute_delta": abs(f1 - metrics["validation_f1"]),
                "validation_ap_absolute_delta": abs(float(average_precision_score(graphs.y[val], inner)) -
                                                    metrics["validation_ap"]),
                "outer_ap_absolute_delta": abs(float(average_precision_score(graphs.y[test], outer)) -
                                               metrics["ap"]),
                "max_outer_score_absolute_delta": float(np.max(np.abs(stored.score.to_numpy() - outer))),
                "outer_predictions_changed_at_saved_threshold":
                    int(np.count_nonzero((outer >= metrics["threshold"]) != stored.prediction)),
            })
        print(f"Verified label isolation and checkpoint inference for fold {fold}", flush=True)

    keys = [key for key in replay[0] if key not in ("fold", "model")]
    maxima = {key: max(row[key] for row in replay) for key in keys}
    write_json(OUT / "data_audit_checkpoint_replay.json", {"runs": replay, "maxima": maxima})
    write_json(OUT / "data_audit_label_isolation.json", {"runs": isolation, "all_passed": True})
    print(f"Verified {len(isolation)} checkpoints after label inversion and removal.")
    print(json.dumps(maxima))


if __name__ == "__main__":
    main()

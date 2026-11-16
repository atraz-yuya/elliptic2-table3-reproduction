"""Independently check the saved Table 2/3 run against the raw Elliptic2 release."""
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix

from dataset import FEATURES, ROOT, TABLE2, sha256, write_json
from paper import reconstruct
from run import FOLDS, METRICS, MODELS, run_lock

HERE = Path(__file__).resolve().parent
RAW = ROOT / "data/elliptic2_raw"
RESULTS = HERE / "results_paper"
AUDIT = RESULTS / "data_audit.json"
ENDPOINTS = ["clId1", "clId2"]
TX_COLUMNS = [*ENDPOINTS, "txId"]


def count_difference(expected, observed):
    expected, observed = set(expected), set(observed)
    return {"missing": len(expected - observed), "extra": len(observed - expected)}


def inventory_from_edges(pool, members, members_by_id, internal, external):
    roles = []
    for row in members.itertuples(index=False):
        if row.ccId in pool:
            roles.append((row.ccId, row.clId, 1))
    arcs = set()
    for (src, dst, _) in internal:
        cc = members_by_id[src]
        if cc in pool:
            arcs.add((cc, src, dst))
    sources = set(members_by_id.index).difference(dst for _, dst, _ in internal)
    sinks = set(members_by_id.index).difference(src for src, _, _ in internal)
    for node, direction, neighbor in external:
        cc = members_by_id[node]
        if cc not in pool:
            continue
        if direction == 0 and node in sources:
            roles.append((cc, neighbor, 2))
            arcs.add((cc, neighbor, node))
        if direction == 1 and node in sinks:
            roles.append((cc, neighbor, 4))
            arcs.add((cc, node, neighbor))
    nodes = {}
    for cc, cluster, role in roles:
        key = (cc, cluster)
        nodes[key] = nodes.get(key, 0) | role
    return nodes, arcs


def check_graphs(graphs, inventory, pool, labels, nodes, arcs):
    ids = sorted(pool)
    assert np.array_equal(graphs.ids, ids)
    assert np.array_equal(graphs.y, labels.loc[ids, "y"].to_numpy())
    node_cc = np.repeat(graphs.ids, np.diff(graphs.node_ptr))
    edge_cc = np.repeat(graphs.ids, np.diff(graphs.edge_ptr))
    role_bits = graphs.roles @ np.array([1, 2, 4])
    observed_nodes = set(zip(node_cc, graphs.entity_ids[graphs.entity_rows], role_bits))
    offset = np.repeat(graphs.node_ptr[:-1], np.diff(graphs.edge_ptr))
    edge_counts = np.repeat(np.diff(graphs.node_ptr), np.diff(graphs.edge_ptr))
    assert (graphs.edges >= 0).all() and (graphs.edges < edge_counts).all()
    src, dst = graphs.entity_ids[graphs.entity_rows[graphs.edges + offset]]
    observed_edges = set(zip(edge_cc, src, dst))
    assert len(observed_nodes) == len(graphs.entity_rows)
    assert len(observed_edges) == len(edge_cc)
    composition = {}
    for label, expected in TABLE2.items():
        subset = inventory[inventory.y == int(label == "suspicious")]
        composition[label] = {"graphs": len(subset)}
        assert len(subset) == expected["graphs"]
        for field in ("member", "funding", "receiving", "nodes", "edges"):
            observed = [float(subset[field].mean()), float(subset[field].std(ddof=1))]
            assert [round(value, 2) for value in observed] == expected[field]
            composition[label][field] = observed
    result = {
        "graph_ids": len(ids), "node_instances": len(nodes), "directed_edges": len(arcs),
        "node_role_difference": count_difference(
            ((cc, cluster, role) for (cc, cluster), role in nodes.items()), observed_nodes),
        "edge_difference": count_difference(arcs, observed_edges),
        "table2": composition,
    }
    return result


def check_attributes(graphs, source_paths):
    attrs = [pd.read_csv(source_paths[key], usecols=["clId", *FEATURES], index_col="clId")
             for key in ("member_attributes", "boundary_attributes")]
    cached = pd.concat(attrs)
    assert cached.index.is_unique
    assert list(cached.columns) == FEATURES
    cached = cached.sort_index()
    raw_path = RAW / "background_nodes.csv"
    total = matched = 0
    row_hits = np.zeros(len(cached), dtype=np.int16)
    mismatch_rows = 0
    graph_attribute_rows = cached.index.get_indexer(graphs.entity_ids)
    assert (graph_attribute_rows >= 0).all()
    np.testing.assert_array_equal(graphs.features, cached.iloc[graph_attribute_rows].to_numpy())
    for chunk in pd.read_csv(raw_path, usecols=["clId", *FEATURES], chunksize=250000):
        total += len(chunk)
        rows = cached.index.get_indexer(chunk.clId)
        keep = rows >= 0
        selected = rows[keep]
        values = chunk.loc[keep, FEATURES].to_numpy()
        mismatch_rows += int(np.count_nonzero((values != cached.iloc[selected].to_numpy()).any(axis=1)))
        np.add.at(row_hits, selected, 1)
        matched += len(selected)
    result = {
        "raw_rows": total, "columns": list(cached.columns), "cached_clusters": len(cached),
        "model_clusters": len(graphs.entity_ids), "raw_rows_matched": matched,
        "cached_ids_missing": int((row_hits == 0).sum()),
        "cached_ids_duplicated_in_raw": int((row_hits > 1).sum()), "mismatched_attribute_rows": mismatch_rows,
        "raw_sha256": sha256(raw_path),
    }
    assert total == 49299864
    return result


def scan_background_edges(archive_path, members_by_id):
    index = members_by_id.index
    cc_ids = members_by_id.to_numpy()
    internal = Counter()
    boundary = set()
    total = boundary_transactions = 0
    with ZipFile(archive_path) as archive:
        candidates = [item for item in archive.infolist()
                      if Path(item.filename).name == "background_edges.csv" and not item.is_dir()]
        (entry,) = candidates
        with archive.open(entry) as stream:
            for chunk in pd.read_csv(stream, usecols=TX_COLUMNS, dtype="int64", chunksize=500000):
                src, dst, tx = (chunk[column].to_numpy() for column in TX_COLUMNS)
                src_rows, dst_rows = index.get_indexer(src), index.get_indexer(dst)
                known_src, known_dst = src_rows >= 0, dst_rows >= 0
                both = known_src & known_dst
                same = np.zeros(len(chunk), dtype=bool)
                same[both] = cc_ids[src_rows[both]] == cc_ids[dst_rows[both]]
                funding = known_dst & ~same
                receiving = known_src & ~same
                internal.update(zip(src[same], dst[same], tx[same]))
                boundary.update((node, 0, neighbor) for node, neighbor in zip(dst[funding], src[funding]))
                boundary.update((node, 1, neighbor) for node, neighbor in zip(src[receiving], dst[receiving]))
                total += len(chunk)
                boundary_transactions += int((funding | receiving).sum())
                if total % 20_000_000 == 0:
                    print(f"Background edge scan: {total:,} transactions", flush=True)
    assert total == 196215606
    return internal, boundary, {
        "raw_rows": total, "boundary_transactions": boundary_transactions,
        "internal_transactions": sum(internal.values()), "unique_boundary_records": len(boundary),
        "zip_member": entry.filename, "zip_member_bytes": entry.file_size, "zip_member_crc32": f"{entry.CRC:08x}",
    }


def check_outputs(graphs, manifest):
    summary = pd.read_csv(RESULTS / "summary.csv")
    split_path = RESULTS / "splits.npz"
    assert sha256(split_path) == manifest["splits_sha256"]
    all_test = []
    by_model = {model: [] for model in MODELS}
    fold_audit = {}
    node_owner = np.repeat(np.arange(len(graphs.ids)), np.diff(graphs.node_ptr))
    role_member = graphs.roles[:, 0]
    with np.load(split_path, allow_pickle=False) as splits:
        np.testing.assert_array_equal(splits["graph_ids"], graphs.ids)
        for fold in range(1, FOLDS + 1):
            fit, val, test = [splits[f"fold{fold}_{part}"] for part in ("fit", "validation", "test")]
            np.testing.assert_array_equal(np.sort(np.r_[fit, val, test]), np.arange(len(graphs.ids)))
            assert (len(fit), len(val), len(test)) == ((78932, 8771, 9745) if fold <= 8 else
                                                       (78933, 8771, 9744))
            all_test.extend(test)
            fit_entities = np.unique(graphs.entity_rows[np.isin(node_owner, fit)])
            test_member = np.isin(node_owner, test) & role_member
            test_external = np.isin(node_owner, test) & ~role_member
            fit_values = graphs.features[fit_entities].astype(np.float64)
            mean, scale = fit_values.mean(axis=0), fit_values.std(axis=0)
            scale[scale == 0] = 1
            shared_members = len(np.intersect1d(fit_entities, graphs.entity_rows[test_member]))
            shared_external = len(np.intersect1d(fit_entities, graphs.entity_rows[test_external]))
            assert shared_members == 0
            fold_audit[str(fold)] = {"fit_clusters": len(fit_entities),
                                     "test_member_clusters_in_fit": shared_members,
                                     "test_external_clusters_in_fit": shared_external}
            for model in MODELS:
                folder = RESULTS / model / f"fold_{fold:02d}"
                metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                scaler = metrics["scaler"]
                assert len(fit_entities) == scaler["unique_fit_clusters"]
                np.testing.assert_allclose(mean, scaler["mean"], rtol=1e-12)
                np.testing.assert_allclose(scale, scaler["scale"], rtol=1e-12)
                pred = pd.read_csv(folder / "outer_predictions.csv", dtype={"score": np.float32})
                np.testing.assert_array_equal(pred.ccId, graphs.ids[test])
                np.testing.assert_array_equal(pred.y, graphs.y[test])
                assert pred.score.between(0, 1).all()
                np.testing.assert_array_equal(pred.prediction, pred.score >= metrics["threshold"])
                tn, fp, fn, tp = confusion_matrix(pred.y, pred.prediction, labels=[0, 1]).ravel()
                assert [tn, fp, fn, tp] == [metrics[key] for key in ("tn", "fp", "fn", "tp")]
                values = [average_precision_score(pred.y, pred.score), tp / max(tp + fp, 1),
                          tp / max(tp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1)]
                np.testing.assert_allclose(values, [metrics[key] for key in METRICS], rtol=1e-12)
                history = pd.read_csv(folder / "validation_history.csv")
                assert history.loc[history.ap.idxmax(), "step"] == metrics["best_step"]
                by_model[model].append(values)
    np.testing.assert_array_equal(np.sort(all_test), np.arange(len(graphs.ids)))
    for model, rows in by_model.items():
        observed = summary[summary.model == model].set_index("metric").loc[list(METRICS)]
        np.testing.assert_allclose(observed["mean"], np.mean(rows, axis=0))
        np.testing.assert_allclose(observed["sd"], np.std(rows, axis=0, ddof=1))
    return {"runs_verified": sum(map(len, by_model.values())), "folds": fold_audit}


def main():
    with run_lock(RESULTS):
        manifest_path = RESULTS / "experiment.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_paths = {}
        for name, item in manifest["sources"]["sources"].items():
            path = ROOT / item["path"]
            assert sha256(path) == item["sha256"]
            source_paths[name] = path
        for name, digest in manifest["code_sha256"].items():
            assert sha256(HERE / name) == digest

        members = pd.read_csv(RAW / "nodes.csv")
        labels = pd.read_csv(RAW / "connected_components.csv").set_index("ccId")
        labels["y"] = labels.ccLabel.map({"licit": 0, "suspicious": 1})
        members_by_id = members.set_index("clId").ccId
        assert members_by_id.index.is_unique
        assert len(labels) == 121810 and labels.index.is_unique
        pool = {cc for position, cc in enumerate(dict.fromkeys(members.ccId), 1)
                if position % 10 not in (8, 9)}
        listed = Counter(pd.read_csv(RAW / "edges.csv", usecols=TX_COLUMNS)
                         .loc[:, TX_COLUMNS].itertuples(index=False, name=None))
        with closing(sqlite3.connect(source_paths["boundary"].as_uri() + "?mode=ro", uri=True)) as con:
            cached = set(con.execute("SELECT node_id, direction, neighbor_id FROM unique_neighbors"))

        graphs, _, inventory, _ = reconstruct()
        saved = pd.read_csv(RESULTS / "graph_inventory.csv")
        pd.testing.assert_frame_equal(saved, inventory, check_dtype=False)
        outputs = check_outputs(graphs, manifest)
        report = {"status": "running", "experiment_sha256": sha256(manifest_path),
                  "data_and_code_hashes_verified": True,
                  "public_labels": labels.ccLabel.value_counts().to_dict(), "outputs": outputs}
        write_json(AUDIT, report)
        print("Verified held-out partitions, fit-only scalers, labels and predictions for all 40 runs.", flush=True)

        report["attributes"] = check_attributes(graphs, source_paths)
        write_json(AUDIT, report)
        print("Raw background node attribute scan complete.", flush=True)

        archive_path = RAW / "background_edges.csv.zip"
        internal, boundary, edge_report = scan_background_edges(archive_path, members_by_id)
        edge_report["internal_missing_transactions"] = sum((listed - internal).values())
        edge_report["internal_extra_transactions"] = sum((internal - listed).values())
        edge_report["boundary_difference"] = count_difference(boundary, cached)
        report["background_edges"] = edge_report
        nodes, arcs = inventory_from_edges(pool, members, members_by_id, internal, boundary)
        report["graph_input"] = check_graphs(graphs, inventory, pool, labels, nodes, arcs)
        attributes = report["attributes"]
        failed = (
            attributes["cached_ids_missing"] or attributes["cached_ids_duplicated_in_raw"] or
            attributes["mismatched_attribute_rows"] or edge_report["internal_missing_transactions"] or
            edge_report["internal_extra_transactions"] or any(edge_report["boundary_difference"].values()) or
            any(report["graph_input"]["node_role_difference"].values()) or
            any(report["graph_input"]["edge_difference"].values())
        )
        report["status"] = "differences_found" if failed else "passed"
        write_json(AUDIT, report)
        print(f"Audit {report['status']}. Report: {AUDIT}", flush=True)
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

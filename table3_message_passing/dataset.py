
"""Prepare Elliptic2 observation graphs from the thesis's read-only caches."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [f"feat#{i}" for i in range(1, 44)]
ROLES = ["member", "funding", "receiving"]
TABLE2 = {
    "licit": {"graphs": 95249, "member": [3.65, 3.17], "funding": [1.59, 2.01],
              "receiving": [1.43, 2.17], "nodes": [6.51, 4.77], "edges": [6.00, 5.75]},
    "suspicious": {"graphs": 2199, "member": [3.80, 2.65], "funding": [1.54, 1.70],
                   "receiving": [1.65, 2.43], "nodes": [6.82, 4.34], "edges": [6.24, 4.75]},
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def pack(labels, members, internal, boundary, attrs):
    """Return contiguous graph-local node and edge blocks with shared feature rows."""
    if labels.ccId.duplicated().any() or not labels.y.isin([0, 1]).all():
        raise ValueError("Expected unique graph IDs with binary labels")
    labels = labels.sort_values("ccId")
    if members.clId.duplicated().any() or set(members.ccId) != set(labels.ccId):
        raise ValueError("Expected unique member IDs and at least one member per graph")
    member_cc = members.set_index("clId").ccId
    internal = internal.copy()
    internal["ccId"] = internal.src.map(member_cc)
    if internal.ccId.isna().any() or not internal.ccId.eq(internal.dst.map(member_cc)).all():
        raise ValueError("Internal edge endpoints must belong to the same target")
    boundary = boundary.copy()
    boundary["ccId"] = boundary.node_id.map(member_cc)
    if (boundary.ccId.isna().any() or not boundary.direction.isin([0, 1]).all()
            or boundary.ccId.eq(boundary.neighbor_id.map(member_cc)).any()):
        raise ValueError("Invalid cached boundary record")

    incoming = boundary.direction.eq(0)
    boundary["src"] = np.where(incoming, boundary.neighbor_id, boundary.node_id)
    boundary["dst"] = np.where(incoming, boundary.node_id, boundary.neighbor_id)
    role_rows = pd.concat([
        members.assign(member=True, funding=False, receiving=False),
        boundary[["ccId", "neighbor_id"]].rename(columns={"neighbor_id": "clId"})
        .assign(member=False, funding=incoming, receiving=~incoming),
    ], ignore_index=True)
    roles = role_rows.groupby(["ccId", "clId"], sort=True)[ROLES].max()
    edges = (pd.concat([internal[["ccId", "src", "dst"]], boundary[["ccId", "src", "dst"]]],
                       ignore_index=True)
             .drop_duplicates().sort_values(["ccId", "src", "dst"]))

    graph_ids = labels.ccId.to_numpy(dtype=np.int64)
    node_graph = np.searchsorted(graph_ids, roles.index.get_level_values("ccId"))
    edge_graph = np.searchsorted(graph_ids, edges.ccId)
    node_counts = np.bincount(node_graph, minlength=len(graph_ids))
    edge_counts = np.bincount(edge_graph, minlength=len(graph_ids))
    node_ptr = np.r_[0, node_counts.cumsum()]
    edge_ptr = np.r_[0, edge_counts.cumsum()]

    node_ids = roles.index.get_level_values("clId").to_numpy(dtype=np.int64)
    entity_ids, entity_rows = np.unique(node_ids, return_inverse=True)
    if not attrs.index.is_unique:
        raise ValueError("Feature table has duplicate cluster IDs")
    features = attrs.reindex(entity_ids)[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Missing or non-finite cluster features")

    local_edges = []
    for endpoint in ["src", "dst"]:
        global_rows = roles.index.get_indexer(pd.MultiIndex.from_frame(edges[["ccId", endpoint]]))
        if (global_rows < 0).any():
            raise ValueError("Edge refers to an absent node")
        local_edges.append(global_rows - node_ptr[edge_graph])

    return Graphs(
        ids=graph_ids, y=labels.y.to_numpy(dtype=np.int8), node_ptr=node_ptr, edge_ptr=edge_ptr,
        entity_ids=entity_ids, entity_rows=entity_rows, features=features,
        roles=roles.to_numpy(dtype=np.bool_), edges=np.stack(local_edges),
    )


def positions(ptr, graph_rows):
    counts = np.diff(ptr)[graph_rows]
    offsets = np.r_[0, counts.cumsum()]
    rows = np.arange(offsets[-1]) + np.repeat(ptr[graph_rows] - offsets[:-1], counts)
    return rows, counts, offsets


class Batch(NamedTuple):
    x: torch.Tensor
    edges: torch.Tensor
    graph: torch.Tensor
    sizes: torch.Tensor
    y: torch.Tensor


class Graphs:
    def __init__(self, **arrays):
        vars(self).update(arrays)

    def fit_rows(self, graph_rows):
        node_rows, _, _ = positions(self.node_ptr, graph_rows)
        return np.unique(self.entity_rows[node_rows])

    def scaler(self, graph_rows):
        fit = self.features[self.fit_rows(graph_rows)].astype(np.float64)
        mean, scale = fit.mean(axis=0), fit.std(axis=0, ddof=0)
        scale[scale == 0] = 1
        return mean, scale

    def overlap(self, fit, evaluation):
        seen = np.zeros(len(self.entity_ids), dtype=bool)
        seen[self.fit_rows(fit)] = True
        node_rows, counts, offsets = positions(self.node_ptr, evaluation)
        shared = np.add.reduceat(seen[self.entity_rows[node_rows]], offsets[:-1])
        return {"shared_graphs": int((shared > 0).sum()), "graphs": len(counts),
                "fraction": float((shared > 0).mean())}

    def batch(self, graph_rows, normalized, device):
        node_rows, sizes, offsets = positions(self.node_ptr, graph_rows)
        edge_rows, edge_counts, _ = positions(self.edge_ptr, graph_rows)
        x = np.empty((len(node_rows), 46), dtype=np.float32)
        x[:, :43] = normalized[self.entity_rows[node_rows]]
        x[:, 43:] = self.roles[node_rows]
        edges = self.edges[:, edge_rows] + np.repeat(offsets[:-1], edge_counts)
        # Each model handles its own self contribution.
        edges = edges[:, edges[0] != edges[1]]
        graph = np.repeat(np.arange(len(graph_rows)), sizes)
        y = self.y[graph_rows].astype(np.float32)
        return Batch(*(torch.from_numpy(a).to(device) for a in (x, edges, graph, sizes, y)))

    def inventory(self):
        inventory = pd.DataFrame({"ccId": self.ids, "y": self.y})
        for i, role in enumerate(ROLES):
            inventory[role] = np.add.reduceat(self.roles[:, i], self.node_ptr[:-1])
        inventory["nodes"] = np.diff(self.node_ptr)
        inventory["edges"] = np.diff(self.edge_ptr)
        inventory["dual_role"] = np.add.reduceat(self.roles[:, 1] & self.roles[:, 2],
                                                 self.node_ptr[:-1])
        return inventory


def prepare():
    raw = ROOT / "data/elliptic2_raw"
    processed = ROOT / "data/processed"
    sources = {
        "labels": raw / "connected_components.csv",
        "members": raw / "nodes.csv",
        "internal": raw / "edges.csv",
        "boundary": processed / "node_classification/boundary_unique_neighbors.sqlite",
        "boundary_provenance": processed / "node_classification/node_boundary_metadata.json",
        "member_attributes": processed / "eda/labeled_internal_nodes.csv",
        "boundary_attributes": processed / "node_classification/onehop_neighbor_attributes.csv",
    }
    fingerprints = {key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                    for key, path in sources.items()}

    labels = pd.read_csv(sources["labels"])
    labels["y"] = labels.ccLabel.map({"licit": 0, "suspicious": 1})
    members = pd.read_csv(sources["members"], usecols=["clId", "ccId"])
    internal = pd.read_csv(sources["internal"], usecols=["clId1", "clId2"])
    internal.columns = ["src", "dst"]
    uri = sources["boundary"].as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as con:
        boundary = pd.read_sql_query("SELECT node_id, direction, neighbor_id FROM unique_neighbors", con)

    boundary_meta = json.loads(sources["boundary_provenance"].read_text(encoding="utf-8"))
    if len(members) != boundary_meta["annotated_nodes"] or len(internal) != boundary_meta["internal_rows"]:
        raise ValueError("Public raw graph row counts differ from the boundary extraction inventory")
    # Validate per-member degrees against the paired extraction output, including members of degree zero.
    degrees_path = processed / "node_classification/node_boundary_features.csv"
    fingerprints["boundary_degrees"] = {"path": str(degrees_path.relative_to(ROOT)),
                                        "sha256": sha256(degrees_path)}
    degrees = pd.read_csv(degrees_path, usecols=["clId", "boundary_in_degree", "boundary_out_degree"])
    degrees = degrees.set_index("clId").sort_index()
    if not np.array_equal(degrees.index.to_numpy(), np.sort(members.clId.to_numpy())):
        raise ValueError("Cached degree inventory and public membership differ")
    actual = (boundary.groupby(["node_id", "direction"]).size().unstack(fill_value=0)
              .reindex(index=degrees.index, columns=[0, 1], fill_value=0))
    if (not boundary.direction.isin([0, 1]).all()
            or not np.array_equal(actual.to_numpy(), degrees.to_numpy())):
        raise ValueError("Boundary cache differs from the extracted per-node degrees")

    member_attrs, boundary_attrs = [
        pd.read_csv(sources[key], usecols=["clId", *FEATURES], index_col="clId")
        for key in ["member_attributes", "boundary_attributes"]
    ]
    shared = member_attrs.index.intersection(boundary_attrs.index)
    if (not member_attrs.index.is_unique or not boundary_attrs.index.is_unique
            or not np.array_equal(member_attrs.loc[shared, FEATURES].to_numpy(),
                                  boundary_attrs.loc[shared, FEATURES].to_numpy())):
        raise ValueError("Inconsistent node attribute caches")
    attrs = pd.concat([member_attrs, boundary_attrs.loc[~boundary_attrs.index.isin(shared)]])
    graphs = pack(labels, members, internal, boundary, attrs)
    return graphs, {"sources": fingerprints, "boundary_extraction": boundary_meta}


def describe(graphs):
    inventory = graphs.inventory()
    composition = {}
    for label, name in enumerate(["licit", "suspicious"]):
        group = inventory[inventory.y == label]
        composition[name] = {"graphs": len(group)}
        for column in [*ROLES, "nodes", "edges"]:
            composition[name][column] = [float(group[column].mean()), float(group[column].std(ddof=1))]
    return inventory, {"public_pool": composition, "paper_table2": TABLE2}

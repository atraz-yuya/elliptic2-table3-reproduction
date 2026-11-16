from pathlib import Path
import sqlite3
import time
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/elliptic2_raw"
OUT_DIR = ROOT / "data/processed/node_classification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP = RAW / "background_edges.csv.zip"
NODES = RAW / "nodes.csv"

OUTPUT = OUT_DIR / "node_boundary_features.csv"
METADATA = OUT_DIR / "node_boundary_metadata.json"
DB = OUT_DIR / "boundary_unique_neighbors.sqlite"

CHUNK_SIZE = 500_000

FEATURES = [
    "boundary_in_degree",
    "boundary_out_degree",
    "boundary_in_tx",
    "boundary_out_tx",
    "boundary_total_tx",
    "boundary_net_tx",
]


def main():
    for path in [OUTPUT, METADATA, DB]:
        if path.exists():
            raise FileExistsError(
                f"Existing file protected: {path}"
            )

    start = time.perf_counter()

    nodes = pd.read_csv(
        NODES,
        usecols=["clId", "ccId"],
        dtype={"clId": "int64", "ccId": "int64"}
    )

    if nodes["clId"].duplicated().any():
        raise ValueError("Duplicate clId in nodes.csv")

    nodes = nodes.sort_values("clId").reset_index(drop=True)

    ids = nodes["clId"].to_numpy(dtype=np.int64)
    ccs = nodes["ccId"].to_numpy(dtype=np.int64)
    index = pd.Index(ids)

    n = len(nodes)

    incoming = np.zeros(n, dtype=np.int64)
    outgoing = np.zeros(n, dtype=np.int64)

    print("=== Input ===", flush=True)
    print("Annotated nodes:", n, flush=True)
    print("ZIP:", ZIP, flush=True)

    conn = sqlite3.connect(DB)

    conn.execute(
        """
        CREATE TABLE unique_neighbors (
            node_id INTEGER NOT NULL,
            direction INTEGER NOT NULL,
            neighbor_id INTEGER NOT NULL,
            PRIMARY KEY (node_id, direction, neighbor_id)
        ) WITHOUT ROWID
        """
    )

    total_rows = 0
    boundary_rows = 0
    internal_rows = 0
    other_rows = 0

    try:
        reader = pd.read_csv(
            ZIP,
            compression="zip",
            usecols=["clId1", "clId2", "txId"],
            dtype={
                "clId1": "int64",
                "clId2": "int64",
                "txId": "int64",
            },
            chunksize=CHUNK_SIZE
        )

        for chunk_number, chunk in enumerate(reader, 1):
            src = chunk["clId1"].to_numpy()
            dst = chunk["clId2"].to_numpy()

            src_pos = index.get_indexer(src)
            dst_pos = index.get_indexer(dst)

            src_known = src_pos >= 0
            dst_known = dst_pos >= 0

            both_known = src_known & dst_known

            same_component = np.zeros(
                len(chunk), dtype=bool
            )

            if both_known.any():
                same_component[both_known] = (
                    ccs[src_pos[both_known]]
                    == ccs[dst_pos[both_known]]
                )

            # サブグラフ外との接続のみ対象
            external_edge = ~same_component

            source_boundary = src_known & external_edge
            target_boundary = dst_known & external_edge

            # 送信元が対象ノード → Boundary outgoing
            np.add.at(
                outgoing,
                src_pos[source_boundary],
                1
            )

            # 受信先が対象ノード → Boundary incoming
            np.add.at(
                incoming,
                dst_pos[target_boundary],
                1
            )

            # 外部接続先の重複を正確に除去する
            # direction 0: incoming
            # direction 1: outgoing
            records = []

            records.extend(
                zip(
                    dst[target_boundary].tolist(),
                    [0] * int(target_boundary.sum()),
                    src[target_boundary].tolist()
                )
            )

            records.extend(
                zip(
                    src[source_boundary].tolist(),
                    [1] * int(source_boundary.sum()),
                    dst[source_boundary].tolist()
                )
            )

            if records:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO unique_neighbors
                    (node_id, direction, neighbor_id)
                    VALUES (?, ?, ?)
                    """,
                    records
                )

            conn.commit()

            total_rows += len(chunk)
            internal_rows += int(same_component.sum())

            boundary_rows += int(
                (source_boundary | target_boundary).sum()
            )

            other_rows += int(
                (~src_known & ~dst_known).sum()
            )

            if chunk_number % 20 == 0:
                elapsed = time.perf_counter() - start

                print(
                    f"Chunks={chunk_number:,} "
                    f"rows={total_rows:,} "
                    f"boundary_rows={boundary_rows:,} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True
                )

        print(
            "\n=== Counting unique external neighbors ===",
            flush=True
        )

        # SQLで重複を除いた相手ノード数を取得
        # 結果は最大でも各ノード×2方向
        degrees = pd.read_sql_query(
            """
            SELECT node_id, direction, COUNT(*) AS degree
            FROM unique_neighbors
            GROUP BY node_id, direction
            """,
            conn
        )

    finally:
        conn.close()

    result = nodes.copy()

    result["boundary_in_tx"] = incoming
    result["boundary_out_tx"] = outgoing

    result["boundary_in_degree"] = 0
    result["boundary_out_degree"] = 0

    if not degrees.empty:
        in_deg = degrees[
            degrees["direction"] == 0
        ].set_index("node_id")["degree"]

        out_deg = degrees[
            degrees["direction"] == 1
        ].set_index("node_id")["degree"]

        result["boundary_in_degree"] = (
            result["clId"].map(in_deg)
            .fillna(0).astype("int64")
        )

        result["boundary_out_degree"] = (
            result["clId"].map(out_deg)
            .fillna(0).astype("int64")
        )

    result["boundary_total_tx"] = (
        result["boundary_in_tx"]
        + result["boundary_out_tx"]
    )

    result["boundary_net_tx"] = (
        result["boundary_out_tx"]
        - result["boundary_in_tx"]
    )

    # 基本的な整合性検証
    if (
        result["boundary_in_degree"]
        > result["boundary_in_tx"]
    ).any():
        raise ValueError("Incoming degree exceeds tx count")

    if (
        result["boundary_out_degree"]
        > result["boundary_out_tx"]
    ).any():
        raise ValueError("Outgoing degree exceeds tx count")

    if result[FEATURES].isna().any().any():
        raise ValueError("Missing structure features")

    result.to_csv(OUTPUT, index=False)

    elapsed = time.perf_counter() - start

    metadata = {
        "source": str(ZIP),
        "chunk_size": CHUNK_SIZE,
        "annotated_nodes": n,
        "background_edges_scanned": total_rows,
        "internal_rows": internal_rows,
        "boundary_rows": boundary_rows,
        "other_rows": other_rows,
        "features": FEATURES,
        "elapsed_seconds": elapsed,
        "test_labels_used": False,
        "definition": (
            "Directed background edges crossing "
            "the annotated subgraph boundary"
        )
    }

    METADATA.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\n=== FINAL SUMMARY ===")
    print("Scanned edges:", f"{total_rows:,}")
    print("Internal edges:", f"{internal_rows:,}")
    print("Boundary edges:", f"{boundary_rows:,}")
    print("Other edges:", f"{other_rows:,}")

    print("\n=== Boundary feature statistics ===")
    print(result[FEATURES].describe().to_string())

    print(f"\nElapsed: {elapsed:.2f}s")
    print("Saved:", OUTPUT)
    print("Saved:", METADATA)
    print("SQLite work database:", DB)


if __name__ == "__main__":
    main()

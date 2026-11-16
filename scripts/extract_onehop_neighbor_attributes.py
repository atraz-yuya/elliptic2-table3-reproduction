from pathlib import Path
import sqlite3
import json
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/elliptic2_raw"

DATA = ROOT / "data/processed/node_classification"

DB = DATA / "boundary_unique_neighbors.sqlite"

SOURCE = RAW / "background_nodes.csv"

OUTPUT = DATA / "onehop_neighbor_attributes.csv"
METADATA = DATA / "onehop_neighbor_attributes_metadata.json"

TEMP = DATA / "onehop_neighbor_attributes.partial.csv"

FEATURES = [f"feat#{i}" for i in range(1, 44)]

CHUNK_SIZE = 200_000


def main():
    for path in [OUTPUT, METADATA, TEMP]:
        if path.exists():
            raise FileExistsError(
                f"Existing file protected: {path}"
            )

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    if not DB.exists():
        raise FileNotFoundError(DB)

    start = time.perf_counter()

    print("=== Loading required neighbor IDs ===", flush=True)

    conn = sqlite3.connect(
        f"file:{DB.resolve()}?mode=ro",
        uri=True
    )

    try:
        neighbor_ids = pd.read_sql_query(
            """
            SELECT DISTINCT neighbor_id
            FROM unique_neighbors
            ORDER BY neighbor_id
            """,
            conn
        )["neighbor_id"].to_numpy(dtype=np.int64)
    finally:
        conn.close()

    if len(neighbor_ids) == 0:
        raise ValueError("No neighbor IDs found")

    if len(np.unique(neighbor_ids)) != len(neighbor_ids):
        raise ValueError("Duplicate neighbor IDs")

    targets = pd.Index(neighbor_ids)

    print("Required unique neighbors:", len(targets))
    print("Source:", SOURCE)
    print("Chunk size:", CHUNK_SIZE)

    # CSV構造を先に検証
    header = pd.read_csv(SOURCE, nrows=0).columns

    required = ["clId", *FEATURES]

    missing = sorted(set(required) - set(header))

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    dtypes = {"clId": "int64"}
    dtypes.update({
        feature: "int32"
        for feature in FEATURES
    })

    scanned = 0
    extracted = 0
    chunks = 0
    written = False

    print("\n=== Scanning background nodes ===", flush=True)

    try:
        reader = pd.read_csv(
            SOURCE,
            usecols=required,
            dtype=dtypes,
            chunksize=CHUNK_SIZE
        )

        for chunk in reader:
            chunks += 1
            scanned += len(chunk)

            mask = chunk["clId"].isin(targets)

            selected = chunk.loc[
                mask, required
            ]

            if not selected.empty:
                selected.to_csv(
                    TEMP,
                    mode="a",
                    header=not written,
                    index=False
                )

                extracted += len(selected)
                written = True

            if chunks % 25 == 0:
                elapsed = time.perf_counter() - start

                print(
                    f"Scanned={scanned:,} "
                    f"Extracted={extracted:,} "
                    f"Elapsed={elapsed:.1f}s",
                    flush=True
                )

        if not written:
            raise ValueError("No matching neighbor attributes")

        print("\n=== Validation ===", flush=True)

        # 抽出したID列だけ再読込して検証
        found = pd.read_csv(
            TEMP,
            usecols=["clId"],
            dtype={"clId": "int64"}
        )["clId"]

        if found.duplicated().any():
            raise ValueError(
                "Duplicate clId in extracted attributes"
            )

        if len(found) != extracted:
            raise ValueError("Extracted row count mismatch")

        found_ids = pd.Index(found)

        missing_ids = targets.difference(found_ids)

        print("Required neighbors:", len(targets))
        print("Extracted neighbors:", len(found_ids))
        print("Missing neighbors:", len(missing_ids))

        if len(missing_ids) > 0:
            print(
                "First missing IDs:",
                missing_ids[:10].tolist()
            )
            raise ValueError(
                "Some neighbor IDs are absent from background_nodes"
            )

        # 全検証が完了してから正式ファイルにする
        TEMP.rename(OUTPUT)

    except Exception:
        print(
            "\nExtraction interrupted. "
            "Partial file is preserved for inspection.",
            flush=True
        )
        raise

    elapsed = time.perf_counter() - start

    metadata = {
        "source": str(SOURCE),
        "neighbor_source": str(DB),
        "required_unique_neighbors": len(targets),
        "scanned_background_nodes": scanned,
        "extracted_neighbor_nodes": extracted,
        "feature_count": len(FEATURES),
        "chunk_size": CHUNK_SIZE,
        "elapsed_seconds": elapsed,
        "labels_used": False,
        "test_labels_used": False,
        "description": (
            "Attributes of unique neighbors across "
            "annotated subgraph boundaries"
        )
    }

    METADATA.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== FINAL SUMMARY ===")
    print(f"Scanned background nodes: {scanned:,}")
    print(f"Unique external neighbors: {extracted:,}")
    print(f"Attributes per node: {len(FEATURES)}")
    print(f"Elapsed: {elapsed:.2f}s")

    print("\nSaved:")
    print(OUTPUT)
    print(METADATA)


if __name__ == "__main__":
    main()

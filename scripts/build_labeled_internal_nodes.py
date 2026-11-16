from pathlib import Path
import json
import os
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/elliptic2_raw"
OUT = ROOT / "data/processed/eda"

INPUT_NODES = RAW / "nodes.csv"
INPUT_LABELS = RAW / "connected_components.csv"
INPUT_FEATURES = RAW / "background_nodes.csv"

OUTPUT = OUT / "labeled_internal_nodes.csv"
META = OUT / "labeled_internal_nodes_metadata.json"

CHUNK_SIZE = 200_000
FEATURES = [f"feat#{i}" for i in range(1, 44)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if OUTPUT.exists() or META.exists():
        raise FileExistsError(
            "結果ファイルが既に存在します。"
            "既存のデータを保護するため停止します。"
        )

    start = time.perf_counter()

    print("=== Loading labels and membership ===", flush=True)

    nodes = pd.read_csv(
        INPUT_NODES,
        usecols=["clId", "ccId"],
        dtype={"clId": "int64", "ccId": "int64"}
    )

    labels = pd.read_csv(
        INPUT_LABELS,
        usecols=["ccId", "ccLabel"]
    )

    if nodes["clId"].duplicated().any():
        raise ValueError("Duplicate clId in nodes.csv")

    if labels["ccId"].duplicated().any():
        raise ValueError("Duplicate ccId in labels")

    labeled = nodes.merge(
        labels,
        on="ccId",
        how="left",
        validate="many_to_one"
    )

    if labeled["ccLabel"].isna().any():
        raise ValueError("Some nodes have no label")

    expected_labels = {"licit", "suspicious"}
    if set(labeled["ccLabel"]) != expected_labels:
        raise ValueError(
            f"Unexpected labels: {set(labeled['ccLabel'])}"
        )

    print("Labeled nodes:", len(labeled), flush=True)
    print(
        labeled["ccLabel"].value_counts().to_string(),
        flush=True
    )

    # 高速に所属ノードを検索するためのID集合
    target_ids = set(labeled["clId"])

    # 小さい対応表を索引として保持
    membership = labeled.set_index("clId")[
        ["ccId", "ccLabel"]
    ]

    total_scanned = 0
    total_matched = 0
    seen_ids = set()

    temp_output = OUTPUT.with_suffix(".csv.tmp")

    try:
        with temp_output.open("w", encoding="utf-8") as f:
            wrote_header = False

            reader = pd.read_csv(
                INPUT_FEATURES,
                chunksize=CHUNK_SIZE
            )

            for chunk in reader:
                total_scanned += len(chunk)

                selected = chunk.loc[
                    chunk["clId"].isin(target_ids)
                ].copy()

                if not selected.empty:
                    if selected["clId"].duplicated().any():
                        raise ValueError(
                            "Duplicate selected clId in chunk"
                        )

                    ids = set(selected["clId"])

                    if seen_ids.intersection(ids):
                        raise ValueError(
                            "Duplicate clId across chunks"
                        )

                    seen_ids.update(ids)

                    selected = selected.join(
                        membership,
                        on="clId",
                        how="left",
                        validate="many_to_one"
                    )

                    if selected["ccLabel"].isna().any():
                        raise ValueError(
                            "Missing label after join"
                        )

                    columns = [
                        "clId", "ccId", "ccLabel", *FEATURES
                    ]

                    selected = selected[columns]

                    selected.to_csv(
                        f,
                        index=False,
                        header=not wrote_header
                    )

                    wrote_header = True
                    total_matched += len(selected)

                if total_scanned % 5_000_000 < CHUNK_SIZE:
                    print(
                        f"Scanned: {total_scanned:,} | "
                        f"Matched: {total_matched:,}",
                        flush=True
                    )

        if total_matched != len(labeled):
            missing = target_ids - seen_ids
            raise ValueError(
                f"Expected {len(labeled)} nodes, "
                f"found {total_matched}. "
                f"Missing IDs: {len(missing)}"
            )

        if len(seen_ids) != len(labeled):
            raise ValueError("Unique node count mismatch")

        # 完全に構築・検証できてから正式ファイル名に変更
        os.replace(temp_output, OUTPUT)

    except Exception:
        if temp_output.exists():
            temp_output.unlink()
        raise

    elapsed = time.perf_counter() - start

    metadata = {
        "source_rows_scanned": total_scanned,
        "labeled_nodes": total_matched,
        "n_features": len(FEATURES),
        "label_counts": {
            k: int(v)
            for k, v in labeled["ccLabel"].value_counts().items()
        },
        "elapsed_seconds": elapsed,
        "label_meaning": (
            "Membership in a labeled subgraph; "
            "not the illicit status of an individual node."
        )
    }

    META.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== COMPLETED ===")
    print(f"Scanned background nodes: {total_scanned:,}")
    print(f"Labeled nodes: {total_matched:,}")
    print(f"Columns: {3 + len(FEATURES)}")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print("Saved:", OUTPUT)
    print("Metadata:", META)


if __name__ == "__main__":
    main()

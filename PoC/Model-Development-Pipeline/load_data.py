# PoC/Model-Development-Pipeline/load_data.py
import os
import pandas as pd
import pandas_gbq

# ------------------ CONFIG ------------------
PROJECT   = "erudite-carving-472018-r5"
DATASETS  = ["curated_demo", "federated_demo"]  # export both datasets as-is
OUT_ROOT  = r"PoC\Data-Pipeline\data"           # output root (Windows-safe path)
# --------------------------------------------

os.makedirs(OUT_ROOT, exist_ok=True)

def gbq(query: str) -> pd.DataFrame:
    """Run a BigQuery SQL and return a pandas DataFrame (no schema changes)."""
    return pandas_gbq.read_gbq(query, project_id=PROJECT, dialect="standard")

def list_tables(dataset: str) -> list:
    """List all base tables and views in a dataset."""
    q = f"""
    SELECT table_name
    FROM `{PROJECT}.{dataset}`.INFORMATION_SCHEMA.TABLES
    WHERE table_type IN ('BASE TABLE', 'VIEW')
    ORDER BY table_name
    """
    df = gbq(q)
    return df["table_name"].tolist()

def export_table(dataset: str, table: str) -> None:
    """Export a single table/view as CSV and Parquet without altering columns."""
    print(f"→ Exporting {dataset}.{table}")
    df = gbq(f"SELECT * FROM `{PROJECT}.{dataset}.{table}`")

    out_dir = os.path.join(OUT_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{table}.csv")
    pq_path  = os.path.join(out_dir, f"{table}.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    print(f"   Saved: {csv_path}")
    print(f"          {pq_path}  (rows={len(df)}, cols={len(df.columns)})")

def main():
    for ds in DATASETS:
        print(f"\n=== DATASET: {ds} ===")
        tables = list_tables(ds)
        if not tables:
            print(f"   (No tables found in {ds})")
            continue
        for tbl in tables:
            export_table(ds, tbl)
    print("\n✓ Finished exporting curated_demo and federated_demo (no column changes).")

if __name__ == "__main__":
    main()

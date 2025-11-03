"""
Exports BigQuery tables to CSV and Parquet without altering schemas.
Reads datasets from config and writes into DATA_OUT_ROOT.
"""

import os
import pandas as pd
import pandas_gbq
from config import GBQ_PROJECT as PROJECT, GBQ_DATASETS as DATASETS, DATA_OUT_ROOT as OUT_ROOT

os.makedirs(OUT_ROOT, exist_ok=True)


def gbq(query: str) -> pd.DataFrame:
    """
    Execute a BigQuery SQL query and return a pandas DataFrame.

    Args:
        query: Standard SQL query string.

    Returns:
        DataFrame containing the query result.
    """
    return pandas_gbq.read_gbq(query, project_id=PROJECT, dialect="standard")


def list_tables(dataset: str) -> list:
    """
    List table and view names within a BigQuery dataset.

    Args:
        dataset: Dataset name within the configured project.

    Returns:
        List of table names as strings.
    """
    q = f"""
    SELECT table_name
    FROM `{PROJECT}.{dataset}`.INFORMATION_SCHEMA.TABLES
    WHERE table_type IN ('BASE TABLE', 'VIEW')
    ORDER BY table_name
    """
    df = gbq(q)
    return df["table_name"].tolist()

def export_table(dataset: str, table: str) -> None:
    """
    Export a single table/view to CSV and Parquet.

    Args:
        dataset: Dataset name.
        table: Table or view name.

    Returns:
        None. Writes files to disk.
    """
    print(f"Exporting {dataset}.{table}")
    df = gbq(f"SELECT * FROM `{PROJECT}.{dataset}.{table}`")

    out_dir = os.path.join(OUT_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{table}.csv")
    pq_path = os.path.join(out_dir, f"{table}.parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    print(f"Saved: {csv_path}")
    print(f"Saved: {pq_path}  (rows={len(df)}, cols={len(df.columns)})")


def main() -> None:
    """
    Export all tables for each dataset listed in config.
    """
    for ds in DATASETS:
        print(f"DATASET: {ds}")
        tables = list_tables(ds)
        if not tables:
            print(f"No tables found in {ds}")
            continue
        for tbl in tables:
            export_table(ds, tbl)
    print("Finished exporting datasets.")


if __name__ == "__main__":
    main()

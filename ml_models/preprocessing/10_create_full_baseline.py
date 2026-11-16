import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TEMP_DIR = ROOT / "datasets" / "processed" / "baseline_temp"
OUTPUT_DIR = ROOT / "datasets" / "processed" / "cicids2017_baseline"

OUTPUT_FILE = OUTPUT_DIR / "baseline_all_records.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if OUTPUT_FILE.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT_FILE}"
    )

temp_glob = (TEMP_DIR / "**" / "*.parquet").as_posix()
output_path = OUTPUT_FILE.as_posix()

con = duckdb.connect()

con.execute(f"""
    COPY (
        SELECT *
        FROM read_parquet('{temp_glob}', union_by_name = true)
    )
    TO '{output_path}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

count = con.execute(
    f"SELECT COUNT(*) FROM read_parquet('{output_path}')"
).fetchone()[0]

print("=" * 60)
print("CYBERSHIELD AI - FULL BASELINE CREATED")
print("=" * 60)

print(f"Records: {count:,}")
print(f"Expected: 2,823,723")
print(f"Output: {OUTPUT_FILE}")

if count != 2_823_723:
    raise RuntimeError("Record count does not match expected value.")

print("\nStep 10 completed successfully.")

con.close()

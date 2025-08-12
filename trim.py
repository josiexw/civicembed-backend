import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# === Config ===
EMBED_DIR   = "data"
FILES       = ["topography", "water", "roads", "vegetation"]
OUT_SUFFIX  = "_trimmed.parquet"        # output name: <file>_trimmed.parquet
BATCH_SIZE  = 100_000

# Switzerland bbox
LAT_MIN, LAT_MAX = 45.8, 47.9
LON_MIN, LON_MAX = 5.96, 10.49

def inside_bbox(batch: pa.RecordBatch) -> pa.ChunkedArray:
    lat = batch.column(batch.schema.get_field_index("lat"))
    lon = batch.column(batch.schema.get_field_index("lon"))
    return pc.and_(
        pc.and_(pc.greater_equal(lat, pa.scalar(LAT_MIN)),
                pc.less_equal(lat,  pa.scalar(LAT_MAX))),
        pc.and_(pc.greater_equal(lon, pa.scalar(LON_MIN)),
                pc.less_equal(lon,  pa.scalar(LON_MAX))),
    )

# ---------- Step 1: count in-bbox rows per file ----------
counts = {}
total_rows = {}
for name in FILES:
    pf = pq.ParquetFile(os.path.join(EMBED_DIR, f"{name}.parquet"))
    total_rows[name] = pf.metadata.num_rows
    kept = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        mask = inside_bbox(batch)
        kept += int(pc.sum(pc.cast(mask, pa.int64())).as_py())
    counts[name] = kept

print("Total rows per file:", {k: f"{v:,}" for k, v in total_rows.items()})
print("Rows inside CH bbox:", {k: f"{v:,}" for k, v in counts.items()})

min_count = min(counts.values())
print(f"\nWill write up to the minimum kept rows across files: {min_count:,}")

# ---------- Step 2: write first `min_count` in-bbox rows to *_trimmed ----------
for name in FILES:
    in_path  = os.path.join(EMBED_DIR, f"{name}.parquet")
    out_path = os.path.join(EMBED_DIR, f"{name}{OUT_SUFFIX}")

    pf = pq.ParquetFile(in_path)
    remaining = min_count
    writer = None
    written = 0

    print(f"\nProcessing {name} → {out_path}")

    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        if remaining <= 0:
            break

        mask = inside_bbox(batch)
        # Quickly skip if no rows in bbox
        if pc.any(mask).as_py() is False:
            continue

        filtered = batch.filter(mask)

        # Cap to remaining
        if len(filtered) > remaining:
            filtered = filtered.slice(0, remaining)

        if writer is None:
            writer = pq.ParquetWriter(out_path, filtered.schema)

        writer.write_batch(filtered)
        written += len(filtered)
        remaining -= len(filtered)
        print(f"  wrote {written:,}/{min_count:,}", end="\r")

    if writer is not None:
        writer.close()

    print(f"  Done. Saved {written:,} rows.")

print("\nAll trimmed files written to bbox-aligned, equal-length Parquets.")

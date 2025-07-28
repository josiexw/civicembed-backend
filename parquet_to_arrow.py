import pyarrow.parquet as pq
import pyarrow as pa

# Read parquet
table = pq.read_table("all_vegetation_embeddings.parquet")

# Save to Arrow IPC format
with pa.OSFile("all_vegetation_embeddings.arrow", "wb") as sink:
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)

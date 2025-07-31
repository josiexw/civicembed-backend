import pyarrow.parquet as pq
import pyarrow as pa

parquet_path = "parquet_filename"
arrow_path = "arrow_filename"

pq_file = pq.ParquetFile(parquet_path)

# Remove embeddings column
all_columns = pq_file.schema.names
columns_to_read = [col for col in all_columns if col != "embeddings"]

# Arrow IPC writer
with pa.OSFile(arrow_path, "wb") as sink:
    writer = None
    for i in range(pq_file.num_row_groups):
        row_group_table = pq_file.read_row_group(i, columns=columns_to_read)
        if writer is None:
            writer = pa.ipc.new_stream(sink, row_group_table.schema)
        writer.write_table(row_group_table)

    if writer is not None:
        writer.close()

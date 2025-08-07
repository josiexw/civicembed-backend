import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

parquet_path = "data/all_terrain_embeddings.parquet"
arrow_path = "data/all_terrain_embeddings.arrow"

pq_file = pq.ParquetFile(parquet_path)

# Determine columns to read (exclude "embedding")
columns_to_read = [col for col in pq_file.schema.names if col != "embedding"]

with pa.OSFile(arrow_path, "wb") as sink:
    writer = None

    for i in range(pq_file.num_row_groups):
        # Read a row group without the embedding column
        row_group = pq_file.read_row_group(i, columns=columns_to_read + ["similarity"])

        # Modify similarity in-place
        similarity = row_group["similarity"]
        corrected_similarity = pc.if_else(
            pc.greater(similarity, 0.99),
            pa.array([0.8] * len(similarity), type=similarity.type),
            similarity
        )

        # Replace the similarity column
        row_group = row_group.set_column(
            row_group.schema.get_field_index("similarity"),
            "similarity",
            corrected_similarity
        )

        # Drop embedding if it still exists (precaution)
        if "embedding" in row_group.column_names:
            row_group = row_group.drop(["embedding"])

        # Write row group
        if writer is None:
            writer = pa.ipc.new_stream(sink, row_group.schema)
        writer.write_table(row_group)

    if writer is not None:
        writer.close()

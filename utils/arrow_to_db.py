import pyarrow as pa
import pyarrow.ipc as ipc
import sqlite3

def arrow_to_df(path):
    with pa.memory_map(path, "r") as source:
        reader = ipc.open_stream(source)
        table = reader.read_all()
    return table.to_pandas()

topography_df = arrow_to_df("data/topography.arrow")
water_df = arrow_to_df("data/water.arrow")
vegetation_df = arrow_to_df("data/vegetation.arrow")
road_df = arrow_to_df("data/roads.arrow")

# === Write to SQLite database ===
conn = sqlite3.connect("data.db")

topography_df.to_sql("topography", conn, if_exists="replace", index=False)
water_df.to_sql("water", conn, if_exists="replace", index=False)
vegetation_df.to_sql("vegetation", conn, if_exists="replace", index=False)
road_df.to_sql("road", conn, if_exists="replace", index=False)

conn.close()
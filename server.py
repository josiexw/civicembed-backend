from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim
import sqlite3
import pandas as pd
import numpy as np
from flask_cors import CORS
import faiss
import traceback

app = Flask(__name__)
CORS(app)
geolocator = Nominatim(user_agent="civicembed")

DB_PATH = "data.db"
SWITZERLAND_BOUNDS = {
    "north": 47.8084,
    "south": 45.818,
    "east": 10.4922,
    "west": 5.9559
}
BIN_SIZE = 0.001

def get_bounding_box(location_name):
    location = geolocator.geocode(location_name, exactly_one=True, addressdetails=False)
    if not location:
        raise ValueError(f"Location '{location_name}' not found.")
    lat, lon = location.latitude, location.longitude
    return {
        "north": min(lat + 0.05, SWITZERLAND_BOUNDS["north"]),
        "south": max(lat - 0.05, SWITZERLAND_BOUNDS["south"]),
        "east": min(lon + 0.05, SWITZERLAND_BOUNDS["east"]),
        "west": max(lon - 0.05, SWITZERLAND_BOUNDS["west"]),
    }

def clamp_bbox(bbox):
    return {
        "north": min(max(bbox["north"], SWITZERLAND_BOUNDS["south"]), SWITZERLAND_BOUNDS["north"]),
        "south": min(max(bbox["south"], SWITZERLAND_BOUNDS["south"]), SWITZERLAND_BOUNDS["north"]),
        "east": min(max(bbox["east"], SWITZERLAND_BOUNDS["west"]), SWITZERLAND_BOUNDS["east"]),
        "west": min(max(bbox["west"], SWITZERLAND_BOUNDS["west"]), SWITZERLAND_BOUNDS["east"]),
    }

def are_adjacent(cell1, cell2, cell_size=0.05):
    return (
        abs(cell1["lat"] - cell2["lat"]) <= cell_size and
        abs(cell1["lng"] - cell2["lng"]) <= cell_size and
        (cell1["lat"] != cell2["lat"] or cell1["lng"] != cell2["lng"])
    )

def merge_bounding_boxes(cells):
    lats = [cell["lat"] for cell in cells]
    lngs = [cell["lng"] for cell in cells]
    cs = 0.05
    return {
        "north": max(lats) + cs / 2,
        "south": min(lats) - cs / 2,
        "east": max(lngs) + cs / 2,
        "west": min(lngs) - cs / 2,
    }

def group_adjacent_cells(df, topK, cell_size=0.05):
    visited = np.zeros(len(df), dtype=bool)
    groups = []
    for i, row in df.iterrows():
        if visited[i]:
            continue
        group = [row]
        visited[i] = True
        for j in range(i + 1, len(df)):
            if visited[j]:
                continue
            if any(are_adjacent(r, df.iloc[j], cell_size) for r in group):
                group.append(df.iloc[j])
                visited[j] = True
        groups.append(group)
        if len(groups) >= topK:
            break
    return [merge_bounding_boxes(group) for group in groups]

def bin_latlon(df, bin_size=BIN_SIZE):
    df["lat_bin"] = (df.index.get_level_values("lat") / bin_size).round().astype(int)
    df["lng_bin"] = (df.index.get_level_values("lng") / bin_size).round().astype(int)
    df = df.reset_index(drop=True).groupby(["lat_bin", "lng_bin"]).mean()
    return df

def load_and_bin_lens_data(conn, lens, bbox=None):
    if bbox:
        query = f"SELECT * FROM {lens} WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
        params = (bbox["south"], bbox["north"], bbox["west"], bbox["east"])
    else:
        query = f"SELECT * FROM {lens}"
        params = ()
    df = pd.read_sql_query(query, conn, params=params)
    df.rename(columns={"lon": "lng"}, inplace=True)
    df.set_index(["lat", "lng"], inplace=True)
    df = bin_latlon(df)
    df.columns = [f"{lens}_{col}" for col in df.columns]
    return df

@app.route("/api/search", methods=["POST"])
def search_topk_cells():
    data = request.get_json()
    location = data["location"]
    topK = int(data["topK"])
    lenses = [l.lower() for l in data.get("lens", [])]

    try:
        bbox = get_bounding_box(location)
        conn = sqlite3.connect(DB_PATH)

        # Load and join local lens data
        dfs = [load_and_bin_lens_data(conn, lens, bbox) for lens in lenses]
        joined_df = pd.concat(dfs, axis=1, join="inner").dropna().reset_index()
        if joined_df.empty:
            return jsonify({"boundingBox": bbox, "topKCells": []})

        joined_df["lat"] = joined_df["lat_bin"] * BIN_SIZE
        joined_df["lng"] = joined_df["lng_bin"] * BIN_SIZE
        joined_df.drop(columns=["lat_bin", "lng_bin"], inplace=True)

        feature_cols = [col for col in joined_df.columns if col not in ("lat", "lng")]
        local_features = joined_df[feature_cols].to_numpy().astype("float32")
        query_vec = np.mean(local_features, axis=0).reshape(1, -1)

        # Load and join full data
        dfs_full = []
        lens_dims = {}
        for lens in lenses:
            df_lens = load_and_bin_lens_data(conn, lens)
            lens_dims[lens] = len(df_lens.columns)
            dfs_full.append(df_lens)
        conn.close()

        full_df = pd.concat(dfs_full, axis=1, join="outer").dropna().reset_index()
        if full_df.empty:
            return jsonify({"boundingBox": bbox, "topKCells": []})

        full_df["lat"] = full_df["lat_bin"] * BIN_SIZE
        full_df["lng"] = full_df["lng_bin"] * BIN_SIZE
        full_df.drop(columns=["lat_bin", "lng_bin"], inplace=True)

        full_features = full_df.drop(columns=["lat", "lng"]).to_numpy().astype("float32")

        index = faiss.IndexFlatL2(full_features.shape[1])
        index.add(full_features)
        distances, indices = index.search(query_vec, topK * 10)

        max_distance = max(distances[0].max(), 1e-6)
        similarities = 1 - distances[0] / max_distance

        topk_df = full_df.iloc[indices[0]].copy()
        topk_df["similarity"] = similarities
        candidate_df = topk_df.reset_index(drop=True)
        topk_bounding_boxes = group_adjacent_cells(candidate_df, topK)

        qvec = query_vec[0]
        lens_similarities = []

        for idx in indices[0]:
            lens_sim_entry = {}
            offset = 0
            for lens in lenses:
                dim = lens_dims[lens]
                vec = full_features[idx][offset:offset+dim]
                qvec_lens = qvec[offset:offset+dim]
                dist = np.linalg.norm(qvec_lens - vec)
                scale = np.linalg.norm(qvec_lens) + 1e-6
                sim = 1 - dist / scale
                lens_sim_entry[lens] = float(sim)
                offset += dim
            lens_similarities.append(lens_sim_entry)

        return jsonify({
            "boundingBox": bbox,
            "topKCells": [clamp_bbox(b) for b in topk_bounding_boxes],
            "similarities": similarities[:len(topk_bounding_boxes)].tolist(),
            "lensSimilarity": lens_similarities[:len(topk_bounding_boxes)]
        })

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim
import sqlite3
import pandas as pd
import numpy as np
from flask_cors import CORS
import time
import faiss

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

def are_adjacent(cell1, cell2, cell_size=0.05):
    lat_diff = abs(cell1["lat"] - cell2["lat"])
    lng_diff = abs(cell1["lng"] - cell2["lng"])
    return lat_diff <= cell_size and lng_diff <= cell_size and (lat_diff + lng_diff > 0)

def merge_bounding_boxes(cells):
    lats = [cell["lat"] for cell in cells]
    lngs = [cell["lng"] for cell in cells]
    cell_size = 0.05
    return {
        "north": max(lats) + cell_size / 2,
        "south": min(lats) - cell_size / 2,
        "east": max(lngs) + cell_size / 2,
        "west": min(lngs) - cell_size / 2,
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

def query_grid_data(lens, bbox):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT * FROM {lens}
    WHERE lat BETWEEN ? AND ?
    AND lon BETWEEN ? AND ?
    """
    df = pd.read_sql_query(query, conn, params=(bbox["south"], bbox["north"], bbox["west"], bbox["east"]))
    conn.close()
    return df

@app.route("/api/search", methods=["POST"])
def search_topk_cells():
    data = request.get_json()
    location = data["location"]
    topK = int(data["topK"])
    lens = data.get("lens").lower()

    try:
        start_total = time.time()

        bbox = get_bounding_box(location)
        local_df = query_grid_data(lens, bbox)

        if local_df.empty:
            return jsonify({"boundingBox": bbox, "topKCells": []})

        feature_cols = [col for col in local_df.columns if col not in ("lat", "lon")]
        local_features = local_df[feature_cols].to_numpy().astype("float32")
        query_vec = np.mean(local_features, axis=0).reshape(1, -1)

        conn = sqlite3.connect(DB_PATH)
        full_df = pd.read_sql_query(
            f"SELECT * FROM {lens} WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?",
            conn,
            params=(SWITZERLAND_BOUNDS["south"], SWITZERLAND_BOUNDS["north"],
                    SWITZERLAND_BOUNDS["west"], SWITZERLAND_BOUNDS["east"])
        )
        conn.close()

        full_df = full_df.rename(columns={"lon": "lng"})
        full_features = full_df[feature_cols].to_numpy().astype("float32")

        # Use FAISS for fast similarity search
        index = faiss.IndexFlatL2(full_features.shape[1])
        index.add(full_features)
        distances, indices = index.search(query_vec, topK * 10)

        topk_df = full_df.iloc[indices[0]].copy()
        topk_df["similarity"] = -distances[0]  # Negate distance to use as similarity

        candidate_df = topk_df.reset_index(drop=True)
        topk_bounding_boxes = group_adjacent_cells(candidate_df, topK)

        print("Search completed in", time.time() - start_total, "seconds")

        return jsonify({
            "boundingBox": bbox,
            "topKCells": topk_bounding_boxes
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

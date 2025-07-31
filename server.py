from flask import Flask, request, jsonify
from geopy.geocoders import Nominatim
import sqlite3
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
geolocator = Nominatim(user_agent="swiss-map")

DB_PATH = "data.db"
SWITZERLAND_BOUNDS = {
    "north": 47.6,
    "south": 45.9,
    "east": 10.3,
    "west": 6.1,
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
        bbox = get_bounding_box(location)
        df = query_grid_data(lens, bbox)

        if df.empty:
            return jsonify({"boundingBox": bbox, "topKCells": []})

        # Compute mean embedding inside bounding box
        features = df[[col for col in df.columns if col not in ("lat", "lon")]]
        mean_vec = features.mean(axis=0).to_numpy().reshape(1, -1)

        # Load all grid cells and compute similarity
        conn = sqlite3.connect(DB_PATH)
        full_df = pd.read_sql_query(f"SELECT * FROM {lens}", conn)
        conn.close()

        feature_cols = [col for col in full_df.columns if col not in ("lat", "lon")]
        all_features = full_df[feature_cols]
        similarities = cosine_similarity(all_features, mean_vec).flatten()

        full_df["similarity"] = similarities
        full_df = full_df.rename(columns={"lon": "lng"})

        # Filter to Swiss bounds first
        swiss_df = full_df[
            (full_df["lat"] >= SWITZERLAND_BOUNDS["south"]) &
            (full_df["lat"] <= SWITZERLAND_BOUNDS["north"]) &
            (full_df["lng"] >= SWITZERLAND_BOUNDS["west"]) &
            (full_df["lng"] <= SWITZERLAND_BOUNDS["east"])
        ]

        # Sort and pick top K within Switzerland
        topk_df = swiss_df.sort_values("similarity", ascending=False).head(topK)

        return jsonify({
            "boundingBox": bbox,
            "topKCells": topk_df[["lat", "lng"]].to_dict(orient="records")
        })

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

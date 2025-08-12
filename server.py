from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import pandas as pd
import numpy as np
import faiss
import traceback

app = Flask(__name__)
CORS(app)

DB_PATH = "data.db"
SWITZERLAND_BOUNDS = {
    "north": 47.8084,
    "south": 45.818,
    "east": 10.4922,
    "west": 5.9559
}
BIN_SIZE = 0.001  # degrees

def clamp_bbox(bbox):
    return {
        "north": min(max(float(bbox["north"]), SWITZERLAND_BOUNDS["south"]), SWITZERLAND_BOUNDS["north"]),
        "south": min(max(float(bbox["south"]), SWITZERLAND_BOUNDS["south"]), SWITZERLAND_BOUNDS["north"]),
        "east":  min(max(float(bbox["east"]),  SWITZERLAND_BOUNDS["west"]),  SWITZERLAND_BOUNDS["east"]),
        "west":  min(max(float(bbox["west"]),  SWITZERLAND_BOUNDS["west"]),  SWITZERLAND_BOUNDS["east"]),
    }

def bin_latlon(df, bin_size=BIN_SIZE):
    df["lat_bin"] = (df.index.get_level_values("lat") / bin_size).round().astype(int)
    df["lng_bin"] = (df.index.get_level_values("lng") / bin_size).round().astype(int)
    df = df.reset_index(drop=True).groupby(["lat_bin", "lng_bin"]).mean(numeric_only=True)
    return df

def load_and_bin_lens_data(conn, lens, bbox=None):
    if bbox:
        query = f"SELECT * FROM {lens} WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
        params = (float(bbox["south"]), float(bbox["north"]), float(bbox["west"]), float(bbox["east"]))
    else:
        query = f"SELECT * FROM {lens}"
        params = ()
    df = pd.read_sql_query(query, conn, params=params)
    df.rename(columns={"lon": "lng"}, inplace=True)
    df.set_index(["lat", "lng"], inplace=True)
    df = bin_latlon(df)
    df.columns = [f"{lens}_{col}" for col in df.columns]
    return df

def make_box_centered(lat: float, lng: float, side_deg: float) -> dict:
    half = side_deg / 2.0
    return {
        "south": float(lat - half),
        "north": float(lat + half),
        "west":  float(lng - half),
        "east":  float(lng + half),
    }

def overlaps(b1: dict, b2: dict) -> bool:
    return not (b1["east"] < b2["west"] or b2["east"] < b1["west"] or
                b1["north"] < b2["south"] or b2["north"] < b1["south"])

def group_candidates_into_boxes(candidate_rows, size_deg: float, topK: int, lenses: list[str]):
    """
    candidate_rows: list of dicts with keys:
      lat, lng, similarity, lens_sim (dict per lens)
    Greedy, non-overlapping grouping: take highest-sim, form a box of side 'size_deg',
    absorb all candidates inside; score box by mean(similarity) of absorbed.
    Also average lens_sim dicts.
    """
    if not candidate_rows:
        return [], [], []

    rows = sorted(candidate_rows, key=lambda r: r["similarity"], reverse=True)

    used = [False] * len(rows)
    boxes, box_sims, box_lens_sims = [], [], []

    for i, r in enumerate(rows):
        if used[i]:
            continue
        box = make_box_centered(r["lat"], r["lng"], size_deg)

        inside_idx = []
        for j, s in enumerate(rows):
            if used[j]:
                continue
            if (box["south"] <= s["lat"] <= box["north"]) and (box["west"] <= s["lng"] <= box["east"]):
                inside_idx.append(j)

        if not inside_idx:
            continue

        if any(overlaps(box, b) for b in boxes):
            continue

        for j in inside_idx:
            used[j] = True

        sims = [float(rows[j]["similarity"]) for j in inside_idx]
        mean_sim = float(np.mean(sims)) if sims else float(r["similarity"])

        lens_accum = {ln: [] for ln in lenses}
        for j in inside_idx:
            lens_sim = rows[j].get("lens_sim", {})
            for ln in lenses:
                v = lens_sim.get(ln)
                if v is not None:
                    lens_accum[ln].append(float(v))
        lens_avg = {ln: (float(np.mean(v)) if len(v) else 0.0) for ln, v in lens_accum.items()}

        boxes.append(box)
        box_sims.append(mean_sim)
        box_lens_sims.append(lens_avg)

        if len(boxes) >= topK:
            break

    return boxes, box_sims, box_lens_sims

@app.route("/api/search", methods=["POST"])
def search_topk_cells():
    try:
        data = request.get_json()
        bbox = data.get("bbox")
        size = float(data.get("size", 0.02))
        topK = int(data["topK"])
        lenses = [l.lower() for l in data.get("lens", [])]

        if not bbox or not isinstance(bbox, dict):
            return jsonify({"error": "bbox is required"}), 400
        if not lenses:
            return jsonify({"error": "lens must be a non-empty list"}), 400

        bbox = {
            "north": float(bbox["north"]),
            "south": float(bbox["south"]),
            "east":  float(bbox["east"]),
            "west":  float(bbox["west"]),
        }
        if bbox["west"] > bbox["east"]:
            bbox["west"], bbox["east"] = bbox["east"], bbox["west"]
        if bbox["south"] > bbox["north"]:
            bbox["south"], bbox["north"] = bbox["north"], bbox["south"]

        conn = sqlite3.connect(DB_PATH)
        dfs_local = [load_and_bin_lens_data(conn, lens, bbox) for lens in lenses]
        joined_df = pd.concat(dfs_local, axis=1, join="inner").dropna().reset_index()
        if joined_df.empty:
            conn.close()
            return jsonify({"boundingBox": bbox, "topKCells": [], "similarities": [], "lensSimilarity": []})

        joined_df["lat"] = joined_df["lat_bin"] * BIN_SIZE
        joined_df["lng"] = joined_df["lng_bin"] * BIN_SIZE
        joined_df.drop(columns=["lat_bin", "lng_bin"], inplace=True)

        feature_cols = [col for col in joined_df.columns if col not in ("lat", "lng")]
        local_features = joined_df[feature_cols].to_numpy().astype("float32")
        # Query vector = mean of local features
        query_vec = np.mean(local_features, axis=0, dtype=np.float32).reshape(1, -1)

        dfs_full = []
        lens_dims = {}
        for lens in lenses:
            df_lens = load_and_bin_lens_data(conn, lens, bbox=None)
            lens_dims[lens] = len(df_lens.columns)
            dfs_full.append(df_lens)
        conn.close()

        full_df = pd.concat(dfs_full, axis=1, join="inner").dropna().reset_index()
        if full_df.empty:
            return jsonify({"boundingBox": bbox, "topKCells": [], "similarities": [], "lensSimilarity": []})

        full_df["lat"] = full_df["lat_bin"] * BIN_SIZE
        full_df["lng"] = full_df["lng_bin"] * BIN_SIZE
        full_df.drop(columns=["lat_bin", "lng_bin"], inplace=True)
        full_features = full_df.drop(columns=["lat", "lng"]).to_numpy().astype("float32")

        index = faiss.IndexFlatL2(full_features.shape[1])
        index.add(full_features)

        oversample = max(200, topK * 100)
        distances, indices = index.search(query_vec, oversample)
        max_distance = float(max(distances[0].max(), 1e-6))
        similarities = 1.0 - (distances[0] / max_distance)

        qvec = query_vec[0]
        candidate_rows = []
        for rank, idx in enumerate(indices[0]):
            cand = {
                "lat": float(full_df.iloc[idx]["lat"]),
                "lng": float(full_df.iloc[idx]["lng"]),
                "similarity": float(similarities[rank]),
                "lens_sim": {}
            }
            offset = 0
            for lens in lenses:
                dim = lens_dims[lens]
                vec = full_features[idx][offset:offset+dim]
                qvec_lens = qvec[offset:offset+dim]
                dist = float(np.linalg.norm(qvec_lens - vec))
                scale = float(np.linalg.norm(qvec_lens) + 1e-6)
                sim = 1.0 - dist / scale
                cand["lens_sim"][lens] = float(sim)
                offset += dim
            candidate_rows.append(cand)

        boxes, box_sims, box_lens_sims = group_candidates_into_boxes(
            candidate_rows, size_deg=size, topK=topK, lenses=lenses
        )

        return jsonify({
            "boundingBox": bbox,
            "requestedSize": size,
            "topKCells": [clamp_bbox(b) for b in boxes],
            "similarities": box_sims,
            "lensSimilarity": box_lens_sims
        })

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

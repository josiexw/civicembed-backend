import os
import numpy as np
import pyarrow.parquet as pq
import faiss
from itertools import combinations

# === Paths ===
embedding_dir = "data"
index_dir = "faiss_indexes"
metadata_path = "metadata.npz"
os.makedirs(index_dir, exist_ok=True)

# === Helper: Load embeddings ===
def load_embeddings(name):
    table = pq.read_table(f"{embedding_dir}/{name}.parquet", columns=["embedding", "lat", "lon"])
    embeddings = np.stack(table["embedding"].to_numpy(zero_copy_only=False)).astype("float32")
    lat = table["lat"].to_numpy()
    lon = table["lon"].to_numpy()
    return embeddings, lat, lon

# === Load all 4 sources ===
topography, lat, lon = load_embeddings("topography")
water, _, _ = load_embeddings("water")
roads, _, _ = load_embeddings("roads")
vegetation, _, _ = load_embeddings("vegetation")

# === Normalize ===
def normalize(x): return faiss.normalize_L2(x.copy())

topography = normalize(topography)
water = normalize(water)
roads = normalize(roads)
vegetation = normalize(vegetation)

# === Build all combinations ===
base = {
    "topography": topography,
    "water": water,
    "roads": roads,
    "vegetation": vegetation,
}

all_combos = {}
for r in range(1, 5):
    for combo in combinations(base.keys(), r):
        name = "+".join(combo)
        avg = np.mean([base[k] for k in combo], axis=0)
        all_combos[name] = normalize(avg)

# === Save metadata once ===
np.savez_compressed(
    metadata_path,
    lat=lat,
    lon=lon,
    count=len(lat)
)
print(f"Saved metadata to {metadata_path}")

# === Helper: Save FAISS index ===
def save_faiss_index(name, vectors):
    d = vectors.shape[1]
    nlist = 256  # number of Voronoi cells (adjust for data scale)
    m = 16       # number of subquantizers for PQ

    quantizer = faiss.IndexFlatIP(d)  # inner product = cosine when normalized
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

    index.train(vectors)
    index.add(vectors)
    index_path = os.path.join(index_dir, f"{name}.index")
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index: {index_path}")

# === Build & Save All 15 Indexes ===
for name, vecs in all_combos.items():
    save_faiss_index(name, vecs)
    
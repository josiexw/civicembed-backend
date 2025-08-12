from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, math, traceback
import numpy as np
import faiss
import pyarrow as pa
import pyarrow.parquet as pq

app = Flask(__name__)
CORS(app)

# --------------------------
# Paths/config
# --------------------------
EMBED_DIR   = "data"                 # where *_trimmed.parquet live (fallback only)
INDEX_DIR   = "faiss_indexes"        # prebuilt indexes like "topography+water.index"
META_PATH   = "metadata.npz"         # lat/lon aligned with FAISS ids (0..N-1)
DIM        = 128
BIN_SIZE    = 0.001

# --------------------------
# Small helpers
# --------------------------
def lenses_key(lenses: list[str]) -> str:
    return "+".join(sorted(lenses))

def index_path_for(lenses: list[str]) -> str:
    return os.path.join(INDEX_DIR, f"{lenses_key(lenses)}.index")

def open_pf_trimmed(name: str) -> pq.ParquetFile:
    path = os.path.join(EMBED_DIR, f"{name}_trimmed.parquet")
    if not os.path.exists(path):
        alt = os.path.join(EMBED_DIR, f"{name}.parquet")
        if not os.path.exists(alt):
            raise FileNotFoundError(f"Parquet for lens '{name}' not found at {path} or {alt}")
        path = alt
    return pq.ParquetFile(path)

def detect_embedding_col(pf: pq.ParquetFile, dim=128) -> str:
    try:
        schema = pf.schema_arrow
    except AttributeError:
        schema = pf.schema.to_arrow_schema()
    for field in schema:
        t = field.type
        if pa.types.is_fixed_size_list(t) and t.list_size == dim and pa.types.is_floating(t.value_type):
            return field.name
        if (pa.types.is_list(t) or pa.types.is_large_list(t)) and pa.types.is_floating(t.value_type):
            if field.name.lower() in ("embedding","embeddings"):
                return field.name
    for field in schema:
        t = field.type
        if (pa.types.is_list(t) or pa.types.is_large_list(t)) and pa.types.is_floating(t.value_type):
            return field.name
    raise RuntimeError(f"No embedding list column found in schema: {schema}")

def to_numpy_embeddings_firstcol(batch: pa.RecordBatch, dim: int) -> np.ndarray:
    if batch.num_columns != 1:
        raise ValueError(f"Expected 1 column, got {batch.num_columns}. Schema: {batch.schema}")
    arr = batch.column(0)
    t = arr.type
    if pa.types.is_fixed_size_list(t) and pa.types.is_floating(t.value_type):
        flat = np.asarray(arr.values.to_numpy(zero_copy_only=False))
        X = flat.reshape(len(arr), -1)
        if X.shape[1] != dim:
            raise ValueError(f"Embedding width {X.shape[1]} != {dim}")
        return X.astype(np.float32, copy=False)
    if (pa.types.is_list(t) or pa.types.is_large_list(t)) and pa.types.is_floating(t.value_type):
        offs = getattr(arr, "offsets", None) or getattr(arr, "value_offsets", None)
        if offs is not None and hasattr(offs, "to_numpy"):
            off = np.asarray(offs.to_numpy(), dtype=np.int64)
            lens = off[1:] - off[:-1]
            if lens.min() == dim and lens.max() == dim:
                flat = np.asarray(arr.values.to_numpy(zero_copy_only=False))
                return flat.reshape(len(arr), dim).astype(np.float32, copy=False)
        pylist = arr.to_pylist()
        X = np.empty((len(pylist), dim), dtype=np.float32)
        for i, row in enumerate(pylist):
            if len(row) != dim:
                raise ValueError(f"Row {i} len {len(row)} != {dim}")
            X[i] = np.asarray(row, dtype=np.float32)
        return X
    raise TypeError(f"Unsupported embedding type: {t}")

def l2_normalize_inplace(x: np.ndarray) -> None:
    faiss.normalize_L2(x)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = a / (np.linalg.norm(a) + 1e-9)
    bn = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(an, bn))

# --------------------------
# Load meta (lat/lon per row)
# --------------------------
META = np.load(META_PATH)
LAT = META["lat"].astype(np.float32, copy=False)
LON = META["lon"].astype(np.float32, copy=False)
N_ROWS = int(META["count"])
assert len(LAT) == len(LON) == N_ROWS, "metadata.npz inconsistent"

# --------------------------
# Cache FAISS indices + Arrow handles
# --------------------------
INDEX_CACHE: dict[str, faiss.Index] = {}
PF_CACHE: dict[str, tuple[pq.ParquetFile, str]] = {}

def get_index(lenses: list[str]) -> faiss.Index:
    key = lenses_key(lenses)
    if key in INDEX_CACHE:
        return INDEX_CACHE[key]
    path = index_path_for(lenses)
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found for lenses {lenses} at {path}")
    idx = faiss.read_index(path)
    INDEX_CACHE[key] = idx
    return idx

def get_pf_and_col(lens: str) -> tuple[pq.ParquetFile, str]:
    if lens in PF_CACHE:
        return PF_CACHE[lens]
    pf = open_pf_trimmed(lens)
    col = detect_embedding_col(pf, dim=DIM)
    PF_CACHE[lens] = (pf, col)
    return pf, col

# --------------------------
# Parquet range read (fallback only)
# --------------------------
def read_range_as_embeddings(pf: pq.ParquetFile, col: str, start: int, end: int) -> np.ndarray:
    want = max(0, end - start)
    if want <= 0:
        return np.empty((0, DIM), dtype=np.float32)
    out = []
    seen = 0
    read_rows = 0
    CHUNK = 200_000
    for b in pf.iter_batches(columns=[col], batch_size=CHUNK):
        left  = start - read_rows
        right = end   - read_rows
        if right <= 0:
            break
        if left >= len(b):
            read_rows += len(b)
            continue
        s = max(0, left)
        e = min(len(b), right)
        if e > s:
            bt = b.slice(s, e - s)
            out.append(to_numpy_embeddings_firstcol(bt, dim=DIM))
            seen += (e - s)
        read_rows += len(b)
        if seen >= want:
            break
    if not out:
        return np.empty((0, DIM), dtype=np.float32)
    return np.vstack(out).astype(np.float32, copy=False)

# --------------------------
# Build query vector: **FAISS reconstruct path**
# --------------------------
def build_query_vector_from_bbox(bbox: dict, lenses: list[str], index: faiss.Index) -> np.ndarray:
    # sanitize bbox
    b = dict(bbox)
    if b["west"] > b["east"]:
        b["west"], b["east"] = b["east"], b["west"]
    if b["south"] > b["north"]:
        b["south"], b["north"] = b["north"], b["south"]

    # 1) ids in bbox
    mask = (
        (LAT >= float(b["south"])) & (LAT <= float(b["north"])) &
        (LON >= float(b["west"]))  & (LON <= float(b["east"]))
    )
    ids = np.nonzero(mask)[0]
    if ids.size == 0:
        print("[query] bbox contains 0 rows")
        return np.empty((0, DIM), dtype=np.float32)

    # 2) reconstruct vectors from the FAISS index (same representation used to search)
    rng = np.random.default_rng(123)
    MAX_IDS = 50_000  # cap for speed; raise if you want even smoother averages
    if ids.size > MAX_IDS:
        ids = np.sort(rng.choice(ids, size=MAX_IDS, replace=False))

    # Some FAISS builds have reconstruct_batch; we fall back to per-id reconstruct
    Q = np.empty((ids.size, DIM), dtype=np.float32)
    have_batch = hasattr(index, "reconstruct_batch")
    if have_batch:
        try:
            index.reconstruct_batch(ids.astype(np.int64), Q)
        except Exception:
            have_batch = False
    if not have_batch:
        # chunked per-id reconstruct to keep GIL switches reasonable
        for s in range(0, ids.size, 2048):
            e = min(ids.size, s + 2048)
            for i, rid in enumerate(ids[s:e], start=s):
                Q[i] = index.reconstruct(int(rid))

    # 3) mean + normalize → query vector
    q = Q.mean(axis=0, dtype=np.float32)[None, :]
    l2_normalize_inplace(q)
    return q

# --------------------------
# Grouping top hits into boxes of side= size (degrees)
# --------------------------
def make_box_centered(lat: float, lon: float, side_deg: float) -> dict:
    half = side_deg / 2.0
    return {
        "south": float(lat - half),
        "north": float(lat + half),
        "west":  float(lon - half),
        "east":  float(lon + half),
    }

def overlaps(b1: dict, b2: dict) -> bool:
    return not (b1["east"] < b2["west"] or b2["east"] < b1["west"] or
                b1["north"] < b2["south"] or b2["north"] < b1["south"])

def group_hits(lat_hits: np.ndarray, lon_hits: np.ndarray, sims: np.ndarray, size: float, topK: int):
    order = np.argsort(-sims)
    used = np.zeros(len(sims), dtype=bool)
    boxes, scores = [], []
    for i in order:
        if used[i]:
            continue
        box = make_box_centered(lat_hits[i], lon_hits[i], size)
        inside = (lat_hits >= box["south"]) & (lat_hits <= box["north"]) & \
                 (lon_hits >= box["west"]) & (lon_hits <= box["east"])
        sc = float(sims[inside].mean()) if inside.any() else float(sims[i])
        used[inside] = True
        if not any(overlaps(box, b) for b in boxes):
            boxes.append(box); scores.append(sc)
        if len(boxes) >= topK:
            break
    return boxes, scores

# --------------------------
# API
# --------------------------
@app.route("/api/search", methods=["POST"])
def search_topk_cells():
    """
    Request JSON:
      {
        "bbox": {"north":..,"south":..,"east":..,"west":..},
        "size": 0.02,  // degrees
        "topK": 5,
        "lens": ["topography","water",...]
      }
    """
    try:
        data = request.get_json(force=True)
        bbox = data["bbox"]
        size = float(data.get("size", 0.02))
        topK = int(data.get("topK", 5))
        lenses = [s.lower() for s in data.get("lens", [])]
        if not lenses:
            return jsonify({"error": "lens must be a non-empty list"}), 400

        # 1) Load FAISS index
        index = get_index(lenses)

        # 2) Build query from FAISS (reconstruct) — robust & fast
        mask_cnt = int(((LAT >= bbox["south"]) & (LAT <= bbox["north"]) &
                        (LON >= bbox["west"])  & (LON <= bbox["east"])).sum())
        print(f"[query] bbox rows ~ {mask_cnt}")
        qvec = build_query_vector_from_bbox(bbox, lenses, index)
        if qvec.size == 0:
            return jsonify({"boundingBox": bbox, "topKCells": [], "similarities": []})

        # 3) Search FAISS (cosine via IP)
        nprobe = min(512, getattr(index, "nlist", 64) // 4 or 64)
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", int(nprobe))
        except Exception:
            pass
        oversample = max(100, topK * 100)
        D, I = index.search(qvec.astype(np.float32), oversample)
        sims_ip = D[0].astype(np.float32)
        ids     = I[0]

        # 4) Map ids -> lat/lon, group
        lat_hits = LAT[ids]; lon_hits = LON[ids]
        boxes, scores = group_hits(lat_hits, lon_hits, sims_ip, size, topK)

        return jsonify({
            "boundingBox": bbox,
            "requestedSize": size,
            "topKCells": boxes,
            "similarities": [float(s) for s in scores],
            "index": {
                "path": index_path_for(lenses),
                "nprobe": int(nprobe),
                "metric": "cosine(IP on L2-normalized)",
            },
            "lensSimilarity": [{} for _ in boxes]
        })

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        faiss.omp_set_num_threads(os.cpu_count())
    except Exception:
        pass
    app.run(debug=True)

import os, math, gc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import faiss
from itertools import combinations

# =====================
# Config
# =====================
EMBED_DIR   = "data"         # expects *_trimmed.parquet
INDEX_DIR   = "faiss_indexes"
META_PATH   = "metadata.npz"
DIM         = 128
TRAIN_SAMPLES = 300_000      # lower -> faster training; keep >= ~10x nlist^(1/2) ideally
BATCH_ROWS    = 400_000      # lower if RAM is tight
M, NBITS      = 8, 8         # IVF-PQ code size (8B/code)
RNG = np.random.default_rng(123)

os.makedirs(INDEX_DIR, exist_ok=True)

# =====================
# Helpers
# =====================
def open_pf(name: str) -> pq.ParquetFile:
    path = os.path.join(EMBED_DIR, f"{name}.parquet")
    pf = pq.ParquetFile(path)
    # Show Arrow schema (not Parquet schema)
    try:
        arrow_schema = pf.schema_arrow
    except AttributeError:
        arrow_schema = pf.schema.to_arrow_schema()
    print(f"{name}.parquet Arrow schema:\n{arrow_schema}", flush=True)
    return pf

def num_rows(pf: pq.ParquetFile) -> int:
    return pf.metadata.num_rows

def l2_normalize(x: np.ndarray) -> None:
    faiss.normalize_L2(x)

def combo_name(keys) -> str:
    return "+".join(keys)

def choose_nlist(N: int) -> int:
    target = int(4 * math.sqrt(N))
    for b in [4096, 8192, 16384, 32768]:
        if b >= target:
            return b
    return 32768

def detect_embedding_col(pf: pq.ParquetFile, dim=128) -> str:
    # Use Arrow schema (not Parquet one)
    try:
        schema = pf.schema_arrow
    except AttributeError:
        schema = pf.schema.to_arrow_schema()

    cand_fixed, cand_list = [], []
    for field in schema:  # pa.Field
        t = field.type
        # fixed_size_list of float32/float64
        if pa.types.is_fixed_size_list(t) and t.list_size == dim and (
            pa.types.is_float32(t.value_type) or pa.types.is_float64(t.value_type)
        ):
            cand_fixed.append(field.name)
        # variable list of float32/float64
        elif (pa.types.is_list(t) or pa.types.is_large_list(t)) and (
            pa.types.is_float32(t.value_type) or pa.types.is_float64(t.value_type)
        ):
            cand_list.append(field.name)

    # Prefer names like 'embedding'/'embeddings'
    for pool in (cand_fixed, cand_list):
        pref = [n for n in pool if n.lower() in ("embedding", "embeddings")]
        if pref: return pref[0]
        if pool: return pool[0]

    raise RuntimeError(f"No float list[dim≈{dim}] embedding column found. Arrow schema: {schema}")

def to_numpy_embeddings_firstcol(batch: pa.RecordBatch, dim: int) -> np.ndarray:
    if batch.num_columns != 1:
        raise ValueError(f"Expected 1 column in batch, got {batch.num_columns}. Schema: {batch.schema}")
    arr = batch.column(0)
    t = arr.type

    # Fixed-size list of floats (float32 or float64)
    if pa.types.is_fixed_size_list(t) and pa.types.is_floating(t.value_type):
        flat = np.asarray(arr.values.to_numpy(zero_copy_only=False))  # <-- .values (no parentheses)
        X = flat.reshape(len(arr), -1)
        if X.shape[1] != dim:
            raise ValueError(f"Embedding width={X.shape[1]} != {dim}")
        return X.astype(np.float32, copy=False)

    # Variable-size list of floats with constant length
    if (pa.types.is_list(t) or pa.types.is_large_list(t)) and pa.types.is_floating(t.value_type):
        # Prefer offsets fast-path
        offs = getattr(arr, "offsets", None) or getattr(arr, "value_offsets", None)
        if offs is not None and hasattr(offs, "to_numpy"):
            off = np.asarray(offs.to_numpy(), dtype=np.int64)
            lens = off[1:] - off[:-1]
            if lens.min() == dim and lens.max() == dim:
                flat = np.asarray(arr.values.to_numpy(zero_copy_only=False))  # <-- .values
                return flat.reshape(len(arr), dim).astype(np.float32, copy=False)
        # Fallback: per-row copy (slower)
        pylist = arr.to_pylist()
        X = np.empty((len(pylist), dim), dtype=np.float32)
        for i, row in enumerate(pylist):
            if len(row) != dim:
                raise ValueError(f"Row {i} length {len(row)} != {dim}")
            X[i] = np.asarray(row, dtype=np.float32)
        return X

    raise TypeError(f"Unsupported embedding Arrow type: {t}")

def stream_batches_aligned_cols(pf_list, colnames, batch_size, N=None):
    iters = [pf.iter_batches(columns=[col], batch_size=batch_size)
             for pf, col in zip(pf_list, colnames)]
    seen = 0
    while True:
        batches = []
        try:
            for it in iters:
                batches.append(next(it))
        except StopIteration:
            return
        L = min(len(b) for b in batches)
        if N is not None:
            L = min(L, N - seen)
        if L <= 0:
            return
        trimmed = [b.slice(0, L) if len(b) != L else b for b in batches]
        seen += L
        yield trimmed
        if N is not None and seen >= N:
            return

# =====================
# Main
# =====================
if __name__ == "__main__":
    try:
        faiss.omp_set_num_threads(os.cpu_count())
    except Exception:
        pass

    topo_pf = open_pf("topography_trimmed")
    water_pf = open_pf("water_trimmed")
    roads_pf = open_pf("roads_trimmed")
    veg_pf   = open_pf("vegetation_trimmed")

    emb_cols = [
        detect_embedding_col(topo_pf, DIM),
        detect_embedding_col(water_pf, DIM),
        detect_embedding_col(roads_pf, DIM),
        detect_embedding_col(veg_pf, DIM),
    ]
    print("Detected embedding columns:", emb_cols, flush=True)

    counts = {
        "topography_trimmed": num_rows(topo_pf),
        "water_trimmed": num_rows(water_pf),
        "roads_trimmed": num_rows(roads_pf),
        "vegetation_trimmed": num_rows(veg_pf),
    }
    print("Row counts (trimmed):", {k: f"{v:,}" for k, v in counts.items()}, flush=True)

    N = min(counts.values())
    if len(set(counts.values())) > 1:
        print(f"[warn] lengths differ; proceeding with N={N:,} (min)", flush=True)

    # Save metadata (lat/lon) once
    print("Saving metadata…", flush=True)
    lat, lon, seen = [], [], 0
    for b in topo_pf.iter_batches(columns=["lat", "lon"], batch_size=BATCH_ROWS):
        if seen >= N: break
        L = min(len(b), N - seen)
        bt = b.slice(0, L)
        lat.append(bt.column(0).to_numpy())
        lon.append(bt.column(1).to_numpy())
        seen += L
    lat = np.concatenate(lat); lon = np.concatenate(lon)
    np.savez_compressed(META_PATH, lat=lat, lon=lon, count=N)
    print(f"Saved metadata to {META_PATH} (N={N:,})", flush=True)
    del lat, lon; gc.collect()

    base_names = ["topography", "water", "roads", "vegetation"]
    combos = [list(ks) for r in range(1,5) for ks in combinations(base_names, r)]

    NLIST = choose_nlist(N)
    print(f"Using nlist={NLIST} for N={N:,}", flush=True)

    # ===== TRAINING (stream + reservoir; zero random seeks) =====
    want = min(TRAIN_SAMPLES, N)
    kept = 0
    seen = 0
    T = W = R = V = None        # materialized reservoirs

    print("Building training cache via reservoir sampling…", flush=True)
    for (bt, bw, br, bv) in stream_batches_aligned_cols(
        [topo_pf, water_pf, roads_pf, veg_pf],
        emb_cols,
        batch_size=BATCH_ROWS,
        N=N
    ):
        Xt = to_numpy_embeddings_firstcol(bt, DIM)
        Xw = to_numpy_embeddings_firstcol(bw, DIM)
        Xr = to_numpy_embeddings_firstcol(br, DIM)
        Xv = to_numpy_embeddings_firstcol(bv, DIM)

        l2_normalize(Xt); l2_normalize(Xw); l2_normalize(Xr); l2_normalize(Xv)

        L = len(Xt)  # <-- define L for this batch

        if kept < want:
            take = min(want - kept, L)
            if T is None:
                # initialize with the first chunk (may be partial)
                T = np.empty((want, DIM), dtype=np.float32)
                W = np.empty((want, DIM), dtype=np.float32)
                R = np.empty((want, DIM), dtype=np.float32)
                V = np.empty((want, DIM), dtype=np.float32)
            T[kept:kept+take] = Xt[:take]
            W[kept:kept+take] = Xw[:take]
            R[kept:kept+take] = Xr[:take]
            V[kept:kept+take] = Xv[:take]
            kept += take
            start = take
        else:
            start = 0

        # Reservoir replacement for remaining rows in this batch
        if start < L and kept >= want:
            g_start = seen + start + 1
            g = np.arange(g_start, g_start + (L - start))
            p = want / g
            mask = RNG.random(L - start) < p
            if np.any(mask):
                r_idx = RNG.integers(0, want, size=mask.sum())
                sel = np.nonzero(mask)[0]
                T[r_idx] = Xt[start + sel]
                W[r_idx] = Xw[start + sel]
                R[r_idx] = Xr[start + sel]
                V[r_idx] = Xv[start + sel]

        seen += L  # <-- bump seen every batch
        del Xt, Xw, Xr, Xv, bt, bw, br, bv
        gc.collect()

    X_topo, X_water, X_roads, X_veg = T, W, R, V
    print(f"Training cache built: {X_topo.shape[0]:,} rows", flush=True)

    def make_Xtrain_for_combo(keys):
        mats = []
        if "topography" in keys: mats.append(X_topo)
        if "water"      in keys: mats.append(X_water)
        if "roads"      in keys: mats.append(X_roads)
        if "vegetation" in keys: mats.append(X_veg)
        if len(mats) == 1:
            return mats[0]
        X = np.mean(mats, axis=0, dtype=np.float32)
        l2_normalize(X)
        return X

    def make_index_ip_ivfpq(d, nlist, m, nbits):
        quant = faiss.IndexFlatIP(d)
        return faiss.IndexIVFPQ(quant, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    # ===== BUILD INDICES (streaming add) =====
    def make_index_ip_ivfpq(d, nlist, m, nbits):
        quant = faiss.IndexFlatIP(d)
        return faiss.IndexIVFPQ(quant, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    for keys in combos:                          # <-- keys is defined here
        name = combo_name(keys)
        out_path = os.path.join(INDEX_DIR, f"{name}.index")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} exists", flush=True)
            continue

        print(f"\n=== Building {name} ===", flush=True)
        Xtrain = make_Xtrain_for_combo(keys)
        idx = make_index_ip_ivfpq(DIM, NLIST, M, NBITS)
        idx.train(Xtrain)
        del Xtrain; gc.collect()

        added = 0
        for (bt, bw, br, bv) in stream_batches_aligned_cols(
            [topo_pf, water_pf, roads_pf, veg_pf],
            emb_cols,
            batch_size=BATCH_ROWS,
            N=N
        ):
            mats = []

            if "topography" in keys:
                Xt = to_numpy_embeddings_firstcol(bt, DIM); l2_normalize(Xt); mats.append(Xt)
            if "water" in keys:
                Xw = to_numpy_embeddings_firstcol(bw, DIM); l2_normalize(Xw); mats.append(Xw)
            if "roads" in keys:
                Xr = to_numpy_embeddings_firstcol(br, DIM); l2_normalize(Xr); mats.append(Xr)
            if "vegetation" in keys:
                Xv = to_numpy_embeddings_firstcol(bv, DIM); l2_normalize(Xv); mats.append(Xv)

            # combine this batch for the selected keys
            if len(mats) == 1:
                Xcombo = mats[0]
            else:
                Xcombo = np.mean(mats, axis=0, dtype=np.float32)
                l2_normalize(Xcombo)

            L = len(Xcombo)
            i, j = added, min(N, added + L)
            if j - i < L:
                Xcombo = Xcombo[: (j - i)]
            ids = np.arange(i, j, dtype=np.int64)
            idx.add_with_ids(Xcombo, ids)
            added = j

            for m in mats: del m
            del Xcombo, ids, bt, bw, br, bv
            gc.collect()

            if added % (BATCH_ROWS * 5) == 0:
                print(f"  added {added:,}/{N:,}", flush=True)

        faiss.write_index(idx, out_path)
        print(f"Saved {out_path}", flush=True)
        del idx; gc.collect()

    print("\nAll indexes done.", flush=True)

"""
ivc_framework.py

A modular implementation of the Phase I-VIII pipeline described for IVC symbol
research. Provides utilities for:
 - Phase I: corpus compilation normalization & vectorization
 - Phase I: adjacency / Markov mapping
 - Phase II: fractal / recursive pattern estimation (box-counting)
 - Phase II: simple CFG-like rule extraction (placeholder)
 - Phase III: clustering and cross-cultural similarity scorers
 - Phase IV: simple harmonic mapping of glyph geometry (radial FFT)
 - Phase V: weighted-probability combination engine (simple aggregator)
 - Phase VII/VIII: validation helpers and synthesis export

Author: Generated for IVC research project
Date: 2025-11-06
"""

from typing import List, Dict, Tuple, Any, Optional
import os
import json
import numpy as np
import cv2
import math
import time
from datetime import datetime

# Optional imports (only when used)
try:
    import networkx as nx
except Exception:
    nx = None

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import pairwise_distances
except Exception:
    KMeans = None
    AgglomerativeClustering = None
    pairwise_distances = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ----------------------------
# Basic IO helpers
# ----------------------------
def list_image_files(folder: str, ext: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif")) -> List[str]:
    """Return sorted list of image file paths from folder."""
    files = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(ext):
            files.append(os.path.join(folder, fn))
    return files

def load_image_gray(path: str) -> np.ndarray:
    """Load image as grayscale uint8."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return img

def save_npz(path: str, **kwargs):
    """Save data arrays/dicts to .npz (numpy) for reproducibility."""
    np.savez_compressed(path, **kwargs)

def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------
# PHASE I: Vectorization & normalization
# ----------------------------
def vectorize_glyph(image: np.ndarray, blur_ksize: int = 3, canny_thresh1: int = 50, canny_thresh2: int = 150,
                    approx_eps_frac: float = 0.01) -> Dict[str, Any]:
    """
    Vectorize a single glyph image.
    - image: grayscale uint8
    Returns a dict with keys:
      - contours (list of ndarray points)
      - approx_polys (list of approx contours)
      - moments (per-contour moments)
      - centroid (x,y) top-level centroid of the largest contour
      - normalized_polylines: list of normalized (x,y) arrays scaled to unit box and re-centered
      - signature_vector: numeric vector capturing curvature/angles/lengths (used for morphometrics)
    """
    if image is None:
        raise ValueError("image must be a numpy array")
    img = image.copy()
    if blur_ksize > 0:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(img, canny_thresh1, canny_thresh2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"contours": [], "approx": [], "centroid": None, "normalized_polylines": [], "signature_vector": np.zeros(16)}
    # pick the largest contour (by area)
    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    main = contours[idx]
    peri = cv2.arcLength(main, True)
    eps = approx_eps_frac * peri
    approx = cv2.approxPolyDP(main, eps, True)
    M = cv2.moments(main)
    if M.get("m00", 0) != 0:
        cx = M["m10"]/M["m00"]
        cy = M["m01"]/M["m00"]
    else:
        cx, cy = np.mean(main[:,0,0]), np.mean(main[:,0,1])
    # normalize to bounding box -> unit square, center at 0
    x,y,w,h = cv2.boundingRect(main)
    if w==0 or h==0:
        norm_polys = []
    else:
        pts = main.reshape(-1,2).astype(np.float32)
        pts[:,0] = (pts[:,0] - (x + w/2)) / max(w,h)
        pts[:,1] = (pts[:,1] - (y + h/2)) / max(w,h)
        norm_polys = [pts]
    # signature: resample contour to fixed length, compute curvature & angles
    def signature_from_contour(c, length=128):
        pts = c.reshape(-1,2).astype(np.float32)
        # cumulative lengths
        dists = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
        cum = np.concatenate([[0], np.cumsum(dists)])
        if cum[-1] == 0:
            return np.zeros(16)
        samp = np.linspace(0, cum[-1], length)
        res = []
        for s in samp:
            idx_ = np.searchsorted(cum, s)
            if idx_ >= len(pts):
                res.append(pts[-1])
            else:
                res.append(pts[idx_])
        res = np.array(res)
        # angles between successive segments
        vecs = res[1:] - res[:-1]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs_n = vecs / norms
        dots = (vecs_n[:-1] * vecs_n[1:]).sum(axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)  # curvature
        # aggregate features
        feats = np.array([
            angles.mean(), angles.std(), np.percentile(angles, 25), np.percentile(angles, 50), np.percentile(angles, 75),
            np.linalg.norm(np.mean(vecs, axis=0)), np.median(norms), len(pts), w/h if h>0 else 0, (cv2.contourArea(main)/(w*h + 1e-9))
        ], dtype=np.float32)
        # pad/trim to 16 dims
        out = np.zeros(16, dtype=np.float32)
        out[:len(feats)] = feats[:16]
        return out
    sig = signature_from_contour(main)
    return {
        "contours": contours,
        "approx": approx,
        "centroid": (float(cx), float(cy)),
        "normalized_polylines": norm_polys,
        "signature_vector": sig,
        "edges": edges
    }

def batch_vectorize_folder(folder: str) -> Dict[str, Dict]:
    """
    Vectorize all images in folder. Returns dict mapping filename -> vectorization dict.
    Also computes a feature matrix (N x D) for later clustering.
    """
    imgs = list_image_files(folder)
    results = {}
    feats = []
    names = []
    for p in imgs:
        try:
            img = load_image_gray(p)
            vec = vectorize_glyph(img)
            results[p] = vec
            feats.append(vec["signature_vector"])
            names.append(os.path.basename(p))
        except Exception as e:
            print("vectorize error", p, e)
            continue
    feature_matrix = np.vstack(feats) if feats else np.zeros((0,16))
    return {"vectors": results, "feature_matrix": feature_matrix, "names": names}

# ----------------------------
# PHASE I: Frequency & adjacency (Markov chain)
# ----------------------------
def build_adjacency_from_sequences(sequences: List[List[str]]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Given a list of symbol sequences (each a list of symbol IDs in observed order),
    build a transition matrix P where P[i,j] = prob(symbol_j | symbol_i).
    Returns (P, index_map)
    """
    symbols = sorted(set([s for seq in sequences for s in seq]))
    idx = {s:i for i,s in enumerate(symbols)}
    n = len(symbols)
    counts = np.zeros((n,n), dtype=float)
    for seq in sequences:
        for a,b in zip(seq[:-1], seq[1:]):
            ia, ib = idx[a], idx[b]
            counts[ia, ib] += 1.0
    # normalize rows
    row_sums = counts.sum(axis=1, keepdims=True) + 1e-12
    P = counts / row_sums
    return P, idx

def markov_stationary_distribution(P: np.ndarray, tol: float = 1e-6, max_iter: int = 10000) -> np.ndarray:
    """Compute stationary distribution by power iteration."""
    n = P.shape[0]
    v = np.ones(n) / n
    for _ in range(max_iter):
        v2 = v.dot(P)
        if np.linalg.norm(v2 - v) < tol:
            return v2
        v = v2
    return v

# ----------------------------
# PHASE II: Fractal & recursive detection (box-counting)
# ----------------------------
def box_count_fractal_dim(binary: np.ndarray, sizes: Optional[List[int]] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate fractal dimension of binary image using box-counting.
    Returns (estimated_dim, sizes_used, counts).
    """
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    binary = (binary > 0).astype(np.uint8)
    h, w = binary.shape
    if sizes is None:
        # powers of 2
        max_power = int(math.floor(math.log2(min(h,w))))
        sizes = [2**i for i in range(max_power, 1, -1)]
    counts = []
    sizes_used = []
    for s in sizes:
        if s == 0:
            continue
        # number of non-empty boxes of size s
        n_h = math.ceil(h / s)
        n_w = math.ceil(w / s)
        cnt = 0
        for i in range(n_h):
            for j in range(n_w):
                y1 = i*s
                x1 = j*s
                y2 = min(h, (i+1)*s)
                x2 = min(w, (j+1)*s)
                block = binary[y1:y2, x1:x2]
                if block.sum() > 0:
                    cnt += 1
        counts.append(cnt)
        sizes_used.append(s)
    # linear fit log(counts) ~ D * log(1/size)
    if len(counts) < 2:
        return 0.0, np.array(sizes_used), np.array(counts)
    logs = np.log(counts)
    inv_sizes = np.log(1.0 / np.array(sizes_used))
    # linear regression
    A = np.vstack([inv_sizes, np.ones_like(inv_sizes)]).T
    coeff, *_ = np.linalg.lstsq(A, logs, rcond=None)
    D = coeff[0]
    return float(D), np.array(sizes_used), np.array(counts)

# ----------------------------
# PHASE II: Context-Free Grammar approximation (lightweight placeholder)
# ----------------------------
def approximate_cfg_from_sequences(sequences: List[List[str]], max_rule_len: int = 4) -> Dict[str, Any]:
    """
    Heuristic extraction of repeated subsequences as 'rules'.
    This is not a full CFG induction algorithm but provides candidate rule sets:
      - frequent n-grams
      - nested repeated motifs
    Returns dict with rule candidates and support counts.
    """
    from collections import Counter, defaultdict
    ngram_counts = Counter()
    for seq in sequences:
        L = len(seq)
        for n in range(2, max_rule_len+1):
            for i in range(L - n + 1):
                gram = tuple(seq[i:i+n])
                ngram_counts[gram] += 1
    # select high-support grams
    rules = [{"rhs": list(g), "count": c} for g,c in ngram_counts.most_common(200) if c >= 2]
    # attempt to find nesting by checking grams that embed each other
    nesting = []
    gram_set = set(ngram_counts.keys())
    for gram in list(ngram_counts.keys())[:200]:
        for other in list(ngram_counts.keys())[:200]:
            if len(other) > len(gram) and tuple(gram) in tuple(other):
                nesting.append({"parent": list(other), "child": list(gram), "parent_count": ngram_counts[other], "child_count": ngram_counts[gram]})
    return {"rules": rules, "nesting": nesting}

# ----------------------------
# PHASE III: Clustering / semantic field grouping
# ----------------------------
def cluster_symbol_features(feature_matrix: np.ndarray, k: int = 5, method: str = "kmeans") -> Dict[str, Any]:
    """
    Cluster numeric feature representations of symbols.
    Returns labels and cluster centers.
    """
    if feature_matrix.size == 0:
        return {"labels": np.array([]), "centers": np.array([])}
    if method == "kmeans":
        if KMeans is None:
            raise ImportError("scikit-learn required for kmeans clustering")
        model = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = model.fit_predict(feature_matrix)
        centers = model.cluster_centers_
    else:
        if AgglomerativeClustering is None:
            raise ImportError("scikit-learn required for hierarchical clustering")
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(feature_matrix)
        centers = None
    return {"labels": labels, "centers": centers}

def compute_similarity_matrix(feature_matrix: np.ndarray, metric: str = "cosine") -> np.ndarray:
    if feature_matrix.size == 0:
        return np.zeros((0,0))
    if pairwise_distances is None:
        raise ImportError("scikit-learn required for pairwise distances")
    D = pairwise_distances(feature_matrix, metric=metric)
    # convert distance to similarity
    S = 1.0 - (D - D.min()) / (D.max() - D.min() + 1e-12)
    return S

# ----------------------------
# PHASE IV: Harmonic mapping (radial signature FFT)
# ----------------------------
def radial_signature_fft(image: np.ndarray, num_rays: int = 360, resample_len: int = 256) -> Dict[str, Any]:
    """
    Compute radial distance signature from centroid and take FFT to find dominant harmonic frequencies.
    Returns: dict {angles, radii, fft_freqs, fft_magnitude, dominant_freqs}
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold and largest contour centroid
    _,th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"angles": np.array([]), "radii": np.array([]), "fft_freqs": np.array([]), "fft_mag": np.array([]), "dominant": []}
    main = max(cnts, key=cv2.contourArea)
    M = cv2.moments(main)
    if M.get("m00",0) == 0:
        cx, cy = np.mean(main[:,:,0]), np.mean(main[:,:,1])
    else:
        cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
    h,w = gray.shape[:2]
    angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    radii = []
    for a in angles:
        # ray sampling outward until boundary or max_dim
        found = False
        for r in np.linspace(0, max(h,w), resample_len):
            x = int(round(cx + r * math.cos(a)))
            y = int(round(cy + r * math.sin(a)))
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            if th[y,x] > 0:
                radii.append(r)
                found = True
                break
        if not found:
            radii.append(0.0)
    radii = np.array(radii)
    # detrend and FFT
    radii_mean = radii - np.mean(radii)
    fft = np.fft.rfft(radii_mean)
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(radii), d=1.0)
    # find peaks
    peaks_idx = np.argsort(mag)[-5:][::-1]
    dominant = [{"freq": float(freqs[i]), "mag": float(mag[i])} for i in peaks_idx if mag[i] > 0]
    return {"angles": angles, "radii": radii, "fft_freqs": freqs, "fft_mag": mag, "dominant": dominant, "centroid": (cx, cy)}

# ----------------------------
# PHASE V: Weighted probability integrator
# ----------------------------
def combine_evidence_for_symbol(symbol_id: str,
                                linguistic_score: float,
                                energetic_score: float,
                                cross_cultural_score: float,
                                weights: Optional[Dict[str,float]] = None) -> float:
    """
    Combine the different evidence streams into a weighted score (0..1).
    weights default: {'linguistic':0.4, 'energetic':0.35, 'cross_cultural':0.25}
    """
    if weights is None:
        weights = {'linguistic':0.4, 'energetic':0.35, 'cross_cultural':0.25}
    s = (weights['linguistic'] * linguistic_score +
         weights['energetic'] * energetic_score +
         weights['cross_cultural'] * cross_cultural_score)
    # clamp 0..1
    return float(max(0.0, min(1.0, s)))

# ----------------------------
# PHASE VII: Validation helpers
# ----------------------------
def cross_test_on_corpora(pipeline_func, corpora_folders: Dict[str,str], **kwargs) -> Dict[str, Any]:
    """
    Run the same pipeline function on multiple corpora (folder paths)
    - pipeline_func: callable(folder_path, **kwargs) -> dict of outputs
    Returns a dict of results per corpus, and some simple aggregations.
    """
    results = {}
    for name, folder in corpora_folders.items():
        try:
            res = pipeline_func(folder, **kwargs)
            results[name] = res
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

# ----------------------------
# VISUALIZATION helpers (if matplotlib available)
# ----------------------------
def plot_adjacency_graph(P: np.ndarray, idx_map: Dict[str,int], top_n: int = 40, figsize=(8,8)):
    if plt is None or nx is None:
        raise ImportError("matplotlib and networkx required for plotting adjacency graph")
    inv = {v:k for k,v in idx_map.items()}
    G = nx.DiGraph()
    n = P.shape[0]
    for i in range(n):
        for j in range(n):
            w = P[i,j]
            if w > 0:
                G.add_edge(inv[i], inv[j], weight=float(w))
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=0.5, iterations=40)
    # draw edges with widths proportional to weight
    edges = G.edges(data=True)
    widths = [d['weight']*5 for (_,_,d) in edges]
    nx.draw_networkx_nodes(G, pos, node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, width=widths, arrowstyle='->', arrowsize=10, edge_color='gray')
    plt.axis('off')
    plt.show()

def plot_fractal_boxcount(sizes: np.ndarray, counts: np.ndarray):
    if plt is None:
        raise ImportError("matplotlib required")
    plt.figure(figsize=(6,4))
    plt.plot(np.log(1.0/sizes), np.log(counts), "o-")
    plt.xlabel("log(1/box size)")
    plt.ylabel("log(box count)")
    plt.title("Box-count fractal fit")
    plt.show()

def plot_radial_signature(angles: np.ndarray, radii: np.ndarray, fft_freqs: np.ndarray, fft_mag: np.ndarray):
    if plt is None:
        raise ImportError("matplotlib required")
    fig, axs = plt.subplots(2,1, figsize=(8,6))
    axs[0].plot(angles, radii)
    axs[0].set_title("Radial distance vs angle")
    axs[1].plot(fft_freqs, fft_mag)
    axs[1].set_title("FFT magnitude")
    plt.show()

# ----------------------------
# Pipeline orchestration helper
# ----------------------------
def run_full_pipeline_on_folder(folder: str, n_clusters: int = 6) -> Dict[str, Any]:
    """
    Runs a simple end-to-end pipeline:
     - vectorize images
     - compute adjacency if sequence files exist (placeholder)
     - compute fractal dims for each glyph
     - cluster signatures
    Returns a dict with results.
    """
    out = {}
    vecs = batch_vectorize_folder(folder)
    out['vector_results'] = vecs
    feats = vecs['feature_matrix']
    out['feature_matrix'] = feats
    # fractal dims
    fractal_dims = {}
    for p, v in vecs['vectors'].items():
        edges = v.get('edges')
        if edges is not None:
            D, sizes, counts = box_count_fractal_dim(edges)
            fractal_dims[p] = {"D": D, "sizes": sizes.tolist(), "counts": counts.tolist()}
    out['fractal_dims'] = fractal_dims
    # clustering
    if feats.size > 0:
        try:
            cl = cluster_symbol_features(feats, k=n_clusters)
            out['clusters'] = cl
        except Exception as e:
            out['clusters'] = {"error": str(e)}
    else:
        out['clusters'] = {"labels": np.array([])}
    return out

# ----------------------------
# Demo runner / __main__
# ----------------------------
if __name__ == "__main__":
    print("ivc_framework module. Sample usage:")
    print(" - from ivc_framework import run_full_pipeline_on_folder")
    print(" - res = run_full_pipeline_on_folder('path/to/glyphs')")
    print("If you want a demo run with sample images, place images in ./samples and rerun.")
    sample_dir = os.path.join(os.getcwd(), "samples")
    if os.path.exists(sample_dir):
        print("Found ./samples, running quick pipeline...")
        st = time.time()
        res = run_full_pipeline_on_folder(sample_dir, n_clusters=4)
        print("Done in %.2fs" % (time.time()-st))
        # save outputs
        save_json("pipeline_output_summary.json", {"timestamp": datetime.now().isoformat(), "summary": {k: (type(v).__name__) for k,v in res.items()}})
        print("Saved pipeline_output_summary.json")
    else:
        print("No samples folder found. Create a folder called 'samples' with glyph images to run demo.")

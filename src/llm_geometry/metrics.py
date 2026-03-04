from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes, svdvals
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def participation_ratio(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    vals = np.linalg.eigvalsh(cov)
    vals = np.clip(vals, 1e-12, None)
    return float((vals.sum() ** 2) / (np.square(vals).sum()))


def anisotropy(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    svals = svdvals(centered)
    energy = np.square(svals)
    return float(energy[0] / np.clip(energy.sum(), 1e-12, None))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(x.T @ y, ord="fro") ** 2
    x_norm = np.linalg.norm(x.T @ x, ord="fro")
    y_norm = np.linalg.norm(y.T @ y, ord="fro")
    denom = np.clip(x_norm * y_norm, 1e-12, None)
    return float(hsic / denom)


def rsa_spearman(x: np.ndarray, y: np.ndarray) -> float:
    dx = pdist(x, metric="cosine")
    dy = pdist(y, metric="cosine")
    corr, _ = spearmanr(dx, dy)
    return float(corr)


def cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    return squareform(pdist(x, metric="cosine"))


def knn_overlap(x: np.ndarray, y: np.ndarray, k: int = 10) -> float:
    n = min(len(x), len(y))
    if n <= 2:
        return float("nan")

    k = min(k, n - 1)
    dx = cosine_distance_matrix(x[:n])
    dy = cosine_distance_matrix(y[:n])

    overlap_scores = []
    for i in range(n):
        nx = set(np.argsort(dx[i])[1 : k + 1].tolist())
        ny = set(np.argsort(dy[i])[1 : k + 1].tolist())
        overlap_scores.append(len(nx.intersection(ny)) / max(len(nx.union(ny)), 1))
    return float(np.mean(overlap_scores))


def procrustes_residual(x: np.ndarray, y: np.ndarray) -> float:
    n = min(len(x), len(y))
    if n <= 2:
        return float("nan")

    x = x[:n]
    y = y[:n]
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    d = min(x.shape[1], y.shape[1])
    x = x[:, :d]
    y = y[:, :d]

    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm < 1e-12 or y_norm < 1e-12:
        return float("nan")

    x = x / x_norm
    y = y / y_norm

    r, _ = orthogonal_procrustes(y, x)
    y_aligned = y @ r
    return float(np.linalg.norm(x - y_aligned, ord="fro") / np.sqrt(n))

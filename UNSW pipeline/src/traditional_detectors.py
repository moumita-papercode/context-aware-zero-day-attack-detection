from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .utils import track


class SignatureBasedDetector:
    def __init__(self, percentile: float = 95, min_violations: int = 1):
        self.percentile = percentile
        self.min_violations = min_violations
        self.normal_ranges = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        normal = X[y == 0]
        if len(normal) == 0:
            track("SBD: no benign samples!")
            return
        lower = np.percentile(normal, 100 - self.percentile, axis=0)
        upper = np.percentile(normal, self.percentile, axis=0)
        self.normal_ranges = (lower, upper)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.normal_ranges is None:
            return np.zeros(len(X), dtype=np.float32)
        lower, upper = self.normal_ranges
        violations = ((X < lower) | (X > upper)).sum(axis=1)
        scores = (violations >= self.min_violations).astype(np.float32)
        return scores


class AnomalyBasedDetector:
    def __init__(self, n_components: int = 5, k: int = 2, epsilon: float = 1.6, random_state: int = 42):
        self.n_components = n_components
        self.k = k
        self.epsilon = epsilon
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.svd = None
        self.normal_profiles = None
        self.nn_index = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        normal = X[y == 0]
        if len(normal) < self.k:
            track("ABD: insufficient benign samples!")
            return
        if len(normal) > 20000:
            normal = normal[np.random.choice(len(normal), 20000, replace=False)]
        X_std = self.scaler.fit_transform(normal)
        n_comp = min(self.n_components, X_std.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.normal_profiles = self.svd.fit_transform(X_std)
        self.nn_index = NearestNeighbors(n_neighbors=self.k + 1)
        self.nn_index.fit(self.normal_profiles)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.nn_index is None or self.svd is None or self.normal_profiles is None:
            return np.zeros(len(X), dtype=np.float32)
        X_std = self.scaler.transform(X)
        X_red = self.svd.transform(X_std)
        dists, idx = self.nn_index.kneighbors(X_red)
        d1 = dists[:, 0]
        nn_dists, _ = self.nn_index.kneighbors(self.normal_profiles[idx[:, 0]])
        d2 = nn_dists[:, 1]
        d2_safe = np.where(d2 == 0, 1e-8, d2)
        ratio = d1 / d2_safe
        scores = (ratio > self.epsilon).astype(np.float32)
        return scores


def train_traditional_detectors(X_train_sel: np.ndarray, y_train: np.ndarray, config: dict):
    track("Training traditional IDS components (train only)...")
    sbd = SignatureBasedDetector(
        percentile=config["sbd_percentile"],
        min_violations=config["sbd_min_violations"],
    )
    abd = AnomalyBasedDetector(
        n_components=config["abd_n_components"],
        k=config["abd_k"],
        epsilon=config["abd_epsilon"],
        random_state=config["random_state"],
    )
    sbd.fit(X_train_sel, y_train)
    abd.fit(X_train_sel, y_train)
    return sbd, abd

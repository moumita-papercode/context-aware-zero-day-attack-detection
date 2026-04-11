from __future__ import annotations

from typing import Dict, List

import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from .utils import track


class ZeroDayFocusedEO:
    def __init__(self, population_size: int = 30, iterations: int = 50, candidate_count: int = 4):
        self.pop_size = population_size
        self.iterations = iterations
        self.candidate_count = candidate_count

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        config: Dict,
        n_features: int = 35,
    ) -> List[str]:
        track("Running Zero-Day Focused EO on train set...")
        n_total_features = X.shape[1]
        population = np.random.randint(0, 2, (self.pop_size, n_total_features))
        best_solution = None
        best_fitness = -np.inf

        for it in range(self.iterations):
            fitness_scores = []
            for individual in population:
                mask = individual.astype(bool)
                if mask.sum() == 0:
                    fitness_scores.append(0.0)
                    continue

                X_sub = X[:, mask]
                clf = xgb.XGBClassifier(
                    max_depth=6,
                    n_estimators=80,
                    random_state=config["random_state"],
                    n_jobs=-1,
                    scale_pos_weight=(len(y) / (y.sum() + 1e-8)),
                )
                try:
                    recall_scores = cross_val_score(clf, X_sub, y, cv=3, scoring="recall", n_jobs=-1)
                    f1_scores = cross_val_score(clf, X_sub, y, cv=3, scoring="f1", n_jobs=-1)
                    score = 0.4 * np.mean(recall_scores) + 0.6 * np.mean(f1_scores)
                except Exception:
                    score = 0.0

                alpha, beta, gamma = 0.8, 0.15, 0.05
                feature_penalty = mask.sum() / n_total_features
                diversity_bonus = 1.0
                fitness = alpha * score - beta * feature_penalty + gamma * diversity_bonus
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = individual.copy()

            top_idx = np.argsort(fitness_scores)[-self.candidate_count :]
            equilibrium_pool = population[top_idx]
            equilibrium_avg = equilibrium_pool.mean(axis=0)
            new_population = []
            for _ in range(self.pop_size):
                lambda_val = np.random.random()
                t = 1 - (it / self.iterations)
                F = np.exp(-lambda_val * t)
                candidate = equilibrium_pool[np.random.randint(len(equilibrium_pool))]
                new_individual = candidate + F * (equilibrium_avg - candidate)
                new_individual = (new_individual > 0.5).astype(int)
                new_population.append(new_individual)
            population = np.array(new_population)

            if it % 10 == 0:
                track(f"EO Iter {it}, Best fitness: {best_fitness:.4f}")

        if best_solution is None:
            raise RuntimeError("EO failed to select any feature subset.")

        mask = best_solution.astype(bool)
        selected_features = [feature_names[i] for i in range(len(feature_names)) if mask[i]]
        if len(selected_features) > n_features:
            temp_clf = xgb.XGBClassifier(max_depth=6, n_estimators=120, random_state=config["random_state"])
            temp_clf.fit(X[:, mask], y)
            importances = temp_clf.feature_importances_
            top = np.argsort(importances)[-n_features:]
            selected_features = [selected_features[i] for i in top]

        track(f"EO selected {len(selected_features)} features, fitness={best_fitness:.4f}")
        return selected_features

from __future__ import annotations

import numpy as np


def advanced_hybrid_fusion(
    tds_proba: np.ndarray,
    model_proba: np.ndarray,
    sbd_proba: np.ndarray,
    abd_proba: np.ndarray,
) -> np.ndarray:
    def platt_scaling(proba: np.ndarray, a: float = 2.5, b: float = -0.3) -> np.ndarray:
        eps = 1e-7
        proba_safe = np.clip(proba, eps, 1 - eps)
        logits = np.log(proba_safe / (1 - proba_safe))
        return 1 / (1 + np.exp(-(a * logits + b)))

    def temperature_scaling(proba: np.ndarray, temp: float = 1.5) -> np.ndarray:
        eps = 1e-7
        proba_safe = np.clip(proba, eps, 1 - eps)
        logits = np.log(proba_safe / (1 - proba_safe))
        return 1 / (1 + np.exp(-logits / temp))

    model_cal = platt_scaling(model_proba, a=3.0, b=-0.1)
    tds_cal = temperature_scaling(tds_proba, temp=1.8)
    sbd_cal = temperature_scaling(sbd_proba, temp=2.0)
    abd_cal = temperature_scaling(abd_proba, temp=1.6)

    ultra_precision_zone = model_proba > 0.75
    high_precision_zone = (model_proba > 0.60) & (model_proba <= 0.75)
    collaboration_zone = ((model_proba > 0.45) & (model_proba <= 0.60) & (tds_proba > 0.55))
    tds_support_zone = ((tds_proba > 0.70) & (model_proba > 0.40) & (model_proba <= 0.60))
    benign_zone = model_proba < 0.35

    model_prior = 0.75
    tds_prior = 0.20
    abd_prior = 0.05

    agreement_score = np.abs(model_proba - tds_proba)
    high_agreement = agreement_score < 0.15
    model_posterior = np.where(high_agreement, model_prior * 0.95, model_prior * 1.05)
    model_posterior = np.clip(model_posterior, 0.70, 0.85)
    tds_posterior = np.where(high_agreement, tds_prior * 1.15, tds_prior * 0.90)
    tds_posterior = np.clip(tds_posterior, 0.10, 0.25)

    total_posterior = model_posterior + tds_posterior + abd_prior + 1e-8
    model_weight = model_posterior / total_posterior
    tds_weight = tds_posterior / total_posterior
    abd_weight = abd_prior / total_posterior
    bayesian_fusion = model_weight * model_cal + tds_weight * tds_cal + abd_weight * abd_cal

    meta_weight_model = np.where(
        (model_proba > 0.6) & (tds_proba > 0.6) & (abd_proba > 0.6),
        0.82,
        np.where(
            (model_proba < 0.4) & (tds_proba < 0.4),
            0.96,
            np.where(
                np.abs(model_proba - 0.5) - np.abs(tds_proba - 0.5) > 0.2,
                0.94,
                np.where(agreement_score < 0.1, 0.88, 0.92),
            ),
        ),
    )
    meta_weight_tds = 1.0 - meta_weight_model
    stacked_fusion = meta_weight_model * model_proba + meta_weight_tds * tds_proba

    gated_fusion = np.where(
        ultra_precision_zone,
        0.99 * model_proba + 0.01 * tds_proba,
        np.where(
            high_precision_zone,
            0.96 * model_proba + 0.04 * tds_proba,
            np.where(
                collaboration_zone,
                0.90 * model_proba + 0.10 * tds_proba,
                np.where(
                    tds_support_zone,
                    0.87 * model_proba + 0.13 * tds_proba,
                    np.where(benign_zone, 0.995 * model_proba + 0.005 * tds_proba, 0.94 * model_proba + 0.06 * tds_proba),
                ),
            ),
        ),
    )

    potential_false_negative = (
        (model_proba > 0.45) & (model_proba < 0.65) & (tds_proba > 0.70) & (abd_proba > 0.60)
    )
    potential_false_positive = (
        (model_proba > 0.55) & (model_proba < 0.70) & (tds_proba < 0.35) & (sbd_proba < 0.30)
    )
    error_corrected = np.where(
        potential_false_negative,
        0.80 * model_proba + 0.20 * tds_proba,
        np.where(potential_false_positive, 0.97 * model_proba + 0.03 * tds_proba, model_proba),
    )

    meta_ensemble = (
        0.40 * gated_fusion
        + 0.25 * stacked_fusion
        + 0.20 * model_proba
        + 0.10 * bayesian_fusion
        + 0.05 * error_corrected
    )
    definitely_benign = model_proba < 0.20
    likely_benign = (model_proba >= 0.20) & (model_proba < 0.40)
    meta_ensemble = np.where(definitely_benign, model_proba * 0.15, meta_ensemble)
    meta_ensemble = np.where(likely_benign, 0.25 * model_proba + 0.75 * meta_ensemble, meta_ensemble)

    safe_boost_zone = ((meta_ensemble > 0.58) & (meta_ensemble < 0.68) & (tds_proba > 0.68) & (model_proba > 0.55))
    meta_ensemble = np.where(safe_boost_zone, meta_ensemble * 1.018, meta_ensemble)
    triple_agree = (model_proba > 0.68) & (tds_proba > 0.65) & (abd_proba > 0.62)
    meta_ensemble = np.where(triple_agree, meta_ensemble * 1.015, meta_ensemble)

    uncertain_attack = (meta_ensemble > 0.48) & (meta_ensemble < 0.55)
    uncertain_benign = (meta_ensemble > 0.40) & (meta_ensemble <= 0.48)
    meta_ensemble = np.where(
        uncertain_attack,
        meta_ensemble * 0.98,
        np.where(uncertain_benign, meta_ensemble * 0.95, meta_ensemble),
    )
    return np.clip(meta_ensemble * 1.005, 0.0, 1.0)



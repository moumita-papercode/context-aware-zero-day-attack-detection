from __future__ import annotations

import numpy as np

from src.config import get_config
from src.data import CICCollectionPreprocessor, load_cic_collection_data, mark_zero_day_attacks, zero_day_train_val_test_split
from src.evaluation import evaluate_model, find_optimal_threshold, results_to_dataframe
from src.feature_selection import ZeroDayFocusedEO
from src.fusion import advanced_hybrid_fusion, hybrid_tds_override
from src.neural_detectors import train_neural_detectors
from src.qwen_ft_detector import QwenFTDetector
from src.traditional_detectors import train_traditional_detectors
from src.utils import setup_environment, track


def _subsample_indices(n: int, k: int | None, random_state: int) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    if k is None or k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def main() -> None:
    config = get_config()
    setup_environment(config["random_state"])
    print("Configuration set!")

    df_raw = load_cic_collection_data(config)
    df_raw, zero_day_types = mark_zero_day_attacks(df_raw, config)
    track(f"Configured zero-day families: {zero_day_types}")

    df_train_raw, df_val_raw, df_test_raw = zero_day_train_val_test_split(df_raw, config)

    preprocessor = CICCollectionPreprocessor()
    df_train, df_val, df_test, feature_cols = preprocessor.fit_transform(df_train_raw, df_val_raw, df_test_raw)

    X_train = df_train[feature_cols].values.astype(np.float32)
    y_train = df_train["label_binary"].values
    X_val = df_val[feature_cols].values.astype(np.float32)
    y_val = df_val["label_binary"].values
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test["label_binary"].values
    track(f"Feature matrices - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    zero_day_eo = ZeroDayFocusedEO(
        population_size=config["eo_population_size"],
        iterations=config["eo_iterations"],
        candidate_count=config["eo_candidate_count"],
    )
    selected_features = zero_day_eo.optimize(X_train, y_train, feature_cols, config, config["n_features"])
    sel_idx = [feature_cols.index(f) for f in selected_features]
    X_train_sel = X_train[:, sel_idx].astype(np.float32)
    X_val_sel = X_val[:, sel_idx].astype(np.float32)
    X_test_sel = X_test[:, sel_idx].astype(np.float32)
    track(f"Selected features: {len(selected_features)}")
    track(f"Train_sel: {X_train_sel.shape}, Val_sel: {X_val_sel.shape}, Test_sel: {X_test_sel.shape}")

    sbd, abd = train_traditional_detectors(X_train_sel, y_train, config)
    sbd_proba_val = sbd.predict_proba(X_val_sel)
    abd_proba_val = abd.predict_proba(X_val_sel)
    tds_proba_val = np.maximum(sbd_proba_val, abd_proba_val)

    sbd_proba = sbd.predict_proba(X_test_sel)
    abd_proba = abd.predict_proba(X_test_sel)
    tds_proba = np.maximum(sbd_proba, abd_proba)

    track(
        f"TDS scores - Val: SBD={sbd_proba_val.mean():.3f}, ABD={abd_proba_val.mean():.3f}, Combined={tds_proba_val.mean():.3f}"
    )
    track(
        f"TDS scores - Test: SBD mean={sbd_proba.mean():.3f}, ABD mean={abd_proba.mean():.3f}, Combined mean={tds_proba.mean():.3f}"
    )

    dnn, gpt2 = train_neural_detectors(X_train_sel, y_train, config)
    dnn_proba_val = dnn.predict_proba(X_val_sel)
    gpt2_proba_val = gpt2.predict_proba(X_val_sel)
    dnn_proba = dnn.predict_proba(X_test_sel)
    gpt2_proba = gpt2.predict_proba(X_test_sel)

    hybrid_dnn_val = advanced_hybrid_fusion(tds_proba_val, dnn_proba_val, sbd_proba_val, abd_proba_val)
    hybrid_dnn = advanced_hybrid_fusion(tds_proba, dnn_proba, sbd_proba, abd_proba)
    hybrid_gpt2_val = advanced_hybrid_fusion(tds_proba_val, gpt2_proba_val, sbd_proba_val, abd_proba_val)
    hybrid_gpt2 = advanced_hybrid_fusion(tds_proba, gpt2_proba, sbd_proba, abd_proba)

    val_idx = _subsample_indices(len(X_val_sel), config.get("llm_val_max"), config["random_state"])
    test_idx = _subsample_indices(len(X_test_sel), config.get("llm_test_max"), config["random_state"])

    y_val_eval = y_val[val_idx]
    y_test_eval = y_test[test_idx]

    sbd_val_eval = sbd_proba_val[val_idx]
    abd_val_eval = abd_proba_val[val_idx]
    tds_val_eval = tds_proba_val[val_idx]
    sbd_test_eval = sbd_proba[test_idx]
    abd_test_eval = abd_proba[test_idx]
    tds_test_eval = tds_proba[test_idx]
    dnn_test_eval = dnn_proba[test_idx]
    gpt2_test_eval = gpt2_proba[test_idx]
    hybrid_dnn_val_eval = hybrid_dnn_val[val_idx]
    hybrid_gpt2_val_eval = hybrid_gpt2_val[val_idx]
    hybrid_dnn_test_eval = hybrid_dnn[test_idx]
    hybrid_gpt2_test_eval = hybrid_gpt2[test_idx]

    df_val_eval = df_val.iloc[val_idx].reset_index(drop=True)
    df_test_eval = df_test.iloc[test_idx].reset_index(drop=True)
    df_val_eval_raw = df_val_raw.iloc[val_idx].reset_index(drop=True)
    df_test_eval_raw = df_test_raw.iloc[test_idx].reset_index(drop=True)
    track(f"Eval subsets -> Val={len(val_idx)}, Test={len(test_idx)}")

    thresholds = {
        "Hybrid1(TDS-DNN)": find_optimal_threshold(y_val_eval, hybrid_dnn_val_eval),
        "Hybrid2(TDS-GPT-2)": find_optimal_threshold(y_val_eval, hybrid_gpt2_val_eval),
    }

    zero_day_mask_eval = df_test_eval["is_zero_day"].values.astype(bool)
    results = [
        evaluate_model(y_test_eval, sbd_test_eval, "SBD", zero_day_mask_eval, threshold=0.5),
        evaluate_model(y_test_eval, abd_test_eval, "ABD", zero_day_mask_eval, threshold=0.5),
        evaluate_model(y_test_eval, tds_test_eval, "TDS(SBD+ABD)", zero_day_mask_eval, threshold=0.5),
        evaluate_model(y_test_eval, dnn_test_eval, "DNN", zero_day_mask_eval, threshold=0.5),
        evaluate_model(y_test_eval, gpt2_test_eval, "GPT-2", zero_day_mask_eval, threshold=0.5),
        evaluate_model(
            y_test_eval,
            hybrid_dnn_test_eval,
            "Hybrid1(TDS-DNN)",
            zero_day_mask_eval,
            threshold=thresholds["Hybrid1(TDS-DNN)"],
        ),
        evaluate_model(
            y_test_eval,
            hybrid_gpt2_test_eval,
            "Hybrid2(TDS-GPT-2)",
            zero_day_mask_eval,
            threshold=thresholds["Hybrid2(TDS-GPT-2)"],
        ),
    ]

    if config.get("enable_qwenft", False):
        qwen = QwenFTDetector(config)
        qwen_proba_val, _ = qwen.predict_attack_proba(
            df_val_eval_raw,
            selected_features,
            batch_size=config.get("qwen_batch_size", 64),
        )
        qwen_proba_test, _ = qwen.predict_attack_proba(
            df_test_eval_raw,
            selected_features,
            batch_size=config.get("qwen_batch_size", 64),
        )
        track(f"QwenFT mean proba (Val):  {qwen_proba_val.mean():.3f}")
        track(f"QwenFT mean proba (Test): {qwen_proba_test.mean():.3f}")

        hybrid_qwen_val = advanced_hybrid_fusion(tds_val_eval, qwen_proba_val, sbd_val_eval, abd_val_eval)
        hybrid_qwen_test = advanced_hybrid_fusion(tds_test_eval, qwen_proba_test, sbd_test_eval, abd_test_eval)
        thresholds["Hybrid3(TDS-Qwen3)"] = find_optimal_threshold(
            y_val_eval,
            hybrid_qwen_val,
            low=0.30,
            high=0.70,
            steps=81,
        )

        results.append(evaluate_model(y_test_eval, qwen_proba_test, "Qwen3", zero_day_mask_eval, threshold=0.5))
        results.append(
            evaluate_model(
                y_test_eval,
                hybrid_qwen_test,
                "Hybrid3(TDS-Qwen3)",
                zero_day_mask_eval,
                threshold=thresholds["Hybrid3(TDS-Qwen3)"],
            )
        )
        qwen.cleanup()

    results_df = results_to_dataframe(results)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC", "ZeroDay_Recall"]
    print("\n" + "=" * 90)
    print("FINAL RESULTS (SBD, ABD, TDS, DNN, GPT-2, Qwen3, Hybrids)")
    print("=" * 90)
    print(results_df[cols].round(4).to_string(index=False))

    print("\nThresholds used:")
    for name, value in thresholds.items():
        print(f"- {name}: {value:.4f}")


if __name__ == "__main__":
    main()

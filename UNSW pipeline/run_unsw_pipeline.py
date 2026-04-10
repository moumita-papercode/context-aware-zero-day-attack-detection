from __future__ import annotations

import numpy as np

from src.config import get_config
from src.data import (
    UNSWPreprocessor,
    load_unsw_nb15_data,
    mark_zero_day_attacks,
    zero_day_train_val_test_split,
)
from src.evaluation import evaluate_model, find_optimal_threshold, results_to_dataframe
from src.feature_selection import ZeroDayFocusedEO
from src.fusion import advanced_hybrid_fusion
from src.neural_detectors import train_neural_detectors
from src.traditional_detectors import train_traditional_detectors
from src.utils import setup_environment, track


def main() -> None:
    setup_environment()
    config = get_config()
    print("Configuration set!")

    df_raw = load_unsw_nb15_data(config)
    df_raw, zero_day_types = mark_zero_day_attacks(df_raw, config)
    track(f"Configured zero-day groups in use: {zero_day_types}")

    df_train_raw, df_val_raw, df_test_raw = zero_day_train_val_test_split(df_raw, config)

    preprocessor = UNSWPreprocessor()
    df_train, df_val, df_test, feature_cols = preprocessor.fit_transform(
        df_train_raw,
        df_val_raw,
        df_test_raw,
    )

    X_train = df_train[feature_cols].values.astype("float32")
    y_train = df_train["label_binary"].values
    X_val = df_val[feature_cols].values.astype("float32")
    y_val = df_val["label_binary"].values
    X_test = df_test[feature_cols].values.astype("float32")
    y_test = df_test["label_binary"].values
    track(f"Feature matrices - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    zero_day_eo = ZeroDayFocusedEO(
        population_size=config["eo_population_size"],
        iterations=config["eo_iterations"],
        candidate_count=config["eo_candidate_count"],
    )
    selected_features = zero_day_eo.optimize(X_train, y_train, feature_cols, config, config["n_features"])
    sel_idx = [feature_cols.index(f) for f in selected_features]

    X_train_sel = X_train[:, sel_idx].astype("float32")
    X_val_sel = X_val[:, sel_idx].astype("float32")
    X_test_sel = X_test[:, sel_idx].astype("float32")
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
        f"TDS scores - Val: SBD={sbd_proba_val.mean():.3f}, "
        f"ABD={abd_proba_val.mean():.3f}, Combined={tds_proba_val.mean():.3f}"
    )
    track(
        f"TDS scores - Test: SBD mean={sbd_proba.mean():.3f}, "
        f"ABD mean={abd_proba.mean():.3f}, Combined mean={tds_proba.mean():.3f}"
    )

    dnn, gpt2 = train_neural_detectors(X_train_sel, y_train, config)

    dnn_proba = dnn.predict_proba(X_test_sel)
    gpt2_proba = gpt2.predict_proba(X_test_sel)

    dnn_proba_val = dnn.predict_proba(X_val_sel)
    gpt2_proba_val = gpt2.predict_proba(X_val_sel)

    hybrid_dnn_val = advanced_hybrid_fusion(tds_proba_val, dnn_proba_val, sbd_proba_val, abd_proba_val)
    hybrid_dnn = advanced_hybrid_fusion(tds_proba, dnn_proba, sbd_proba, abd_proba)

    hybrid_gpt2_val = advanced_hybrid_fusion(tds_proba_val, gpt2_proba_val, sbd_proba_val, abd_proba_val)
    hybrid_gpt2 = advanced_hybrid_fusion(tds_proba, gpt2_proba, sbd_proba, abd_proba)

    thresholds = {
        "Hybrid1(TDS-DNN)": find_optimal_threshold(y_val, hybrid_dnn_val),
        "Hybrid2(TDS-GPT-2)": find_optimal_threshold(y_val, hybrid_gpt2_val),
    }

    zero_day_mask = df_test["is_zero_day"].values.astype(bool)
    results = []
    results.append(evaluate_model(y_test, sbd_proba, "SBD", zero_day_mask))
    results.append(evaluate_model(y_test, abd_proba, "ABD", zero_day_mask))
    results.append(evaluate_model(y_test, tds_proba, "TDS(SBD+ABD)", zero_day_mask))
    results.append(evaluate_model(y_test, dnn_proba, "DNN", zero_day_mask))
    results.append(evaluate_model(y_test, gpt2_proba, "GPT-2", zero_day_mask))
    results.append(
        evaluate_model(
            y_test,
            hybrid_dnn,
            "Hybrid1(TDS-DNN)",
            zero_day_mask,
            threshold=thresholds["Hybrid1(TDS-DNN)"],
        )
    )
    results.append(
        evaluate_model(
            y_test,
            hybrid_gpt2,
            "Hybrid2(TDS-GPT-2)",
            zero_day_mask,
            threshold=thresholds["Hybrid2(TDS-GPT-2)"],
        )
    )

    results_df = results_to_dataframe(results)
    print("\n" + "=" * 90)
   
    print("=" * 90)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1_Score", "AUC_ROC", "ZeroDay_Recall"]
    print(results_df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

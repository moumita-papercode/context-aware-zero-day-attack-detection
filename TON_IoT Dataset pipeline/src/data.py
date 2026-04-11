from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from .utils import track


DataFrames = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


class TONIoTPreprocessor:
    def __init__(self) -> None:
        self.feature_cols: List[str] = []
        self.scaler = MinMaxScaler()
        self.encoders: Dict[str, LabelEncoder] = {}

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        track("Preprocessing TON-IoT (train -> val -> test)...")
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_test = df_test.copy()

        for frame in (df_train, df_val, df_test):
            frame["label_binary"] = frame["is_attack"].astype(int)

        exclude_cols = [
            "label_binary",
            "is_attack",
            "is_benign",
            "attack_type",
            "is_zero_day",
            "split",
            "type_norm",
        ]
        identifier_cols = [
            "ts", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "service", "flag"
        ]
        exclude_cols.extend([c for c in identifier_cols if c in df_train.columns])

        feature_cols = [c for c in df_train.columns if c not in exclude_cols]
        track(f"Initial feature candidates (train): {len(feature_cols)}")

        cat_cols = df_train[feature_cols].select_dtypes(include=["object"]).columns.tolist()
        for col in tqdm(cat_cols, desc="Encoding categoricals (train)"):
            le = LabelEncoder()
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            df_val[col] = le.transform(
                df_val[col].astype(str).where(df_val[col].astype(str).isin(le.classes_), le.classes_[0])
            )
            df_test[col] = le.transform(
                df_test[col].astype(str).where(df_test[col].astype(str).isin(le.classes_), le.classes_[0])
            )
            self.encoders[col] = le

        for col in tqdm(feature_cols, desc="Converting to numeric"):
            df_train[col] = pd.to_numeric(df_train[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df_val[col] = pd.to_numeric(df_val[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        num_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        medians = df_train[num_cols].median()
        df_train[num_cols] = df_train[num_cols].fillna(medians)
        df_val[num_cols] = df_val[num_cols].fillna(medians)
        df_test[num_cols] = df_test[num_cols].fillna(medians)

        constant_cols = [c for c in num_cols if df_train[c].nunique() <= 1]
        if constant_cols:
            track(f"Removing constant columns (train): {constant_cols}")
            feature_cols = [c for c in feature_cols if c not in constant_cols]

        if len(feature_cols) > 1:
            corr = df_train[feature_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
            if high_corr_cols:
                track(f"Removing highly correlated columns (train): {high_corr_cols}")
                feature_cols = [c for c in feature_cols if c not in high_corr_cols]

        feature_cols = [
            c for c in feature_cols if c in df_train.columns and np.issubdtype(df_train[c].dtype, np.number)
        ]
        track(f"Final feature count: {len(feature_cols)}")

        df_train[feature_cols] = self.scaler.fit_transform(df_train[feature_cols])
        df_val[feature_cols] = self.scaler.transform(df_val[feature_cols])
        df_test[feature_cols] = self.scaler.transform(df_test[feature_cols])

        self.feature_cols = feature_cols
        return df_train, df_val, df_test, feature_cols


def _normalize_type_value(value: object) -> str:
    if pd.isna(value):
        return "-"
    text = str(value).strip()
    if text == "":
        return "-"
    return text.lower()


def load_toniot_single_csv(config: Dict) -> pd.DataFrame:
    track("Loading TON-IoT single CSV dataset...")
    df = pd.read_csv(config["files"]["data"])
    track(f"Total dataset: {len(df)} samples")

    if "label" in df.columns:
        label_series = df["label"]
    else:
        label_series = df.iloc[:, -1]

    label_str = label_series.astype(str).str.strip().str.lower()
    is_normal = label_str.isin(["normal", "0"])
    df["is_attack"] = (~is_normal).astype(int)
    df["is_benign"] = 1 - df["is_attack"]
    df["attack_type"] = np.where(df["is_attack"] == 1, "Attack", "Normal")
    df["split"] = "train"

    if "type" in df.columns:
        df["type_norm"] = df["type"].apply(_normalize_type_value)
    else:
        df["type_norm"] = np.where(df["is_attack"] == 1, "attack", "normal")

    track(
        f"Attack distribution: Normal={int((df['is_attack'] == 0).sum())}, "
        f"Attack={int((df['is_attack'] == 1).sum())}"
    )
    return df


def mark_zero_day_attacks(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    track("Marking zero-day attack samples (20% of attacks held out)...")
    df = df.copy()
    attack_mask = df["is_attack"] == 1
    attack_samples = df[attack_mask]

    if len(attack_samples) == 0:
        track("WARNING: No attack samples found!")
        df["is_zero_day"] = False
        return df, []

    n_zero_day = max(1, int(0.2 * len(attack_samples)))
    zero_day_indices = np.random.choice(attack_samples.index, size=n_zero_day, replace=False)
    df["is_zero_day"] = False
    df.loc[zero_day_indices, "is_zero_day"] = True

    observed_zero_types = []
    if "type_norm" in df.columns:
        observed_zero_types = sorted(df.loc[df["is_zero_day"], "type_norm"].dropna().unique().tolist())

    track(f"Zero-day samples: {df['is_zero_day'].sum()} / {len(df)}")
    return df, observed_zero_types


def zero_day_train_val_test_split(df: pd.DataFrame, config: Dict) -> DataFrames:
    track("Creating proper zero-day-aware train/val/test split...")
    df_zero_day = df[df["is_zero_day"]].copy()
    df_non_zero_day = df[~df["is_zero_day"]].copy()

    df_normal = df_non_zero_day[df_non_zero_day["is_attack"] == 0]
    df_known_attack = df_non_zero_day[df_non_zero_day["is_attack"] == 1]

    if len(df_known_attack) == 0:
        track("WARNING: No known attacks for training! Using all non-zero-day as training pool.")
        df_train_full = df_non_zero_day
        df_val = df_non_zero_day.sample(
            n=min(max(1, len(df_non_zero_day) // 5), len(df_non_zero_day)),
            random_state=config["random_state"],
        )
        df_test_non_zd = df_non_zero_day.drop(df_val.index)
    else:
        df_train_full, df_temp = train_test_split(
            df_non_zero_day,
            test_size=0.3,
            random_state=config["random_state"],
            stratify=df_non_zero_day["is_attack"],
        )
        df_val, df_test_non_zd = train_test_split(
            df_temp,
            test_size=0.5,
            random_state=config["random_state"],
            stratify=df_temp["is_attack"],
        )

    df_test = pd.concat([df_test_non_zd, df_zero_day], ignore_index=True)

    max_train = int(config["max_rows_total"] * 0.7)
    max_val = int(config["max_rows_total"] * 0.15)
    max_test = int(config["max_rows_total"] * 0.15)

    df_train = (
        df_train_full.sample(n=max_train, random_state=config["random_state"])
        if len(df_train_full) > max_train
        else df_train_full
    )
    if len(df_val) > max_val:
        df_val = df_val.sample(n=max_val, random_state=config["random_state"])
    if len(df_test) > max_test:
        df_test = df_test.sample(n=max_test, random_state=config["random_state"])

    track(
        f"Train: {len(df_train)} samples ({df_train['is_attack'].mean():.1%} attacks, "
        f"{df_train['is_zero_day'].sum()} zero-day)"
    )
    track(
        f"Val  : {len(df_val)} samples ({df_val['is_attack'].mean():.1%} attacks, "
        f"{df_val['is_zero_day'].sum()} zero-day)"
    )
    track(
        f"Test : {len(df_test)} samples ({df_test['is_attack'].mean():.1%} attacks, "
        f"{df_test['is_zero_day'].sum()} zero-day)"
    )

    assert df_train["is_zero_day"].sum() == 0, "Train set contains no zero-day samples."
    assert df_val["is_zero_day"].sum() == 0, "Validation set contains  no zero-day samples."
    assert df_test["is_zero_day"].sum() > 0, "Test set should contain zero-day samples."
    return df_train, df_val, df_test

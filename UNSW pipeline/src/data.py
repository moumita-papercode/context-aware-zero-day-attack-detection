from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from .utils import track


DataFrames = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


class UNSWPreprocessor:
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
        track("Preprocessing UNSW-NB15 (train -> val -> test)...")
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
            "attack_cat",
            "label",
        ]
        identifier_cols = ["id", "srcip", "sport", "dstip", "dsport", "proto", "service", "state"]
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
            df_train[col] = pd.to_numeric(df_train[col], errors="coerce")
            df_val[col] = pd.to_numeric(df_val[col], errors="coerce")
            df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

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


def load_unsw_nb15_data(config: Dict) -> pd.DataFrame:
    track("Loading UNSW-NB15 dataset...")
    df_train = pd.read_csv(config["files"]["train"])
    df_test = pd.read_csv(config["files"]["test"])
    track(f"Training set: {len(df_train)} samples")
    track(f"Testing  set: {len(df_test)} samples")

    df_train["split"] = "train"
    df_test["split"] = "test"
    df = pd.concat([df_train, df_test], ignore_index=True)

    if "label" in df.columns:
        df["is_attack"] = df["label"].astype(int)
        df["is_benign"] = 1 - df["is_attack"]
    else:
        df["is_attack"] = (df["attack_cat"] != "Normal").astype(int)
        df["is_benign"] = (df["attack_cat"] == "Normal").astype(int)

    if "attack_cat" in df.columns:
        df["attack_type"] = df["attack_cat"].fillna("Normal")
    else:
        df["attack_type"] = df["is_attack"].map({0: "Normal", 1: "Attack"})

    track(f"Attack distribution: {df['attack_type'].value_counts().to_dict()}")
    return df



def mark_zero_day_attacks(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    track("Marking zero-day attack families...")
    df = df.copy()
    zero_day_types: List[str] = []

    for pattern in config["zero_day_attack_types"]:
        matches = [a for a in df["attack_type"].unique() if pattern.lower() in str(a).lower()]
        zero_day_types.extend(matches)

    zero_day_types = sorted(set(zero_day_types))
    if not zero_day_types:
        track("WARNING: No configured zero-day attack types found; using random subset of attacks.")
        attack_idx = df[df["is_attack"] == 1].index
        holdout_count = max(1, int(0.3 * len(attack_idx)))
        zero_idx = np.random.choice(attack_idx, holdout_count, replace=False)
        df["is_zero_day"] = False
        df.loc[zero_idx, "is_zero_day"] = True
    else:
        track(f"Zero-day families: {zero_day_types}")
        df["is_zero_day"] = df["attack_type"].isin(zero_day_types)

    track(f"Zero-day samples: {df['is_zero_day'].sum()} / {len(df)}")
    return df, zero_day_types



def zero_day_train_val_test_split(df: pd.DataFrame, config: Dict) -> DataFrames:
    track("Creating proper zero-day-aware train/val/test split...")

    df_train_orig = df[df["split"] == "train"].copy()
    df_test_orig = df[df["split"] == "test"].copy()

    df_train_full = df_train_orig[~df_train_orig["is_zero_day"]].copy()
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=0.2,
        random_state=config["random_state"],
        stratify=df_train_full["is_attack"],
    )

    df_zero_day_all = df[df["is_zero_day"]].copy()
    df_test = pd.concat([df_test_orig, df_zero_day_all], ignore_index=True).drop_duplicates(ignore_index=True)

    max_train = int(config["max_rows_total"] * 0.7)
    max_val = int(config["max_rows_total"] * 0.15)
    max_test = int(config["max_rows_total"] * 0.15)

    if len(df_train) > max_train:
        df_train = df_train.sample(n=max_train, random_state=config["random_state"])
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

    assert df_train["is_zero_day"].sum() == 0, "Train set not contains zero-day samples."
    assert df_val["is_zero_day"].sum() == 0, "Validation set not contains zero-day samples."
    assert df_test["is_zero_day"].sum() > 0, "Test set should contain zero-day samples."
    track("Zero-day split verification passed!")

    return df_train, df_val, df_test

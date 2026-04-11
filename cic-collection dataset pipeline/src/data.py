from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from .utils import track


DataFrames = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def canonicalize_type(value: object) -> str:
    if pd.isna(value):
        return "Other"
    text = str(value).strip().lower()
    if text == "":
        return "Other"
    if "benign" in text or text == "normal":
        return "Benign"
    if "web attack" in text or "xss" in text or "sql" in text or "webattack" in text:
        return "Webattack"
    if "infiltration" in text:
        return "Infiltration"
    if "bruteforce" in text or "ftp" in text or "ssh" in text:
        return "Bruteforce"
    if "bot" in text:
        return "Botnet"
    if "portscan" in text:
        return "Portscan"
    if "ddos" in text:
        return "DDoS"
    if "dos" in text:
        return "DoS"
    return "Other"


class CICCollectionPreprocessor:
    def __init__(self) -> None:
        self.feature_cols: List[str] = []
        self.scaler = MinMaxScaler()

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        track("Preprocessing CIC Collection (train -> val -> test)...")
        df_train = df_train.copy()
        df_val = df_val.copy()
        df_test = df_test.copy()

        for frame in (df_train, df_val, df_test):
            frame["label_binary"] = frame["is_attack"].astype(int)

        exclude_cols = [
            "label_binary",
            "label",
            "is_attack",
            "is_benign",
            "attack_type",
            "is_zero_day",
            "split",
            "Label",
            "type",
            "type_norm",
            "ClassLabel",
        ]
        feature_cols = [c for c in df_train.columns if c not in exclude_cols]
        track(f"Initial feature candidates (train): {len(feature_cols)}")

        for col in feature_cols:
            df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
            df_val[col] = df_val[col].replace([np.inf, -np.inf], np.nan)
            df_test[col] = df_test[col].replace([np.inf, -np.inf], np.nan)

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


def _categorize_attack(label: object) -> str:
    label_lower = str(label).strip().lower()
    if "benign" in label_lower:
        return "Normal"
    if "ddos" in label_lower or ("dos" in label_lower and "ddos" not in label_lower):
        return "DoS"
    if "portscan" in label_lower:
        return "PortScan"
    if "bot" in label_lower:
        return "Bot"
    if "web attack" in label_lower or "xss" in label_lower or "sql" in label_lower:
        return "Web Attack"
    if "infiltration" in label_lower:
        return "Infiltration"
    if "bruteforce" in label_lower or "ftp" in label_lower or "ssh" in label_lower:
        return "Brute Force"
    if "heartbleed" in label_lower:
        return "Heartbleed"
    if "backdoor" in label_lower:
        return "Backdoor"
    return "Other"


def load_cic_collection_data(config: Dict) -> pd.DataFrame:
    track("Loading CIC Collection dataset...")
    df = pd.read_parquet(config["files"]["data"])
    track(f"Total dataset: {len(df)} samples")

    if "Label" not in df.columns:
        raise KeyError("Expected a 'Label' column in the CIC Collection parquet file.")

    df["Label"] = df["Label"].astype(str).str.strip()
    df["attack_type"] = df["Label"].apply(_categorize_attack)
    df["is_attack"] = (df["attack_type"] != "Normal").astype(int)
    df["is_benign"] = 1 - df["is_attack"]
    df["label"] = df["is_attack"].astype(int)
    df["type"] = df["Label"].apply(canonicalize_type)
    df["type_norm"] = df["type"].astype(str).str.lower()
    df["split"] = "train"

    track(f"Attack distribution: {df['attack_type'].value_counts().to_dict()}")
    return df


def mark_zero_day_attacks(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, List[str]]:
    track("Marking zero-day attack families...")
    df = df.copy()
    zero_day_types = list(config.get("zero_day_attack_types", []))
    track(f"Zero-day families: {zero_day_types}")
    df["is_zero_day"] = df["attack_type"].isin(zero_day_types)
    track(f"Zero-day samples: {df['is_zero_day'].sum()} / {len(df)}")
    return df, zero_day_types


def zero_day_train_val_test_split(df: pd.DataFrame, config: Dict) -> DataFrames:
    track("Creating proper zero-day-aware train/val/test split...")
    df_zero_day = df[df["is_zero_day"]].copy()
    df_non_zero_day = df[~df["is_zero_day"]].copy()

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

    assert df_train["is_zero_day"].sum() == 0, "Train set contains not zero-day samples."
    assert df_val["is_zero_day"].sum() == 0, "Validation set not contains zero-day samples."
    assert df_test["is_zero_day"].sum() > 0, "Test set should contain zero-day samples."
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

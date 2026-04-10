from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Config, GPT2Model

from .utils import DEVICE


class SimpleDNN(nn.Module):
    def __init__(self, input_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.scaler = StandardScaler()

    def _pad_if_needed(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] < self.input_dim:
            X = np.concatenate([X, np.zeros((X.shape[0], self.input_dim - X.shape[1]))], axis=1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, batch_size: int = 128) -> None:
        X = self._pad_if_needed(X)
        X_scaled = self.scaler.fit_transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
        self.to(DEVICE)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        for _ in range(epochs):
            self.train()
            idx = torch.randperm(len(X_t), device=DEVICE)
            for i in range(0, len(X_t), batch_size):
                xb = X_t[idx[i : i + batch_size]]
                yb = y_t[idx[i : i + batch_size]]
                opt.zero_grad()
                out = self.net(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._pad_if_needed(X)
        X_scaled = self.scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
        self.eval()
        with torch.no_grad():
            probs = self.net(X_t).cpu().numpy().flatten()
        return np.clip(probs, 0.01, 0.99)


class GPT2FlowDetector(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.seq_len = config["gpt2_sequence_length"]
        self.gpt2_config = GPT2Config(
            vocab_size=1,
            n_positions=self.seq_len,
            n_embd=config["gpt2_hidden_size"],
            n_layer=config["gpt2_num_layers"],
            n_head=config["gpt2_num_heads"],
            n_inner=config["gpt2_hidden_size"] * 4,
            resid_pdrop=config["gpt2_dropout"],
            embd_pdrop=config["gpt2_dropout"],
            attn_pdrop=config["gpt2_dropout"],
        )
        self.gpt2 = GPT2Model(self.gpt2_config)
        self.feature_proj = nn.Linear(1, self.gpt2_config.n_embd)
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2_config.n_embd, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.scaler = StandardScaler()

    def _create_sequences(self, X: np.ndarray, stride: int = 4) -> np.ndarray:
        _, n_features = X.shape
        L = self.seq_len
        sequences = []
        for start in range(0, n_features - L + 1, stride):
            sequences.append(X[:, start : start + L])
        if not sequences:
            pad_width = ((0, 0), (0, max(0, L - n_features)))
            sequences = [np.pad(X, pad_width, mode="constant")]
        return np.stack(sequences, axis=1)

    def forward(self, seq_batch: torch.Tensor) -> torch.Tensor:
        _, n_seq, _ = seq_batch.shape
        seq_batch = seq_batch.unsqueeze(-1)
        emb = self.feature_proj(seq_batch)
        seq_outputs = []
        for i in range(n_seq):
            s = emb[:, i, :, :]
            out = self.gpt2(inputs_embeds=s)
            h = out.last_hidden_state
            rep = (h[:, -1, :] + h.mean(dim=1) + h.max(dim=1).values) / 3
            seq_outputs.append(rep)
        combined = torch.stack(seq_outputs, dim=1).mean(dim=1)
        return self.classifier(combined)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 8, batch_size: int = 32, stride: int = 4) -> None:
        X_scaled = self.scaler.fit_transform(X_train)
        seq_np = self._create_sequences(X_scaled, stride=stride)
        seq_t = torch.tensor(seq_np, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
        self.to(DEVICE)
        opt = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.BCELoss()
        self.train()
        n = len(seq_t)
        for _ in range(epochs):
            idx = torch.randperm(n, device=DEVICE)
            for i in range(0, n, batch_size):
                xb = seq_t[idx[i : i + batch_size]]
                yb = y_t[idx[i : i + batch_size]]
                opt.zero_grad()
                preds = self.forward(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()

    def predict_proba(self, X: np.ndarray, batch_size: int = 64, stride: int = 4) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        seq_np = self._create_sequences(X_scaled, stride=stride)
        seq_t = torch.tensor(seq_np, dtype=torch.float32, device=DEVICE)
        self.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(seq_t), batch_size):
                xb = seq_t[i : i + batch_size]
                pb = self.forward(xb)
                probs.extend(pb.cpu().numpy().flatten())
        return np.clip(np.array(probs), 0.01, 0.99)


def train_neural_detectors(
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    config: Dict,
):
    dnn = SimpleDNN()
    dnn.fit(X_train_sel, y_train)

    gpt2 = GPT2FlowDetector(config)
    gpt2.fit(X_train_sel, y_train)

    return dnn, gpt2

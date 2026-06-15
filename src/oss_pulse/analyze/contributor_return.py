"""LSTM-based contributor return prediction.

For each contributor with 2+ PRs, builds a sequence of their PR history
and uses an LSTM to predict whether they will submit another PR after
their most recent one.  This captures temporal dependencies (momentum,
burnout patterns) that flat-feature classifiers miss.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from oss_pulse.visualize.style import PALETTE, save_fig, setup_style

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 20
HIDDEN_DIM = 32
NUM_FEATURES = 5
EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 64
SEED = 42
XGBOOST_BASELINE_AUC = 0.916  # from abandonment classifier (PR-centric features)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_author_sequences(
    df: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    """Build per-author PR sequences with features and targets.

    Returns a dict keyed by author, each value has:
      - features: (seq_len, NUM_FEATURES) array
      - targets: (seq_len,) array  -- 1 if author has another PR after this one
    """
    # Deduplicate to one row per PR
    pr_df = df.drop_duplicates(subset=["repo_name", "pr_number", "author"], keep="last")

    # Filter non-bots with 2+ PRs
    pr_df = pr_df[~pr_df["is_bot"]].copy()
    author_counts = pr_df.groupby("author").size()
    valid_authors = author_counts[author_counts >= 2].index
    pr_df = pr_df[pr_df["author"].isin(valid_authors)].copy()

    # Sort by author then date
    pr_df = pr_df.sort_values(["author", "pr_created_at"]).reset_index(drop=True)

    # Feature engineering
    pr_df["was_merged"] = (pr_df["pr_outcome"] == "merged").astype(float)
    pr_df["log_additions"] = np.log1p(pr_df["additions"].clip(lower=0))
    pr_df["log_deletions"] = np.log1p(pr_df["deletions"].clip(lower=0))
    pr_df["hours_to_merge"] = pr_df["time_to_merge_hours"].fillna(0.0).clip(lower=0)

    # Days since last PR (per author)
    pr_df["days_since_last"] = (
        pr_df.groupby("author")["pr_created_at"]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
        .fillna(0.0)
    )

    feature_cols = [
        "was_merged",
        "log_additions",
        "log_deletions",
        "hours_to_merge",
        "days_since_last",
    ]

    sequences: dict[str, dict[str, np.ndarray]] = {}
    for author, group in pr_df.groupby("author"):
        feats = group[feature_cols].values.astype(np.float32)
        # Target: 1 if there is at least one more PR after this one
        targets = np.ones(len(group), dtype=np.float32)
        targets[-1] = 0.0  # last PR -> no return (yet)
        sequences[str(author)] = {"features": feats, "targets": targets}

    return sequences


# ---------------------------------------------------------------------------
# Dataset & collation
# ---------------------------------------------------------------------------


class ContributorDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch dataset wrapping author PR sequences."""

    def __init__(
        self,
        sequences: dict[str, dict[str, np.ndarray]],
        max_len: int = MAX_SEQ_LEN,
    ) -> None:
        self.items: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for _author, data in sequences.items():
            feats = torch.from_numpy(data["features"])
            tgts = torch.from_numpy(data["targets"])
            seq_len = min(len(feats), max_len)
            # Truncate to last max_len items (most recent history)
            feats = feats[-max_len:]
            tgts = tgts[-max_len:]
            self.items.append((feats, tgts, seq_len))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.items[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences and return (padded_features, padded_targets, lengths)."""
    feats_list = [item[0] for item in batch]
    tgts_list = [item[1] for item in batch]
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)

    padded_feats = pad_sequence(feats_list, batch_first=True, padding_value=0.0)
    padded_tgts = pad_sequence(tgts_list, batch_first=True, padding_value=-1.0)

    return padded_feats, padded_tgts, lengths


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ContributorReturnLSTM(nn.Module):
    """LSTM predicting whether a contributor will return after each PR."""

    def __init__(
        self,
        input_dim: int = NUM_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Pack for efficient computation
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Apply linear to each timestep
        logits = self.fc(output).squeeze(-1)  # (batch, seq_len)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    model: ContributorReturnLSTM,
    train_loader: DataLoader,  # type: ignore[type-arg]
    epochs: int = EPOCHS,
    lr: float = LR,
) -> list[float]:
    """Train the LSTM model, return per-epoch losses."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for feats, tgts, lengths in train_loader:
            optimizer.zero_grad()
            logits = model(feats, lengths)

            # Mask: only compute loss on non-padded positions
            mask = tgts >= 0
            loss_unreduced = criterion(logits, tgts)
            loss = (loss_unreduced * mask).sum() / mask.sum()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            n_samples += mask.sum().item()

        avg_loss = total_loss / max(n_samples, 1)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch + 1:2d}/{epochs}  loss={avg_loss:.4f}")

    return epoch_losses


def evaluate_model(
    model: ContributorReturnLSTM,
    test_loader: DataLoader,  # type: ignore[type-arg]
) -> dict[str, float]:
    """Evaluate the model on the test set. Return AUC and accuracy."""
    model.eval()
    all_probs: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for feats, tgts, lengths in test_loader:
            logits = model(feats, lengths)
            probs = torch.sigmoid(logits)
            mask = tgts >= 0
            all_probs.extend(probs[mask].numpy().tolist())
            all_targets.extend(tgts[mask].numpy().tolist())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))

    return {"auc": auc, "accuracy": acc}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_auc_comparison(
    lstm_auc: float,
    xgb_auc: float = XGBOOST_BASELINE_AUC,
) -> plt.Figure:
    """Bar chart comparing LSTM vs XGBoost AUC."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["XGBoost\n(flat features)", "LSTM\n(sequential)"]
    aucs = [xgb_auc, lstm_auc]
    colors = [PALETTE["secondary"], PALETTE["primary"]]

    bars = ax.bar(models, aucs, color=colors, width=0.5, edgecolor="white")

    # Add value labels on bars
    for bar, auc_val in zip(bars, aucs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{auc_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_ylabel("AUC-ROC")
    ax.set_title("Contributor Return Prediction: LSTM vs XGBoost Baseline")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color=PALETTE["grid"], linewidth=0.8, linestyle="--", alpha=0.7)

    # Annotation
    delta = lstm_auc - xgb_auc
    sign = "+" if delta > 0 else ""
    ax.annotate(
        f"{sign}{delta:.3f} AUC",
        xy=(1, lstm_auc),
        xytext=(1.35, (lstm_auc + xgb_auc) / 2),
        fontsize=12,
        fontweight="bold",
        color=PALETTE["success"] if delta > 0 else PALETTE["accent"],
        arrowprops=dict(
            arrowstyle="->",
            color=PALETTE["success"] if delta > 0 else PALETTE["accent"],
            lw=1.5,
        ),
        ha="left",
        va="center",
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    set_seed()

    data_path = Path("data/processed/pr_events_featured.parquet")
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} PR events loaded")

    # Build sequences
    print("\nBuilding author PR sequences...")
    sequences = build_author_sequences(df)
    print(f"  {len(sequences):,} authors with 2+ PRs")

    # Train/test split by author (80/20)
    authors = list(sequences.keys())
    random.shuffle(authors)
    split_idx = int(0.8 * len(authors))
    train_authors = set(authors[:split_idx])
    test_authors = set(authors[split_idx:])

    train_seqs = {a: sequences[a] for a in train_authors}
    test_seqs = {a: sequences[a] for a in test_authors}
    print(f"  Train: {len(train_seqs):,} authors, Test: {len(test_seqs):,} authors")

    # Create datasets and loaders
    train_ds = ContributorDataset(train_seqs)
    test_ds = ContributorDataset(test_seqs)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Train
    print("\nTraining LSTM...")
    model = ContributorReturnLSTM()
    train_model(model, train_loader)

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader)
    lstm_auc = metrics["auc"]
    lstm_acc = metrics["accuracy"]

    print(f"\n{'=' * 50}")
    print(f"  LSTM AUC:      {lstm_auc:.3f}")
    print(f"  LSTM Accuracy: {lstm_acc:.3f}")
    print(f"  XGBoost AUC:   {XGBOOST_BASELINE_AUC:.3f}  (baseline)")
    delta = lstm_auc - XGBOOST_BASELINE_AUC
    sign = "+" if delta > 0 else ""
    print(f"  Delta:         {sign}{delta:.3f}")
    print(f"{'=' * 50}")

    # Interpretation
    print("\nInterpretation:")
    if delta > 0:
        print(
            f"  The LSTM outperforms XGBoost by {delta:.3f} AUC, suggesting"
            " that sequential patterns in contributor behavior (momentum,"
            " declining engagement, merge streaks) carry predictive signal"
            " that flat per-PR features miss."
        )
    elif delta > -0.02:
        print(
            f"  The LSTM is within {abs(delta):.3f} AUC of XGBoost."
            " Sequential patterns provide comparable signal to flat features,"
            " and the two approaches may capture complementary information."
        )
    else:
        print(
            f"  The LSTM underperforms XGBoost by {abs(delta):.3f} AUC."
            " With this dataset size, the sequential structure may not"
            " provide enough additional signal to overcome the LSTM's"
            " higher capacity requirements."
        )

    # Plot
    setup_style()
    fig = plot_auc_comparison(lstm_auc)
    out_path = save_fig(fig, "nn_contributor_return")
    print(f"\nFigure saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

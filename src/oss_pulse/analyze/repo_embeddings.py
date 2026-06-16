"""Learn repo embeddings from contributor overlap and visualize hidden structure."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE

from oss_pulse.visualize.style import PALETTE, save_fig, setup_style


def build_contributor_repo_matrix(
    pr_df: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, int], list[str]]:
    """Build author -> set of repos mapping from non-bot authors.

    Returns:
        author_repos: dict mapping author to set of repo names
        repo_to_idx: dict mapping repo name to integer index
        idx_to_repo: list where position = index, value = repo name
    """
    non_bot = pr_df[~pr_df["is_bot"]]
    author_repos: dict[str, set[str]] = defaultdict(set)
    for author, repo in zip(non_bot["author"], non_bot["repo_name"], strict=True):
        author_repos[author].add(repo)

    # Only keep authors contributing to 2+ repos (they form positive pairs)
    author_repos = {a: repos for a, repos in author_repos.items() if len(repos) >= 2}

    all_repos = sorted({r for repos in author_repos.values() for r in repos})
    repo_to_idx = {r: i for i, r in enumerate(all_repos)}
    return author_repos, repo_to_idx, all_repos


def generate_training_pairs(
    author_repos: dict[str, set[str]],
    repo_to_idx: dict[str, int],
    all_repos: list[str],
    neg_ratio: int = 3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate positive and negative pairs for training.

    Positive pairs: repos co-contributed by the same author.
    Negative pairs: random repo pairs not co-contributed (sampled at neg_ratio:1).
    """
    rng = random.Random(seed)

    # Build set of positive pairs for fast lookup
    positive_set: set[tuple[int, int]] = set()
    for repos in author_repos.values():
        idxs = sorted(repo_to_idx[r] for r in repos)
        for a, b in combinations(idxs, 2):
            positive_set.add((a, b))

    positives = list(positive_set)
    print(f"  Positive pairs: {len(positives)}")

    # Sample negatives
    n_repos = len(all_repos)
    negatives: list[tuple[int, int]] = []
    target_neg = len(positives) * neg_ratio
    attempts = 0
    max_attempts = target_neg * 20
    while len(negatives) < target_neg and attempts < max_attempts:
        a = rng.randint(0, n_repos - 1)
        b = rng.randint(0, n_repos - 1)
        if a == b:
            attempts += 1
            continue
        pair = (min(a, b), max(a, b))
        if pair not in positive_set:
            negatives.append(pair)
        attempts += 1

    print(f"  Negative pairs: {len(negatives)}")

    # Combine into tensors
    all_pairs = positives + negatives
    labels = [1.0] * len(positives) + [0.0] * len(negatives)

    repo_a = torch.tensor([p[0] for p in all_pairs], dtype=torch.long)
    repo_b = torch.tensor([p[1] for p in all_pairs], dtype=torch.long)
    targets = torch.tensor(labels, dtype=torch.float32)

    # Shuffle
    perm = torch.randperm(len(all_pairs))
    return repo_a[perm], repo_b[perm], targets[perm]


class RepoEmbeddingModel(nn.Module):
    """Simple embedding model: score = sigmoid(dot product of embeddings)."""

    def __init__(self, n_repos: int, dim: int = 16):
        super().__init__()
        self.embeddings = nn.Embedding(n_repos, dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, repo_a: torch.Tensor, repo_b: torch.Tensor) -> torch.Tensor:
        emb_a = self.embeddings(repo_a)
        emb_b = self.embeddings(repo_b)
        score = (emb_a * emb_b).sum(dim=-1)
        return torch.sigmoid(score)


def train_model(
    model: RepoEmbeddingModel,
    repo_a: torch.Tensor,
    repo_b: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 2048,
) -> list[float]:
    """Train the embedding model with BCE loss."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = len(targets)
    losses: list[float] = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            a_batch = repo_a[idx]
            b_batch = repo_b[idx]
            t_batch = targets[idx]

            pred = model(a_batch, b_batch)
            loss = criterion(pred, t_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return losses


def visualize_embeddings(
    embeddings: np.ndarray,
    repo_names: list[str],
    language_map: dict[str, str],
    pr_counts: dict[str, int],
    n_annotate: int = 10,
) -> plt.Figure:
    """Project embeddings to 2D with t-SNE and create scatter plot."""
    # t-SNE projection
    perplexity = min(30, len(repo_names) - 1)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        max_iter=1000,
    )
    coords = tsne.fit_transform(embeddings)

    # Get language for each repo
    languages = [language_map.get(r, "Unknown") for r in repo_names]
    unique_langs = sorted(set(languages))

    # Color map: use a good qualitative palette
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_langs))
    lang_to_color = {lang: cmap(i) for i, lang in enumerate(unique_langs)}

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot each language group
    for lang in unique_langs:
        mask = [lg == lang for lg in languages]
        x = coords[mask, 0]
        y = coords[mask, 1]
        ax.scatter(
            x,
            y,
            c=[lang_to_color[lang]],
            label=lang,
            s=60,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    # Annotate top repos by PR count
    repo_pr_counts = [(pr_counts.get(r, 0), i, r) for i, r in enumerate(repo_names)]
    repo_pr_counts.sort(reverse=True)
    top_repos = repo_pr_counts[:n_annotate]

    for _, idx, name in top_repos:
        short_name = name.split("/")[-1]  # just the repo part
        ax.annotate(
            short_name,
            (coords[idx, 0], coords[idx, 1]),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
            xytext=(0, 8),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=PALETTE["secondary"],
                alpha=0.85,
            ),
        )

    ax.set_xlabel("t-SNE dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE dimension 2", fontsize=12)
    ax.set_title(
        "Repo Embeddings from Contributor Overlap",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    # Legend outside the plot
    ax.legend(
        title="Language",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        title_fontsize=10,
        frameon=True,
    )

    ax.grid(True, alpha=0.3)
    return fig


def analyze_clusters(
    embeddings: np.ndarray,
    repo_names: list[str],
    language_map: dict[str, str],
) -> None:
    """Print analysis of the embedding space."""
    from scipy.spatial.distance import cdist

    distances = cdist(embeddings, embeddings, metric="cosine")
    n = len(repo_names)

    print("\n" + "=" * 70)
    print("REPO EMBEDDING ANALYSIS")
    print("=" * 70)

    # Find closest pairs
    print("\nTop 15 closest repo pairs (by cosine similarity):")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((distances[i, j], i, j))
    pairs.sort()

    for rank, (dist, i, j) in enumerate(pairs[:15], 1):
        lang_i = language_map.get(repo_names[i], "?")
        lang_j = language_map.get(repo_names[j], "?")
        cross = " [CROSS-LANGUAGE]" if lang_i != lang_j else ""
        print(
            f"  {rank:2d}. {repo_names[i]:40s} ({lang_i:12s}) <-> "
            f"{repo_names[j]:40s} ({lang_j:12s}) "
            f"dist={dist:.3f}{cross}"
        )

    # Cross-language neighbors
    print("\nCross-language close pairs (different language, distance < median):")
    median_dist = np.median([d for d, _, _ in pairs])
    cross_pairs = [
        (d, i, j)
        for d, i, j in pairs
        if language_map.get(repo_names[i], "?") != language_map.get(repo_names[j], "?")
        and d < median_dist
    ]
    cross_pairs.sort()
    for dist, i, j in cross_pairs[:20]:
        lang_i = language_map.get(repo_names[i], "?")
        lang_j = language_map.get(repo_names[j], "?")
        print(
            f"  {repo_names[i]:40s} ({lang_i:12s}) <-> "
            f"{repo_names[j]:40s} ({lang_j:12s}) "
            f"dist={dist:.3f}"
        )

    # Cluster analysis via DBSCAN on the cosine distance matrix
    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=median_dist * 0.6, min_samples=2, metric="precomputed")
    labels = db.fit_predict(distances)
    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()

    print(f"\nCluster analysis (DBSCAN, eps={median_dist * 0.6:.3f}):")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points (unclustered repos): {n_noise}")

    for cid in sorted(set(labels) - {-1}):
        members = [repo_names[i] for i in range(n) if labels[i] == cid]
        langs = [language_map.get(r, "?") for r in members]
        unique = set(langs)
        multi = " [MULTI-LANGUAGE]" if len(unique) > 1 else ""
        print(f"\n  Cluster {cid}{multi} ({', '.join(sorted(unique))}):")
        for m in members:
            print(f"    - {m} ({language_map.get(m, '?')})")

    # Summary insight
    print("\n" + "=" * 70)
    print("INSIGHT SUMMARY")
    print("=" * 70)
    print(
        f"\nThe embedding space reveals {n_clusters} distinct clusters among {n} repos."
    )
    if cross_pairs:
        print(
            f"\n{len(cross_pairs)} cross-language pairs sit closer than the median "
            f"distance, suggesting contributor overlap transcends programming "
            f"language boundaries. These repos are 'siblings' not because they "
            f"share a tech stack, but because the same people work on them."
        )
    print(
        "\nThis reveals hidden structure that metadata like language or stars "
        "cannot capture: the social graph of contributors creates clusters "
        "that reflect communities of practice, organizational affiliations, "
        "and shared problem domains rather than surface-level language choices."
    )


def main() -> None:
    """Run the full pipeline."""
    print("Loading data...")
    pr_df = pd.read_parquet("data/processed/pr_events_featured.parquet")
    repos_df = pd.read_parquet("data/raw/top_repos.parquet")

    # Language map
    language_map = dict(
        zip(repos_df["repo_name"], repos_df["language"].fillna("Unknown"), strict=True)
    )

    # PR counts per repo (for annotation sizing)
    pr_counts = pr_df["repo_name"].value_counts().to_dict()

    print("\nBuilding contributor-repo matrix...")
    author_repos, repo_to_idx, all_repos = build_contributor_repo_matrix(pr_df)
    n_repos = len(all_repos)
    print(f"  Multi-repo authors: {len(author_repos)}")
    print(f"  Repos with cross-contributors: {n_repos}")

    print("\nGenerating training pairs...")
    repo_a, repo_b, targets = generate_training_pairs(
        author_repos, repo_to_idx, all_repos
    )

    print(f"\nTraining embedding model (dim=16, {n_repos} repos)...")
    model = RepoEmbeddingModel(n_repos, dim=16)
    losses = train_model(model, repo_a, repo_b, targets, epochs=30, lr=0.01)
    print(f"  Final loss: {losses[-1]:.4f}")

    # Extract embeddings
    with torch.no_grad():
        embeddings = model.embeddings.weight.numpy().copy()

    print("\nProjecting to 2D with t-SNE...")
    setup_style()

    fig = visualize_embeddings(
        embeddings, all_repos, language_map, pr_counts, n_annotate=10
    )
    path = save_fig(fig, "nn_repo_embeddings", fmt="svg")
    print(f"\nFigure saved to {path}")
    plt.close(fig)

    # Analysis
    analyze_clusters(embeddings, all_repos, language_map)

    # Compute cluster count for stats (mirrors analyze_clusters logic)
    from scipy.spatial.distance import cdist
    from sklearn.cluster import DBSCAN

    distances = cdist(embeddings, embeddings, metric="cosine")
    median_dist = float(np.median(distances[np.triu_indices(len(all_repos), k=1)]))
    db = DBSCAN(eps=median_dist * 0.6, min_samples=2, metric="precomputed")
    cluster_labels = db.fit_predict(distances)
    n_clusters = int(len(set(cluster_labels) - {-1}))

    # Update stats.json
    stats_path = Path("data/processed/stats.json")
    stats_data: dict[str, Any] = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats_data = json.load(f)

    stats_data["repo_embeddings"] = {
        "n_repos_embedded": len(all_repos),
        "n_multi_repo_authors": len(author_repos),
        "n_clusters": n_clusters,
        "final_loss": round(losses[-1], 4),
    }

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=2)
    print(f"\nUpdated {stats_path}")


if __name__ == "__main__":
    main()

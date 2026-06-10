"""Repo discovery and organization-type classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

COMPANY_ORGS: set[str] = {
    "google",
    "facebook",
    "meta",
    "microsoft",
    "apple",
    "amazon",
    "netflix",
    "uber",
    "airbnb",
    "twitter",
    "x",
    "vercel",
    "hashicorp",
    "elastic",
    "grafana",
    "datadog",
    "stripe",
    "shopify",
    "cloudflare",
    "supabase",
    "pytorch",
    "openai",
    "anthropic",
    "denoland",
    "jetbrains",
}

FOUNDATION_ORGS: set[str] = {
    "apache",
    "linux",
    "cncf",
    "kubernetes",
    "nodejs",
    "python",
    "rust-lang",
    "golang",
    "eclipse",
    "mozilla",
    "w3c",
    "ietf",
    "llvm",
    "gnome",
    "kde",
    "freedesktop",
}


def classify_org(org_name: str) -> str:
    """Classify a GitHub org as company, foundation, or community."""
    lower = org_name.lower()
    if lower in COMPANY_ORGS:
        return "company"
    if lower in FOUNDATION_ORGS:
        return "foundation"
    return "community"


def discover_repos(client: object, n: int = 200) -> pd.DataFrame:
    """Discover top repos using a BigQueryClient and classify them."""
    from oss_pulse.extract.bigquery import BigQueryClient

    assert isinstance(client, BigQueryClient)
    repos_df = client.extract_top_repos(n=n)
    repos_df["org"] = repos_df["repo_name"].str.split("/").str[0]
    repos_df["org_type"] = repos_df["org"].apply(classify_org)
    return repos_df


def load_repo_config(
    path: Path = Path("config/repos.csv"),
) -> pd.DataFrame:
    """Load manually curated repo classifications."""
    return pd.read_csv(path)


def merge_repo_metadata(
    discovered: pd.DataFrame,
    manual: pd.DataFrame,
) -> pd.DataFrame:
    """Merge auto-discovered repos with manual overrides."""
    if "repo" in manual.columns:
        manual = manual.rename(columns={"repo": "repo_name"})

    merged = discovered.merge(
        manual[["repo_name", "language", "category"]],
        on="repo_name",
        how="left",
    )
    override_cols = ["org_type"]
    for col in override_cols:
        if col in manual.columns:
            overrides = manual.set_index("repo_name")[col]
            mask = merged["repo_name"].isin(overrides.index)
            merged.loc[mask, col] = merged.loc[mask, "repo_name"].map(overrides)
    return merged


if __name__ == "__main__":
    config = load_repo_config()
    print(f"Loaded {len(config)} repos from config")
    print(config.head())

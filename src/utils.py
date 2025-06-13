"""Utility helpers for data loading, summarisation and quick EDA plots.
Minimal comments – keep code lean for notebook usage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = [
    "load_data",
    "basic_info",
    "numeric_summary",
    "plot_histograms",
    "plot_boxplots",
    "correlation_heatmap",
]


def load_data(
    filepath: str | Path,
    *,
    datetime_cols: Sequence[str] | None = ("TransactionMonth",),
    delimiter: str = "|",
) -> pd.DataFrame:
    """Load CSV-like motor-insurance dataset.

    Parameters
    ----------
    filepath : str | Path
        Location of data file.
    datetime_cols : list[str] | None, default ("TransactionMonth",)
        Columns to coerce to datetime if present.
    delimiter : str, default "|"
        Column separator used in the raw text file.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, delimiter=delimiter, low_memory=False)

    # Coerce date columns when present
    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def basic_info(df: pd.DataFrame) -> None:
    """Print quick shape, dtypes and missing counts."""
    print("Rows:", len(df), "\tColumns:", df.shape[1])
    print("\nDtype counts:\n", df.dtypes.value_counts())
    print("\nTop missing columns:\n", df.isna().sum().sort_values(ascending=False).head(20))


def numeric_summary(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    """Return describe() summary for selected numeric columns."""
    return df.loc[:, numeric_cols].describe().T


def _prep_axes(n_items: int, n_cols: int = 3, figsize: tuple[int, int] = (14, 4)):
    rows = -(-n_items // n_cols)
    fig, axs = plt.subplots(rows, n_cols, figsize=figsize, squeeze=False)
    return fig, axs.flatten()


def plot_histograms(df: pd.DataFrame, cols: Sequence[str], *, kde: bool = True) -> None:
    """Histograms for numeric features."""
    fig, axes = _prep_axes(len(cols))
    sns.set_style("whitegrid")

    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), ax=ax, kde=kde)
        ax.set_title(col)

    plt.tight_layout()


def plot_boxplots(
    df: pd.DataFrame,
    cols: Sequence[str],
    *,
    group_col: str | None = None,
) -> None:
    """Box plots per column, optionally grouped by categorical feature."""
    fig, axes = _prep_axes(len(cols))
    sns.set_style("whitegrid")

    for ax, col in zip(axes, cols):
        if group_col and group_col in df.columns:
            sns.boxplot(x=df[group_col], y=df[col], ax=ax)
            ax.set_xlabel(group_col)
        else:
            sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()


def correlation_heatmap(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Heatmap of Pearson correlations among *cols*."""
    corr = df.loc[:, cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()

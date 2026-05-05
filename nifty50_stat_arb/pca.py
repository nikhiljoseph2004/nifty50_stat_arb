"""
Principal component analysis utilities for Nifty 50 return data.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_RETURNS_PATH = os.path.join(PROJECT_ROOT, "data", "nifty50", "returns.csv")
DEFAULT_PCA_COMPONENTS_PATH = os.path.join(PROJECT_ROOT, "data", "nifty50", "pca_components.csv")


def load_returns(returns_path: str = DEFAULT_RETURNS_PATH) -> pd.DataFrame:
    """Load return data from disk."""
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    returns.sort_index(inplace=True)
    return returns


def compute_pca(
    returns: pd.DataFrame,
    train_fraction: float = 0.8,
    variance_threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute PCA on the first train_fraction of returns using Ledoit-Wolf covariance."""
    if returns.empty:
        raise ValueError("Returns data is empty")

    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be between 0 and 1")

    train_size = max(2, int(len(returns) * train_fraction))
    training_returns = returns.iloc[:train_size].copy()

    if training_returns.shape[0] < 2:
        raise ValueError("At least two rows are required to estimate covariance")

    lw = LedoitWolf()
    lw.fit(training_returns.to_numpy())
    covariance = pd.DataFrame(
        lw.covariance_,
        index=training_returns.columns,
        columns=training_returns.columns,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.to_numpy())

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_variance = eigenvalues.sum()
    explained_ratio = eigenvalues / total_variance
    cumulative_ratio = np.cumsum(explained_ratio)

    component_count = int(np.searchsorted(cumulative_ratio, variance_threshold) + 1)

    ranked_table = pd.DataFrame(
        eigenvectors[:, :component_count].T,
        columns=covariance.columns,
    )
    ranked_table.insert(0, "cumulative_variance_pct", cumulative_ratio[:component_count] * 100)
    ranked_table.insert(0, "explained_variance_pct", explained_ratio[:component_count] * 100)
    ranked_table.insert(0, "eigenvalue", eigenvalues[:component_count])
    ranked_table.insert(0, "component", [f"PC{i}" for i in range(1, component_count + 1)])

    covariance_df = pd.DataFrame(covariance, index=covariance.index, columns=covariance.columns)
    training_slice = training_returns

    return covariance_df, training_slice, ranked_table


def print_pca_summary(
    returns_path: str = DEFAULT_RETURNS_PATH,
    train_fraction: float = 0.8,
    variance_threshold: float = 0.99,
    output_csv_path: str | None = DEFAULT_PCA_COMPONENTS_PATH,
) -> pd.DataFrame:
    """Load returns, compute PCA, and print the retained eigenvector table."""
    returns = load_returns(returns_path)
    covariance, training_returns, ranked_table = compute_pca(
        returns,
        train_fraction=train_fraction,
        variance_threshold=variance_threshold,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", lambda value: f"{value:.6f}")

    print(f"Loaded returns: {returns.shape[0]} rows x {returns.shape[1]} assets")
    print(
        f"Training window: {training_returns.index[0].date()} to {training_returns.index[-1].date()} "
        f"({training_returns.shape[0]} rows, first {train_fraction:.0%} of data)"
    )
    print(f"Ledoit-Wolf covariance matrix shape: {covariance.shape}")
    print(
        f"Retained {len(ranked_table)} components to explain "
        f"{ranked_table['cumulative_variance_pct'].iloc[-1]:.2f}% of variance"
    )

    if output_csv_path:
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        ranked_table.to_csv(output_csv_path, index=False)
        print(f"Saved retained component table to {output_csv_path}")

    summary_table = ranked_table[
        [
            "component",
            "eigenvalue",
            "explained_variance_pct",
            "cumulative_variance_pct",
        ]
    ]

    print()
    print(summary_table.to_string(index=False))

    return ranked_table


def main() -> None:
    """CLI entry point for PCA analysis."""
    parser = argparse.ArgumentParser(
        description="Compute PCA on the first 80% of the Nifty 50 returns data"
    )
    parser.add_argument(
        "--returns-path",
        type=str,
        default=DEFAULT_RETURNS_PATH,
        help="Path to the returns CSV. Default: data/nifty50_returns.csv",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of rows to use for the covariance estimate. Default: 0.8",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.99,
        help="Minimum cumulative explained variance to retain. Default: 0.99",
    )
    parser.add_argument(
        "--output-csv-path",
        type=str,
        default=DEFAULT_PCA_COMPONENTS_PATH,
        help=(
            "Path to save retained PCA components with metrics and asset weights. "
            "Use an empty string to skip CSV export. "
            "Default: data/nifty50_pca_components.csv"
        ),
    )
    args = parser.parse_args()

    print_pca_summary(
        returns_path=args.returns_path,
        train_fraction=args.train_fraction,
        variance_threshold=args.variance_threshold,
        output_csv_path=args.output_csv_path or None,
    )


if __name__ == "__main__":
    main()
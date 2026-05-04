"""
Plot explained and cumulative variance from PCA component CSV files.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
DEFAULT_NIFTY50_PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots", "nifty50")
DEFAULT_VARIANCE_THRESHOLD_PCT = 99.0


def load_pca_components(csv_path: str) -> pd.DataFrame:
    """Load PCA component table from CSV."""
    components = pd.read_csv(csv_path)
    required_cols = {
        "component",
        "explained_variance_pct",
        "cumulative_variance_pct",
    }
    missing = required_cols.difference(components.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {csv_path}: {missing_str}")

    return components


def components_to_threshold(
    components: pd.DataFrame,
    variance_threshold_pct: float = DEFAULT_VARIANCE_THRESHOLD_PCT,
) -> pd.DataFrame:
    """Return leading components through the one that reaches threshold variance."""
    cumulative = components["cumulative_variance_pct"].to_numpy()
    threshold_index = int((cumulative < variance_threshold_pct).sum())
    if threshold_index >= len(components):
        return components.copy()

    return components.iloc[: threshold_index + 1].copy()


def build_output_path(csv_path: str, plots_dir: str = DEFAULT_PLOTS_DIR) -> str:
    """Create output image path using input CSV base name plus _plot."""
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    file_name = f"{base_name}_plot.png"
    return os.path.join(plots_dir, file_name)


def plot_eigenvalue_profile(
    csv_path: str,
    plots_dir: str = DEFAULT_PLOTS_DIR,
    variance_threshold_pct: float = DEFAULT_VARIANCE_THRESHOLD_PCT,
) -> str:
    """Create and save the dual-axis PCA variance plot."""
    components = load_pca_components(csv_path)
    selected = components_to_threshold(
        components,
        variance_threshold_pct=variance_threshold_pct,
    )

    x_labels = selected["component"].astype(str).tolist()
    explained = selected["explained_variance_pct"].to_numpy()
    cumulative = selected["cumulative_variance_pct"].to_numpy()
    x_positions = list(range(len(selected)))

    os.makedirs(plots_dir, exist_ok=True)
    output_path = build_output_path(csv_path, plots_dir=plots_dir)

    fig, left_axis = plt.subplots(figsize=(12, 6))
    right_axis = left_axis.twinx()

    left_axis.bar(
        x_positions,
        explained,
        color="#4e79a7",
        alpha=0.85,
        label="Explained Variance (%)",
    )
    right_axis.plot(
        x_positions,
        cumulative,
        color="#e15759",
        marker="o",
        linewidth=2.0,
        label="Cumulative Variance (%)",
    )

    left_axis.set_xlabel("Principal Components")
    left_axis.set_ylabel("Explained Variance (%)")
    right_axis.set_ylabel("Cumulative Variance (%)")
    left_axis.set_xticks(x_positions)
    left_axis.set_xticklabels(x_labels, rotation=45, ha="right")
    left_axis.set_title(
        f"PCA Variance Profile ({os.path.basename(csv_path)})\n"
        f"Components through {variance_threshold_pct:.0f}% cumulative variance"
    )

    left_axis.grid(axis="y", linestyle="--", alpha=0.3)
    right_axis.set_ylim(0, max(100.0, float(cumulative.max()) + 1.0))

    left_handles, left_labels = left_axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    left_axis.legend(left_handles + right_handles, left_labels + right_labels, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_position_counts(
    results: "pd.DataFrame",
    plots_dir: str = DEFAULT_NIFTY50_PLOTS_DIR,
    output_filename: str = "position_counts_plot.png",
    initial_capital: float = 1_000_000.0,
) -> str:
    """
    Plot long/short position counts over time with cumulative PnL in rupees
    overlaid on a secondary right y-axis.

    Long counts are positive, short counts are negated so both sit on a
    symmetric y-axis with zero in the middle.
    """
    required = {"long_count", "short_count", "cumulative_return"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Results DataFrame missing columns: {sorted(missing)}")

    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, output_filename)

    fig, ax = plt.subplots(figsize=(14, 5))

    # --- left axis: position counts ---
    ax.plot(
        results.index,
        results["long_count"],
        color="#4e79a7",
        linewidth=1.5,
        label="Long positions",
    )
    ax.plot(
        results.index,
        -results["short_count"],
        color="#e15759",
        linewidth=1.5,
        label="Short positions (negated)",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(results.index, results["long_count"], 0, alpha=0.15, color="#4e79a7")
    ax.fill_between(results.index, -results["short_count"], 0, alpha=0.15, color="#e15759")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of positions")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # --- right axis: cumulative PnL in rupees ---
    ax2 = ax.twinx()
    pnl_rupees = results["cumulative_return"] * initial_capital
    ax2.plot(
        results.index,
        pnl_rupees,
        color="#59a14f",
        linewidth=2.0,
        linestyle="-",
        label=f"Cumulative PnL (₹, capital=₹{initial_capital:,.0f})",
    )
    ax2.axhline(0, color="#59a14f", linewidth=0.5, linestyle=":")
    ax2.set_ylabel("Cumulative PnL (₹)", color="#59a14f")
    ax2.tick_params(axis="y", labelcolor="#59a14f")
    ax2.yaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}")
    )

    # combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    ax.set_title("Long / Short Position Counts and Cumulative PnL Over Time")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def main() -> None:
    """CLI entry point for eigenvalue plotting."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dual-axis plot of explained and cumulative variance "
            "from a PCA component CSV"
        )
    )
    parser.add_argument(
        "--pca-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "nifty50_pca_components.csv"),
        help="Path to PCA components CSV. Default: data/nifty50_pca_components.csv",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=DEFAULT_PLOTS_DIR,
        help="Directory where plot image is saved. Default: plots",
    )
    parser.add_argument(
        "--variance-threshold-pct",
        type=float,
        default=DEFAULT_VARIANCE_THRESHOLD_PCT,
        help="Cumulative variance cutoff for included PCs. Default: 99",
    )

    args = parser.parse_args()
    output_path = plot_eigenvalue_profile(
        csv_path=args.pca_csv,
        plots_dir=args.plots_dir,
        variance_threshold_pct=args.variance_threshold_pct,
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
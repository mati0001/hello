#!/usr/bin/env python3
"""
CSV Data Profiler
-----------------
Profiles CSV files by generating summaries of structure, data types,
missing values, statistics, duplicates, and value distributions.

Usage:
    python csv_profiler.py                      # interactive file picker from ~/Downloads
    python csv_profiler.py path/to/file.csv     # profile a specific file
    python csv_profiler.py --dir /some/folder   # pick from a custom folder
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate


# ── helpers ──────────────────────────────────────────────────────────────────

def find_csv_files(directory: str) -> list[str]:
    """Return a sorted list of CSV file paths found in *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".csv")
    )


def pick_file(directory: str) -> str:
    """Let the user choose a CSV file from *directory* interactively."""
    csv_files = find_csv_files(directory)
    if not csv_files:
        print(f"No CSV files found in: {directory}")
        sys.exit(1)

    print(f"\nCSV files in {directory}:\n")
    for i, path in enumerate(csv_files, 1):
        size_kb = os.path.getsize(path) / 1024
        print(f"  [{i}] {os.path.basename(path)}  ({size_kb:,.1f} KB)")

    while True:
        choice = input("\nSelect a file number (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            return csv_files[int(choice) - 1]
        print("Invalid choice. Try again.")


# ── profiling sections ───────────────────────────────────────────────────────

def section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def profile_overview(df: pd.DataFrame, filepath: str) -> None:
    section("1. OVERVIEW")
    file_size = os.path.getsize(filepath) / 1024
    rows = [
        ["File", os.path.basename(filepath)],
        ["File size", f"{file_size:,.1f} KB"],
        ["Rows", f"{len(df):,}"],
        ["Columns", f"{len(df.columns):,}"],
        ["Total cells", f"{df.size:,}"],
        ["Duplicate rows", f"{df.duplicated().sum():,}"],
        ["Total missing values", f"{df.isna().sum().sum():,}"],
        ["Memory usage", f"{df.memory_usage(deep=True).sum() / 1024:,.1f} KB"],
    ]
    print(tabulate(rows, tablefmt="simple"))


def profile_column_types(df: pd.DataFrame) -> None:
    section("2. COLUMN DATA TYPES")
    rows = []
    for col in df.columns:
        rows.append([col, str(df[col].dtype)])
    print(tabulate(rows, headers=["Column", "Dtype"], tablefmt="simple"))

    print("\nType summary:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"  {dtype}: {count} column(s)")


def profile_missing_values(df: pd.DataFrame) -> None:
    section("3. MISSING VALUES")
    total = len(df)
    rows = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = (n_missing / total * 100) if total else 0
        rows.append([col, n_missing, f"{pct:.1f}%"])
    print(tabulate(rows, headers=["Column", "Missing", "% Missing"], tablefmt="simple"))


def profile_numeric_stats(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return

    section("4. NUMERIC COLUMN STATISTICS")
    stats = df[numeric_cols].describe().T
    stats["skew"] = df[numeric_cols].skew()
    stats["kurtosis"] = df[numeric_cols].kurtosis()
    stats = stats.round(2)
    print(tabulate(stats, headers="keys", tablefmt="simple"))

    # zeros and negatives
    print("\nZeros & negatives:")
    rows = []
    for col in numeric_cols:
        n_zero = (df[col] == 0).sum()
        n_neg = (df[col] < 0).sum()
        rows.append([col, n_zero, n_neg])
    print(tabulate(rows, headers=["Column", "Zeros", "Negatives"], tablefmt="simple"))


def profile_categorical_stats(df: pd.DataFrame) -> None:
    cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    if not cat_cols:
        return

    section("5. CATEGORICAL COLUMN STATISTICS")
    rows = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        top_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
        top_freq = df[col].value_counts().iloc[0] if n_unique > 0 else 0
        avg_len = df[col].dropna().astype(str).str.len().mean()
        rows.append([col, n_unique, top_val, top_freq, f"{avg_len:.1f}"])
    print(tabulate(
        rows,
        headers=["Column", "Unique", "Top Value", "Top Freq", "Avg Length"],
        tablefmt="simple",
    ))


def profile_value_distributions(df: pd.DataFrame, top_n: int = 5) -> None:
    cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    if not cat_cols:
        return

    section("6. TOP VALUE DISTRIBUTIONS (categorical)")
    for col in cat_cols:
        print(f"\n  -- {col} --")
        vc = df[col].value_counts().head(top_n)
        total = len(df)
        rows = []
        for val, cnt in vc.items():
            pct = cnt / total * 100
            bar = "#" * int(pct / 2)
            rows.append([val, cnt, f"{pct:.1f}%", bar])
        print(tabulate(rows, headers=["Value", "Count", "%", ""], tablefmt="simple"))


def profile_correlations(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return

    section("7. NUMERIC CORRELATIONS (Pearson)")
    corr = df[numeric_cols].corr().round(2)
    print(tabulate(corr, headers="keys", tablefmt="simple", showindex=True))

    # highlight strong correlations
    print("\nStrong correlations (|r| >= 0.7, excluding self):")
    found = False
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            r = corr.loc[c1, c2]
            if abs(r) >= 0.7:
                print(f"  {c1}  <->  {c2}:  {r}")
                found = True
    if not found:
        print("  (none)")


def plot_top_correlations(df: pd.DataFrame, filepath: str, top_n: int = 10) -> None:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        print("\n  Not enough numeric columns to plot correlations.")
        return

    section("9. TOP CORRELATION PAIRS (graph)")

    corr = df[numeric_cols].corr()

    # collect all unique pairs with their absolute correlation
    pairs = []
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            r = corr.loc[c1, c2]
            pairs.append((f"{c1}  vs  {c2}", r))

    # sort by absolute value, take top N
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    pairs = pairs[:top_n]

    if not pairs:
        print("  No column pairs to plot.")
        return

    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.6)))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Pearson Correlation")
    ax.set_title(f"Top {len(pairs)} Most Correlated Column Pairs")
    ax.set_xlim(-1.05, 1.05)
    ax.axvline(x=0, color="gray", linewidth=0.5)

    # add value labels on bars
    for bar, val in zip(bars, values[::-1]):
        x_pos = bar.get_width() + (0.03 if val >= 0 else -0.03)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=9, fontweight="bold")

    plt.tight_layout()

    # save next to the CSV file
    base = os.path.splitext(filepath)[0]
    output_path = f"{base}_top_correlations.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Graph saved to: {output_path}")


def profile_sample(df: pd.DataFrame, n: int = 5) -> None:
    section("8. SAMPLE DATA (first rows)")
    print(tabulate(df.head(n), headers="keys", tablefmt="simple", showindex=False))


# ── main ─────────────────────────────────────────────────────────────────────

def profile_csv(filepath: str) -> None:
    """Run the full profiling pipeline on a CSV file."""
    print(f"\nLoading {filepath} ...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows x {len(df.columns):,} columns.\n")

    profile_overview(df, filepath)
    profile_column_types(df)
    profile_missing_values(df)
    profile_numeric_stats(df)
    profile_categorical_stats(df)
    profile_value_distributions(df)
    profile_correlations(df)
    plot_top_correlations(df, filepath)
    profile_sample(df)

    print(f"\n{'=' * 60}")
    print("  Profiling complete.")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a CSV file.")
    parser.add_argument("file", nargs="?", help="Path to a CSV file")
    parser.add_argument(
        "--dir",
        default=os.path.expanduser("~/Downloads"),
        help="Directory to scan for CSV files (default: ~/Downloads)",
    )
    args = parser.parse_args()

    if args.file:
        filepath = args.file
    else:
        filepath = pick_file(args.dir)

    if not os.path.isfile(filepath):
        print(f"Error: file not found: {filepath}")
        sys.exit(1)

    profile_csv(filepath)


if __name__ == "__main__":
    main()

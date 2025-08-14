#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot sentence frequency per year (no keyword filtering, **no CSV output**) — Seaborn version.

- Reads a CSV containing long texts (e.g., speeches), splits each text into
  sentences (robust PDF layout unwrapping), aggregates the total number of
  sentences per calendar year, and plots the totals using **seaborn**.
- Optionally saves a PNG chart; does **not** write any CSV.

Example
  python sentence_frequency_over_years_no_csv_seaborn.py \
      --input speechesEdit.csv \
      --text-col speechContent \
      --date-col date \
      --out-plot sentence_frequency_by_year.png --show

Notes
- Requires seaborn (install: `pip install seaborn`).
- Windows/PyCharm safe (__main__ guard + freeze_support()).
"""
from __future__ import annotations
import argparse
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

# --- NEW: seaborn import and theme ---
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
except Exception as e:
    raise SystemExit(
        "Seaborn is required for plotting. Install it with `pip install seaborn`.\n"
        f"Original import error: {e}"
    )

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ---------------------------
# Sentence splitting & layout unwrapping
# ---------------------------
# Fixed-width lookbehind for sentence end, then consume closing quotes/brackets.
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])[\"”»')\]]*\s+")

NL_NORM_RE = re.compile(r"\r\n?")
SOFT_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
SINGLE_NL_RE = re.compile(r"(?<!\n)\n(?!\n)")  # single \n not part of a blank-line paragraph break
MULTISPACE_RE = re.compile(r"\s{2,}")


def unwrap_layout(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = NL_NORM_RE.sub("\n", text)
    t = SOFT_HYPHEN_LINEBREAK_RE.sub(r"\1\2", t)  # dehyphenate across line break
    t = SINGLE_NL_RE.sub(" ", t)                    # join soft line wraps
    t = MULTISPACE_RE.sub(" ", t)
    return t.strip()


# ---------------------------
# IO
# ---------------------------

def read_csv(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, engine="pyarrow")
    except Exception:
        return pd.read_csv(csv_path)


# ---------------------------
# Core logic
# ---------------------------

def count_sentences(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    clean = unwrap_layout(text)
    parts = [s for s in SENT_SPLIT_RE.split(clean) if s and s.strip()]
    return len(parts)


def aggregate_sentence_counts_by_year(df: pd.DataFrame, *, text_col: str, date_col: str) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Input CSV must contain a '{text_col}' column.")
    # Ensure date is datetime for reliable year extraction
    if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        with pd.option_context("mode.chained_assignment", None):
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            except Exception:
                df[date_col] = pd.NaT

    counter: Counter[int] = Counter()

    it = df.iterrows()
    if tqdm is not None:
        it = tqdm(it, total=len(df), desc="Counting sentences")

    for _, row in it:
        text = row.get(text_col, None)
        n_sent = count_sentences(text)
        if n_sent == 0:
            continue
        year = None
        if date_col in df.columns:
            d = row.get(date_col, pd.NaT)
            if pd.notna(d):
                try:
                    year = pd.Timestamp(d).year
                except Exception:
                    year = None
        if year is not None:
            counter[year] += n_sent

    if not counter:
        return pd.DataFrame(columns=["year", "num_sentences"])  # empty

    years = sorted(counter.keys())
    full_range = range(min(years), max(years) + 1)
    counts = pd.Series({y: counter.get(y, 0) for y in full_range}, name="num_sentences")
    out = counts.rename_axis("year").reset_index()
    return out


def plot_counts(df_year: pd.DataFrame, *, out_png: str | None, show: bool) -> None:
    if df_year.empty:
        print("No sentence counts to plot.")
        return
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=df_year, x="year", y="num_sentences", marker="o")
    ax.set_title("Anzahl Sätze (pro Jahr)")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Anzahl Sätze ")
    # Optional: emphasize horizontal grid (whitegrid theme already provides one)
    ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot to: {out_png}")
    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------
# CLI
# ---------------------------

def main(input_csv: str, *, text_col: str, date_col: str, out_plot: str | None, show: bool) -> None:
    df = read_csv(input_csv)
    counts = aggregate_sentence_counts_by_year(df, text_col=text_col, date_col=date_col)
    plot_counts(counts, out_png=out_plot, show=show)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    ap = argparse.ArgumentParser(description="Plot sentence frequency per year (no CSV output, seaborn plot)")
    ap.add_argument("--input", "-i", default="speechesEdit.csv", help="Input CSV with texts")
    ap.add_argument("--text-col", default="speechContent", help="Column containing the full text")
    ap.add_argument("--date-col", default="date", help="Column with a parseable date")
    ap.add_argument("--out-plot", default="sentence_frequency_by_year.png", help="Where to save the plot (PNG)")
    ap.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = ap.parse_args()

    main(
        args.input,
        text_col=args.text_col,
        date_col=args.date_col,
        out_plot=args.out_plot,
        show=args.show,
    )

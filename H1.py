#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword hits per year — using a *pre-filtered* sentence CSV (Seaborn version).

This is a drop-in rewrite of your "accelerated keyword scanner" that no longer
re-parses speeches. Instead, it reads the CSV that already contains only the
sentences with ≥1 keyword (e.g. `sentences_with_keywords.csv`). Each row is one
hit, so yearly totals are simple row counts by year.

Changes vs. matplotlib version
- Uses **seaborn** for the yearly line plot (with a whitegrid theme).
- Matplotlib is still imported only to create/show the figure; plotting itself is
  done with seaborn's `lineplot`.

Outputs
- Line plot: yearly sum of keyword‑matching sentences (via seaborn)
- Optional per‑speech hit counts CSV (one row per speech with its hits)

CLI
  python keyword_hits_from_prefiltered_csv_seaborn.py \
      --input sentences_with_keywords.csv \
      --date-col date \
      --speech-id-col id \
      --per-speech-output hits_per_speech.csv

Notes
- Windows/PyCharm safe (__main__ guard + freeze_support()).
- Tries pandas read_csv(engine="pyarrow"); falls back gracefully.
- Requires seaborn (install with: `pip install seaborn`).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# Progress bar is optional here (IO-bound and grouping is fast), but available
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# --- NEW: seaborn import and theme ---
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
except Exception as e:
    raise SystemExit(
        "Seaborn is required for plotting. Install it with `pip install seaborn`.\n"
        f"Original import error: {e}"
    )

DEFAULT_INPUT = "sentences_with_keywords.csv"


def read_sentences_csv(path: str, date_col: str = "date") -> pd.DataFrame:
    read_kwargs = {}
    if date_col:
        read_kwargs["parse_dates"] = [date_col]
    try:
        return pd.read_csv(path, engine="pyarrow", **read_kwargs)
    except Exception:
        try:
            return pd.read_csv(path, **read_kwargs)
        except Exception:
            # Final fallback without parse_dates
            return pd.read_csv(path)


def ensure_year_column(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if "year" not in out.columns:
        if date_col in out.columns:
            if not pd.api.types.is_datetime64_any_dtype(out[date_col]):
                with pd.option_context("mode.chained_assignment", None):
                    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
            out["year"] = out[date_col].dt.year
        else:
            # No date, no year; create a placeholder for grouping
            out["year"] = pd.NA
    return out


def plot_hits_per_year(df_sentences: pd.DataFrame, year_col: str = "year") -> None:
    if year_col not in df_sentences.columns:
        print("Hinweis: Keine 'year'-Spalte vorhanden – überspringe Plot.")
        return

    # Each row is a hit => count rows per year
    hits_per_year = (
        df_sentences
        .dropna(subset=[year_col])
        .groupby(year_col, as_index=True)
        .size()
        .sort_index()
    )

    if hits_per_year.empty:
        print("Keine Daten zum Plotten gefunden (prüfe 'year').")
        return

    # Prepare tidy frame for seaborn
    hits_df = hits_per_year.reset_index(name="hits").rename(columns={year_col: "year"})

    plt.figure(figsize=(10, 4))
    ax = sns.lineplot(data=hits_df, x="year", y="hits", marker="o")
    ax.set_title("Sätze mit migrationsbezogenen Stichworten (pro Jahr)")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Anzahl Sätze")
    plt.tight_layout()
    plt.show()


def compute_hits_per_speech(df_sentences: pd.DataFrame, speech_id_col: str, date_col: str = "date") -> pd.DataFrame:
    """Aggregate the sentence‑level rows into one row per speech with its hit count.
    Requires a speech identifier column (e.g., 'id'). Keeps the first non-null
    metadata values for other columns present in the input (excluding 'sentence').
    """
    if speech_id_col not in df_sentences.columns:
        raise KeyError(f"Spalte '{speech_id_col}' nicht in der Eingabe gefunden.")

    # Count sentences per speech
    hits = (
        df_sentences
        .groupby(speech_id_col, as_index=False)
        .size()
        .rename(columns={"size": "hits"})
    )

    # Bring back a single representative row per speech (metadata) and merge hits
    meta_cols = [c for c in df_sentences.columns if c not in {"sentence"}]
    meta = (
        df_sentences[meta_cols]
        .sort_values([speech_id_col, date_col] if date_col in meta_cols else [speech_id_col])
        .drop_duplicates(subset=[speech_id_col], keep="first")
    )

    out = meta.merge(hits, on=speech_id_col, how="left")

    # Ensure year column exists for optional downstream grouping
    out = ensure_year_column(out, date_col=date_col)
    return out


def main(
    input_csv: str = DEFAULT_INPUT,
    date_col: str = "date",
    speech_id_col: Optional[str] = "id",
    per_speech_output: Optional[str] = None,
) -> None:
    df = read_sentences_csv(input_csv, date_col=date_col)

    if "sentence" not in df.columns:
        raise SystemExit("Die Eingabedatei muss eine Spalte 'sentence' enthalten (eine Zeile = ein Satz).")

    df = ensure_year_column(df, date_col=date_col)

    # Plot yearly totals (row counts by year) with seaborn
    plot_hits_per_year(df, year_col="year")

    # Optional: write per‑speech aggregation with 'hits' like the original pipeline
    if speech_id_col and per_speech_output:
        agg = compute_hits_per_speech(df, speech_id_col=speech_id_col, date_col=date_col)
        agg.to_csv(per_speech_output, index=False)
        print(f"Gespeichert: {per_speech_output} (Reden: {len(agg):,})")


if __name__ == "__main__":
    # Windows-safe guard
    import multiprocessing as mp
    import argparse

    mp.freeze_support()

    ap = argparse.ArgumentParser(description="Plot yearly keyword hits from a pre-filtered sentence CSV (seaborn)")
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Pfad zur Satz-CSV (enthält Spalte 'sentence')")
    ap.add_argument("--date-col", default="date", help="Name der Datums-Spalte (falls vorhanden)")
    ap.add_argument("--speech-id-col", default="id", help="Spalte, die eine Rede identifiziert (für per-speech Aggregation)")
    ap.add_argument("--per-speech-output", default=None, help="Wenn gesetzt, schreibe Aggregat je Rede (CSV)")
    args = ap.parse_args()

    main(
        input_csv=args.input,
        date_col=args.date_col,
        speech_id_col=args.speech_id_col if args.speech_id_col else None,
        per_speech_output=args.per_speech_output,
    )

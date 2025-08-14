#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacked sentiment bars per year (seaborn) —
BOTTOM → TOP = negative, neutral, positive.

This version fixes the stack order so that:
  - the **bottom** area is **negative** (red),
  - the **middle** area is **neutral** (grey),
  - the **top** area is **positive** (green).

Fixed colors (green/grey/red) and no transparency.
Edit the constants below to point to your CSV and columns.

CHANGE REQUESTED:
- Force the legend title to "Kategorie" even though the hue column is named "category".
"""
import pandas as pd
import matplotlib.pyplot as plt

# --- seaborn import and theme ---
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
except Exception as e:
    raise SystemExit(
        "Seaborn is required for plotting. Install it with `pip install seaborn`.\n"
        f"Original import error: {e}"
    )

# ======== EDIT THIS ========
INPUT_CSV = "sentences_agree_sentiws_textblob.csv"  # <- set your file here
DATE_COL = "date"                  # if 'year' doesn't exist, we'll derive it from this
CATEGORY_COL = "category_sentiws"  # which column holds the sentiment category
TITLE = "Verteilung der Sentimente (pro Jahr)"
SAVE_PATH = None                   # e.g., "yearly_distribution.png" to save instead of show
# ============================

# Color mapping (supports common English/German/short labels)
CATEGORY_COLOR_MAP = {
    "positive": "#2ca02c",  # green
    "positiv":  "#2ca02c",
    "pos":      "#2ca02c",

    "neutral":  "#7f7f7f",  # grey
    "neu":      "#7f7f7f",
    "0":        "#7f7f7f",

    "negative": "#d62728",  # red
    "negativ":  "#d62728",
    "neg":      "#d62728",
    "-1":       "#d62728",
}

# Desired **stacking** bottom→top: NEGATIVE, NEUTRAL, POSITIVE
POS_ALIASES = ["positive", "positiv", "pos", "+1"]
NEU_ALIASES = ["neutral", "neu", "0"]
NEG_ALIASES = ["negative", "negativ", "neg", "-1"]


def read_any_csv(path: str, date_col: str | None) -> pd.DataFrame:
    read_kwargs = {}
    if date_col:
        read_kwargs["parse_dates"] = [date_col]
    try:
        return pd.read_csv(path, engine="pyarrow", **read_kwargs)
    except Exception:
        try:
            return pd.read_csv(path, **read_kwargs)
        except Exception:
            return pd.read_csv(path)


def ensure_year(df: pd.DataFrame, date_col: str | None) -> None:
    if "year" not in df.columns and date_col and date_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            with pd.option_context("mode.chained_assignment", None):
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year


def main():
    df = read_any_csv(INPUT_CSV, date_col=DATE_COL)

    if "sentence" not in df.columns:
        raise SystemExit("Die CSV muss eine Spalte 'sentence' enthalten (für die Zählung).")

    # Ensure year exists or derive it
    if "year" not in df.columns and (not DATE_COL or DATE_COL not in df.columns):
        print("Hinweis: Keine 'year'- und keine gültige 'date'-Spalte vorhanden – überspringe Plot.")
        return
    ensure_year(df, date_col=DATE_COL)

    # Build pivot (year x category -> counts)
    dist = (
        df
        .pivot_table(index="year", columns=CATEGORY_COL, values="sentence", aggfunc="count", fill_value=0)
        .sort_index()
    )

    if dist.empty:
        print("Keine Daten zum Plotten gefunden.")
        return

    # Long form for seaborn stacked bars
    long_df = (
        dist.reset_index()
            .melt(id_vars="year", var_name="category", value_name="count")
    )
    long_df = long_df[long_df["count"] > 0]
    if long_df.empty:
        print("Keine zählbaren Kategorien nach dem Umformen.")
        return

    # Which labels are present?
    cats_present = [str(c) for c in long_df["category"].unique().tolist()]

    # Build desired bottom→top order: NEG, NEU, POS (only those present)
    stack_bottom_to_top = (
        [c for c in NEG_ALIASES if c in cats_present] +
        [c for c in NEU_ALIASES if c in cats_present] +
        [c for c in POS_ALIASES if c in cats_present]
    )
    if not stack_bottom_to_top:
        stack_bottom_to_top = cats_present

    # seaborn.histplot quirk: the first hue often ends up on TOP in the visual stack.
    # To render bottom→top = NEG, NEU, POS, we pass the REVERSED order to hue_order.
    hue_order_for_plot = list(reversed(stack_bottom_to_top))

    # Palette only for present categories
    palette = {c: CATEGORY_COLOR_MAP.get(c, "#999999") for c in cats_present}

    # Ensure categorical dtype for consistent ordering
    long_df["category"] = pd.Categorical(long_df["category"], categories=hue_order_for_plot, ordered=True)

    plt.figure(figsize=(10, 5))
    ax = sns.histplot(
        data=long_df,
        x="year",
        weights="count",
        hue="category",              # <-- column stays 'category'
        hue_order=hue_order_for_plot,
        multiple="stack",
        discrete=True,
        shrink=0.85,
        element="bars",
        stat="count",
        common_norm=False,
        palette=palette,
        alpha=1.0,  # no transparency
        legend=True,
    )

    # --- Ensure legend title is "Kategorie" regardless of hue column name ---
    # If seaborn already created a legend, rename its title first.
    existing_leg = ax.get_legend()
    if existing_leg is not None:
        existing_leg.set_title("Kategorie")

    ax.set_title(TITLE)
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Anzahl Sätze")

    # Legend in human order (bottom→top = NEG, NEU, POS)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        valid = {l: h for h, l in zip(handles, labels) if l in stack_bottom_to_top}
        ordered_labels = [l for l in stack_bottom_to_top if l in valid]
        ordered_handles = [valid[l] for l in ordered_labels]
        leg = ax.legend(ordered_handles, ordered_labels, title="Kategorie", frameon=False)
        if leg is not None:
            leg.set_title("Kategorie")  # double‑ensure title

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if SAVE_PATH:
        plt.savefig(SAVE_PATH, bbox_inches="tight", dpi=150)
        print(f"Gespeichert: {SAVE_PATH}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

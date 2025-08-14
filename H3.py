#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse fertiger (vor-klassifizierter) Satz-CSV — Seaborn

Änderung (Barplot-Stil wie Referenz):
- Gestapelter Jahres-Balkenplot entspricht nun exakt dem gewünschten Stil:
  * Bottom → Top = NEGATIV (rot), NEUTRAL (grau), POSITIV (grün)
  * Feste Farben (keine Transparenz)
  * Legendentitel immer "Kategorie" – unabhängig vom Spaltennamen
  * Reihenfolge der Legende = NEG, NEU, POS

Weitere Eigenschaften bleiben erhalten:
- Linienplot pro Fraktion mit echten Gaps (z. B. FDP 2013→2017 nicht verbunden)
- "Andere"-Fraktion ausgeschlossen
- Übliche Parteifarben
"""
from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# ---------------------------
# CLI / Defaults
# ---------------------------
DEFAULT_INPUT = "sentences_agree_sentiws_textblob.csv"
FACTION_MAP = {23: "SPD", 13: "FDP", 3: "Grüne", 4: "Union", 0: "AfD", 6: "Linke"}
CATEGORY_ORDER = ["negativ", "neutral", "positiv"]
FACTION_ORDER = ["SPD", "Union", "Grüne", "FDP", "Linke", "AfD"]
PARTY_COLORS = {
    "SPD": "#E3000F",   # Rot
    "Union": "#111111", # Schwarz/Dunkelgrau
    "Grüne": "#1A963A", # Grün
    "FDP": "#FFCC00",   # Gelb
    "Linke": "#BE3075", # Magenta
    "AfD": "#009EE0",   # Blau
}

# Feste Farben für Sentiment-Kategorien (de/en/aliases)
CATEGORY_COLOR_MAP = {
    "positive": "#2ca02c",  # grün
    "positiv":  "#2ca02c",
    "pos":      "#2ca02c",
    "+1":       "#2ca02c",

    "neutral":  "#7f7f7f",  # grau
    "neu":      "#7f7f7f",
    "0":        "#7f7f7f",

    "negative": "#d62728",  # rot
    "negativ":  "#d62728",
    "neg":      "#d62728",
    "-1":       "#d62728",
}
POS_ALIASES = ["positive", "positiv", "pos", "+1"]
NEU_ALIASES = ["neutral", "neu", "0"]
NEG_ALIASES = ["negative", "negativ", "neg", "-1"]

# Seaborn Theme
sns.set_theme(context="notebook", style="whitegrid")


def read_sentences_csv(path: str, date_col: str | None) -> pd.DataFrame:
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


def ensure_category(df: pd.DataFrame, category_col: str, polarity_col: str) -> None:
    if category_col not in df.columns:
        if polarity_col not in df.columns:
            raise SystemExit("Weder Kategorie noch Polarity vorhanden – gib eine davon an.")

        def to_cat(x: float) -> str:
            try:
                if x < -0.01:
                    return "negativ"
                if x > 0.01:
                    return "positiv"
            except Exception:
                pass
            return "neutral"

        df[category_col] = df[polarity_col].apply(to_cat)


def ensure_faction(df: pd.DataFrame, faction_col: str, faction_id_col: str | None) -> None:
    if faction_col not in df.columns:
        if faction_id_col and faction_id_col in df.columns:
            df[faction_col] = df[faction_id_col].map(FACTION_MAP)
        else:
            df[faction_col] = pd.NA
    df[faction_col] = df[faction_col].astype("string")


def export_pivot(df: pd.DataFrame, pivot: pd.DataFrame, export_dir: str, name: str) -> None:
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"{name}.csv")
    pivot.to_csv(path, encoding="utf-8")
    print(f"Exportiert: {path}")


def _filter_known_parties(df: pd.DataFrame, faction_col: str) -> pd.DataFrame:
    # behalte nur definierte Parteien
    mask = df[faction_col].isin(FACTION_ORDER)
    return df.loc[mask].copy()


def _categorical_faction(df: pd.DataFrame, faction_col: str) -> pd.DataFrame:
    df = df.copy()
    df[faction_col] = pd.Categorical(df[faction_col], categories=FACTION_ORDER, ordered=True)
    return df


# ---------------------------
# BARPLOT (Jahr) — Stil wie Referenz
# ---------------------------

def plot_dist_year(df: pd.DataFrame, year_col: str, category_col: str, export_dir: str | None) -> None:
    if year_col not in df.columns:
        print("Hinweis: Keine 'year'-Spalte vorhanden – überspringe Jahresplot.")
        return

    # Pivot: year × category -> counts (für Export und Konsistenz)
    dist_year = (
        df.pivot_table(index=year_col, columns=category_col, values="sentence", aggfunc="count", fill_value=0)
          .sort_index()
    )
    if export_dir:
        export_pivot(df, dist_year, export_dir, "dist_year")

    if dist_year.empty:
        print("Keine Daten zum Plotten gefunden (Jahr/Kategorie).")
        return

    # Long-Form für histplot mit Gewichten
    long_df = (
        dist_year.reset_index()
                 .melt(id_vars=year_col, var_name="category", value_name="count")
    )
    long_df = long_df[long_df["count"] > 0]
    if long_df.empty:
        print("Keine zählbaren Kategorien nach dem Umformen.")
        return

    # Welche Labels sind vorhanden?
    cats_present = [str(c) for c in long_df["category"].unique().tolist()]

    # Bottom→Top Ordnung: NEG, NEU, POS (nur vorhandene)
    stack_bottom_to_top = (
        [c for c in NEG_ALIASES if c in cats_present] +
        [c for c in NEU_ALIASES if c in cats_present] +
        [c for c in POS_ALIASES if c in cats_present]
    )
    if not stack_bottom_to_top:
        stack_bottom_to_top = cats_present

    # Für histplot hue_order so wählen, dass visuell Bottom→Top = NEG, NEU, POS erscheint
    hue_order_for_plot = list(reversed(stack_bottom_to_top))

    # Farbpalette nur für vorhandene Kategorien
    palette = {c: CATEGORY_COLOR_MAP.get(c, "#999999") for c in cats_present}

    # kategorischer Typ zur stabilen Reihenfolge
    long_df["category"] = pd.Categorical(long_df["category"], categories=hue_order_for_plot, ordered=True)

    year_vals = sorted(long_df[year_col].dropna().unique())

    plt.figure(figsize=(10, 5))
    ax = sns.histplot(
        data=long_df,
        x=year_col,
        weights="count",
        hue="category",               # Name bleibt "category"
        hue_order=hue_order_for_plot,
        multiple="stack",
        discrete=True,
        shrink=0.85,
        element="bars",
        stat="count",
        common_norm=False,
        palette=palette,
        alpha=1.0,                     # keine Transparenz
        legend=True,
    )

    # Legendentitel erzwingen
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Kategorie")

    # Legende in menschlicher Reihenfolge (NEG, NEU, POS)
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        # Map Label -> Handle; filtere auf bekannte Reihenfolge
        mapping = {l: h for h, l in zip(handles, labels) if l in stack_bottom_to_top}
        ordered_labels = [l for l in stack_bottom_to_top if l in mapping]
        ordered_handles = [mapping[l] for l in ordered_labels]
        leg = ax.legend(ordered_handles, ordered_labels, title="Kategorie", frameon=False)
        if leg is not None:
            leg.set_title("Kategorie")

    ax.set_title("Verteilung der Sentimente (pro Jahr)")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Anzahl Sätze")
    ax.set_xticks(year_vals)
    ax.set_xticklabels([str(int(y)) for y in year_vals], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ---------------------------
# LINEPLOT (Ø-Polarity) — mit echten Gaps
# ---------------------------

def plot_mean_year_faction(df: pd.DataFrame, year_col: str, faction_col: str, polarity_col: str, export_dir: str | None) -> None:
    if year_col not in df.columns or polarity_col not in df.columns:
        print("Hinweis: Für Ø-Sentiment pro Jahr/Fraktion werden 'year' und 'polarity' benötigt.")
        return

    df2 = _categorical_faction(_filter_known_parties(df, faction_col), faction_col)

    # Vollständige Jahrsliste
    years = sorted(df2[year_col].dropna().unique())

    # Mittelwerte je Jahr/Fraktion → Pivot mit NaN für fehlende Kombinationen
    pivot = (
        df2.groupby([year_col, faction_col])[polarity_col].mean()
           .unstack(faction_col)
           .reindex(index=years)
    )

    if export_dir:
        export_pivot(df2, pivot, export_dir, "mean_year_faction")

    # Zeichnen: pro Partei eine Linie mit NaN → Lücken
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    drawn = False
    cols_ordered = [p for p in FACTION_ORDER if p in pivot.columns]
    for party in cols_ordered:
        y = pivot[party]
        if y.notna().sum() == 0:
            continue
        ax.plot(
            pivot.index,
            y.values,
            marker='o',         # immer Punkte
            linewidth=1.8,
            color=PARTY_COLORS.get(party, None),
            label=party,
        )
        drawn = True

    if not drawn:
        print("Hinweis: Keine Daten für bekannte Parteien vorhanden.")
        return

    ax.axhline(0, linewidth=0.8)
    ax.set_title("Ø-Sentiment pro Jahr und Fraktion")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Durchschnittlicher Sentiment-Score")
    ax.set_xticks(pivot.index)
    ax.set_xticklabels([str(int(y)) for y in pivot.index], rotation=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Fraktion", ncol=2)
    plt.tight_layout()
    plt.show()


# ---------------------------
# BARPLOT (Fraktionen, alle Jahre)
# ---------------------------

def plot_dist_faction(df: pd.DataFrame, faction_col: str, category_col: str, export_dir: str | None) -> None:
    df2 = _categorical_faction(_filter_known_parties(df, faction_col), faction_col)

    dist_faction = df2.pivot_table(index=faction_col, columns=category_col, values="sentence", aggfunc="count", fill_value=0)
    dist_faction = dist_faction.loc[[p for p in FACTION_ORDER if p in dist_faction.index]]

    if export_dir:
        export_pivot(df2, dist_faction, export_dir, "dist_faction")

    # Für Konsistenz: gleiche Kategorie-Farben und Reihenfolge (NEG, NEU, POS)
    long_df = (
        dist_faction.reset_index()
                    .melt(id_vars=faction_col, var_name="category", value_name="count")
    )
    long_df = long_df[long_df["count"] > 0]
    cats_present = [str(c) for c in long_df["category"].unique().tolist()]
    stack_bottom_to_top = (
        [c for c in NEG_ALIASES if c in cats_present] +
        [c for c in NEU_ALIASES if c in cats_present] +
        [c for c in POS_ALIASES if c in cats_present]
    )
    hue_order_for_plot = list(reversed(stack_bottom_to_top)) if stack_bottom_to_top else None
    palette = {c: CATEGORY_COLOR_MAP.get(c, "#999999") for c in cats_present}

    long_df["category"] = pd.Categorical(long_df["category"], categories=hue_order_for_plot, ordered=True)

    plt.figure(figsize=(10, 5))
    ax = sns.histplot(
        data=long_df,
        x=faction_col,
        weights="count",
        hue="category",
        hue_order=hue_order_for_plot,
        multiple="stack",
        discrete=True,
        shrink=0.85,
        element="bars",
        stat="count",
        common_norm=False,
        palette=palette,
        alpha=1.0,
        legend=True,
    )
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title("Kategorie")
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels and stack_bottom_to_top:
        mapping = {l: h for h, l in zip(handles, labels) if l in stack_bottom_to_top}
        ordered_labels = [l for l in stack_bottom_to_top if l in mapping]
        ordered_handles = [mapping[l] for l in ordered_labels]
        leg = ax.legend(ordered_handles, ordered_labels, title="Kategorie", frameon=False)
        if leg is not None:
            leg.set_title("Kategorie")

    ax.set_title("Verteilung der Sentimente nach Fraktion")
    ax.set_xlabel("Fraktion")
    ax.set_ylabel("Anzahl Sätze")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ---------------------------
# MAIN
# ---------------------------

def main(
    input_csv: str,
    date_col: str,
    polarity_col: str,
    category_col: str,
    faction_col: str,
    faction_id_col: str,
    export_dir: str | None,
) -> None:
    df = read_sentences_csv(input_csv, date_col=date_col if date_col else None)

    required_cols = {"sentence"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Eingabe muss Spalte(n) {missing} enthalten.")

    ensure_year(df, date_col=date_col)
    ensure_category(df, category_col=category_col, polarity_col=polarity_col)
    ensure_faction(df, faction_col=faction_col, faction_id_col=faction_id_col)

    # Plots
    plot_dist_year(df, year_col="year", category_col=category_col, export_dir=export_dir)
    plot_mean_year_faction(df, year_col="year", faction_col=faction_col, polarity_col=polarity_col, export_dir=export_dir)
    plot_dist_faction(df, faction_col=faction_col, category_col=category_col, export_dir=export_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyse vor-klassifizierter Satz-CSV — Seaborn (Barplot-Stil wie Referenz)")
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Pfad zur vor-klassifizierten CSV")
    ap.add_argument("--date-col", default="date", help="Name der Datums-Spalte (falls vorhanden)")

    # CHANGE IF AGREE OR NORMAL csv file
    ap.add_argument("--polarity-col", default="polarity_sentiws", help="Spalte mit Polarity (falls vorhanden)")
    ap.add_argument("--category-col", default="category_sentiws", help="Spalte mit Sentiment-Kategorie")
    ap.add_argument("--faction-col", default="faction", help="Spalte mit Fraktionsnamen; wird erzeugt, wenn fehlt")
    ap.add_argument("--faction-id-col", default="factionId", help="Spalte mit Fraktions-ID für Mapping → Name")
    ap.add_argument("--export-dir", default=None, help="Optional: Verzeichnis für Pivot-CSV-Exporte")
    args = ap.parse_args()

    main(
        input_csv=args.input,
        date_col=args.date_col,
        polarity_col=args.polarity_col,
        category_col=args.category_col,
        faction_col=args.faction_col,
        faction_id_col=args.faction_id_col,
        export_dir=args.export_dir,
    )

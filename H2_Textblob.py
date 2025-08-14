#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextBlob-DE–basierte Sentimentanalyse für *vorgefilterte* Satz-CSV.

Was das Skript macht
- Liest eine CSV mit genau einer Zeile pro Satz (Spalte: `sentence`).
- Berechnet Sentiment-Polarität pro Satz via TextBlob-DE (keine erneute Satzsegmentierung).
- Erhält alle Metadaten-Spalten und ergänzt: `polarity`, `category` (negativ/neutral/positiv).
- Stellt sicher, dass `year` existiert (falls `date` vorhanden ist, wird Jahr abgeleitet).
- Speichert eine angereicherte CSV und zeigt ein gestapeltes Balkendiagramm pro Jahr.

Nutzung
  python textblob_from_prefiltered_sentences.py \
      --input sentences_with_keywords.csv \
      --output sentences_with_sentiment_textblob.csv

Hinweise
- Benötigt `textblob-de` (Installation: `pip install textblob-de`).
- Optional: Fortschrittsbalken & Parallelisierung mit `tqdm` (`pip install tqdm`).
- Windows/PyCharm-sicher (__main__-Guard + freeze_support()).
- `read_csv` probiert engine="pyarrow" für Speed; fällt bei Bedarf zurück.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# ---------------------------------------------------------
# 0) Abhängigkeiten prüfen
# ---------------------------------------------------------
try:
    from textblob_de import TextBlob as TextBlobDE
except Exception as exc:
    raise SystemExit(
        "Fehlende Abhängigkeit: 'textblob-de'. Installiere mit 'pip install textblob-de'.\n"
        f"Ursprünglicher Fehler: {exc}"
    )

try:
    from tqdm.auto import tqdm
    from tqdm.contrib.concurrent import process_map  # parallel + progress
except Exception:
    tqdm = None
    process_map = None
    print("Tipp: 'pip install tqdm', um Fortschritt & schnellere Parallelisierung zu sehen.")

# ---------------------------------------------------------
# 1) Defaults (können via CLI überschrieben werden)
# ---------------------------------------------------------
IN_CSV = "sentences_with_keywords.csv"
OUT_CSV = "sentences_with_sentiment_textblob.csv"

# ---------------------------------------------------------
# 2) Sentiment-Helfer
# ---------------------------------------------------------

def textblob_sentence_polarity(text: str) -> float:
    """Gibt Polarität in [-1.0, 1.0] zurück (TextBlob-DE)."""
    blob = TextBlobDE(text or "")
    # TextBlob-DE liefert: polarity ∈ [-1, 1]
    return float(blob.sentiment.polarity or 0.0)


def sentiment_category(p: float) -> str:
    # leichte Toleranz um 0 herum
    if p < -0.01:
        return "negativ"
    if p > 0.01:
        return "positiv"
    return "neutral"

# ---------------------------------------------------------
# 3) I/O- und Preprocessing-Helfer
# ---------------------------------------------------------

def read_sentences_csv(path: str) -> pd.DataFrame:
    # Versuche 'date' zu parsen, falls vorhanden, für die spätere Jahresableitung
    try:
        return pd.read_csv(path, engine="pyarrow", parse_dates=["date"])  # ignoriert, wenn 'date' fehlt
    except Exception:
        try:
            return pd.read_csv(path, parse_dates=["date"])  # Fallback
        except ValueError:
            return pd.read_csv(path)


def ensure_year_column(df: pd.DataFrame) -> None:
    if "year" not in df.columns and "date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            with pd.option_context("mode.chained_assignment", None):
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

# ---------------------------------------------------------
# 4) Polaritätsberechnung (vektorisiert/parallel)
# ---------------------------------------------------------

def compute_polarity_series(texts: List[str], use_processes: bool = True) -> List[float]:
    if use_processes and process_map is not None:
        max_workers = min(32, (os.cpu_count() or 2))
        return process_map(
            textblob_sentence_polarity,
            texts,
            max_workers=max_workers,
            chunksize=500,
            desc="Berechne TextBlob-DE"
        )
    # Fallback: sequentiell (ggf. mit tqdm)
    iterator = texts
    if tqdm is not None:
        iterator = tqdm(texts, total=len(texts), desc="Berechne TextBlob-DE")
    return [textblob_sentence_polarity(s) for s in iterator]

# ---------------------------------------------------------
# 5) Plot
# ---------------------------------------------------------

def plot_yearly_distribution(sent_df: pd.DataFrame) -> None:
    if "year" not in sent_df.columns:
        print("Hinweis: Keine 'year'-Spalte vorhanden – überspringe Plot.")
        return
    dist = (
        sent_df
        .pivot_table(index="year", columns="category", values="sentence", aggfunc="count", fill_value=0)
        .sort_index()
    )
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    dist.plot(kind="bar", stacked=True, width=0.85, ax=ax)
    plt.title("Verteilung der Sentiment‑Kategorien (TextBlob‑DE) in Stichwort‑Sätzen pro Jahr")
    plt.xlabel("Jahr")
    plt.ylabel("Anzahl Sätze")
    plt.legend(title="Kategorie")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 6) Main
# ---------------------------------------------------------

def main(input_csv: str = IN_CSV, output_csv: str = OUT_CSV, use_processes: bool = True) -> None:
    df = read_sentences_csv(input_csv)
    if "sentence" not in df.columns:
        raise SystemExit("Die Eingabedatei muss eine Spalte 'sentence' enthalten.")

    ensure_year_column(df)

    texts = df["sentence"].astype(str).tolist()
    if os.getenv("FORCE_SEQUENTIAL", "0") == "1":
        use_processes = False

    polarities = compute_polarity_series(texts, use_processes=use_processes)

    out = df.copy()
    out["polarity"] = polarities
    out["category"] = out["polarity"].apply(sentiment_category)

    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Gespeichert: {output_csv} (Zeilen: {len(out):,})")

    plot_yearly_distribution(out)


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    mp.freeze_support()

    ap = argparse.ArgumentParser(description="TextBlob-DE-Sentiment über vorgefilterte Satz-CSV berechnen")
    ap.add_argument("--input", "-i", default=IN_CSV, help="Pfad zur Satz-CSV (muss 'sentence' enthalten)")
    ap.add_argument("--output", "-o", default=OUT_CSV, help="Pfad zur Ausgabedatei")
    ap.add_argument("--no-parallel", action="store_true", help="Parallelisierung deaktivieren")
    args = ap.parse_args()

    main(input_csv=args.input,
         output_csv=args.output,
         use_processes=not args.no_parallel)

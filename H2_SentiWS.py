#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SentiWS-based sentiment for *pre-filtered* sentence CSVs.

What this does
- Reads a CSV that already contains one row per sentence (column: `sentence`).
- Computes SentiWS polarity per sentence (no re-splitting or keyword filtering).
- Preserves all metadata columns and adds: `polarity`, `category` (negativ/neutral/positiv).
- Ensures/derives a `year` column if `date` exists.
- Saves an enriched CSV and shows a stacked bar chart by year.

Usage
  python sentiws_from_prefiltered_sentences.py \
      --input sentences_with_keywords.csv \
      --sentiws SentiWS_v2.0 \
      --output sentences_with_sentiment_sentiws.csv

Notes
- Windows/PyCharm safe (__main__ guard + freeze_support()).
- Optional parallelism via tqdm.contrib.concurrent.process_map.
- Tries engine="pyarrow" for faster read_csv; falls back if unavailable.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

# ---------------------------------------------------------
# 0) Progress bar + optional parallel mapping
# ---------------------------------------------------------
try:
    from tqdm.auto import tqdm
    from tqdm.contrib.concurrent import process_map  # parallel + progress
except Exception:
    tqdm = None
    process_map = None
    print("Tipp: 'pip install tqdm', um Fortschritt & schnellere Parallelisierung zu sehen.")

# ---------------------------------------------------------
# 1) Defaults (can be overridden by CLI)
# ---------------------------------------------------------
SENTIWS_DIR = "SentiWS_v2.0"
IN_CSV = "sentences_with_keywords.csv"
OUT_CSV = "sentences_with_sentiment_sentiws.csv"

# Tokenization for SentiWS lookup
TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß\-]+", re.UNICODE)

# ---------------------------------------------------------
# 2) Lazy-loaded SentiWS lexicon
# ---------------------------------------------------------
_SENTIWS_LEX: Dict[str, float] = None
_LINE_RE = re.compile(r"^(.+?)\|([A-Z]{2,})\t([-+]?\d+[\.,]?\d*)\t?(.*)$")


def load_sentiws_lexicon(base_dir: str) -> Dict[str, float]:
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"SentiWS-Verzeichnis nicht gefunden: {base_dir}. Bitte mit --sentiws angeben."
        )
    lex: Dict[str, float] = {}
    for fname in os.listdir(base_dir):
        if not fname.lower().endswith(".txt"):
            continue
        with open(os.path.join(base_dir, fname), "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                m = _LINE_RE.match(line)
                if not m:
                    continue
                lemma, _pos, score_str, forms = m.groups()
                score = float(score_str.replace(",", "."))
                forms_set = {lemma.lower()}
                if forms:
                    forms_set.update({w.strip().lower() for w in forms.split(",") if w.strip()})
                for w in forms_set:
                    lex[w] = score
    return lex


def get_lex() -> Dict[str, float]:
    global _SENTIWS_LEX
    if _SENTIWS_LEX is None:
        _SENTIWS_LEX = load_sentiws_lexicon(SENTIWS_DIR)
    return _SENTIWS_LEX

# ---------------------------------------------------------
# 3) Sentiment helpers
# ---------------------------------------------------------

def sentiws_sentence_polarity(text: str) -> float:
    """Mean of SentiWS scores over tokens present; returns 0.0 if none found."""
    lex = get_lex()
    toks = [t.lower() for t in TOKEN_RE.findall(text or "")]
    scores = [lex[t] for t in toks if t in lex]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def sentiment_category(p: float) -> str:
    if p < -0.01:
        return "negativ"
    if p > 0.01:
        return "positiv"
    return "neutral"

# ---------------------------------------------------------
# 4) I/O helpers
# ---------------------------------------------------------

def read_sentences_csv(path: str) -> pd.DataFrame:
    # Try to parse 'date' if present for plotting/deriving year
    try:
        return pd.read_csv(path, engine="pyarrow", parse_dates=["date"])  # ignored if 'date' missing
    except Exception:
        try:
            return pd.read_csv(path, parse_dates=["date"])  # falls back gracefully
        except ValueError:
            return pd.read_csv(path)


def ensure_year_column(df: pd.DataFrame) -> None:
    if "year" not in df.columns and "date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            with pd.option_context("mode.chained_assignment", None):
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

# ---------------------------------------------------------
# 5) Polarity computation (vectorized/parallel)
# ---------------------------------------------------------

def compute_polarity_series(texts: List[str], use_processes: bool = True) -> List[float]:
    if use_processes and process_map is not None:
        max_workers = min(32, (os.cpu_count() or 2))
        return process_map(sentiws_sentence_polarity, texts, max_workers=max_workers, chunksize=500, desc="Berechne SentiWS")
    # Fallback sequential
    iterator = texts
    if tqdm is not None:
        iterator = tqdm(texts, total=len(texts), desc="Berechne SentiWS")
    return [sentiws_sentence_polarity(s) for s in iterator]

# ---------------------------------------------------------
# 6) Plot
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
    plt.title("Verteilung der Sentiment‑Kategorien (SentiWS) in Stichwort‑Sätzen pro Jahr")
    plt.xlabel("Jahr")
    plt.ylabel("Anzahl Sätze")
    plt.legend(title="Kategorie")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 7) Main
# ---------------------------------------------------------

def main(input_csv: str = IN_CSV, sentiws_dir: str = SENTIWS_DIR, output_csv: str = OUT_CSV, use_processes: bool = True) -> None:
    global SENTIWS_DIR
    SENTIWS_DIR = sentiws_dir  # set for get_lex()

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

    ap = argparse.ArgumentParser(description="SentiWS-Sentiment über vorgefilterte Satz-CSV berechnen")
    ap.add_argument("--input", "-i", default=IN_CSV, help="Pfad zur Satz-CSV (muss 'sentence' enthalten)")
    ap.add_argument("--sentiws", "-s", default=SENTIWS_DIR, help="Pfad zum SentiWS-Verzeichnis (z. B. SentiWS_v2.0)")
    ap.add_argument("--output", "-o", default=OUT_CSV, help="Pfad zur Ausgabedatei")
    ap.add_argument("--no-parallel", action="store_true", help="Parallelisierung deaktivieren")
    args = ap.parse_args()

    main(input_csv=args.input,
         sentiws_dir=args.sentiws,
         output_csv=args.output,
         use_processes=not args.no_parallel)

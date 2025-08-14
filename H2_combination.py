#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finde alle Sätze (inkl. Metadaten/Features), bei denen **SentiWS** und **TextBlob**
dieselbe Sentiment-Kategorie vergeben haben.

Eingaben (CSV)
- sentences_with_sentiment_sentiws.csv  (mind. Spalten: sentence, category[, polarity, ...])
- sentences_with_sentiment_textblob.csv (dito)

Ausgabe
- sentences_agree_sentiws_textblob.csv  — vereinigte Metadaten + category/polarity beider Methoden

Join-Logik (anpassbar per --join-keys)
Versucht automatisch sinnvolle Schlüssel in dieser Reihenfolge:
  1) id,row_index,sentence_index
  2) id,date,sentence
  3) id,sentence
  4) date,sentence
  5) sentence

Wichtig
- **Alle Metadaten/Features** erscheinen im Output **ohne** Prä-/Suffixe – selbst bei Namenskonflikten.
  Bei Konflikten wird die **SentiWS**-Spalte bevorzugt, ansonsten die **TextBlob**-Spalte.
- Die Spalte **sentence** heißt im Output immer genau so (kein *_sentiws oder *_textblob).
- **Nur** die Spalten **category**/**polarity** werden mit Suffixen `_sentiws` und `_textblob` ausgegeben.

Beispiel
  python agreeing_sentences_sentiws_vs_textblob.py \
      --sentiws sentences_with_sentiment_sentiws.csv \
      --textblob sentences_with_sentiment_textblob.csv \
      --out sentences_agree_sentiws_textblob.csv

Optional
  --join-keys id,sentence           # eigene Join-Schlüssel setzen
  --cat-col-sentiws category        # Kategoriespalte in der SentiWS-CSV
  --cat-col-textblob category       # Kategoriespalte in der TextBlob-CSV
  --pol-col-sentiws polarity        # Polarityspalte in der SentiWS-CSV (optional)
  --pol-col-textblob polarity       # Polarityspalte in der TextBlob-CSV (optional)
"""
from __future__ import annotations
import argparse
from typing import List, Set
import pandas as pd

DEFAULT_SENTIWS = "sentences_with_sentiment_sentiws.csv"
DEFAULT_TEXTBLOB = "sentences_with_sentiment_textblob.csv"
DEFAULT_OUT = "sentences_agree_sentiws_textblob.csv"


def read_csv(path: str) -> pd.DataFrame:
    # 'date' ggf. parsen, wenn vorhanden
    try:
        return pd.read_csv(path, engine="pyarrow", parse_dates=["date"])  # ignoriert, falls 'date' fehlt
    except Exception:
        try:
            return pd.read_csv(path, parse_dates=["date"])  # Fallback
        except Exception:
            return pd.read_csv(path)


def auto_join_keys(df_sw: pd.DataFrame, df_tb: pd.DataFrame) -> List[str]:
    candidates = [
        ["id", "row_index", "sentence_index"],
        ["id", "date", "sentence"],
        ["id", "sentence"],
        ["date", "sentence"],
        ["sentence"],
    ]
    for keys in candidates:
        if all(k in df_sw.columns and k in df_tb.columns for k in keys):
            return keys
    # letzter Rückfall: größter sinnvoller Schnitt beider Spaltenmengen, der 'sentence' enthält
    inter = [c for c in df_sw.columns if c in df_tb.columns]
    if "sentence" in inter:
        return ["sentence"]
    raise SystemExit("Keine passenden Join-Schlüssel gefunden. Bitte --join-keys angeben.")


def main(
    sentiws_csv: str,
    textblob_csv: str,
    out_csv: str,
    join_keys_arg: str | None,
    cat_col_sentiws: str,
    cat_col_textblob: str,
    pol_col_sentiws: str | None,
    pol_col_textblob: str | None,
) -> None:
    df_sw = read_csv(sentiws_csv)
    df_tb = read_csv(textblob_csv)

    # Join-Schlüssel bestimmen
    if join_keys_arg:
        join_keys = [k.strip() for k in join_keys_arg.split(",") if k.strip()]
    else:
        join_keys = auto_join_keys(df_sw, df_tb)
    if not join_keys:
        raise SystemExit("Leere Join-Schlüssel übergeben.")

    # Zusammenführen (inner join) — behalte Konflikte mit Suffixen
    merged = pd.merge(
        df_sw,
        df_tb,
        how="inner",
        on=join_keys,
        suffixes=("_sentiws", "_textblob"),
        copy=False,
    )

    # Sicherstellen, dass 'sentence' ohne Suffix existiert (falls nicht Key)
    if "sentence" not in merged.columns:
        if "sentence_sentiws" in merged.columns:
            merged["sentence"] = merged["sentence_sentiws"]
        elif "sentence_textblob" in merged.columns:
            merged["sentence"] = merged["sentence_textblob"]

    # Spaltennamen für Kategorie/Polarity im Merge-Resultat
    cat_sw = f"{cat_col_sentiws}_sentiws" if cat_col_sentiws in df_sw.columns else cat_col_sentiws
    cat_tb = f"{cat_col_textblob}_textblob" if cat_col_textblob in df_tb.columns else cat_col_textblob

    if cat_sw not in merged.columns or cat_tb not in merged.columns:
        raise SystemExit(
            f"Kategoriespalten nicht gefunden (gesucht: '{cat_sw}' und '{cat_tb}'). Prüfe Parameter/CSV-Spalten."
        )

    agree = merged[merged[cat_sw] == merged[cat_tb]].copy()

    # Optionale Polarity-Spaltennamen im Merge-Resultat
    pol_sw = None
    pol_tb = None
    if pol_col_sentiws and pol_col_sentiws in df_sw.columns:
        pol_sw = f"{pol_col_sentiws}_sentiws"
    if pol_col_textblob and pol_col_textblob in df_tb.columns:
        pol_tb = f"{pol_col_textblob}_textblob"

    # --- Features/Metadaten ohne Suffixe herstellen ---
    def feature_set(df: pd.DataFrame, cat_col: str, pol_col: str | None) -> Set[str]:
        drop = set(join_keys)
        drop.add(cat_col)
        if pol_col:
            drop.add(pol_col)
        return {c for c in df.columns if c not in drop}

    feat_sw = feature_set(df_sw, cat_col_sentiws, pol_col_sentiws)
    feat_tb = feature_set(df_tb, cat_col_textblob, pol_col_textblob)
    features = [c for c in dict.fromkeys([*df_sw.columns, *df_tb.columns]) if c in (feat_sw | feat_tb)]

    # Für jede Feature-Spalte unsuffigierte Ausgabe erzeugen (SentiWS bevorzugt, sonst TextBlob)
    for c in features:
        if c in join_keys:
            continue
        if c == "sentence":
            # sentence bereits unsuffigiert vorhanden (durch Join-Key oder oben hergestellt)
            continue
        if c in agree.columns:
            # existiert schon ohne Suffix (z. B. wenn es nur in einer Quelle vorkam)
            continue
        sw_col = f"{c}_sentiws"
        tb_col = f"{c}_textblob"
        if sw_col in agree.columns:
            agree[c] = agree[sw_col]
        elif tb_col in agree.columns:
            agree[c] = agree[tb_col]
        # Falls beides fehlt, ignorieren

    # --- Ausgabespalten zusammenstellen ---
    cols_out: List[str] = []

    # 1) Join-Keys ohne Suffixe (Reihenfolge beibehalten)
    cols_out.extend([k for k in join_keys if k in agree.columns])

    # 2) sentence explizit früh platzieren, falls nicht schon in Join-Keys
    if "sentence" in agree.columns and "sentence" not in cols_out:
        cols_out.append("sentence")

    # 3) Alle unsuffigierten Feature-Spalten
    for c in features:
        if c not in cols_out and c in agree.columns and not c.endswith(("_sentiws", "_textblob")):
            if c not in {cat_col_sentiws, cat_col_textblob, pol_col_sentiws, pol_col_textblob}:
                cols_out.append(c)

    # 4) Kategorien/Polarities beider Methoden (mit Suffixen)
    if cat_sw not in cols_out:
        cols_out.append(cat_sw)
    if cat_tb not in cols_out:
        cols_out.append(cat_tb)
    if pol_sw and pol_sw in agree.columns and pol_sw not in cols_out:
        cols_out.append(pol_sw)
    if pol_tb and pol_tb in agree.columns and pol_tb not in cols_out:
        cols_out.append(pol_tb)

    # 5) Suffixierte Feature-Spalten aus der Ausgabe entfernen (nur cat/pol dürfen Suffixe haben)
    allowed_suffix_cols = {cat_sw, cat_tb}
    if pol_sw:
        allowed_suffix_cols.add(pol_sw)
    if pol_tb:
        allowed_suffix_cols.add(pol_tb)
    cols_out = [c for c in cols_out if (not c.endswith("_sentiws") and not c.endswith("_textblob")) or c in allowed_suffix_cols]

    # Deduplizieren & speichern
    seen = set()
    cols_final = [c for c in cols_out if (c in agree.columns and not (c in seen or seen.add(c)))]

    agree.loc[:, cols_final].to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Gespeichert: {out_csv}  (Zeilen: {len(agree):,} | Join-Keys: {join_keys})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sätze speichern, bei denen SentiWS- und TextBlob-Kategorie übereinstimmen")
    ap.add_argument("--sentiws", default=DEFAULT_SENTIWS, help="Pfad zur SentiWS-CSV")
    ap.add_argument("--textblob", default=DEFAULT_TEXTBLOB, help="Pfad zur TextBlob-CSV")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Pfad für die Ergebnis-CSV")
    ap.add_argument("--join-keys", default=None, help="Kommagetrennte Liste von Join-Schlüsseln (z. B. id,sentence)")
    ap.add_argument("--cat-col-sentiws", default="category", help="Kategoriespalte in der SentiWS-CSV")
    ap.add_argument("--cat-col-textblob", default="category", help="Kategoriespalte in der TextBlob-CSV")
    ap.add_argument("--pol-col-sentiws", default="polarity", help="(Optional) Polarityspalte in der SentiWS-CSV")
    ap.add_argument("--pol-col-textblob", default="polarity", help="(Optional) Polarityspalte in der TextBlob-CSV")
    args = ap.parse_args()

    main(
        sentiws_csv=args.sentiws,
        textblob_csv=args.textblob,
        out_csv=args.out,
        join_keys_arg=args.join_keys,
        cat_col_sentiws=args.cat_col_sentiws,
        cat_col_textblob=args.cat_col_textblob,
        pol_col_sentiws=args.pol_col_sentiws,
        pol_col_textblob=args.pol_col_textblob,
    )

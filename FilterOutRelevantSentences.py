#!/usr/bin/env python3
"""
Extractor (V3) — fixes the Python `re` error: "re.error: repetition not allowed
inside lookbehind" / "look-behind requires fixed-width pattern" by using a
**fixed-width** lookbehind and consuming any closing quotes/brackets **outside**
the lookbehind.

Also keeps the layout unwrapping (join soft linebreaks, dehyphenate) to avoid
partial sentences from PDF text.
"""
from __future__ import annotations
import argparse
import re
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ---------------------------
# Keywords + normalization (same list)
# ---------------------------

# pass, wahlentscheid, landesminist, neid,
STICHWORTE = [
    "fluchtling","staatsangehor","weiterbild","asylbewerb","zuwander","aussiedl","asyl","asylrecht",
    "migration","migrant","abschieb","asylverfahr","ruckfuhr","einreis","zuwand","herkunftsland",
    "kontingent","einwander","fluchtursach","asylantrag","aufenthaltsrecht","asylbewerberleistungsgesetz",
    "ubersiedl","migrantinn","familiennachzug","asylsuch","auslandergesetz","migrationshintergrund",
    "integrationspolit","fachkraftemangel","auslanderrecht","auslanderbehord","zuwanderungsgesetz","drittstaat",
    "aussengrenz","zuzug","fluchtlingspolit","nationalitat","loyalitat","spataussiedl","zuflucht",
    "bundesvertriebenengesetz","gastarbeit","fluchtlingskonvention","fluchtlingsstrom","einwanderungsland",
    "zustrom","burgerkriegsfluchtling","visum","visa","fluchtlingslag","auslanderpolit",
    "sprachkurs","bleiberecht","asylpolit","einwanderungsgesetz","volljahr","bundesunternehm",
    "anerkennungsverfahr","zwangslag","richtlinienvorschlag","frontex","asylkompromiss","aufenthaltstitel",
    "migrationspolit","sachleist","wahlalt","aufenthaltsstatus","fluchtlingskris","scheng","abschiebehaft",
    "burgerkrieg","sozialcharta","syr","sprachforder","fluchtlingskommissar","volkszugehor",
    "willkommenskultur","aufenthaltsgesetz","wanderungsbeweg","abschiebestopp","aufenthaltserlaubnis",
    "herkunftsstaat","staatsangehorigkeitsgesetz","zentrumsfraktion","altfall","optionspflicht","schutzsuch",
    "sprachkenntnis","staatsangehorigkeitsrecht","bleibeperspektiv","punktesyst","asylverfahrensgeset","windel",
    "nachzug","flüchtlinge","ausländer","flüchtlingen","zuwanderung","vertriebenen","ausländern","asylbewerber",
    "migranten","migration","heimatvertriebenen","aussiedler","einwanderung","ansiedler","vertriebene",
    "zuwanderer","asylbewerbern","flüchtling","heimatvertriebene","sowjetzonenflüchtlinge","aussiedlern",
    "einwanderer","asylsuchenden","asylsuchende","bürgerkriegsflüchtlinge","zuwanderern","ansiedlern",
    "migrantinnen","vertriebener","emigranten","kriegsflüchtlinge","ausländerinnen","immigranten"
]


_TRANSLATE = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "ae", "Ö": "oe", "Ü": "ue", "ß": "ss"})

def normalize(s: str) -> str:
    return s.translate(_TRANSLATE).lower()

STICHWORTE_NORM = sorted({normalize(w) for w in STICHWORTE})
KW_RE = re.compile("(?:" + "|".join(map(re.escape, STICHWORTE_NORM)) + ")")

# ---------------------------------------------------------------------
# Sentence splitting WITHOUT variable-length lookbehind
# We use a fixed-width lookbehind for the end-of-sentence mark, then
# consume any closing quotes/brackets in the separator itself.
# ---------------------------------------------------------------------
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])[\"”»')\]]*\s+")

# --- Layout unwrapping (join soft wraps, remove hyphenation) ---
NL_NORM_RE = re.compile(r"\r\n?")
SOFT_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
SINGLE_NL_RE = re.compile(r"(?<!\n)\n(?!\n)")  # single \n not part of a blank-line paragraph break
MULTISPACE_RE = re.compile(r"\s{2,}")

def unwrap_layout(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = NL_NORM_RE.sub("\n", text)
    t = SOFT_HYPHEN_LINEBREAK_RE.sub(r"\1\2", t)  # dehyphenate across line break
    t = SINGLE_NL_RE.sub(" ", t)                   # join soft line wraps
    t = MULTISPACE_RE.sub(" ", t)
    return t.strip()

# IO helpers

def read_data(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, engine="pyarrow")
    except Exception:
        return pd.read_csv(csv_path)


def find_matches(sentence_norm: str) -> list[str]:
    return sorted(set(KW_RE.findall(sentence_norm)))


def extract(df: pd.DataFrame, text_col: str = "speechContent", date_col: str = "date") -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f"Input CSV must contain a '{text_col}' column.")

    out_rows = []
    it = df.iterrows()
    if tqdm is not None:
        it = tqdm(it, total=len(df), desc="Extract sentences")

    date_is_datetime = date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col])

    for row_idx, row in it:
        full_text = row.get(text_col, None)
        if not isinstance(full_text, str) or not full_text.strip():
            continue

        clean = unwrap_layout(full_text)
        sentences = SENT_SPLIT_RE.split(clean)

        meta = row.to_dict()
        meta.pop(text_col, None)

        for s_idx, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            sent_norm = normalize(sent)
            if KW_RE.search(sent_norm) is None:
                continue
            matched = find_matches(sent_norm)

            rec = {**meta,
                   "row_index": row_idx,
                   "sentence_index": s_idx,
                   "sentence": sent,
                   "matched_keywords_norm": "|".join(matched),
                   "num_keywords": len(matched)}

            if date_col in meta:
                try:
                    d = meta[date_col]
                    if not date_is_datetime:
                        d = pd.to_datetime(d, errors="coerce")
                    if pd.notna(d):
                        rec["year"] = pd.Timestamp(d).year
                except Exception:
                    pass

            out_rows.append(rec)

    if not out_rows:
        return pd.DataFrame(columns=[c for c in df.columns if c != text_col] +
                                   ["row_index","sentence_index","sentence","matched_keywords_norm","num_keywords","year"])

    out_df = pd.DataFrame(out_rows)

    added = ["row_index","sentence_index","sentence","matched_keywords_norm","num_keywords","year"]
    meta_cols = [c for c in out_df.columns if c not in added]
    ordered = meta_cols + [c for c in added if c in out_df.columns]
    return out_df.loc[:, ordered]


def main(input_csv: str, output_csv: str, text_col: str = "speechContent", date_col: str = "date") -> None:
    df = read_data(input_csv)
    if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        with pd.option_context("mode.chained_assignment", None):
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            except Exception:
                pass

    out_df = extract(df, text_col=text_col, date_col=date_col)
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(out_df):,} sentences with keywords to: {output_csv}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", default="speechesEdit.csv")
    ap.add_argument("--output", "-o", default="sentences_with_keywords.csv")
    ap.add_argument("--text-col", default="speechContent")
    ap.add_argument("--date-col", default="date")
    args = ap.parse_args()

    main(args.input, args.output, text_col=args.text_col, date_col=args.date_col)

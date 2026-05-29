"""
Prepare external test sets for the JD relevance classifier.

Pulls three span-annotated job-description datasets from the Hugging Face Hub
(green, skillspan, sayfullina), converts their token+tag span annotations into
binary sentence-level relevance labels (skill or knowledge span present -> 1,
otherwise 0), and writes one TSV per dataset alongside the original training
data.

The training set itself is not produced here. This script is the public,
runnable companion to the model training pipeline -- the original training
corpus cannot be redistributed, so only the externally-sourced test data is
materialised.

Output (all under config.PREPARED_DATA_DIR):
    green_test.tsv
    ss_test.tsv
    say_test.tsv

Each file: Sentence \t label  (label in {0, 1})

Run:
    python prepare_test_data.py
"""

import os
from collections import Counter

import pandas as pd
from datasets import load_dataset

import config

# BIO tags that count as a positive (relevant) span. Mirrors the tag schemes
# used across the three source datasets.
VIABLE_TAGS = ["B-SKILL", "I-SKILL", "B", "B-SOFT"]


def contains_any(lst, targets):
    """True if any element of `targets` appears in `lst`."""
    return any(t in lst for t in targets)


def span_to_binary(example, tag_columns):
    """Set example['label']=1 if any listed tag column contains a viable tag."""
    relevant = any(contains_any(example[col], VIABLE_TAGS) for col in tag_columns)
    return {"label": int(relevant)}


def tokens_to_sentence(example):
    return {"Sentence": " ".join(example["tokens"])}


def clean(df):
    """Cast label to int, drop empty/whitespace-only sentences, reset index."""
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["Sentence"])
    df = df[df["Sentence"].str.strip() != ""]
    return df.reset_index(drop=True)


def prepare_split(ds_split, tag_columns):
    """HF split -> cleaned [Sentence, label] DataFrame."""
    ds = ds_split.map(lambda x: span_to_binary(x, tag_columns))
    ds = ds.map(tokens_to_sentence)
    df = ds.to_pandas()[["Sentence", "label"]]
    return clean(df)


def main():
    print("Loading datasets from Hugging Face Hub ...")
    green = load_dataset("jjzha/green")
    ss = load_dataset("jjzha/skillspan")
    say = load_dataset("jjzha/sayfullina")

    # tag columns differ across datasets -- skillspan has both skill and knowledge
    sources = [
        ("green_test.tsv", green["test"], ["tags_skill"]),
        ("ss_test.tsv", ss["test"], ["tags_skill", "tags_knowledge"]),
        ("say_test.tsv", say["test"], ["tags_skill"]),
    ]

    os.makedirs(config.PREPARED_DATA_DIR, exist_ok=True)

    for filename, split, tag_columns in sources:
        df = prepare_split(split, tag_columns)
        out_path = os.path.join(config.PREPARED_DATA_DIR, filename)
        df.to_csv(out_path, sep="\t", index=False)
        dist = Counter(df["label"])
        print(f"{filename:<16} {len(df):>5} rows  | label dist: {dict(dist)}  -> {out_path}")


if __name__ == "__main__":
    main()

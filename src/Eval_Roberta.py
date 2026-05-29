"""
Reproduce test-set metrics for the trained RoBERTa relevance classifier
without retraining. Loads the saved model via RelevanceClassifier and runs
the same report() the final training script printed.

Usage:
    python eval_roberta.py --model-id Aleksandruz/Binary-Relevance-RoBERTa
"""

import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config
from Inference_Roberta import RelevanceClassifier


def load_test(path):
    df = pd.read_csv(path, sep="\t", dtype=object)
    df = df.dropna(subset=["Sentence"])
    df = df[df["Sentence"].str.strip() != ""]
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def report(name, labels, probs_pos, threshold):
    for tag, thr in [("argmax@0.5", 0.5), (f"tuned@{threshold:.3f}", threshold)]:
        preds = (probs_pos >= thr).astype(int)
        pc = f1_score(labels, preds, average=None)
        print(
            f"  {name:<16} [{tag:<14}] acc={accuracy_score(labels, preds):.4f} "
            f"f1_macro={f1_score(labels, preds, average='macro'):.4f} "
            f"f1_0={pc[0]:.4f} f1_1={pc[1]:.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Aleksandruz/Binary-Relevance-RoBERTa",
                    help="HF repo id or local path")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    clf = RelevanceClassifier(args.model_id, device=args.device)
    print(f"Loaded {args.model_id}  |  threshold = {clf.threshold:.3f}\n")

    print("=== Test metrics ===")
    for name, path in [
        # ("test_expert", config.TEST_EXPERT_DATA),
        # ("test_crossannot", config.TEST_CROSSANNOT_EXPANDED),
        ("test_say", config.TEST_SAY_DATA),
        ("test_ss", config.TEST_SS_DATA),
        ("test_green", config.TEST_GREEN_DATA),
    ]:
        df = load_test(path)
        probs_pos = clf.predict(df["Sentence"].tolist(), mode="prob")[:, 1]
        report(name, df["label"].to_numpy(), probs_pos, clf.threshold)


if __name__ == "__main__":
    main()
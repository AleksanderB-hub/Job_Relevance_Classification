"""
Final training for the RoBERTa relevance classifier (Stage 1).

Reads best_params.json from the Optuna search, trains ONE model on the train
split, refits the decision threshold on the val split, evaluates on the
held-out test sets, then merges the LoRA adapter into the base model and saves
a standalone HF model.

The tuned threshold is written into the model config as `decision_threshold`,
so it travels with the model and is picked up automatically at inference.
A copy is also saved as threshold.json.

Note: the deployed model is trained on the 80% train split (the 20% val split
is held out so the threshold isn't fit on training data). Test numbers come
from the separate test_expert / test_crossannot sets.

Run:
    python final_train_roberta.py
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import config
from Roberta_search import (
    BASE_MODEL,
    LORA_TARGETS,
    NUM_EPOCHS,
    PER_DEVICE_BS,
    SEARCH_DIR,
    SEED,
    SentenceDataset,
    best_threshold,
    build_datasets,
    compute_metrics,
)

FINAL_DIR = os.path.join(config.OUTPUT_DIR, "roberta_final")


def load_test(path):
    """Read a test TSV, dropping rows with missing/empty Sentence (mirrors prior band-aid)."""
    df = pd.read_csv(path, sep="\t", dtype=object)
    df = df.dropna(subset=["Sentence"])
    df = df[df["Sentence"].str.strip() != ""]
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def pos_probs(trainer, df, tokenizer):
    """Positive-class probabilities [N] for a dataframe."""
    logits = trainer.predict(SentenceDataset(df, tokenizer)).predictions
    return torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]


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
    ap = argparse.ArgumentParser(description="Final RoBERTa training + threshold fit + HF export.")
    ap.add_argument("--params", default=os.path.join(SEARCH_DIR, "best_params.json"))
    ap.add_argument("--push", action="store_true", help="push merged model to the HF Hub")
    ap.add_argument("--repo-id", default=None, help="<user>/<repo> (required with --push)")
    args = ap.parse_args()

    with open(args.params) as f:
        best = json.load(f)
    p = best["params"]
    print("Best params:\n" + json.dumps(p, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset, val_dataset, val_labels = build_datasets(tokenizer)

    model_config = AutoConfig.from_pretrained(BASE_MODEL, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, config=model_config)
    model = get_peft_model(
        model,
        LoraConfig(
            r=p["lora_r"],
            lora_alpha=2 * p["lora_r"],
            target_modules=LORA_TARGETS,
            lora_dropout=p["lora_dropout"],
            bias="none",
            task_type="SEQ_CLS",
        ),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=os.path.join(FINAL_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=p["grad_accum"],
        learning_rate=p["learning_rate"],
        warmup_ratio=p["warmup_ratio"],
        weight_decay=p["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to="none",
        seed=SEED,
        fp16=torch.cuda.is_available(),
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)],
    )
    trainer.train()

    # Refit decision threshold on the held-out val split
    val_probs = pos_probs(trainer, val_dataset.dataframe, tokenizer)
    threshold, val_f1 = best_threshold(val_probs, val_labels)
    print(f"\nRefit threshold = {threshold:.3f}  (val macro-F1 = {val_f1:.4f})")

    # Held-out test evaluation
    print("\n=== Test metrics ===")
    for name, path in [
        # ("test_expert", config.TEST_EXPERT_DATA),
        # ("test_crossannot", config.TEST_CROSSANNOT_EXPANDED),
        ("test_say", config.TEST_SAY_DATA),
        ("test_ss", config.TEST_SS_DATA),
        ("test_green", config.TEST_GREEN_DATA),
    ]:
        df = load_test(path)
        report(name, df["label"].to_numpy(), pos_probs(trainer, df, tokenizer), threshold)

    # Merge LoRA into the base model -> standalone model loadable without PEFT
    merged = model.merge_and_unload()
    merged.config.decision_threshold = float(threshold)

    os.makedirs(FINAL_DIR, exist_ok=True)
    merged.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    with open(os.path.join(FINAL_DIR, "threshold.json"), "w") as f:
        json.dump({"decision_threshold": float(threshold)}, f, indent=2)
    print(f"\nSaved merged model + tokenizer to {FINAL_DIR}")

    if args.push:
        if not args.repo_id:
            raise SystemExit("--push requires --repo-id <user>/<repo>")
        merged.push_to_hub(args.repo_id)
        tokenizer.push_to_hub(args.repo_id)
        print(f"Pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
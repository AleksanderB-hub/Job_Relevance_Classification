"""
Optuna hyperparameter search for the RoBERTa + LoRA binary classifier (Stage 1:
JD sentence relevance).

Each trial:
  - samples hyperparameters,
  - fine-tunes roberta-base + LoRA on the train split,
  - selects the best epoch by val macro-F1 (argmax) via early stopping,
  - tunes a decision threshold on the val positive-class probabilities,
  - returns the threshold-tuned val macro-F1 as the objective value.

Median pruning kills unpromising trials early using per-epoch val F1.

Outputs (under RESULTS_DIR/roberta_search/):
  best_params.json   best hyperparameters + tuned threshold
  study.db           Optuna SQLite storage (resumable via --resume)
  trials.csv         all trials

Run:
  python search_roberta.py --n-trials 30
  python search_roberta.py --n-trials 30 --timeout 7200   # stop after 2h
"""

import argparse
import gc
import json
import os
from functools import partial

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

import config

# === Fixed settings ===
BASE_MODEL = "roberta-base"
LORA_TARGETS = ["query", "value"]
MAX_LENGTH = 128
NUM_EPOCHS = 8            # upper bound; early stopping ends most runs sooner
PER_DEVICE_BS = 8         # safe for roberta-base @128 tokens on 12-24GB
SEED = 42

SEARCH_DIR = os.path.join(config.RESULTS_DIR, "roberta_search")
os.makedirs(SEARCH_DIR, exist_ok=True)


class SentenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        example = self.dataframe.iloc[idx]
        encoding = self.tokenizer(
            example["Sentence"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(int(example["label"]), dtype=torch.long)
        return encoding


def compute_metrics(eval_pred):
    """Argmax macro-F1 — used for early stopping / best-epoch selection only."""
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {"f1": f1_score(labels, preds, average="macro")}


def best_threshold(probs_pos, labels):
    """Sweep the positive-class probability threshold; return (threshold, macro_f1)."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs_pos >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


class OptunaPruningCallback(TrainerCallback):
    """Report per-epoch val F1 to Optuna and prune unpromising trials."""

    def __init__(self, trial, metric="eval_f1"):
        self.trial = trial
        self.metric = metric
        self.epoch = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric not in metrics:
            return
        self.trial.report(metrics[self.metric], self.epoch)
        self.epoch += 1
        if self.trial.should_prune():
            raise optuna.TrialPruned()


def build_datasets(tokenizer):
    df = pd.read_csv(config.TRAIN_DATA, sep="\t", dtype=object)
    df["label"] = df["label"].astype(int)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    val_labels = val_df["label"].to_numpy()
    return (
        SentenceDataset(train_df, tokenizer),
        SentenceDataset(val_df, tokenizer),
        val_labels,
    )


def objective(trial, train_dataset, val_dataset, val_labels):
    # --- search space ---
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    lora_r = trial.suggest_categorical("lora_r", [4, 8, 16, 32])
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_steps", 0.0, 0.1)
    grad_accum = trial.suggest_categorical("grad_accum", [1, 2, 4])

    model_config = AutoConfig.from_pretrained(BASE_MODEL, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, config=model_config
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            target_modules=LORA_TARGETS,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
        ),
    )

    trial_dir = os.path.join(SEARCH_DIR, f"trial_{trial.number}")
    training_args = TrainingArguments(
        output_dir=trial_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
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
        callbacks=[
            OptunaPruningCallback(trial),
            EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001),
        ],
    )

    try:
        trainer.train()
        logits = trainer.predict(val_dataset).predictions
        probs_pos = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        threshold, tuned_f1 = best_threshold(probs_pos, val_labels)
        trial.set_user_attr("threshold", threshold)
        result = tuned_f1
    finally:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Optuna search for RoBERTa+LoRA relevance classifier.")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None, help="overall timeout in seconds")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_dataset, val_dataset, val_labels = build_datasets(tokenizer)

    study = optuna.create_study(
        study_name="roberta_relevance",
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_warmup_steps=1),
        storage=f"sqlite:///{os.path.join(SEARCH_DIR, 'study.db')}",
        load_if_exists=True,
    )
    study.optimize(
        partial(objective, train_dataset=train_dataset, val_dataset=val_dataset, val_labels=val_labels),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    best = study.best_trial
    summary = {
        "base_model": BASE_MODEL,
        "lora_targets": LORA_TARGETS,
        "max_length": MAX_LENGTH,
        "num_epochs": NUM_EPOCHS,
        "per_device_batch_size": PER_DEVICE_BS,
        "best_trial": best.number,
        "best_val_macro_f1_tuned": best.value,
        "threshold": best.user_attrs.get("threshold", 0.5),
        "params": best.params,
    }
    with open(os.path.join(SEARCH_DIR, "best_params.json"), "w") as f:
        json.dump(summary, f, indent=2)
    study.trials_dataframe().to_csv(os.path.join(SEARCH_DIR, "trials.csv"), index=False)

    print("\n=== Search complete ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
# Binary Relevance RoBERTa — Training Shell

Public companion to [`Aleksandruz/Binary-Relevance-RoBERTa`](https://huggingface.co/Aleksandruz/Binary-Relevance-RoBERTa)
on the Hugging Face Hub.

This repository contains the **training and evaluation code** for a RoBERTa
sentence-level classifier that labels job-description sentences as
*relevant* (1) or *not relevant* (0) for downstream skill/knowledge
extraction. The model itself lives on the Hub; this repo documents and
reproduces the process.

> **Note on training data.** The primary annotated corpus used to train this
> model cannot be redistributed, so the `prepare_test_data.py` script only
> materialises the publicly-available test sets (from
> [skillspan](https://huggingface.co/datasets/jjzha/skillspan),
> [green](https://huggingface.co/datasets/jjzha/green),
> [sayfullina](https://huggingface.co/datasets/jjzha/sayfullina)). The
> training scripts are included for transparency and reuse on your own data.

## Repository Structure

\`\`\`
.
├── src/
│   ├── config.py                  # shared paths / project root resolution
│   ├── Roberta_Search.py          # Optuna hyperparameter search
│   ├── Full_Roberta_binary.py     # final training + threshold fit + HF push
│   ├── Inference_Roberta.py       # RelevanceClassifier wrapper (prob / clean)
│   ├── Eval_Roberta.py            # evaluate the published model on test sets
│   └── prepare_test_data.py       # build the three public test TSVs
├── Data/Skills/prepared_data/     # public test sets land here
├── results/                       # search artefacts (best_params, trials)
├── requirements.txt
└── README.md
\`\`\`

## Quickstart

Clone, set up an environment, and reproduce the test-set metrics against the
published model:

\`\`\`bash
git clone https://github.com/<your-user>/<repo>.git
cd <repo>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# fetch and prepare the three public test sets
python src/prepare_test_data.py

# evaluate the published HF model on those test sets
python src/Eval_Roberta.py
\`\`\`

\`Eval_Roberta.py\` defaults to the Hub model
\`Aleksandruz/Binary-Relevance-RoBERTa\`. Override with \`--model-id <path-or-id>\`
and pick the device with \`--device cuda|cpu\`.

## Scripts

| Script | Purpose |
| --- | --- |
| \`prepare_test_data.py\` | Loads three span-annotated JD datasets from the Hub, converts their token-level tags to binary sentence-level relevance labels, and writes one TSV per dataset to \`Data/Skills/prepared_data/\`. |
| \`Roberta_Search.py\` | Optuna search over LoRA rank/dropout, learning rate, warmup, weight decay, gradient accumulation. Writes \`results/best_params.json\` and a resumable SQLite study. Requires a training TSV (not included). |
| \`Full_Roberta_binary.py\` | Loads \`best_params.json\`, fine-tunes once, refits the decision threshold on the held-out val split, evaluates on the test sets, merges the LoRA adapter into the base model, and optionally pushes to the Hub (\`--push --repo-id ...\`). |
| \`Inference_Roberta.py\` | Lightweight \`RelevanceClassifier\` wrapper. Two modes: \`prob\` returns \`[N, 2]\` probabilities; \`clean\` returns \`[N]\` 0/1 labels using the tuned threshold stored in the model's \`config.json\`. |
| \`Eval_Roberta.py\` | Reproduces the test-set metrics (accuracy, macro-F1, per-class F1) at both argmax 0.5 and the tuned threshold, against any local or Hub model. |

## Using the Model

The pushed model is a standalone \`RobertaForSequenceClassification\` — no PEFT
needed at inference. The tuned decision threshold rides along inside
\`config.json\` as \`decision_threshold\`.

\`\`\`python
from src.Inference_Roberta import RelevanceClassifier

clf = RelevanceClassifier("Aleksandruz/Binary-Relevance-RoBERTa", device="cuda")

sentences = ["Manage a team of engineers.", "Free snacks provided."]
clf.predict(sentences, mode="clean")   # array([1, 0])
clf.predict(sentences, mode="prob")    # [[P(0), P(1)], ...]
\`\`\`

Or via plain \`transformers\`:

\`\`\`python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("Aleksandruz/Binary-Relevance-RoBERTa")
tokenizer = AutoTokenizer.from_pretrained("Aleksandruz/Binary-Relevance-RoBERTa")
threshold = getattr(model.config, "decision_threshold", 0.5)
\`\`\`

## Training Pipeline (for reference)

If you have your own training data in the same \`Sentence \t label\` TSV
format, the full pipeline is:

\`\`\`bash
# 1. hyperparameter search
python src/Roberta_Search.py --n-trials 30

# 2. final training + evaluation + (optional) Hub push
python src/Full_Roberta_binary.py
python src/Full_Roberta_binary.py --push --repo-id <user>/<repo>
\`\`\`

Configure paths in \`src/config.py\`. The project root is auto-detected from
the location of \`config.py\`, so the scripts work regardless of where the
repo is cloned.

## Model Details

See the [model card on Hugging Face](https://huggingface.co/Aleksandruz/Binary-Relevance-RoBERTa)
for architecture, training data composition, hyperparameters, and test-set
metrics.

## License

[FILL: e.g. MIT, Apache-2.0]

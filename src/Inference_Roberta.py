"""
Single-model inference for the RoBERTa relevance classifier (Stage 1).

Loads ONE fine-tuned model from the Hugging Face Hub (or a local path) by ID,
on a configurable device, and returns:
    mode="prob"  -> [N, 2] float32 array of per-class probabilities (cols = [P(0), P(1)])
    mode="clean" -> [N]    int array of 0/1 using the tuned decision threshold

The decision threshold is read from the model config (`decision_threshold`,
written at training time); pass `threshold=` to override, else falls back to 0.5.

Inputs must already be sentence-split the same way as training (spaCy blank
pipeline + rule-based sentencizer). Use `RelevanceClassifier.sentencize(text)`
to reproduce that split if you're starting from raw text.

Example:
    clf = RelevanceClassifier("user/jd-relevance-roberta", device="cuda")
    clf.predict(["Manage a team of engineers.", "Free snacks provided."])      # -> array([1, 0])
    clf.predict(["Manage a team of engineers."], mode="prob")                   # -> array([[0.02, 0.98]])
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RelevanceClassifier:
    def __init__(self, model_id, device="cuda", threshold=None, max_length=128, batch_size=32):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device).eval()

        self.max_length = max_length
        self.batch_size = batch_size

        cfg_thr = getattr(self.model.config, "decision_threshold", None)
        if threshold is not None:
            self.threshold = float(threshold)
        elif cfg_thr is not None:
            self.threshold = float(cfg_thr)
        else:
            self.threshold = 0.5

    @torch.no_grad()
    def _probs(self, sentences):
        out = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            # float() guards against fp16/quantized logits before softmax
            out.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict(self, sentences, mode="clean"):
        """sentences: str or list[str]. Returns probs [N,2] (prob) or labels [N] (clean)."""
        if mode not in ("prob", "clean"):
            raise ValueError("mode must be 'prob' or 'clean'")
        if isinstance(sentences, str):
            sentences = [sentences]
        sentences = list(sentences)
        if not sentences:
            return np.empty((0, 2), dtype=np.float32) if mode == "prob" else np.empty((0,), dtype=int)

        probs = self._probs(sentences)
        if mode == "prob":
            return probs
        return (probs[:, 1] >= self.threshold).astype(int)

    @staticmethod
    def sentencize(text):
        """Reproduce the training-time split: spaCy blank pipeline + sentencizer."""
        import spacy

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return [s.text.strip() for s in nlp(text).sents if s.text.strip()]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Classify JD sentences as relevant (1) / not (0).")
    ap.add_argument("--model-id", required=True, help="HF repo id or local path")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--mode", default="clean", choices=["clean", "prob"])
    ap.add_argument("--text", required=True, help="raw text; will be sentence-split")
    args = ap.parse_args()

    clf = RelevanceClassifier(args.model_id, device=args.device)
    sents = RelevanceClassifier.sentencize(args.text)
    result = clf.predict(sents, mode=args.mode)
    for sent, r in zip(sents, result):
        print(f"{r}\t{sent}")
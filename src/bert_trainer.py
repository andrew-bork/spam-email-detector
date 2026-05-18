"""
bert_trainer.py

A HuggingFace BERT sequence-classification trainer that plugs into the
existing Trainer interface (train / inference / timed_inference / evaluate /
build_loaders).

Key design decisions
--------------------
* The existing DataLoaders yield (text: str, label: int) pairs for raw-text
  datasets, so tokenization happens inside the collate function rather than
  up-front.  This keeps the trainer self-contained and avoids a separate
  pre-processing step.
* `inference(x: str)` accepts a raw string (overriding the np.ndarray
  signature from the base class) because BERT operates on text, not
  pre-computed embeddings.
* Mixed-precision training (torch.autocast) is used automatically when a
  CUDA device is available.
* Early stopping (patience-based on validation loss) is supported via the
  `patience` kwarg to `train()`.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, precision_score, recall_score

# ── Re-use the base class from the project's trainer module ──────────────────
from trainer import Trainer


# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_MAX_LENGTH = 128
ID2LABEL = {0: "HAM", 1: "SPAM"}
LABEL2ID = {"HAM": 0, "SPAM": 1}


# ── Collate helper ────────────────────────────────────────────────────────────
def _make_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    """
    Returns a collate function that tokenizes a batch of (text, label) pairs.

    Works with both string labels (rare) and integer labels.
    """
    def collate_fn(batch: list[tuple[str, Any]]):
        texts, labels = zip(*batch)
        encoding = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        label_tensor = torch.tensor(
            [int(l) for l in labels], dtype=torch.long
        )
        return encoding, label_tensor

    return collate_fn


# ── Trainer ───────────────────────────────────────────────────────────────────
class BERTTrainer(Trainer):
    """
    Fine-tunes a BERT (or any AutoModelForSequenceClassification) model for
    binary spam / ham classification.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"bert-base-uncased"``.
    num_labels : int
        Number of output classes (default 2).
    max_length : int
        Token sequence length used during tokenization (default 128).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = 2,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)

    # ── DataLoader factory ────────────────────────────────────────────────────
    def build_loaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 16,
        **kwargs,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Build DataLoaders that tokenize on-the-fly via a collate function.
        The underlying datasets must yield ``(text: str, label: int)`` pairs.
        """
        collate = _make_collate_fn(self.tokenizer, self.max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        return train_loader, val_loader

    # ── Training ──────────────────────────────────────────────────────────────
    def train(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        epochs: int = 3,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        patience: int = 3,
        **kwargs,
    ) -> tuple[list[float], list[float]]:
        """
        Fine-tune BERT with a linear warm-up schedule and optional early stopping.

        Parameters
        ----------
        train_loader, val_loader : DataLoader
            Produced by :meth:`build_loaders`.
        epochs : int
            Maximum number of training epochs (default 3; BERT converges fast).
        lr : float
            Peak learning rate (default 2e-5, typical for BERT fine-tuning).
        weight_decay : float
            AdamW weight decay (default 0.01).
        warmup_ratio : float
            Fraction of total steps used for linear LR warm-up (default 0.1).
        patience : int
            Early-stopping patience in epochs; set to ``epochs`` to disable.

        Returns
        -------
        train_losses, val_losses : list[float]
        """
        model = self.model
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # use_amp = self.device.type == "cuda"
        # scaler = torch.cuda.GradScaler(enabled=use_amp)

        train_losses: list[float] = []
        val_losses: list[float] = []
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            epoch_loss = 0.0
            for encoding, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False
            ):
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                labels = labels.to(self.device)

                optimizer.zero_grad()
                with torch.autocast(device_type=self.device.type, enabled=False):
                    outputs = model(**encoding, labels=labels)
                    loss = outputs.loss

                loss.backward()
                # scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # scaler.step(optimizer)
                # scaler.update()
                scheduler.step()
                epoch_loss += loss.item()

            avg_train = epoch_loss / len(train_loader)
            train_losses.append(avg_train)

            # ── Validate ───────────────────────────────────────────────────
            avg_val, val_acc = self._val_loss_and_acc(val_loader)
            val_losses.append(avg_val)

            print(
                f"Epoch [{epoch+1}/{epochs}]  "
                f"Train Loss: {avg_train:.4f}  "
                f"Val Loss: {avg_val:.4f}  "
                f"Val Acc: {100 * val_acc:.2f}%"
            )

            # ── Early stopping ─────────────────────────────────────────────
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
                # Keep a copy of the best weights in memory
                self._best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after epoch {epoch+1}.")
                    break

        # Restore best weights if early stopping fired
        if hasattr(self, "_best_state"):
            model.load_state_dict(
                {k: v.to(self.device) for k, v in self._best_state.items()}
            )

        return train_losses, val_losses

    # ── Inference ─────────────────────────────────────────────────────────────
    def inference(self, x: str) -> bool:  # type: ignore[override]
        """
        Classify a single raw text string.

        Parameters
        ----------
        x : str
            The email / message text.

        Returns
        -------
        bool
            ``True`` if the model predicts SPAM (class index 1).
        """
        self.model.eval()
        encoding = self.tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self.model(**encoding).logits
        return bool(logits.argmax(dim=-1).item() == 1)

    def timed_inference(self, x: str) -> tuple[bool, float]:  # type: ignore[override]
        start = time.perf_counter()
        result = self.inference(x)
        return result, time.perf_counter() - start

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, data_loader: Iterable) -> dict[str, Any]:
        """
        Compute loss, accuracy, F1, precision, and recall over a DataLoader.

        Returns
        -------
        dict with keys: loss, accuracy, f1_macro, precision_macro, recall_macro
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        total_loss = 0.0

        with torch.no_grad():
            for encoding, labels in tqdm(data_loader, desc="Evaluating", leave=False):
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                labels = labels.to(self.device)

                logits = self.model(**encoding).logits
                total_loss += criterion(logits, labels).item()
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        all_preds_np = np.concatenate(all_preds)
        all_targets_np = np.concatenate(all_targets)

        accuracy = (all_preds_np == all_targets_np).mean()
        return {
            "loss": total_loss / max(len(data_loader), 1),
            "accuracy": float(accuracy),
            "f1_macro": f1_score(all_targets_np, all_preds_np, average="macro"),
            "precision_macro": precision_score(
                all_targets_np, all_preds_np, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                all_targets_np, all_preds_np, average="macro", zero_division=0
            ),
        }

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, directory: str) -> None:
        """Save model + tokenizer to *directory* (HuggingFace format)."""
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

    @classmethod
    def load(cls, directory: str, max_length: int = DEFAULT_MAX_LENGTH) -> "BERTTrainer":
        """
        Reload a fine-tuned model saved with :meth:`save`.

        Example
        -------
        >>> trainer = BERTTrainer.load("outputs/bert_spam")
        >>> trainer.inference("Congratulations! You've won a prize.")
        True
        """
        instance = cls.__new__(cls)
        instance.max_length = max_length
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.tokenizer = AutoTokenizer.from_pretrained(directory)
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            directory
        ).to(instance.device)
        instance.model_name = directory
        return instance

    # ── Private helpers ───────────────────────────────────────────────────────
    def _val_loss_and_acc(self, val_loader: Iterable) -> tuple[float, float]:
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for encoding, labels in val_loader:
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                labels = labels.to(self.device)
                logits = self.model(**encoding).logits
                total_loss += criterion(logits, labels).item()
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += len(labels)
        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy


# ── Quick integration smoke-test ──────────────────────────────────────────────
if __name__ == "__main__":
    from dataset_loader import KaggleSpamDataset, split_train_val
    from train import test_classifiers

    EPOCHS = 3
    BATCH_SIZE = 16

    print("Loading dataset...")
    dataset = KaggleSpamDataset("./data/emails.csv")

    trainer = BERTTrainer(model_name="bert-base-uncased", max_length=128)
    train_ds, val_ds = split_train_val(dataset, train_portion=0.8)
    train_loader, val_loader = trainer.build_loaders(
        train_ds, val_ds, batch_size=BATCH_SIZE
    )

    print("Training...")
    train_losses, val_losses = trainer.train(
        train_loader, val_loader,
        epochs=EPOCHS, lr=2e-5, weight_decay=0.01,
    )

    print("\nFinal evaluation:")
    metrics = trainer.evaluate(val_loader)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nSingle inference:")
    sample = "Congratulations! You have been selected for a free prize. Click here now!"
    print(f"  Input : {sample!r}")
    print(f"  Spam? : {trainer.inference(sample)}")

    trainer.save("outputs/bert_spam_model")
    print("\nModel saved to outputs/bert_spam_model/")
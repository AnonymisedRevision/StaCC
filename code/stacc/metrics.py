"""Scoring.

The theory of the paper is stated for the Brier loss, while the headline number
readers want is accuracy, so both are computed everywhere and reported side by
side rather than one standing in for the other.  Expected calibration error is
included for the same reason.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             roc_auc_score)


def brier(probs, labels, num_classes=None):
    """Multiclass Brier score, the mean squared error against the one-hot label.

    This is the loss for which the risk and ambiguity decompositions of the
    paper are identities rather than approximations.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    k = num_classes or probs.shape[1]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy(probs, labels):
    """Plain accuracy, without going through scikit-learn.

    ``evaluate`` below is the full report and is the right thing at the end of a
    run.  Inside a stream loop it is the wrong thing: it also computes two
    F1 variants, a precision, a binned calibration error and an AUC, none of
    which is read per batch, and on a long stream of small batches that
    dominates the whole run.  This and :func:`brier` are what the loop calls.
    """
    probs = np.asarray(probs)
    return float(np.mean(probs.argmax(axis=1) == np.asarray(labels)))


def acc_and_brier(probs, labels):
    """Both per-batch numbers in one pass over the array."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    rows = np.arange(len(labels))
    hit = float(np.mean(probs.argmax(axis=1) == labels))
    sq = float(np.mean(np.sum(probs * probs, axis=1) - 2.0 * probs[rows, labels] + 1.0))
    return hit, sq


def ece(probs, labels, n_bins=15):
    """Expected calibration error over equal-width confidence bins."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def evaluate(probs, labels, num_classes=None):
    """Everything the paper reports for a single predictor on a single split."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    preds = probs.argmax(axis=1)
    k = num_classes or probs.shape[1]

    out = {
        "acc": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="weighted",
                                           zero_division=0)),
        "brier": brier(probs, labels, k),
        "ece": ece(probs, labels),
        "n": int(len(labels)),
    }
    # AUC needs every class present, which a target domain does not guarantee
    try:
        present = np.unique(labels)
        if len(present) == 2:
            out["auc"] = float(roc_auc_score(labels, probs[:, present[1]]))
        elif len(present) > 2 and len(present) == k:
            out["auc"] = float(roc_auc_score(labels, probs, multi_class="ovr",
                                             average="weighted"))
        else:
            out["auc"] = float("nan")
    except ValueError:
        out["auc"] = float("nan")
    return out


class EarlyStopping:
    """Stop when the monitored score has not improved for `patience` epochs."""

    def __init__(self, patience=5, min_delta=0.0, mode="max"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stop = False

    def _better(self, cur):
        if self.best is None:
            return True
        return (cur < self.best - self.min_delta if self.mode == "min"
                else cur > self.best + self.min_delta)

    def step(self, score):
        if self._better(score):
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# --------------------------------------------------------------------------- #
# choosing what to select checkpoints on
# --------------------------------------------------------------------------- #
SELECT_METRICS = ("acc", "brier")


def resolve_select_metric(name, num_classes):
    """The metric everything downstream is scored on.

    Accuracy, on every benchmark and at every label cardinality.  Two reasons to
    fix it rather than adapt it.  A metric that changes with the label space is
    not comparable across a table, and the certificate of Corollary 4 bounds the
    Brier risk, so the two quantities worth reporting are the loss the theory
    speaks about and the number a practitioner reads.  Both are produced by
    ``evaluate``; nothing selects on a per-class average.
    """
    if name != "auto":
        if name not in SELECT_METRICS:
            raise ValueError(f"unknown select metric {name!r}; expected one of "
                             f"{SELECT_METRICS} or 'auto'")
        return name, "requested explicitly"
    return "acc", f"{num_classes} classes, scored on accuracy throughout"


def chance_level(num_classes):
    """Accuracy of a uniform guess, the floor the headroom is measured from."""
    return 1.0 / float(max(num_classes, 1))


def cost_of_trusting_validation(target_scores, val_scores, num_classes):
    """What a practitioner gives up by choosing the member validation ranks first.

    Reported two ways, because the raw gap is not comparable across label
    cardinalities: giving up 0.10 accuracy on a two-class task and on a
    1,139-class task are not the same event.  The normalised form divides by the
    oracle's headroom above chance, so it reads as the share of the achievable
    above-chance accuracy that trusting validation forfeits, and is comparable
    across the table.
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    val_scores = np.asarray(val_scores, dtype=np.float64)
    oracle = float(target_scores.max())
    picked = float(target_scores[int(np.argmax(val_scores))])
    chance = chance_level(num_classes)
    headroom = max(oracle - chance, 1e-12)
    return {
        "oracle": oracle,
        "picked": picked,
        "cost": oracle - picked,
        "cost_normalised": (oracle - picked) / headroom,
        "chance": chance,
        "headroom": oracle - chance,
    }

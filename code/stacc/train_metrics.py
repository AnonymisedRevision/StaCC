"""Metrics of the pool trainer.

Kept exactly as they were when the released pools were trained, because they
decide which checkpoint of each member is stored and whether the member is
accepted, and a pool rebuilt with different ones would be a different pool.
The method, the comparators and every table use ``stacc.metrics`` instead.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
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


def geometric_mean_recall(labels, preds):
    """Geometric mean of per-class recall.

    Used for checkpoint selection because it refuses to be satisfied by a model
    that abandons a class, which plain accuracy will tolerate whenever the
    classes are unbalanced.  It is zero as soon as any class has zero recall,
    so a small floor keeps it usable as a selection signal early in training.
    """
    cm = confusion_matrix(labels, preds)
    denom = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.where(denom > 0, np.diag(cm) / np.maximum(denom, 1), np.nan)
    recalls = recalls[~np.isnan(recalls)]
    if len(recalls) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(np.clip(recalls, 1e-6, None)))))


def evaluate(probs, labels, num_classes=None):
    """Everything the paper reports for a single predictor on a single split."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    preds = probs.argmax(axis=1)
    k = num_classes or probs.shape[1]

    # Average over the classes that actually occur in this split, and not over
    # sklearn's default of unique(y_true) union unique(y_pred).
    #
    # The default makes the denominator depend on what the predictor happened to
    # predict: a member that occasionally names a class the split does not
    # contain adds it to the average at F1 zero and is scored below an otherwise
    # identical member that does not.  On iWildCam's id_val, where 71 of 182
    # classes occur, that is worth a factor of two in macro F1 between members of
    # equal accuracy, and macro F1 is what selects the checkpoint and ranks the
    # pool.  Predicting an absent class is still punished, through the recall of
    # whichever real class was missed instead; it is just not punished twice and
    # by a variable amount.
    #
    # The weighted figures are unchanged by this, to the last bit of the sum:
    # they weight by support in y_true, so a class that never occurs already
    # carried no weight.  Their label set is fixed too only for consistency.
    #
    # The "y_pred contains classes not in y_true" warning is not from any of
    # these; it comes from balanced_accuracy_score, which says so whenever the
    # prediction names a class the split does not contain.  That one is correct
    # as it stands -- it averages recall over the classes in y_true, which is
    # what it should do -- so the warning stays, and on a long-tailed split it
    # is expected rather than a symptom.
    present = np.unique(labels)

    out = {
        "acc": float(accuracy_score(labels, preds)),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=present,
                                   zero_division=0)),
        "f1": float(f1_score(labels, preds, average="weighted", labels=present,
                             zero_division=0)),
        "precision": float(precision_score(labels, preds, average="weighted",
                                           labels=present, zero_division=0)),
        "gm": geometric_mean_recall(labels, preds),
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
SELECT_METRICS = ("gm", "balanced_acc", "macro_f1", "acc")


def resolve_select_metric(name, num_classes):
    """Pick the checkpoint-selection metric, refusing ones that degenerate.

    The geometric mean of per-class recall is the right signal when the label
    space is small: it refuses to be satisfied by a model that abandons a class,
    which plain accuracy will tolerate.  It is the wrong signal once the label
    space is large and long-tailed, because a single class with zero recall
    sends it to the floor for every member, and selection becomes arbitrary.
    Above ten classes we therefore fall back to macro F1, which penalises
    abandoned classes without collapsing on them.
    """
    if name != "auto":
        if name not in SELECT_METRICS:
            raise ValueError(f"unknown select metric {name!r}; expected one of "
                             f"{SELECT_METRICS} or 'auto'")
        return name, f"requested explicitly"
    if num_classes <= 10:
        return "gm", f"{num_classes} classes, geometric-mean recall is informative"
    return "macro_f1", (f"{num_classes} classes, geometric-mean recall would sit at "
                        "the floor for nearly every member")

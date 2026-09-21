"""Randomised construction of a diverse pool.

The pool is built the way GeNeX builds one: a hyperparameter range is fixed in
advance, each member draws from it independently, and the member is kept only if
it clears a floor on the source validation split.  Nothing about the target
domain enters, so the pool is honest with respect to the shift.

What matters for selection is not that members are individually strong but
that they disagree.  A pool of near-duplicates has a small signature diameter
and leaves a selection rule nothing to choose between.  Every draw is therefore recorded, and pool_health() reports
the disagreement actually achieved rather than assuming randomisation delivered
it.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import models
from .augment import build_setups, eval_transform
#from .augment2 import build_setups, eval_transform
from .data import Transformed
from .train_metrics import EarlyStopping, evaluate, resolve_select_metric

# --------------------------------------------------------------------------- #
# the range each member draws from
# --------------------------------------------------------------------------- #
# SEARCH_SPACE = {
#     "arch": models.ARCHITECTURES,
#     "augment": None,                      # filled from build_setups()
#     "pretrained": [True, True, True, True],   # mostly warm starts, some cold // all must be imagenet pre-trained weights here
#     "lr": [1e-3],#[1e-3, 5e-3],[5e-4, 1e-4, 5e-5, 1e-5],
#     "weight_decay": [0.0],#[0.0, 1e-4],#[1e-4],#[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0.0],
#     "optimizer": ["adam"], #["adam", "adamw"], # no "sgd", it gets too delay; adam better
#     "batch_size": [128],#[128, 256],#[64, 128],#[16, 32, 64],
#     "sched_patience": [3, 4],
#     "sched_factor": [0.5, 0.7],
#     "per_epoch_sched": [True],   # if True, sched_patience/sched_factor are ignored
#     "stop_patience": [5, 6],
#     "label_smoothing": [0.00],#[0.08],#[0.0, 0.0, 0.05, 0.1],
#     "drop_rate": [0.00],#[0.2],#[0.0, 0.0, 0.1, 0.2],
#     "epochs": [3],
# }

SEARCH_SPACE = {
    "arch": models.ARCHITECTURES,
    "augment": None,                      # filled from build_setups()
    "pretrained": [True, True, True, True],   # mostly warm starts, some cold // all must be imagenet pre-trained weights here
    "lr": [5e-4],#[1e-3, 5e-3],[5e-4, 1e-4, 5e-5, 1e-5],
    "weight_decay": [0.0],#[0.0, 1e-4],#[1e-4],#[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0.0],
    "optimizer": ["adam"], #["adam", "adamw"], # no "sgd", it gets too delay; adam better
    "batch_size": [128],#[128, 256],#[64, 128],#[16, 32, 64],
    "sched_patience": [3, 4],
    "sched_factor": [0.5, 0.7],
    "per_epoch_sched": [False],   # if True, sched_patience/sched_factor are ignored
    "stop_patience": [5, 6],
    "label_smoothing": [0.00],#[0.00],#[0.0, 0.0, 0.05, 0.1],
    "drop_rate": [0.0],#[0.0],#[0.0, 0.0, 0.1, 0.2],
    "epochs": [3],
}


@dataclass
class MemberSpec:
    """One draw from the search space, enough to reproduce the member exactly."""
    member_id: int
    seed: int
    arch: str
    augment: str
    pretrained: bool
    lr: float
    weight_decay: float
    optimizer: str
    batch_size: int
    sched_patience: int
    sched_factor: float
    stop_patience: int
    label_smoothing: float
    drop_rate: float
    epochs: int
    # Defaulted, so that a manifest or checkpoint written before this field
    # existed still reconstructs, and so that a SEARCH_SPACE without the key
    # keeps drawing the plateau scheduler exactly as it did.
    per_epoch_sched: bool = False

    def line(self):
        return (f"[{self.member_id:03d}] {self.arch:22} {self.augment:16} "
                f"pre={int(self.pretrained)} lr={self.lr:<7g} wd={self.weight_decay:<7g} "
                f"{self.optimizer:5} bs={self.batch_size:<4} ls={self.label_smoothing}")


def draw_spec(member_id, rng, setup_names, space=None):
    space = dict(space or SEARCH_SPACE)
    space["augment"] = setup_names
    pick = lambda k: space[k][rng.randrange(len(space[k]))]
    return MemberSpec(
        member_id=member_id,
        seed=rng.randrange(2 ** 31 - 1),
        arch=pick("arch"),
        augment=pick("augment"),
        pretrained=pick("pretrained"),
        lr=pick("lr"),
        weight_decay=pick("weight_decay"),
        optimizer=pick("optimizer"),
        batch_size=pick("batch_size"),
        sched_patience=pick("sched_patience"),
        sched_factor=pick("sched_factor"),
        stop_patience=pick("stop_patience"),
        label_smoothing=pick("label_smoothing"),
        drop_rate=pick("drop_rate"),
        epochs=pick("epochs"),
        # Read whether it is there or not, and accept it as a bare True/False as
        # well as a list to draw from, since it is a switch rather than an axis
        # the pool is meant to vary over.
        per_epoch_sched=_pick_flag(space, "per_epoch_sched", rng),
    )


def _pick_flag(space, key, rng):
    """A SEARCH_SPACE entry that may be absent, a scalar, or a list to draw from."""
    if key not in space:
        return False
    v = space[key]
    if isinstance(v, (list, tuple)):
        return bool(v[rng.randrange(len(v))]) if v else False
    return bool(v)


class HalveEachEpoch:
    """Multiply the learning rate by ``factor`` at the end of every epoch.

    1e-3, 5e-4, 2.5e-4, and so on: a fixed geometric decay that does not wait to
    be told the score has stopped improving.  ReduceLROnPlateau only cuts the
    rate after ``sched_patience`` epochs without progress, which on a short run
    means it may never fire at all; over three epochs it cannot fire more than
    once.  This one always fires, so a run that is only a few epochs long still
    ends at a rate small enough to settle.

    ``step`` takes and ignores a score, so it is a drop-in for the plateau
    scheduler at the same call site and the training loop needs no branch.
    """

    def __init__(self, optimizer, factor=0.5):
        if not 0.0 < factor < 1.0:
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        self.optimizer = optimizer
        self.factor = float(factor)

    def step(self, *_):
        for group in self.optimizer.param_groups:
            group["lr"] *= self.factor

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


def _make_optimizer(spec, params):
    if spec.optimizer == "adam":
        return torch.optim.Adam(params, lr=spec.lr, weight_decay=spec.weight_decay)
    if spec.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=spec.lr, weight_decay=spec.weight_decay)
    return torch.optim.SGD(params, lr=spec.lr, momentum=0.9,
                           weight_decay=spec.weight_decay, nesterov=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, amp=True):
    """Softmax outputs and labels for a whole split.

    Half precision occasionally overflows on a single awkward input, and a
    non-finite logit becomes a NaN probability that then poisons every
    downstream statistic.  A batch containing non-finite logits is therefore
    recomputed in full precision rather than passed on.
    """
    model.eval()
    use_amp = amp and device.type == "cuda"
    probs, labels, recomputed = [], [], 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        if not torch.isfinite(logits).all():
            logits = model(x).float()          # again, without autocast
            recomputed += 1
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(np.asarray(y))
    if recomputed:
        print(f"    recomputed {recomputed} batch(es) in fp32 after an overflow")
    return np.concatenate(probs), np.concatenate(labels)


def train_member(spec, bench, device, size=224, workers=4, floor=0.0,
                 amp=True, verbose=True, select_metric="gm", chance_mult=1.25,
                 train_subset=None):
    """Train one member, returning (model, best_val_metrics) or (None, None).

    The checkpoint kept is the epoch with the best source-validation score under
    ``select_metric``.  Selection never sees a target domain.

    The acceptance test is deliberately weak.  Under distribution shift a strong
    source-validation score is not evidence of a strong target score, so gating
    hard on validation would select for exactly the validation overfitting the
    pool is supposed to survive.  What validation can be trusted for is the
    negative direction: a member that cannot beat chance on data drawn from its
    own training distribution will not do better on a shifted one, and it
    contributes nothing but noise to the signature space.  ``chance_mult``
    therefore sets a floor just above chance, and nothing more.
    """
    seed_everything(spec.seed)
    setups = build_setups(size)

    source_train = bench.source_train if train_subset is None else train_subset
    train_ds = Transformed(source_train, setups[spec.augment])
    val_ds = Transformed(bench.source_val, eval_transform(size))
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin, drop_last=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=max(64, spec.batch_size), shuffle=False,
                            num_workers=workers, pin_memory=pin,
                            persistent_workers=workers > 0)

    model = models.build(spec.arch, bench.num_classes, spec.pretrained,
                         spec.drop_rate, device, size)
    opt = _make_optimizer(spec, model.parameters())
    # per_epoch_sched takes precedence: when it is set, sched_patience and
    # sched_factor are not consulted at all, and the rate halves every epoch.
    sched = (HalveEachEpoch(opt, factor=0.5) if spec.per_epoch_sched
             else torch.optim.lr_scheduler.ReduceLROnPlateau(
                 opt, mode="max", patience=spec.sched_patience,
                 factor=spec.sched_factor))
    lossf = nn.CrossEntropyLoss(label_smoothing=spec.label_smoothing)
    stopper = EarlyStopping(patience=spec.stop_patience, mode="max")
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    best_score, best_state, best_metrics, dead = -1.0, None, None, 0
    for ep in range(spec.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()

        probs, labels = predict(model, val_loader, device, amp)
        m = evaluate(probs, labels, bench.num_classes)
        score = m[select_metric]
        sched.step(score)

        if verbose:
            print(f"    ep {ep + 1:02d}/{spec.epochs}  loss {running / max(1, len(train_loader)):.4f}"
                  f"  val acc {m['acc']:.4f}  {select_metric} {score:.4f}"
                  f"  brier {m['brier']:.4f}")

        if score > best_score:
            best_score = score
            best_metrics = m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # a member that never leaves chance is not worth more epochs
        if m["acc"] <= 1.0 / bench.num_classes + 1e-6:
            dead += 1
            if dead >= 5:
                if verbose:
                    print("    abandoned: stuck at chance")
                break
        else:
            dead = 0
            if stopper.step(score):
                if verbose:
                    print(f"    early stop at epoch {ep + 1}")
                break

    if best_state is None or best_metrics is None:
        return None, None
    # The floor is applied to mean per-class recall rather than to plain
    # accuracy, since a model predicting only the most frequent class can clear
    # an accuracy floor while carrying no information.  Mean per-class recall
    # scores chance guessing and single-class collapse alike at 1/k.
    chance = 1.0 / bench.num_classes
    if best_metrics["balanced_acc"] < chance_mult * chance:
        if verbose:
            print(f"    rejected: val balanced acc {best_metrics['balanced_acc']:.4f} "
                  f"is not {chance_mult:g}x chance ({chance_mult * chance:.4f})")
        return None, best_metrics
    if floor > 0.0 and best_metrics[select_metric] < floor:
        if verbose:
            print(f"    rejected: val {select_metric} {best_metrics[select_metric]:.4f}"
                  f" below floor {floor}")
        return None, best_metrics
    model.load_state_dict(best_state)
    model.eval()
    return model, best_metrics


# --------------------------------------------------------------------------- #
# pool-level diagnostics
# --------------------------------------------------------------------------- #
def pool_health(signatures):
    """Is the pool actually diverse?

    ``signatures`` is (M, n, C): softmax outputs of M members on n shared
    samples.  Returns the quantities that decide whether the geometry has
    anything to work with, all computed in the flattened signature space the
    method operates in.
    """
    M = len(signatures)
    flat = signatures.reshape(M, -1).astype(np.float64)

    # a single non-finite entry would otherwise turn every statistic below into
    # NaN, so report it and compute the geometry on the usable members rather
    # than losing the whole diagnostic to one bad patch
    finite = np.isfinite(flat).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        print(f"    WARNING: {n_bad} member(s) have non-finite signatures and are "
              "excluded from the geometry statistics")
    geo = flat[finite]
    centroid = geo.mean(axis=0)
    radii = np.linalg.norm(geo - centroid, axis=1)

    preds = signatures.argmax(axis=2)
    disagree = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            d = float(np.mean(preds[i] != preds[j]))
            disagree[i, j] = disagree[j, i] = d
    off = disagree[~np.eye(M, dtype=bool)]

    # how far the nearest neighbour of each member is, relative to the spread:
    # if this is tiny the pool is a cloud of duplicates
    dist = np.linalg.norm(geo[:, None, :] - geo[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    nn_dist = dist.min(axis=1)

    return {
        "n_members": int(M),
        "n_nonfinite_members": n_bad,
        "mean_radius": float(radii.mean()),
        "radius_spread": float(radii.std()),
        "mean_pairwise_disagreement": float(off.mean()),
        "min_pairwise_disagreement": float(off.min()),
        "max_pairwise_disagreement": float(off.max()),
        "mean_nn_distance": float(nn_dist.mean()),
        "nn_over_radius": float(nn_dist.mean() / max(radii.mean(), 1e-12)),
    }


class PoolRegistry:
    """The manifest: one row per member, written as it goes.

    Training a pool takes hours, so the registry is appended to after every
    member rather than at the end, and a run can resume by reading it back.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.models_dir = os.path.join(out_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "manifest.jsonl")

    def existing_ids(self):
        if not os.path.exists(self.path):
            return set()
        ids = set()
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["member_id"])
        return ids

    def member_path(self, member_id):
        return os.path.join(self.models_dir, f"member_{member_id:03d}.pt")

    def append(self, spec, metrics, accepted, seconds):
        row = {**asdict(spec), "accepted": bool(accepted),
               "seconds": round(seconds, 1),
               "val": metrics or {}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def rows(self, accepted_only=True):
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not accepted_only or r.get("accepted"):
                    out.append(r)
        return out

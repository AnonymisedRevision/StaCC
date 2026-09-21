"""Reading a built pool from disk.

A pool directory holds a manifest of every draw, the accepted members'
checkpoints, and, once the caching pass has run, their predictions.  This module
reads all three and knows nothing about how they were produced, so a pool built
by any procedure can be consumed as long as it has this shape.

    <pool>/config.json                the invocation that built it
    <pool>/manifest.jsonl             one row per draw, accepted or not
    <pool>/models/member_XXX.pt       architecture and weights
    <pool>/predictions/               clean outputs, cached once
    <pool>/stability/<family>/        outputs under perturbation, cached once
"""
from __future__ import annotations

import json
import os

import numpy as np


class Pool:
    """A pool directory, with lazy access to whatever has been cached."""

    def __init__(self, root):
        self.root = root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"pool directory not found: {root}")
        cfg = os.path.join(root, "config.json")
        self.config = json.load(open(cfg, encoding="utf-8")) if os.path.exists(cfg) else {}

    # ---------------------------------------------------------------- members
    @property
    def models_dir(self):
        return os.path.join(self.root, "models")

    def member_path(self, member_id):
        return os.path.join(self.models_dir, f"member_{member_id:03d}.pt")

    def rows(self, accepted_only=True):
        path = os.path.join(self.root, "manifest.jsonl")
        if not os.path.exists(path):
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not accepted_only or r.get("accepted"):
                    out.append(r)
        return out

    # ------------------------------------------------------------ predictions
    @property
    def predictions_dir(self):
        return os.path.join(self.root, "predictions")

    def index(self):
        path = os.path.join(self.predictions_dir, "index.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{self.root} has no cached predictions; run the caching pass first"
            )
        return json.load(open(path, encoding="utf-8"))

    def member_ids(self):
        return list(self.index()["members"])

    def load_clean(self, splits=None):
        """Cached clean outputs as {split: (M, n, C)}, plus labels and the index.

        Members are stacked in the order the index records, which is the order
        every array in this codebase uses.
        """
        idx = self.index()
        labels = dict(np.load(os.path.join(self.predictions_dir, "labels.npz")))
        preds = {}
        for mid in idx["members"]:
            z = np.load(os.path.join(self.predictions_dir, f"member_{mid:03d}.npz"))
            for name in z.files:
                if splits is not None and name not in splits:
                    continue
                preds.setdefault(name, []).append(z[name])
        preds = {k: np.stack(v) for k, v in preds.items()}
        return preds, labels, idx

    # -------------------------------------------------------------- stability
    def stability_dir(self, tag):
        return os.path.join(self.root, "stability", tag)

    def has_stability(self, tag):
        return os.path.exists(os.path.join(self.stability_dir(tag), "index.json"))

    def stability_tags(self):
        base = os.path.join(self.root, "stability")
        if not os.path.isdir(base):
            return []
        return sorted(d for d in os.listdir(base) if self.has_stability(d))

    def stability_index(self, tag):
        path = os.path.join(self.stability_dir(tag), "index.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no cached stability run {tag!r} in {self.root}")
        return json.load(open(path, encoding="utf-8"))

    def load_stability(self, tag, families=None):
        """One cached perturbation run.

        Returns ``(clean, perturbed, index)`` where ``clean`` is (M, n, C) and
        ``perturbed`` maps a family name to (M, P, n, C).  The clean outputs are
        recomputed in the same pass as the perturbed ones and through the same
        pipeline, so the displacement being measured is due to the perturbation
        and not to any difference in preprocessing.

        The index records which query points the run covered, since a large
        target split is normally subsampled and every downstream quantity has to
        be restricted to the same points.
        """
        idx = self.stability_index(tag)
        want = list(idx["families"]) if families is None else list(families)
        missing = [f for f in want if f not in idx["families"]]
        if missing:
            raise KeyError(
                f"run {tag!r} holds {sorted(idx['families'])}, not {missing}"
            )
        d = self.stability_dir(tag)
        clean, pert = [], {f: [] for f in want}
        for mid in idx["members"]:
            z = np.load(os.path.join(d, f"member_{mid:03d}.npz"))
            clean.append(z["clean"])
            for f in want:
                pert[f].append(z[f])
        return (
            np.stack(clean),
            {f: np.stack(v) for f, v in pert.items()},
            idx,
        )

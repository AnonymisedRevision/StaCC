"""Cache every pool member's outputs on a query split, clean and perturbed.

This is the only part of the method that needs a GPU, and it is paid once.  With
the outputs on disk the certificate, the objective, the selection, every
baseline and every ablation run in seconds from numpy, so the method can be
iterated on without touching a model again.

The pipeline is resize, to-tensor, perturb, normalise.  Perturbation happens in
[0, 1] pixel space before ImageNet normalisation, so that the radius eps means
the same thing on every benchmark, and the clean pass goes through the identical
pipeline with the perturbation omitted.  Measuring displacement against a clean
pass computed here rather than against an earlier cache removes any chance that
what is being measured is a difference in preprocessing.

A large target split is normally subsampled, because the cost is (1 + sum of
draws over families) forward passes per member and the method only ever sees a
query batch at a time.  The subset is drawn once from a fixed seed and recorded,
so every member and every family covers exactly the same query points.

Outputs are written per member, immediately after each is computed, so an
interruption costs one member rather than the whole pass.  Re-running skips
whatever is already on disk.

    python precompute_stability.py --pool <pool> --split target \
        --families gauss:0.03:2,photometric:0.15:2 --limit 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from stacc import data as D
from stacc import models
from stacc.perturb import adversarial_displacement, build_family, normalise
from stacc.registry import Pool


# Cached probabilities are stored at half precision.  A cache holds one array of
# n x C per member per draw, so at 1,139 classes it is roughly five hundred times
# the size of a two-class one, and float32 would put a single benchmark into the
# tens of gigabytes.  Half precision costs about 1e-3 on a Brier risk, which is
# three orders below the quantities the method compares, and everything is
# widened back to float64 the moment it is read.
STORE_DTYPE = np.float16


class PixelView(Dataset):
    """Images resized and scaled to [0, 1], with normalisation left to the GPU."""

    def __init__(self, base, indices, size):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.tf = T.Compose([T.Resize((size, size), antialias=True), T.ToTensor()])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, target = self.base[int(self.indices[i])]
        return self.tf(img), int(target)


def materialise(base, subset, size, batch, workers):
    """Decode the query subset once into a single uint8 tensor.

    The alternative, iterating a DataLoader per member per draw, spends most of
    its time spawning worker processes and re-decoding the same images: with
    forty members and several draws each that is hundreds of passes over data
    that never changes.  Decoding once and keeping the result resident turns the
    rest of the job into pure GPU work.

    Storing uint8 is lossless here rather than a compromise, because the resize
    happens on the PIL image and the subsequent to-tensor step produces exactly
    k/255 for integer k, which round-trips through uint8 unchanged.
    """
    ds = PixelView(base, subset, size)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        persistent_workers=False)
    chunks = [(x * 255.0).round().to(torch.uint8) for x, _ in loader]
    return torch.cat(chunks)


def iter_batches(buf, batch):
    """Slices of the resident buffer, as [0, 1] float tensors on demand."""
    for s in range(0, len(buf), batch):
        yield buf[s:s + batch]


def _labels_of(base, subset):
    """Labels for a subset without decoding the images.

    Every loader in this codebase exposes its targets as an array under one of
    a few names; falling back to indexing the dataset would open and decode one
    image per query point, which costs more than the clean forward pass does.
    """
    for attr in ("targets", "y", "labels"):
        vals = getattr(base, attr, None)
        if vals is not None and not callable(vals):
            arr = np.asarray(vals)
            if arr.ndim == 1 and len(arr) == len(base):
                return arr[subset].astype(np.int16)
    return np.array([int(base[int(i)][1]) for i in subset], dtype=np.int16)


def family_key(name, eps):
    """Cache key for one family at one radius, e.g. ``stain@0.15``.

    The radius is part of the identity because eps is the parameter the theory
    trades validity against tightness over, so a run normally caches the same
    family at several radii and they must not collide.
    """
    return f"{name}@{eps:g}"


def parse_families(spec):
    """``gauss:0.03:2,photometric:0.15:4`` -> [(key, name, eps, draws), ...].

    ``adv`` is accepted as a name and means projected gradient ascent on the
    member's own signature displacement, which needs gradients and is therefore
    handled separately from the random families.
    """
    out, seen = [], set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        bits = part.split(":")
        if len(bits) != 3:
            raise argparse.ArgumentTypeError(f"expected name:eps:draws, got {part!r}")
        name, eps, draws = bits[0], float(bits[1]), int(bits[2])
        if draws < 1:
            raise argparse.ArgumentTypeError(f"{name}: draws must be at least 1")
        key = family_key(name, eps)
        if key in seen:
            raise argparse.ArgumentTypeError(f"duplicate family {key}")
        seen.add(key)
        out.append((key, name, eps, draws))
    if not out:
        raise argparse.ArgumentTypeError("no families requested")
    return out


def _forward(model, x, amp, device):
    """Softmax outputs, recomputing in full precision if autocast overflowed.

    A non-finite logit becomes a NaN probability that then poisons every
    statistic downstream, so a batch that overflows is redone rather than
    passed on.
    """
    with torch.no_grad():
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(normalise(x))
        logits = logits.float()
        if not torch.isfinite(logits).all():
            logits = model(normalise(x)).float()
        return torch.softmax(logits, dim=1).cpu().numpy()


def _clean_pass(model, buf, batch, device, amp):
    """Outputs on the unperturbed query points, through the shared pipeline."""
    probs = []
    for chunk in iter_batches(buf, batch):
        x = chunk.to(device, non_blocking=True).float() / 255.0
        probs.append(_forward(model, x, amp, device))
    return np.concatenate(probs).astype(STORE_DTYPE)


def _random_pass(model, buf, batch, device, family, amp, seed):
    """One draw from a random perturbation family, over the whole subset."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    probs, applied = [], []
    for chunk in iter_batches(buf, batch):
        x = chunk.to(device, non_blocking=True).float() / 255.0
        xp = family.draw(x, generator=gen)
        applied.append(float((xp - x).flatten(1).pow(2).mean(dim=1).sqrt().mean()))
        probs.append(_forward(model, xp, amp, device))
    return np.concatenate(probs).astype(STORE_DTYPE), float(np.mean(applied))


def _adversarial_pass(model, buf, batch, device, eps, steps, seed):
    """One adversarial estimate of the worst case, needing gradients."""
    probs, applied = [], []
    for b, chunk in enumerate(iter_batches(buf, batch)):
        x = chunk.to(device, non_blocking=True).float() / 255.0
        xp = adversarial_displacement(model, x, eps, steps=steps, seed=seed + b)
        applied.append(float((xp - x).flatten(1).pow(2).mean(dim=1).sqrt().mean()))
        with torch.no_grad():
            logits = model(normalise(xp)).float()
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs).astype(STORE_DTYPE), float(np.mean(applied))


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pool", required=True, help="a built pool directory")
    p.add_argument("--split", required=True, help="which target split to cache")
    p.add_argument("--families", type=parse_families, required=True,
                   help="comma-separated name:eps:draws, e.g. gauss:0.03:2")
    p.add_argument("--tag", default=None, help="name of this cached run")
    p.add_argument("--benchmark", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="subsample the split to this many query points (0 = all)")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--adv-steps", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    pool = Pool(a.pool)
    cfg = pool.config
    benchmark = a.benchmark or cfg.get("benchmark")
    root = a.data or cfg.get("data")
    size = a.image_size or cfg.get("image_size", 224)
    if benchmark is None or root is None:
        raise SystemExit("benchmark and data root must be given or present in config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench = D.load(benchmark, root, cfg.get("val_frac", 0.2), cfg.get("split_seed", 0))
    # ``source_val`` is accepted alongside the target splits.  A benchmark that
    # publishes no out-of-distribution validation domain still has to have its
    # settings fixed somewhere, and its in-distribution validation split is the
    # only honest place left, since the shifted test split is the one thing
    # nothing may be tuned on.
    if a.split == "source_val":
        base = bench.source_val
    elif a.split in bench.targets:
        base = bench.targets[a.split]
    else:
        raise SystemExit(
            f"{a.split!r} is neither 'source_val' nor a target split of "
            f"{benchmark}; available: {sorted(bench.targets)}"
        )

    n_full = len(base)
    rng = np.random.default_rng(a.subset_seed)
    if a.limit and a.limit < n_full:
        subset = np.sort(rng.choice(n_full, size=a.limit, replace=False))
    else:
        subset = np.arange(n_full)

    tag = a.tag or f"{a.split}_n{len(subset)}"
    out_dir = pool.stability_dir(tag)
    os.makedirs(out_dir, exist_ok=True)

    # Refuse to write a second split into a directory that already holds
    # another one.  Members are skipped when their file exists, so reusing a tag
    # across splits would leave the earlier split's predictions in place under
    # the later split's index and labels, and every number computed from the
    # result would be silently wrong rather than obviously broken.
    old_index = os.path.join(out_dir, "index.json")
    if os.path.exists(old_index) and not a.force:
        prev = json.load(open(old_index, encoding="utf-8"))
        clash = []
        if prev.get("split") != a.split:
            clash.append(f"split {prev.get('split')!r} vs {a.split!r}")
        if prev.get("subset") != subset.tolist():
            clash.append(f"a different subset of {prev.get('n_full')} points")
        if clash:
            raise SystemExit(
                f"tag {tag!r} already holds " + "; ".join(clash) + ".\n"
                "Choose a different --tag, or pass --force to recompute the "
                "whole directory."
            )

    try:
        member_ids = pool.member_ids()
    except FileNotFoundError:
        member_ids = [r["member_id"] for r in pool.rows(accepted_only=True)]

    labels = _labels_of(base, subset)

    total_passes = 1 + sum(draws for _, _, _, draws in a.families)
    print(f"{benchmark}  ->  {out_dir}")
    print(f"  split {a.split}: {len(subset)} of {n_full} query points, image size {size}")
    print(f"  {len(member_ids)} members, {total_passes} passes each "
          f"({', '.join(f'{k}x{d}' for k, _, _, d in a.families)})")
    print(f"  device {device}")

    t_dec = time.time()
    buf = materialise(base, subset, size, a.batch, a.workers)
    print(f"  decoded once into {buf.numel() / 1e6:.0f} MB resident "
          f"in {time.time() - t_dec:.0f}s\n", flush=True)

    applied_log = {}
    t0, done = time.time(), 0
    for i, mid in enumerate(member_ids, 1):
        path = os.path.join(out_dir, f"member_{mid:03d}.npz")
        if os.path.exists(path) and not a.force:
            print(f"  [{i}/{len(member_ids)}] member {mid:03d} already cached")
            continue
        model = models.load_member(pool.member_path(mid), device)
        blob = {"clean": _clean_pass(model, buf, a.batch, device, a.amp)}
        for key, name, eps, draws in a.families:
            per_draw, applied = [], []
            for k in range(draws):
                # zlib.crc32 rather than hash(): Python randomises string
                # hashing per process, which would make draws irreproducible.
                seed = 10_000 * mid + int(zlib.crc32(key.encode())) % 997 + k
                if name == "adv":
                    pr, ap = _adversarial_pass(model, buf, a.batch, device, eps,
                                               a.adv_steps, seed)
                else:
                    pr, ap = _random_pass(model, buf, a.batch, device,
                                          build_family(name, eps), a.amp, seed)
                per_draw.append(pr)
                applied.append(ap)
            blob[key] = np.stack(per_draw)
            applied_log.setdefault(key, []).append(float(np.mean(applied)))
        np.savez_compressed(path, **blob)
        del model
        torch.cuda.empty_cache()
        done += 1
        el = time.time() - t0
        print(f"  [{i}/{len(member_ids)}] member {mid:03d} cached  "
              f"({el / done / 60:.2f} min each, "
              f"{(len(member_ids) - i) * el / done / 60:.0f} min left)")

    index = {
        "benchmark": benchmark,
        "split": a.split,
        "image_size": size,
        "members": list(member_ids),
        "subset": subset.tolist(),
        "n_full": int(n_full),
        "num_classes": bench.num_classes,
        "subset_seed": a.subset_seed,
        "families": {
            key: {"family": name, "eps": eps, "draws": draws,
                  "mean_applied_displacement": float(np.mean(applied_log[key]))
                  if key in applied_log else None,
                  "adv_steps": a.adv_steps if name == "adv" else None}
            for key, name, eps, draws in a.families
        },
    }
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "labels.npz"), labels=labels)

    gb = sum(os.path.getsize(os.path.join(out_dir, f))
             for f in os.listdir(out_dir)) / 1e9
    print(f"\ncached {len(member_ids)} members, {gb:.2f} GB in {out_dir}")
    for name, vals in applied_log.items():
        print(f"  {name:14} mean applied pixel displacement {np.mean(vals):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

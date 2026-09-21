"""Cache every pool member's clean outputs on the source validation split.

The stability cache of ``precompute_stability.py`` already holds each member's
clean and perturbed outputs on the query subset, so the only thing still missing
downstream is the source term of the certified radius, which is a per-member
Brier risk on ``source_val``.  That is what this caches, and nothing else.

Two things are deliberately narrower than the caching pass this file replaces.
It writes one split rather than all of them, because the target splits are
already covered and writing them twice at 1,139 classes would run to tens of
gigabytes.  And it subsamples, because the quantity being estimated is a scalar
per member whose error falls as one over the square root of the sample, so ten
thousand rows and forty thousand differ by well under a thousandth of a Brier
risk while costing four times as much.

Outputs land in ``<pool>/predictions`` in the layout ``stacc.registry.Pool``
expects, are written per member so an interruption costs one member, and a
re-run skips whatever is on disk.

    python precompute_clean.py --pool <pool> --data <packed benchmark>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from precompute_stability import (STORE_DTYPE, _clean_pass, _labels_of,
                                  materialise)
from stacc import data as D
from stacc import models
from stacc.registry import Pool

SPLIT = "source_val"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--pool", required=True, help="a built pool directory")
    p.add_argument("--benchmark", default=None)
    p.add_argument("--data", default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=10000,
                   help="subsample source_val to this many rows (0 = all)")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=256)
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
        raise SystemExit("benchmark and data root must be given or in config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench = D.load(benchmark, root, cfg.get("val_frac", 0.2),
                   cfg.get("split_seed", 0))
    base = bench.source_val

    n_full = len(base)
    rng = np.random.default_rng(a.subset_seed)
    if a.limit and a.limit < n_full:
        subset = np.sort(rng.choice(n_full, size=a.limit, replace=False))
    else:
        subset = np.arange(n_full)

    out_dir = pool.predictions_dir
    os.makedirs(out_dir, exist_ok=True)

    # The same guard the stability cache carries.  Members are skipped when their
    # file exists, so a directory written from a different subset would pair one
    # member's outputs with another's rows and every source risk downstream would
    # be quietly wrong rather than visibly broken.
    index_path = os.path.join(out_dir, "index.json")
    if os.path.exists(index_path) and not a.force:
        prev = json.load(open(index_path, encoding="utf-8"))
        complete = (SPLIT in prev.get("splits", {}) and all(
            os.path.exists(os.path.join(out_dir, f"member_{m:03d}.npz"))
            for m in prev.get("members", [])))
        if complete:
            # Camelyon17's first pool was cached by an earlier pass that wrote
            # the target splits here too.  Rewriting the index from this run
            # would drop them for no gain, so a complete directory is left alone.
            print(f"{out_dir} already holds {SPLIT} for "
                  f"{len(prev['members'])} members; nothing to do")
            return 0
        if prev.get("subset") is not None and prev["subset"] != subset.tolist():
            raise SystemExit(
                f"{out_dir} already holds a different subset of {SPLIT}. "
                "Pass --force to recompute it, or match the earlier --limit "
                "and --subset-seed."
            )

    try:
        member_ids = pool.member_ids()
    except FileNotFoundError:
        member_ids = [r["member_id"] for r in pool.rows(accepted_only=True)]

    labels = _labels_of(base, subset)

    print(f"{benchmark}  ->  {out_dir}")
    print(f"  {SPLIT}: {len(subset)} of {n_full} rows, image size {size}")
    print(f"  {len(member_ids)} members, 1 pass each, device {device}")

    t_dec = time.time()
    buf = materialise(base, subset, size, a.batch, a.workers)
    print(f"  decoded once into {buf.numel() / 1e6:.0f} MB resident "
          f"in {time.time() - t_dec:.0f}s\n", flush=True)

    t0, done = time.time(), 0
    for i, mid in enumerate(member_ids, 1):
        path = os.path.join(out_dir, f"member_{mid:03d}.npz")
        if os.path.exists(path) and not a.force:
            print(f"  [{i}/{len(member_ids)}] member {mid:03d} already cached")
            continue
        model = models.load_member(pool.member_path(mid), device)
        probs = _clean_pass(model, buf, a.batch, device, a.amp)
        np.savez_compressed(path, **{SPLIT: probs.astype(STORE_DTYPE)})
        del model
        torch.cuda.empty_cache()
        done += 1
        el = time.time() - t0
        print(f"  [{i}/{len(member_ids)}] member {mid:03d} cached  "
              f"({el / done:.1f}s each, "
              f"{(len(member_ids) - i) * el / done / 60:.1f} min left)")

    index = {
        "benchmark": benchmark,
        "image_size": size,
        "members": list(member_ids),
        "splits": {SPLIT: int(len(subset))},
        "num_classes": bench.num_classes,
        "subset": subset.tolist(),
        "n_full": int(n_full),
        "subset_seed": a.subset_seed,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, "labels.npz"),
                        **{SPLIT: labels})

    gb = sum(os.path.getsize(os.path.join(out_dir, f))
             for f in os.listdir(out_dir)) / 1e9
    print(f"\ncached {len(member_ids)} members, {gb:.2f} GB in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

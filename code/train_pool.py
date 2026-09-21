"""Build and store the diverse pool.

    python train_pool.py --benchmark camelyon17 --data ../data/camelyon17 \
        --pool-size 40 --image-size 96 --out ../pools/camelyon17_seed_1

Every member is trained on the source domains only, kept at its best
source-validation checkpoint, and written to disk with the draw that produced
it.  Target domains are scored once at the end for diagnostics, never used for
selection.  The run is resumable: re-invoking with the same output directory
continues from the last member in the manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from stacc import data as D
from stacc import models, pool
from stacc.augment import build_setups, eval_transform
from stacc.data import Transformed
from stacc.train_metrics import evaluate


def boolflag(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True,
                   help="'PACS:sketch', 'VLCS:SUN09', 'OfficeHome:Clipart', "
                        "or a WILDS task such as 'camelyon17'")
    p.add_argument("--data", default="./data", help="dataset root")
    p.add_argument("--out", default=None,
                   help="output directory (default: pools/<benchmark>)")
    p.add_argument("--pool-size", type=int, default=40,
                   help="number of members to ACCEPT")
    p.add_argument("--max-attempts", type=int, default=0,
                   help="cap on draws; 0 means 3x pool-size")
    p.add_argument("--select-metric", default="auto",
                   help="checkpoint-selection metric: auto, gm, balanced_acc, "
                        "macro_f1 or acc. 'auto' uses gm below 11 classes and "
                        "macro_f1 above, since gm sits at the floor for every "
                        "member once the label space is large")
    p.add_argument("--chance-mult", type=float, default=1.25,
                   help="a member is kept only if its source-validation accuracy "
                        "beats this multiple of chance. This is a noise filter, "
                        "not a quality gate: under shift a high validation score "
                        "is not evidence of a high target score, so gating hard "
                        "on validation would select for validation overfitting")
    p.add_argument("--floor", type=float, default=0.0,
                   help="optional absolute floor on the selection metric, applied "
                        "on top of --chance-mult; 0 disables it")
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="fraction of the source training split each member sees. "
                        "Below 1 this both bounds the cost of a large pool and "
                        "adds another axis of disagreement, since members drawn "
                        "with different seeds see different subsets")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                   help="override the batch sizes members draw from. Small "
                        "inputs leave memory unused at the default sizes, so a "
                        "96-pixel benchmark wants larger batches than a "
                        "224-pixel one")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of each source domain held out for validation")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--pool-seed", type=int, default=1234,
                   help="seed for the hyperparameter draws")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", type=boolflag, default=True)
    p.add_argument("--signature-samples", type=int, default=512,
                   help="samples used for the pool-diversity diagnostic")
    p.add_argument("--score-targets", type=boolflag, default=True,
                   help="score every accepted member on the target domains "
                        "(diagnostic only; never used for selection)")
    p.add_argument("--dry-run", type=boolflag, default=False,
                   help="print the draws and the data summary, train nothing")
    p.add_argument("--rescore-only", type=boolflag, default=False,
                   help="skip training and recompute the diagnostics for an "
                        "existing pool, e.g. after a scoring bug is fixed")
    return p.parse_args(argv)


def main(argv=None):
    # a redirected log is useless if it only appears when the run ends
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out or os.path.join("pools", args.benchmark.replace(":", "_"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"device       {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    bench = D.load(args.benchmark, args.data, args.val_frac, args.split_seed)
    print(bench.describe())

    from stacc.train_metrics import resolve_select_metric
    select_metric, why = resolve_select_metric(args.select_metric, bench.num_classes)
    print(f"selection    {select_metric} ({why})")
    print(f"acceptance   val accuracy > {args.chance_mult:g} x chance = "
          f"{args.chance_mult / bench.num_classes:.4f}")

    space = dict(pool.SEARCH_SPACE)
    if args.batch_sizes:
        space["batch_size"] = list(args.batch_sizes)
        print(f"batch sizes  {space['batch_size']} (overridden)")

    setups = list(build_setups(args.image_size).keys())
    reg = pool.PoolRegistry(out_dir)
    done = reg.existing_ids()
    accepted = len(reg.rows(accepted_only=True))
    if done:
        print(f"resuming: {len(done)} draws on record, {accepted} accepted")

    if args.rescore_only:
        if accepted == 0:
            print("nothing to rescore in this directory")
            return 1
        print(f"rescoring {accepted} stored members, no training")
        diagnose(args, bench, reg, device, out_dir, select_metric)
        return 0

    rng = random.Random(args.pool_seed)
    max_attempts = args.max_attempts or 3 * args.pool_size

    if args.dry_run:
        print("\nfirst 10 draws:")
        for i in range(10):
            print("  " + pool.draw_spec(i, rng, setups, space).line())
        print(f"\nwould write to {out_dir}")
        return 0

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "device": str(device),
                   "num_classes": bench.num_classes,
                   "held_out": bench.held_out}, f, indent=1)

    attempt = 0
    while accepted < args.pool_size and attempt < max_attempts:
        spec = pool.draw_spec(attempt, rng, setups, space)
        attempt += 1
        if spec.member_id in done:
            continue
        print(f"\n{spec.line()}")
        t0 = time.time()
        try:
            subset = None
            if args.train_frac < 1.0:
                n = len(bench.source_train)
                k = max(1, int(round(n * args.train_frac)))
                idx = np.random.default_rng(spec.seed).choice(n, k, replace=False)
                subset = Subset(bench.source_train, idx.tolist())
            model, metrics = pool.train_member(
                spec, bench, device, args.image_size, args.workers,
                args.floor, args.amp, select_metric=select_metric,
                chance_mult=args.chance_mult, train_subset=subset)
        except torch.cuda.OutOfMemoryError:
            print("    out of memory at this batch size, skipping draw")
            torch.cuda.empty_cache()
            reg.append(spec, None, False, time.time() - t0)
            continue
        secs = time.time() - t0

        if model is None:
            reg.append(spec, metrics, False, secs)
            continue
        models.save_member(model, reg.member_path(spec.member_id),
                           extra={"spec": spec.__dict__, "val": metrics})
        reg.append(spec, metrics, True, secs)
        accepted += 1
        print(f"    kept ({accepted}/{args.pool_size})  val acc {metrics['acc']:.4f}"
              f"  {select_metric} {metrics[select_metric]:.4f}  [{secs / 60:.1f} min]")
        del model
        torch.cuda.empty_cache()

    print(f"\npool complete: {accepted} members accepted from {attempt} draws")
    if accepted == 0:
        print("nothing was accepted; lower --floor or check the data root")
        return 1

    diagnose(args, bench, reg, device, out_dir, select_metric)
    return 0


def diagnose(args, bench, reg, device, out_dir, select_metric=None):
    """Measure the diversity the randomisation actually produced."""
    if select_metric is None:
        from stacc.train_metrics import resolve_select_metric
        select_metric, _ = resolve_select_metric(args.select_metric, bench.num_classes)
    rows = reg.rows(accepted_only=True)
    print(f"\nscoring {len(rows)} members for diagnostics")

    # a fixed subset of the target domain, shared by every member, so the
    # signatures are directly comparable
    tgt_name, tgt = next(iter(bench.targets.items()))
    n = min(args.signature_samples, len(tgt))
    idx = np.random.default_rng(0).choice(len(tgt), n, replace=False)
    sig_ds = Transformed(Subset(tgt, idx.tolist()), eval_transform(args.image_size))
    sig_loader = DataLoader(sig_ds, batch_size=64, shuffle=False,
                            num_workers=args.workers)

    target_loaders = {}
    if args.score_targets:
        for name, ds in bench.targets.items():
            target_loaders[name] = DataLoader(
                Transformed(ds, eval_transform(args.image_size)),
                batch_size=64, shuffle=False, num_workers=args.workers)

    # Scoring a member against a large target split is the expensive part of a
    # run, so a member already present in the report is not scored again.  Adding
    # ten members to a pool of forty then costs ten evaluations rather than
    # fifty.  Signature rows are reused the same way, keyed by member id so that
    # a changed pool cannot silently misalign them with the wrong member.
    prev_entries, prev_sig = {}, {}
    report_path = os.path.join(out_dir, "pool_report.json")
    sig_path = os.path.join(out_dir, "signatures.npy")
    ids_path = os.path.join(out_dir, "signature_ids.npy")
    if os.path.exists(report_path):
        try:
            old = json.load(open(report_path, encoding="utf-8"))
            needed = set(target_loaders) | {"val"}
            for e in old.get("members", []):
                if needed.issubset(e.keys()):
                    prev_entries[e["member_id"]] = e
            if (old.get("signature_samples") == int(n)
                    and old.get("signature_domain") == tgt_name
                    and os.path.exists(sig_path) and os.path.exists(ids_path)):
                arr = np.load(sig_path)
                ids = np.load(ids_path)
                if len(arr) == len(ids):
                    prev_sig = {int(i): arr[j] for j, i in enumerate(ids)}
        except Exception as exc:
            print(f"  could not reuse the previous report ({exc}); scoring all")
            prev_entries, prev_sig = {}, {}

    reused = sum(1 for r in rows if r["member_id"] in prev_entries)
    if reused:
        print(f"  reusing {reused} already-scored member(s), "
              f"scoring {len(rows) - reused}")

    signatures, sig_ids, per_member = [], [], []
    for r in rows:
        mid = r["member_id"]
        cached = prev_entries.get(mid)
        if cached is not None and mid in prev_sig:
            entry = dict(cached)
            entry["arch"], entry["augment"] = r["arch"], r["augment"]
            entry["val"] = r["val"]
            per_member.append(entry)
            signatures.append(prev_sig[mid])
            sig_ids.append(mid)
            continue

        m = models.load_member(reg.member_path(mid), device)
        probs, labels = pool.predict(m, sig_loader, device, args.amp)
        signatures.append(probs)
        sig_ids.append(mid)
        entry = {"member_id": mid, "arch": r["arch"], "augment": r["augment"],
                 "val": r["val"]}
        if cached is not None:
            # scores survived but the signature did not; keep the expensive part
            for name in target_loaders:
                entry[name] = cached[name]
        else:
            for name, loader in target_loaders.items():
                p, y = pool.predict(m, loader, device, args.amp)
                entry[name] = evaluate(p, y, bench.num_classes)
            print(f"    scored member {mid:03d}")
        per_member.append(entry)
        del m
        torch.cuda.empty_cache()

    signatures = np.stack(signatures)
    np.save(sig_path, signatures)
    np.save(ids_path, np.asarray(sig_ids, dtype=np.int64))
    health = pool.pool_health(signatures)

    report = {"benchmark": args.benchmark, "health": health,
              "signature_samples": int(n), "signature_domain": tgt_name,
              "members": per_member}

    print("\npool diversity")
    for k, v in health.items():
        print(f"  {k:32} {v}")
    if health["mean_pairwise_disagreement"] < 0.05:
        print("\n  WARNING: members agree on more than 95% of samples. A pool this")
        print("  uniform leaves a selection rule nothing to choose between. Widen")
        print("  the search space before running the method.")

    if args.score_targets:
        # rank and score on the SELECTION metric, not on accuracy: where one
        # class dominates a split, accuracy rewards single-class collapse and
        # would make a degenerate member look like the best one
        key = select_metric
        val_acc = np.array([e["val"][key] for e in per_member])
        for name in bench.targets:
            accs = np.array([e[name][key] for e in per_member if name in e])
            print(f"\n  {name}: target accuracy over members  "
                  f"min {accs.min():.4f}  mean {accs.mean():.4f}  max {accs.max():.4f}")
            print(f"  spread {accs.max() - accs.min():.4f}  "
                  "(a wide spread is what makes selection worth doing)")

            # Does source validation already rank the pool the way the target
            # does?  Where it does, no label-free rule can add anything over
            # simply trusting validation, and the benchmark cannot separate this
            # method from that baseline.  This is the quantity that decides
            # whether the problem the paper poses exists on a given dataset.
            if len(accs) > 2:
                corr = float(np.corrcoef(val_acc, accs)[0, 1])
                pick = float(accs[int(np.argmax(val_acc))])
                cost = float(accs.max() - pick)
                report["health"][f"{name}_val_target_corr_{key}"] = corr
                report["health"][f"{name}_validation_selection_cost_{key}"] = cost
                print(f"  corr(source val, target) {corr:+.3f}")
                print(f"  picking by validation gives {pick:.4f} against an "
                      f"oracle {accs.max():.4f}, a cost of {cost:.4f}")
                if cost < 0.01:
                    print("  NOTE: validation already identifies the best member here,")
                    print("  so this benchmark cannot separate a label-free rule from")
                    print("  a validation baseline. Expect small margins.")

    with open(os.path.join(out_dir, "pool_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())

"""Run StaCC and every comparator on a cached pool, and report the diagnostics.

Reads the clean and perturbed outputs cached by ``precompute_stability.py``,
the clean source-validation outputs cached alongside the pool, and nothing else.
No model is loaded and no target label reaches the selection; labels are read
once per batch, at the end, to score what the label-free rules already chose.

The stream is the target split cut into query batches.  Every ``--period``
batches the certificates are recomputed, the objective is rebuilt and the
committee is reselected, which is the online operation of the paper; on the
remaining batches the standing committee is applied.

    python run_stacc.py --pool <pool> --tag <cached run> --family photometric \
        --k 3 --n-q 500 --period 10 --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from stacc import baselines as B
from stacc import calibrate
from stacc import certificate as CERT
from stacc import committee as COM
from stacc.metrics import (acc_and_brier, accuracy, brier, evaluate,
                           resolve_select_metric)
from stacc.registry import Pool


def batches(n, size, rng, shuffle=True):
    """Cut the query stream into batches.

    Shuffling is not cosmetic here.  Some target splits arrive sorted by class,
    Camelyon17's two hospital splits among them, so contiguous batches would be
    single-class and every method would be scored against a constant label.  The
    caller is warned if a batch turns out to contain one class anyway.
    """
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, n - size + 1, size):
        yield idx[s:s + size]


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
_CACHE = {}
_VAL_CACHE = {}


def load_everything(pool_dir, tag, family, calibrated=True):
    """Clean and perturbed target outputs, source validation outputs, labels.

    Memoised, because a sweep runs the same cache through dozens of settings and
    re-reading forty members of compressed outputs per cell dominates everything
    else.  Nothing downstream mutates these arrays, so sharing them is safe.
    """
    key = (os.path.abspath(pool_dir), tag, family, bool(calibrated))
    if key in _CACHE:
        return _CACHE[key]
    pool = Pool(pool_dir)
    clean, pert, idx = pool.load_stability(tag, families=[family])
    perturbed = pert[family]
    y = np.load(os.path.join(pool.stability_dir(tag), "labels.npz"))["labels"].astype(int)

    preds, labels, pidx = pool.load_clean(splits=["source_val"])
    val_p = preds["source_val"]
    val_y = labels["source_val"].astype(int)

    # Optional: a TENT-adapted comparator, read only if a ``tent.npz`` cache is
    # present.  Absent by default, since it is the only comparator that
    # needs a GPU pass of its own.  It is deliberately left uncalibrated: the
    # temperature was fitted to the unadapted member and adaptation moves the
    # weights, so applying it would be calibrating one model with another
    # model's constant.  Accuracy is invariant to temperature in any case.
    tent_path = os.path.join(pool.stability_dir(tag), "tent.npz")
    tent = None
    if os.path.exists(tent_path):
        z = np.load(tent_path)
        tent = {"probs": z["probs"].astype(np.float64), "covered": z["covered"]}

    if list(pidx["members"]) != list(idx["members"]):
        raise SystemExit(
            "member order differs between the prediction cache and the stability "
            "cache; recompute one of them"
        )

    temps = None
    if calibrated:
        temps = calibrate.fit_pool(val_p, val_y)
        val_p = calibrate.apply_pool(val_p, temps)
        clean = calibrate.apply_pool(clean, temps)
        perturbed = calibrate.apply_pool(perturbed, temps)

    out = {
        "clean": clean,
        "perturbed": perturbed,
        "tent": tent,
        "y": y,
        "val_p": val_p,
        "val_y": val_y,
        "index": idx,
        "temperatures": temps,
        "pool": pool,
    }
    _CACHE[key] = out
    return out


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run(args):
    d = load_everything(args.pool, args.tag, args.family, calibrated=not args.no_calibrate)
    clean, perturbed, y = d["clean"], d["perturbed"], d["y"]
    val_p, val_y = d["val_p"], d["val_y"]
    tent = d.get("tent")
    # Use only the first ``draws`` cached draws when asked.  Draws are seeded by
    # member, family and draw index, so with the same caching batch size the first
    # P draws of a longer cache are exactly the draws a P-draw cache would hold,
    # and a sweep over P needs only the largest cache.
    draws = int(getattr(args, "draws", 0) or 0)
    if draws:
        if draws > perturbed.shape[1]:
            raise SystemExit(f"asked for {draws} draws, the cache holds "
                             f"{perturbed.shape[1]}")
        perturbed = perturbed[:, :draws]
    M, N, C = clean.shape
    P = perturbed.shape[1]
    if args.k > M:
        raise SystemExit(f"committee of {args.k} needs at least that many members, pool has {M}")

    metric, why = resolve_select_metric("auto", C)
    # The paper reports both: accuracy is what a deployment cares about, the
    # Brier risk is the loss every bound in the theory is stated in, so the
    # certificate and the score have to be readable in the same units.  Both are
    # computed directly rather than through ``evaluate``, whose F1 variants,
    # calibration error and AUC are never read per batch and which, on a stream
    # cut into many small batches, costs more than everything else combined.
    score = accuracy

    # source-validation quantities, computed once
    val_brier = np.array([brier(val_p[i], val_y) for i in range(M)])
    val_scores = np.array([score(val_p[i], val_y) for i in range(M)])
    floor = float(val_brier.max())
    prior = np.bincount(val_y, minlength=C) / len(val_y)

    c_stat = CERT.concentration_slack(args.n_q, len(val_y), M, args.eta)
    c_used = args.c if args.c is not None else 0.0

    # Tuning asks one question, namely what a setting is worth, and the twenty
    # comparators answer a different one at several times the cost.  The lean
    # path keeps the method, its two variants and the full-pool reference it is
    # quoted against, and computes nothing else.
    lean = bool(getattr(args, "lean", False))

    # The validation-only rules do not depend on the query batch at all, so they
    # are settled once here rather than recomputed at every reselection.  Greedy
    # forward selection in particular costs k*M scorings of the whole validation
    # split, which on a large split dominates everything else in the loop, and
    # across a sweep it would be repeated per cell, so it is memoised too.
    vkey = (os.path.abspath(args.pool), args.tag, bool(args.no_calibrate),
            int(args.k), float(args.lam))
    if lean:
        # Greedy forward selection costs k*M scorings of the whole validation
        # split and only feeds a comparator row, so it is skipped here.
        val_top_k = B.best_by_source(val_scores, args.k)
        val_best = B.best_by_source(val_scores, 1)[0]
        val_greedy, val_l2 = None, None
    elif vkey in _VAL_CACHE:
        val_top_k, val_best, val_greedy, val_l2 = _VAL_CACHE[vkey]
    else:
        val_top_k = B.best_by_source(val_scores, args.k)
        val_best = B.best_by_source(val_scores, 1)[0]
        val_greedy = B.greedy_selection(val_p, val_y, args.k, score)
        val_l2 = B.l2_model_averaging(val_p, val_y, lam=args.lam)
        _VAL_CACHE[vkey] = (val_top_k, val_best, val_greedy, val_l2)
    # ATC and the difference of confidences both rank members by a source-side
    # quantity adjusted by a batch-side one.  Only the second half changes as
    # the stream advances, so the first is fitted here.
    atc = B.atc_fit(val_p, val_y)

    if not getattr(args, "quiet", False):
        print(f"pool {args.pool}")
        print(f"  cached run {args.tag}: split {d['index']['split']}, "
              f"{N} query points, {M} members, {C} classes")
        print(f"  family {args.family} "
              f"eps={d['index']['families'][args.family]['eps']:g} "
              f"draws={P}, sensitivity estimator '{args.mode}'")
        print(f"  committee k={args.k}, batch {args.n_q}, reselect every {args.period}, "
              f"rule '{args.rule}'{' injective' if not args.allow_repeats else ''}")
        print(f"  scoring on {metric} ({why})")
        print(f"  calibration {'off' if args.no_calibrate else 'on'}"
              + ("" if args.no_calibrate else
                 f" (temperature median {np.median(d['temperatures']):.3f})"))
        print(f"  conservativeness c={c_used:g} "
              f"(computable concentration part {c_stat:.4f})")
        print(f"  constant-predictor risk floor 1-||prior||^2 = "
              f"{CERT.constant_predictor_floor(prior):.4f}\n")

    methods = [
        "stacc", "stacc_uniform", "stacc_floor", "stacc_floor_uniform",
        "topk_certified", "topk_stability", "max_ambiguity",
        "full_pool", "topk_entropy", "spectral_meta", "atc_weighted",
        "agreement_weighted", "confidence_ens", "k_medoids", "dpp",
        "random_committee", "snd_topk", "doc_topk", "augcon_topk",
        "topk_source_val", "greedy_source_val", "best_source_val", "l2_averaging",
        "oracle_member", "oracle_stacking",
    ]
    if tent is not None:
        methods.insert(methods.index("best_source_val") + 1, "tent_best_val")
    if lean:
        methods = ["stacc", "stacc_uniform", "stacc_floor", "full_pool"]
    agg = {m: [] for m in methods}
    agg_brier = {m: [] for m in methods}
    diag = []

    rng = np.random.default_rng(args.seed)
    standing = None

    for b, idx in enumerate(batches(N, args.n_q, rng, shuffle=True)):
        if args.max_batches and b >= args.max_batches:
            break
        batch = clean[:, idx, :]
        pbatch = perturbed[:, :, idx, :]
        yb = y[idx]
        if len(np.unique(yb)) < 2 and b == 0:
            print("  WARNING: query batches contain a single class, so every "
                  "method is scored against a constant label")
        G = COM.gram(batch)
        pair_d2 = COM.squared_distances(G)

        if b % args.period == 0:
            s = CERT.stability_score(batch, pbatch, mode=args.mode)
            rho = CERT.certified_radii(val_brier, s, c=c_used)
            rho_floor = CERT.certified_radii(floor, s, c=c_used)
            # A member that has collapsed to one class on the target is
            # perfectly stable there, so stability-only ranking puts it first.
            # The per-member source term excludes it; the floor variant, which
            # discards that term, needs this label-free guard instead.
            dead = CERT.degenerate_members(batch, args.max_class_share)
            if len(dead) and b == 0 and not getattr(args, "quiet", False):
                print(f"  excluded {len(dead)} member(s) that predict one "
                      f"class on the query batch: {dead.tolist()}")
            standing = {
                "excluded": int(len(dead)),
                "s": s,
                "rho": rho,
                "rho_floor": rho_floor,
                "weighted": COM.frank_wolfe(G, rho ** 2, args.k, rule=args.rule,
                                            injective=not args.allow_repeats,
                                            exclude=dead),
                "uniform": COM.frank_wolfe(G, rho ** 2, args.k, rule="uniform",
                                           injective=not args.allow_repeats,
                                           exclude=dead),
                "floor": COM.frank_wolfe(G, rho_floor ** 2, args.k, rule=args.rule,
                                         injective=not args.allow_repeats,
                                         exclude=dead),
                "floor_uniform": COM.frank_wolfe(G, rho_floor ** 2, args.k,
                                                 rule="uniform",
                                                 injective=not args.allow_repeats,
                                                 exclude=dead),
                # Two distinct ablations, kept apart because they fail
                # differently.  Deleting the ambiguity term leaves top-k by
                # certified risk, which keeps the source term and is safe.
                # Deleting the source term as well leaves top-k by the raw
                # stability score, which is the rule Proposition 7 says collapses
                # onto members that are constant where the score is measured; it
                # is reported without the degeneracy guard, so that the collapse
                # is visible rather than hidden.
                "topk_cert": COM.topk_certified(rho ** 2, args.k),
                "topk_stab": np.argsort(s)[:args.k],
                "max_amb": COM.max_ambiguity(G, args.k),
                "falsified": len(CERT.falsified_pairs(G, rho)),
                "loc": CERT.localisation_bound(G, rho),
            }
            if not lean:
                standing.update({
                    "snd": np.argsort(
                        -B.soft_neighbourhood_density(batch, rng=rng))[:args.k],
                    # Two label-free accuracy estimators used as ranking rules,
                    # the nearest competitors to ranking by certified radius.
                    "doc": np.argsort(
                        -B.difference_of_confidences(atc, batch))[:args.k],
                    "augcon": np.argsort(
                        -B.augmentation_consistency(batch, pbatch))[:args.k],
                })

        st = standing
        # Every member's accuracy on this batch, used by the oracle row and by
        # three of the diagnostics below.  One pass, reused; it was previously
        # recomputed for each of them.
        member_scores = np.array([score(batch[i], yb) for i in range(M)])
        out = {
            "stacc": B.weighted(batch, st["weighted"].weights),
            "stacc_uniform": B.uniform(batch, st["uniform"].members),
            "stacc_floor": B.weighted(batch, st["floor"].weights),
            "full_pool": B.uniform(batch),
        }
        if not lean:
            out.update({
                "stacc_floor_uniform": B.uniform(batch, st["floor_uniform"].members),
                "topk_certified": B.uniform(batch, st["topk_cert"]),
                "topk_stability": B.uniform(batch, st["topk_stab"]),
                "max_ambiguity": B.uniform(batch, st["max_amb"]),
                "topk_entropy": B.uniform(batch, B.lowest_entropy(batch, args.k)),
                "spectral_meta": B.weighted(batch, B.spectral_meta_learner(batch)),
                "atc_weighted": B.weighted(batch, B.atc_weights(atc, batch)),
                "agreement_weighted": B.weighted(batch, B.agreement_weights(batch)),
                "confidence_ens": B.weighted(batch, B.confidence_ensemble(batch)),
                # Both take the member-to-member squared distances the Gram
                # matrix already holds, rather than the signatures themselves.
                "k_medoids": B.uniform(batch, B.k_medoids(pair_d2, args.k, rng)),
                "dpp": B.uniform(batch, B.dpp_greedy(pair_d2, args.k)),
                "random_committee": B.uniform(batch,
                                              B.random_committee(M, args.k, rng)),
                "snd_topk": B.uniform(batch, st["snd"]),
                "doc_topk": B.uniform(batch, st["doc"]),
                "augcon_topk": B.uniform(batch, st["augcon"]),
                "topk_source_val": B.uniform(batch, val_top_k),
                "greedy_source_val": B.uniform(batch, val_greedy),
                "best_source_val": batch[val_best],
                "l2_averaging": B.weighted(batch, val_l2),
                "oracle_member": batch[int(np.argmax(member_scores))],
                "oracle_stacking": B.weighted(
                    batch, COM.oracle_stacking(G, batch, yb)),
            })
        if tent is not None and tent["covered"][idx].all():
            out["tent_best_val"] = tent["probs"][idx]
        for name, probs in out.items():
            if name not in agg:
                continue
            a, br = acc_and_brier(probs, yb)
            agg[name].append(a)
            agg_brier[name].append(br)

        # ---- diagnostics: labels enter only here, and only to check the theory
        risks = COM.realised_risks(G, batch, yb)
        cw = st["weighted"]
        realised = COM.weighted_risk(G, batch, yb, cw.weights)
        diag.append({
            "batch": int(b),
            "excluded": int(st["excluded"]),
            "J": float(cw.J),
            "gap": float(cw.gap),
            "realised_brier": float(realised),
            "committee_covered": bool(realised <= cw.J + 1e-12),
            "member_coverage": float(np.mean(risks <= st["rho"] ** 2 + 1e-12)),
            "median_slack": float(np.median(st["rho"] ** 2 - risks)),
            "ambiguity": float(cw.ambiguity),
            "certified_mean": float(cw.certified_mean),
            "diversity_paid": bool(cw.J < float((st["rho"] ** 2).min())),
            "diam": float(cw.diam),
            # How loose the localisation is, in one number: a ratio near or
            # above one means every member's ball swallows its neighbours and
            # the intersection is doing no work.
            "median_radius": float(np.median(st["rho"])),
            "median_pair_distance": float(np.median(
                np.sqrt(np.maximum(COM.squared_distances(G), 0.0))[
                    np.triu_indices(M, 1)])),
            "falsified_pairs": int(st["falsified"]),
            "diam_K_upper": float(st["loc"]["diam_K_upper"]),
            "minimax_floor_upper": float(st["loc"]["minimax_floor_upper"]),
            "committee": [int(m) for m in cw.members],
            "oracle_member": int(np.argmax(member_scores)),
            "captures_oracle": bool(
                int(np.argmax(member_scores)) in set(cw.members.tolist())),
            "stability_target_corr": float(
                np.corrcoef(-st["s"], member_scores)[0, 1]),
            "certified_target_corr": float(
                np.corrcoef(-st["rho"], member_scores)[0, 1]),
            "val_target_corr": float(np.corrcoef(val_scores, member_scores)[0, 1]),
        })

    return agg, diag, metric, {
        "brier": {k: list(map(float, v)) for k, v in agg_brier.items()},
        "val_brier": val_brier.tolist(),
        "val_scores": val_scores.tolist(),
        "floor": floor,
        "c_stat": c_stat,
        "c_used": c_used,
        "prior": prior.tolist(),
        "temperatures": None if d["temperatures"] is None else d["temperatures"].tolist(),
        "draws": int(P),
        "n_query_points": int(N),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def report(agg, diag, metric, extra, args):
    # Accuracy and the Brier risk side by side, because they answer different
    # questions.  Accuracy is what a deployment cares about; the Brier risk is
    # the loss every bound in the paper is stated in, so it is the column that
    # can be read against J and the certified radii without a change of units.
    brier_agg = extra.get("brier", {})
    print(f"{'method':24} {'mean ' + metric:>10} {'sd':>7} "
          f"{'Brier':>9} {'sd':>7}   vs full pool (acc / Brier)")
    print("-" * 84)
    base = float(np.mean(agg["full_pool"]))
    base_b = float(np.mean(brier_agg["full_pool"])) if "full_pool" in brier_agg else None
    for name, values in agg.items():
        v = np.asarray(values)
        bv = np.asarray(brier_agg.get(name, []), dtype=float)
        bm = bv.mean() if bv.size else float("nan")
        bs = bv.std() if bv.size else float("nan")
        if name == "full_pool":
            tail = ""
        else:
            # A lower Brier risk is better, so its margin is signed the other
            # way round and printed as base minus this, keeping "positive is
            # better" true in both columns.
            db = "" if base_b is None or not bv.size else f" / {base_b - bm:+.4f}"
            tail = f"   {v.mean() - base:+.4f}{db}"
        print(f"{name:24} {v.mean():>10.4f} {v.std():>7.4f} "
              f"{bm:>9.4f} {bs:>7.4f}{tail}")

    n = len(diag)
    get = lambda k: np.array([x[k] for x in diag], dtype=float)
    print(f"\ncertificate diagnostics over {n} batches")
    print(f"  member coverage  E(sigma_i) <= U_i       {get('member_coverage').mean():.1%}")
    print(f"  committee coverage  E(sigma_w) <= J(w)   "
          f"{np.mean([x['committee_covered'] for x in diag]):.1%}")
    print(f"  median slack                             {get('median_slack').mean():+.4f}")
    print(f"  falsified pairs (Prop. 7)                "
          f"{get('falsified_pairs').mean():.1f} of {len(extra['val_brier'])**2//2}")
    print(f"  certified risk J(w)                      {get('J').mean():.4f}")
    print(f"  realised Brier of the committee          {get('realised_brier').mean():.4f}")
    print(f"  Frank-Wolfe gap                          {get('gap').mean():.4f}")
    print(f"  ambiguity A(w)                           {get('ambiguity').mean():.4f}")
    print(f"  certified mean sum_i w_i rho_i^2         {get('certified_mean').mean():.4f}")
    print(f"  diversity certified to pay (eq. 16)      "
          f"{np.mean([x['diversity_paid'] for x in diag]):.1%}")
    print(f"  pool diameter                            {get('diam').mean():.4f}")
    print(f"  diam(K) upper bound                      {get('diam_K_upper').mean():.4f}")
    print(f"  committee contains the oracle member     "
          f"{np.mean([x['captures_oracle'] for x in diag]):.1%}")

    print(f"\nrank information: correlation of member target score with")
    print(f"  source validation score                  {get('val_target_corr').mean():+.3f}")
    print(f"  negated stability score  -s_i            {get('stability_target_corr').mean():+.3f}")
    print(f"  negated certified radius -rho_i          {get('certified_target_corr').mean():+.3f}")

    # Equation (24): the selection pass is a real expense that a committee-size
    # figure alone would hide, so cost is reported amortised over the period.
    M = len(extra["val_brier"])
    P = extra["draws"]
    amortised = M * (1 + P) / args.period + args.k
    print(f"\ncost, in model evaluations per query batch")
    print(f"  full-pool ensemble                       {M}")
    print(f"  StaCC, M(1+P)/T + k with P={P}, T={args.period}, k={args.k}"
          f"    {amortised:.1f}")
    print(f"  ratio                                    {amortised / M:.2f}x")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True)
    p.add_argument("--tag", required=True, help="a cached perturbation run")
    p.add_argument("--family", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-q", type=int, default=500)
    p.add_argument("--period", type=int, default=20)
    p.add_argument("--mode", choices=["worst", "mean"], default="worst")
    p.add_argument("--rule", choices=["line", "fw", "uniform"], default="line")
    p.add_argument("--c", type=float, default=0.2,
                   help="conservativeness constant c; 0.2 as deployed")
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--max-class-share", type=float, default=0.85,
                   help="exclude members predicting one class on this share of "
                        "the batch; 1.0 disables the guard")
    p.add_argument("--allow-repeats", action="store_true")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--draws", type=int, default=0,
                   help="use only the first P cached draws; 0 uses all")
    p.add_argument("--lean", action="store_true",
                   help="score the method, its variants and the full-pool "
                        "reference only, skipping every comparator and oracle; "
                        "this is what tuning needs and it is several times "
                        "faster")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    agg, diag, metric, extra = run(a)
    report(agg, diag, metric, extra, a)

    if a.out:
        payload = {
            "config": vars(a),
            "metric": metric,
            "scores": {k: list(map(float, v)) for k, v in agg.items()},
            "brier": extra.pop("brier"),
            "diagnostics": diag,
            "pool": extra,
        }
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
            f.write("\n")
        os.replace(tmp, a.out)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

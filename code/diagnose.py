"""The two diagnostics that decide whether the method has anything to work with.

Neither needs a committee, so both are run before any selection is attempted.

The first asks whether the stability signal carries information about target
performance where source validation does not.  It reports, for every cached
family and radius, the pool-wide rank correlation between target score and the
negated stability score, set beside the correlation with source validation.  The
paper's premise is that the second is clearly positive on the pools at which the
first is near zero or negative; if no family achieves that, the premise fails on
this benchmark and nothing downstream will rescue it.

The second asks whether the certificates are valid and how tight.  Validity and
tightness trade off against each other through the radius: a small eps gives a
sharp bound that the data refuses, a large one gives a bound the data respects
but which says little.  The falsification test of Proposition 6 marks the radii
at which the transport assumption is refuted outright, using no labels, and the
coverage column then confirms with labels what the test already knew.

    python diagnose.py --pool <pool> --tag <cached run>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from stacc import calibrate
from stacc import certificate as CERT
from stacc import committee as COM
from stacc.metrics import (accuracy, brier, cost_of_trusting_validation,
                           resolve_select_metric)
from stacc.registry import Pool


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--mode", choices=["worst", "mean"], default="worst")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--c", type=float, default=0.2)
    p.add_argument("--min-share", type=float, default=0.6,
                   help="least fraction of the certified radius the stability "
                        "term must contribute for a family to be admissible")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    pool = Pool(a.pool)
    idx = pool.stability_index(a.tag)
    keys = list(idx["families"])
    clean, pert, _ = pool.load_stability(a.tag)
    y = np.load(os.path.join(pool.stability_dir(a.tag), "labels.npz"))["labels"].astype(int)
    preds, labels, _ = pool.load_clean(splits=["source_val"])
    val_p, val_y = preds["source_val"], labels["source_val"].astype(int)

    if not a.no_calibrate:
        T = calibrate.fit_pool(val_p, val_y)
        val_p = calibrate.apply_pool(val_p, T)
        clean = calibrate.apply_pool(clean, T)
        pert = {k: calibrate.apply_pool(v, T) for k, v in pert.items()}

    M, N, C = clean.shape
    metric, _ = resolve_select_metric("auto", C)
    # ``accuracy`` rather than ``evaluate``: the latter also computes two F1
    # variants, a precision, a binned calibration error and a one-vs-rest AUC,
    # none of which is read here, and at RxRx1's 1,139 classes that AUC alone
    # costs more than the whole diagnostic.
    target_score = np.array([accuracy(clean[i], y) for i in range(M)])
    val_score = np.array([accuracy(val_p[i], val_y) for i in range(M)])
    val_brier = np.array([brier(val_p[i], val_y) for i in range(M)])
    target_brier = np.array([brier(clean[i], y) for i in range(M)])
    G = COM.gram(clean)

    print(f"pool {a.pool}")
    print(f"  cached run {a.tag}: split {idx['split']}, {N} query points, "
          f"{M} members, {C} classes, scoring on {metric}")
    print(f"  calibration {'off' if a.no_calibrate else 'on'}, "
          f"sensitivity estimator '{a.mode}', c={a.c:g}")

    cost = cost_of_trusting_validation(target_score, val_score, C)
    print(f"\nthe precondition: does trusting validation cost anything here?")
    print(f"  classes {C}, chance {cost['chance']:.4f}")
    print(f"  oracle member target {metric}                {cost['oracle']:.4f}")
    print(f"  target {metric} of the best member by val    {cost['picked']:.4f}")
    print(f"  cost of trusting validation                {cost['cost']:+.4f}")
    print(f"  the same as a share of above-chance headroom "
          f"{cost['cost_normalised']:.3f}")
    print(f"  corr(val score, target score)   pearson    "
          f"{np.corrcoef(val_score, target_score)[0, 1]:+.3f}"
          f"   spearman {spearmanr(val_score, target_score).statistic:+.3f}")

    print(f"\ndoes stability carry the signal validation is missing?")
    head = (f"  {'family':18} {'applied':>8} {'mean s':>8} {'pearson':>8} {'spearman':>9}"
            f" {'cert.r':>8} {'flip.r':>8} {'cover':>7} {'slack':>8} {'falsif':>7} {'J*':>8}")
    print(head)
    print("  " + "-" * (len(head) - 2))

    rows = []
    for key in keys:
        s = CERT.stability_score(clean, pert[key], mode=a.mode)
        phi = CERT.flip_rate(clean, pert[key])
        rho = CERT.certified_radii(val_brier, s, c=a.c)
        cover = float(np.mean(target_brier <= rho ** 2 + 1e-12))
        slack = float(np.median(rho ** 2 - target_brier))
        fals = len(CERT.falsified_pairs(G, rho))
        loc = CERT.localisation_bound(G, rho)
        pear = float(np.corrcoef(-s, target_score)[0, 1])
        spear = float(spearmanr(-s, target_score).statistic)
        cert_r = float(np.corrcoef(-rho, target_score)[0, 1])
        # the decision-level score of Remark 24, reported as an alternative
        # ranking signal rather than fed into J, whose units are Brier
        flip_r = float(np.corrcoef(-phi, target_score)[0, 1])
        applied = idx["families"][key].get("mean_applied_displacement")
        rows.append({
            "family": key, "applied": applied, "mean_s": float(s.mean()),
            "mean_flip_rate": float(phi.mean()),
            "pearson_stability": pear, "spearman_stability": spear,
            "pearson_certified": cert_r, "pearson_flip": flip_r,
            "spearman_flip": float(spearmanr(-phi, target_score).statistic),
            "coverage": cover, "median_slack": slack,
            "falsified_pairs": int(fals), "J_star": float(loc["J_star"]),
            "diam_K_upper": float(loc["diam_K_upper"]),
        })
        print(f"  {key:18} {applied if applied is None else f'{applied:8.4f}'} "
              f"{s.mean():8.4f} {pear:+8.3f} {spear:+9.3f} {cert_r:+8.3f} {flip_r:+8.3f} "
              f"{cover:7.1%} {slack:+8.4f} {fals:7d} {loc['J_star']:8.4f}")

    print(f"\n  reading: a family is a candidate when its stability correlation is "
          f"clearly positive,\n  and its certificate is usable when coverage is near "
          f"100% with no falsified pairs.\n  Those two pull in opposite directions "
          f"through eps, which is the trade-off to report.")

    # The correlation column above uses target labels, so it diagnoses the
    # choice of family but cannot make it in a deployment.  The certified
    # optimum can: every family's bound is valid, so the tightest is valid too.
    scores = {r["family"]: CERT.stability_score(clean, pert[r["family"]], mode=a.mode)
              for r in rows}
    radii = {k: CERT.certified_radii(val_brier, s, c=a.c) for k, s in scores.items()}
    shares = {k: CERT.stability_share(val_brier, s, c=a.c) for k, s in scores.items()}
    chosen, ranking = CERT.select_family(G, radii, shares, min_share=a.min_share)
    by_corr = max(rows, key=lambda r: r["pearson_stability"])["family"]
    for r in rows:
        r["stability_share"] = shares[r["family"]]
    print(f"\nchoosing the family without labels, by certified risk")
    for r in sorted(ranking, key=lambda x: x["J_star"]):
        mark = " <- chosen" if r["family"] == chosen else ""
        why = []
        if r["falsified_pairs"]:
            why.append("falsified")
        if r["stability_share"] is not None and r["stability_share"] < a.min_share:
            why.append(f"share {r['stability_share']:.2f}")
        flag = f"  (discarded: {', '.join(why)})" if why else ""
        print(f"  {r['family']:18} J* {r['J_star']:8.4f}   "
              f"share {r['stability_share']:.2f}   "
              f"diam(K) <= {r['diam_K_upper']:.4f}{flag}{mark}")
    if chosen is None:
        print("  no family is admissible, so the honest answer is to abstain and "
              "average the whole pool")
    print(f"  chosen without labels, by certified risk   {chosen}")
    print(f"  ranked first by correlation, using labels  {by_corr}")
    print("  the two are proxies for the same thing and need not coincide; the "
          "certified\n  criterion is the one available in a deployment.")

    if a.out:
        payload = {"pool": a.pool, "tag": a.tag, "metric": metric,
                   "config": vars(a), "rows": rows,
                   "label_free_family": chosen,
                   "best_family_by_correlation": by_corr,
                   "family_ranking": ranking,
                   "val_target_pearson": float(np.corrcoef(val_score, target_score)[0, 1]),
                   "cost_of_trusting_validation": cost["cost"],
                   "cost_normalised": cost["cost_normalised"],
                   "chance": cost["chance"],
                   "num_classes": int(C),
                   "oracle": cost["oracle"],
                   "target_score": target_score.tolist(),
                   "val_score": val_score.tolist()}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

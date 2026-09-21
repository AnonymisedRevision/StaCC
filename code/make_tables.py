"""Turn the result JSONs into the LaTeX table bodies the paper expects.

Typing numbers into a table by hand is how transcription errors get published,
so nothing here is manual.  The script reads whatever ``run_all.py`` wrote under
``../results``, aggregates the three pools of each benchmark, and prints table
bodies ready to paste.

Every headline number is a mean over pools, with the spread across pools beside
it, and every comparison against the full-pool ensemble carries a paired
bootstrap interval over query batches.  A difference is set in bold only when
that interval excludes zero, which is the discipline the review asks for and the
reason no single run is ever quoted on its own.

    python make_tables.py                     # every benchmark with results
    python make_tables.py --which ood_val     # the tuning split instead
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import benchmarks as B

# The order the paper's main table uses, with its display names.  Anything a run
# produced that is not listed here is printed as a comment rather than dropped,
# so a new comparator shows up without editing this file.
MAIN_ROWS = [
    ("deployed", "StaCC, as deployed"),
    (None, None),
    ("stacc", "StaCC, weighted"),
    ("stacc_uniform", "StaCC, uniform"),
    ("stacc_floor", "StaCC, floor variant$^{\\dagger}$"),
    (None, None),
    ("topk_certified", "Top-$k$ certified$^{\\dagger}$"),
    ("topk_stability", "Top-$k$ stability$^{\\dagger\\dagger}$"),
    ("max_ambiguity", "Max-ambiguity$^{\\dagger}$"),
    (None, None),
    ("full_pool", "Full-pool ensemble"),
    ("topk_entropy", "Top-$k$ entropy"),
    ("spectral_meta", "Spectral meta-learner~\\citep{parisi2014ranking}"),
    ("atc_weighted", "ATC-weighted~\\citep{garg2022leveraging}"),
    ("agreement_weighted", "Agreement-weighted~\\citep{baek2022agreement}"),
    ("confidence_ens", "Confidence ensemble~\\citep{zoppi2025confidence}"),
    ("l2_averaging", "$L_2$ model averaging~\\citep{zhu2024stability}"),
    ("k_medoids", "$k$-medoids~\\citep{li2023classifier}"),
    ("dpp", "DPP~\\citep{kulesza2012determinantal}"),
    ("random_committee", "Random committee"),
    ("snd_topk", "SND~\\citep{saito2021tune}"),
    (None, None),
    ("doc_topk", "Diff.\\ of confidences~\\citep{guillory2021predicting}"),
    ("augcon_topk", "Augmentation consistency~\\citep{deng2021does}"),
    ("tent_best_val", "TENT-adapted best by val.~\\citep{wang2021tent}"),
    (None, None),
    ("topk_source_val", "Top-$k$ by val.\\ "),
    ("greedy_source_val", "Greedy selection~\\citep{caruana2004ensemble}"),
    ("best_source_val", "Best single by val.\\ "),
    (None, None),
    ("oracle_member", "\\emph{Oracle member}"),
    ("oracle_stacking", "\\emph{Oracle stacking}"),
]

ORACLES = {"oracle_member", "oracle_stacking"}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find(name, seed, which, kind):
    """The result file for one run, whichever way its seed is spelled.

    Pool directories and hand-written output paths do not always agree on
    ``seed3`` against ``seed_3``, and a table silently missing a third of its
    runs is worse than one that fails, so both spellings are tried.
    """
    cands = [B.result_path(name, seed, which, kind)]
    if seed.startswith("seed") and not seed.startswith("seed_"):
        cands.append(B.result_path(name, "seed_" + seed[4:], which, kind))
    elif seed.startswith("seed_"):
        cands.append(B.result_path(name, "seed" + seed[5:], which, kind))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def columns(which="ood_test", names=None):
    """One column per benchmark, carrying every pool that produced results.

    The diagnostics file is optional.  Everything the tables need about the
    precondition is recoverable from the run itself, since the oracle member and
    the best-by-validation member are both scored rows; only the per-family
    signal table genuinely needs ``diagnose.py`` to have been run.
    """
    out = []
    for name in (names or list(B.BENCHMARKS)):
        b = B.BENCHMARKS[name]
        if not b.get(which):
            continue
        runs = []
        for seed in B.SEEDS:
            main = _find(name, seed, which, "main")
            if main is None:
                continue
            diag = _find(name, seed, which, "diag")
            runs.append({"seed": seed, "main": load(main),
                         "diag": load(diag) if diag else None})
        if runs:
            out.append({"name": name, "label": label_of(name, which),
                        "short": b["label"], "classes": b["classes"],
                        "runs": runs})
    return out


def label_of(name, which):
    b = B.BENCHMARKS[name]
    pretty = {"ood_test_hospital2": "hospital~2",
              "ood_val_hospital1": "hospital~1",
              "ood_test_crc_val_7k": "CRC-VAL-HE-7K"}
    split = b[which]
    return f"{b['label']}, {pretty.get(split, chr(92) + 'texttt{' + split.replace('_', chr(92) + '_') + '}')}"


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def fmt(x, places=3):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{places}f}"


def cell(values, places=3, bold=False):
    """Mean over pools, with the spread across pools when there is more than one."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "--"
    body = fmt(v.mean(), places)
    if bold:
        body = f"\\mathbf{{{body}}}"
    if v.size > 1:
        return f"${body}_{{\\pm{fmt(v.std(ddof=0), places)}}}$"
    return f"${body}$"


def bootstrap_ci(a, b, reps=10000, alpha=0.05, seed=0):
    """Paired bootstrap over batches for the mean difference a - b.

    Paired because the two rules see the same query batches, which removes the
    batch-to-batch variation that dominates the raw spread and is the reason an
    unpaired interval on these numbers would be uninformative.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan")
    d = a[:n] - b[:n]
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, n, size=(reps, n))].mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def pooled(col, key, field="scores"):
    """The per-batch series for one method, concatenated over pools."""
    out = []
    for r in col["runs"]:
        s = r["main"].get(field, {})
        if key in s:
            out.append(np.asarray(s[key], float))
    return np.concatenate(out) if out else np.array([])


def per_pool_means(col, key, field="scores"):
    return [float(np.mean(r["main"][field][key]))
            for r in col["runs"] if key in r["main"].get(field, {})]


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def table_benchmarks(cols):
    print("% ---- Table 1: the benchmarks (static, from the registry) ----------")
    for col in cols:
        b = B.BENCHMARKS[col["name"]]
        print(f"% {b['label']:11} C={b['classes']:>5}  test={b['ood_test']}  "
              f"val={b['ood_val']}  query subset {b['limit']}, "
              f"n_q {b['n_q']}, batches {b['limit'] // b['n_q']}")
    print()


def table_precondition(cols):
    """The cost of trusting validation, read off the run itself.

    The oracle member and the member ranked first on source validation are both
    scored rows of every run, so their gap needs no separate diagnostic pass.
    The correlation comes from the per-batch diagnostic the run already logs.
    """
    print("% ---- Table: the precondition ---------------------------------------")
    for col in cols:
        C = col["classes"]
        corr, orac, cost, norm = [], [], [], []
        for r in col["runs"]:
            s = r["main"]["scores"]
            o = float(np.mean(s["oracle_member"]))
            v = float(np.mean(s["best_source_val"]))
            orac.append(o)
            cost.append(o - v)
            norm.append((o - v) / max(o - 1.0 / C, 1e-12))
            corr.append(float(np.mean([d["val_target_corr"]
                                       for d in r["main"]["diagnostics"]])))
        print(f"{col['label']:32} & ${C:,}$ & "
              f"{cell(corr, 2)} & {cell(orac)} & {cell(cost)} & {cell(norm)} \\\\")
    print()


def table_signal(cols, families=None):
    print("% ---- Table: does stability carry the missing signal? ----------------")
    if not any(r["diag"] for col in cols for r in col["runs"]):
        print("% no diagnostics files; run diagnose.py on the same tags to fill "
              "this table")
        print()
        return
    fams = families
    if fams is None:
        fams = []
        for col in cols:
            for r in col["runs"]:
                if not r["diag"]:
                    continue
                for row in r["diag"]["rows"]:
                    if row["family"] not in fams:
                        fams.append(row["family"])
    print("% families in order: " + ", ".join(fams))
    for col in cols:
        corr = [r["diag"]["val_target_pearson"] for r in col["runs"] if r["diag"]]
        cells = []
        for f in fams:
            vals = [row["pearson_stability"]
                    for r in col["runs"] if r["diag"] for row in r["diag"]["rows"]
                    if row["family"] == f]
            cells.append(cell(vals, 2) if vals else "--")
        print(f"{col['label']:32} & {cell(corr, 2)} & " + " & ".join(cells) + " \\\\")
    print()


def table_family(cols):
    print("% ---- Table: the label-free choice of family -------------------------")
    for col in cols:
        picks, js, shares = [], [], []
        for r in col["runs"]:
            d = r["diag"]
            if d is None:
                # No diagnostic pass, but the run records the family it was
                # given, which is what the label-free rule returned.
                picks.append(r["main"]["config"]["family"])
                js.append(float(np.mean([x["J"] for x in r["main"]["diagnostics"]])))
                continue
            f = d.get("label_free_family")
            picks.append("abstain" if f is None else f)
            by = {x["family"]: x for x in d["rows"]}
            if f in by:
                js.append(by[f]["J_star"])
                shares.append(by[f]["stability_share"])
        chosen = ", ".join(sorted(set(picks)))
        print(f"{col['label']:32} & \\texttt{{{chosen}}} & "
              f"{cell(js)} & {cell(shares, 2)} & \\num & \\num & \\num & \\num \\\\")
    print("% the per-family accuracies are filled by: python sensitivity.py menu")
    print()


def table_main(cols, field="scores", places=3):
    what = "accuracy" if field == "scores" else "Brier risk"
    print(f"% ---- Table: main results, {what} ------------------------------")

    # The deployed row honours the label-free precondition: where no family is
    # admissible the method abstains, and the row reports what it defers to.
    for col in cols:
        for r in col["runs"]:
            s = r["main"][field]
            abstains = bool(r["diag"]) and r["diag"].get("label_free_family") is None
            s["deployed"] = list(s["full_pool"] if abstains else s["stacc"])
            if abstains:
                print(f"% {col['label']} {r['seed']}: abstained, "
                      f"deployed row is the full pool")

    # bold the best label-free entry per column, oracles excluded
    best = []
    for col in cols:
        cands = {}
        for key, _ in MAIN_ROWS:
            if key is None or key in ORACLES:
                continue
            m = per_pool_means(col, key, field)
            if m:
                cands[key] = np.mean(m)
        if not cands:
            best.append(None)
        elif field == "scores":
            best.append(max(cands, key=cands.get))
        else:
            best.append(min(cands, key=cands.get))

    seen = set()
    for key, name in MAIN_ROWS:
        if key is None:
            print("\\midrule")
            continue
        if not any(per_pool_means(c, key, field) for c in cols):
            continue
        seen.add(key)
        cells = []
        for j, col in enumerate(cols):
            m = per_pool_means(col, key, field)
            cells.append("--" if not m else cell(m, places, bold=(key == best[j])))
        print(f"{name:48} & " + " & ".join(cells) + " \\\\")

    extra = sorted({k for c in cols for r in c["runs"]
                    for k in r["main"][field]} - seen - {"deployed"})
    for key in extra:
        cells = [cell(per_pool_means(c, key, field), places) for c in cols]
        print(f"% not in the row list: {key:22} & " + " & ".join(cells) + " \\\\")
    print()


def table_validity(cols):
    print("% ---- Table: certificate diagnostics ---------------------------------")
    for col in cols:
        d = [x for r in col["runs"] for x in r["main"]["diagnostics"]]
        g = lambda k: [float(x[k]) for x in d if k in x]
        ratio = [float(x["median_radius"]) / max(float(x["median_pair_distance"]), 1e-12)
                 for x in d if "median_radius" in x]
        print(f"{col['label']:32} & "
              f"{cell(g('member_coverage'), 2)} & "
              f"{cell(g('median_slack'))} & "
              f"{cell([float(x['committee_covered']) for x in d], 2)} & "
              f"{cell(g('gap'))} & "
              f"{cell(g('falsified_pairs'), 1)} & "
              f"{cell(ratio, 1) if ratio else '--'} \\\\")
    print()


def margins(cols):
    """Not a paper table: the margins and intervals the narrative should quote."""
    print("% ---- margins the narrative should quote, with paired bootstrap CIs ---")
    for col in cols:
        base = pooled(col, "full_pool")
        print(f"% {col['label']}  ({len(col['runs'])} pools, "
              f"{len(base)} batches total)")
        refs = [("full_pool", "vs full pool"),
                ("topk_source_val", "vs top-k by val"),
                ("best_source_val", "vs best single by val"),
                ("topk_certified", "vs top-k certified"),
                ("oracle_member", "vs oracle member")]
        for variant in ("deployed", "stacc", "stacc_uniform"):
            a = pooled(col, variant)
            if a.size == 0:
                continue
            line = f"%   {variant:16} {a.mean():.4f}"
            for other, tag in refs:
                b = pooled(col, other)
                if b.size == 0:
                    continue
                lo, hi = bootstrap_ci(a, b)
                star = "*" if (lo > 0 or hi < 0) else " "
                line += f"   {tag} {a.mean() - b.mean():+.4f} [{lo:+.4f},{hi:+.4f}]{star}"
            print(line)
        # the best non-ablation label-free comparator, which the paper leads with
        pool_free = ["full_pool", "topk_entropy", "spectral_meta", "atc_weighted",
                     "agreement_weighted", "confidence_ens", "l2_averaging",
                     "k_medoids", "dpp", "random_committee", "snd_topk",
                     "doc_topk", "augcon_topk"]
        means = {k: np.mean(per_pool_means(col, k)) for k in pool_free
                 if per_pool_means(col, k)}
        if means:
            top = max(means, key=means.get)
            a, b = pooled(col, "deployed"), pooled(col, top)
            if a.size and b.size:
                lo, hi = bootstrap_ci(a, b)
                print(f"%   best non-ablation label-free rule is {top} at "
                      f"{means[top]:.4f}; margin {a.mean() - b.mean():+.4f} "
                      f"[{lo:+.4f},{hi:+.4f}]")
        # how often the ambiguity term is certified to pay, eq. (15)
        d = [x for r in col["runs"] for x in r["main"]["diagnostics"]]
        if d:
            print(f"%   condition (15) holds on "
                  f"{np.mean([x['diversity_paid'] for x in d]):.1%} of batches")
    print()


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--which", default="ood_test", choices=["ood_test", "ood_val"])
    p.add_argument("--benchmarks", default=None)
    p.add_argument("--families", default=None)
    a = p.parse_args(argv)

    names = [s.strip() for s in a.benchmarks.split(",")] if a.benchmarks else None
    cols = columns(a.which, names)
    if not cols:
        print(f"no results under {B.RESULTS} for {a.which}; run run_all.py first")
        return 1
    print(f"% generated from {B.RESULTS}, split {a.which}, "
          f"{sum(len(c['runs']) for c in cols)} runs\n")

    table_benchmarks(cols)
    table_precondition(cols)
    table_signal(cols, a.families.split(",") if a.families else None)
    table_family(cols)
    table_main(cols, "scores")
    table_main(cols, "brier")
    table_validity(cols)
    margins(cols)
    return 0


if __name__ == "__main__":
    sys.exit(main())

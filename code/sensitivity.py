"""Sensitivity of StaCC to every constant, written straight into LaTeX tables.

Each job is one ``python sensitivity.py <job>`` command, so it runs the same way
in PowerShell, cmd or bash, and each job rewrites only its own rows of the tables
in ``new_paper/sensitivity_tables``.  Nothing else is written: no JSON, no result
file.  The tables are the record, and rows a job has not reached keep their red
``\\num`` placeholders.

Every row a job fills ends in a LaTeX comment ``% @<id>`` naming it, which is how
the next job finds its own rows.  The comment is invisible in the compiled paper,
so each file can be ``\\input`` as it stands.  A job writes its rows only after
every pool it covers has finished, so an interrupted job leaves the table as it
was rather than half filled with a one-pool mean.

    python sensitivity.py list        # every job, and whether its cache exists
    python sensitivity.py h1-c        # one sweep; fills its rows when done

Settings are the deployed ones of Table 2 except the one constant a job varies.
On hospital 1 the perturbation family is chosen afresh at every setting by the
label-free rule, as the deployed procedure would.  On the four test columns it is
held at stain jitter 0.15, and the share table says whether that choice survives.
"""
from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from argparse import Namespace

import numpy as np

import benchmarks as B
import run_stacc
from stacc import calibrate
from stacc import certificate as CERT
from stacc import committee as COM
from stacc.metrics import brier
from stacc.registry import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.normpath(os.path.join(_HERE, "..", "sensitivity_tables"))

# The deployed constants of Table 2.  Every job varies exactly one of them.
DEPLOYED = dict(k=10, c=0.2, period=20, n_q=500, mode="worst", rule="line",
                min_share=0.6, max_class_share=0.85, eta=0.1, lam=1.0, seed=1,
                draws=0)
M_POOL = 40          # members per pool, for the evaluation counts of eq. (cost)
P_DEPLOYED = 3       # draws in the main caches
TEST_FAMILY = "stain@0.15"
PGD_TAG = "estimator_h1_n5000"

AXES = {
    # axis: (argument it sets, values swept, deployed value)
    "c": ("c", [0.0, 0.1, 0.2, 0.4, 1.0], 0.2),
    "k": ("k", [3, 5, 10, 20], 10),
    "T": ("period", [1, 5, 20], 20),
    "P": ("draws", [1, 3, 10], 3),
    "tol": ("max_class_share", [0.70, 0.85, 0.95, 0.99, 1.0], 0.85),
    "varsigma": ("min_share", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 0.6),
    "mode": ("mode", ["worst", "mean"], "worst"),
}
AXIS_LABEL = {"c": "$c$", "k": "$k$", "T": "$T$", "P": "$P$",
              "tol": r"$\mathrm{tol}$", "varsigma": r"$\varsigma$",
              "mode": "estimator"}
H1_AXES = ["c", "k", "T", "P", "tol", "varsigma", "mode"]
TEST_AXES = ["c", "k", "T", "P", "tol", "mode"]
TEST_COLUMNS = ["camelyon17", "rxrx1", "iwildcam", "kather"]
SHARE_COLUMNS = ["h1", "camelyon17", "rxrx1", "iwildcam", "kather"]

# Table 4 at the deployed constants, accuracy and certified risk, so each test
# sweep can say whether its deployed row reproduces the main result.
REFERENCE = {"camelyon17": (0.918, 0.270), "rxrx1": (0.283, 2.030),
             "iwildcam": (0.593, 0.686), "kather": (0.981, 0.398)}

H1_FILE = "tab_sens_h1.tex"
TEST_FILE = "tab_sens_test.tex"
SHARE_FILE = "tab_sens_share.tex"
PGD_FILE = "tab_sens_estimator.tex"
MENU_FILE = "tab_family_menu.tex"

PH = r"\num"


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def vid(v):
    """Stable text for a swept value, used in row ids."""
    return v if isinstance(v, str) else f"{v:g}"


def evaluations(k=DEPLOYED["k"], T=DEPLOYED["period"], P=P_DEPLOYED):
    """Model evaluations per query batch, M(1+P)/T + k."""
    return int(round(M_POOL * (1 + P) / T + k))


def value_label(axis, v, brackets):
    star = "^{\\star}" if v == AXES[axis][2] else ""
    if axis == "mode":
        name = {"worst": "worst", "mean": "average"}[v]
        return name + ("$^{\\star}$" if star else "")
    if axis == "tol":
        return "off" if v >= 1.0 else f"${1.0 - v:.2f}{star}$"
    text = f"${vid(v)}{star}$"
    if brackets and axis == "k":
        text += f" $[{evaluations(k=v)}]$"
    elif brackets and axis == "T":
        text += f" $[{evaluations(T=v)}]$"
    elif brackets and axis == "P":
        text += f" $[{evaluations(P=v)}]$"
    return text


SHORT = {"stain": "stain", "photometric": "photo", "gauss": "gauss",
         "linf": r"$\ell_\infty$", "adv": "PGD"}


def family_label(key):
    if key is None:
        return "abstain"
    name, eps = key.split("@")
    eps = eps[1:] if eps.startswith("0.") else eps
    return f"{SHORT.get(name, name)} ${eps}$"


def finite(xs):
    return [float(x) for x in xs if x is not None and np.isfinite(x)]


def num(xs, places=3):
    xs = finite(xs)
    return "--" if not xs else f"${np.mean(xs):.{places}f}$"


def mean_sd(xs, places=3, bold=False):
    xs = finite(xs)
    if not xs:
        return "--"
    m, s = float(np.mean(xs)), float(np.std(xs))
    body = f"{m:.{places}f}"
    if bold:
        body = f"\\mathbf{{{body}}}"
    if len(xs) == 1:
        return f"${body}$"
    return f"${body}_{{\\pm{s:.{places}f}}}$"


def menu_families():
    out = []
    for part in B.FAMILIES.split(","):
        name, eps, _ = part.split(":")
        out.append(f"{name}@{float(eps):g}")
    return out


# --------------------------------------------------------------------------- #
# the tables
# --------------------------------------------------------------------------- #
NOTE = ("% Generated by code/sensitivity.py.  Rows ending in \"% @id\" are rewritten\n"
        "% by its jobs, so captions may be edited freely but those rows may not.\n")


def _row(row_id, cells):
    return " & ".join(cells) + r" \\ % @" + row_id


def template_h1():
    head = NOTE + r"""\begin{table}[!ht]
\scriptsize\centering
\caption{Sensitivity on Camelyon17 hospital~1, the domain the constants were fixed
on, one constant at a time with the others at their deployed values, mean over
three pools. Accuracy, with its spread across pools, and Brier risk are those of
the deployed committee; $J$ is its nominal certified risk, gap the Frank--Wolfe
duality gap, the coverages are as in Table~\ref{tab:validity}, and excl.\ is the
mean number of members the dispersion guard removes per reselection. The family
is the one the label-free rule of Section~\ref{sec:choose-family} returns at that
setting; where it abstains, accuracy and Brier risk are those of the full-pool
ensemble it defers to. Brackets give model evaluations per query batch by
\eqref{eq:cost}. $\star$ marks the deployed value.}
\label{tab:sens-h1}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccccc}
\toprule
axis & value & family & accuracy & Brier & $J(\wv)$ & gap & member cov. & committee cov. & excl. \\
\midrule
"""
    lines = []
    for n, axis in enumerate(H1_AXES):
        if n:
            lines.append(r"\midrule")
        for i, v in enumerate(AXES[axis][1]):
            label = AXIS_LABEL[axis] if i == 0 else ""
            lines.append(_row(f"h1:{axis}={vid(v)}",
                              [label, value_label(axis, v, True)] + [PH] * 8))
    return head + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""


def template_test():
    head = NOTE + r"""\begin{table}[!ht]
\scriptsize\centering
\caption{Sensitivity on the official test split of each benchmark, family held at
stain jitter with $\varepsilon=0.15$, one constant at a time with the others at
their deployed values, mean over three pools. Each cell is accuracy / nominal
certified risk $J(\wv)$. $\star$ marks the deployed value, whose rows reproduce
Table~\ref{tab:main}. RxRx1 has six query batches, so there $T=5$ and $T=20$
differ by a single reselection. A dash marks a benchmark for which the 10-draw
cache was not built.}
\label{tab:sens-test}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}
\toprule
axis & value & Camelyon17, h.~2 & RxRx1 & iWildCam & Kather \\
\midrule
"""
    cell = r"\num\,/\,\num"
    lines = []
    for n, axis in enumerate(TEST_AXES):
        if n:
            lines.append(r"\midrule")
        for i, v in enumerate(AXES[axis][1]):
            label = AXIS_LABEL[axis] if i == 0 else ""
            lines.append(_row(f"test:{axis}={vid(v)}",
                              [label, value_label(axis, v, False)] + [cell] * 4))
    return head + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""


def template_share():
    head = NOTE + r"""\begin{table}[!ht]
\scriptsize\centering
\caption{Stability share $\mathrm{sh}(f)$ of \eqref{eq:share} for each family in the
menu at the deployed constants, mean over three pools, with the family's rank by
mean certified optimum $\min_{\wv}J_f$ in brackets, among families no pool
falsifies (fals.\ otherwise). Since the chosen family is the best-ranked family
whose share clears $\varsigma$, and the method abstains once none does, this table
determines the whole $\varsigma$ sweep on every column. Photo is photometric
jitter.}
\label{tab:sens-share}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccccc}
\toprule
family & Camelyon17, h.~1 & Camelyon17, h.~2 & RxRx1 & iWildCam & Kather \\
\midrule
"""
    cell = r"\num~(\num)"
    lines = [_row(f"share:{fam}", [family_label(fam)] + [cell] * 5)
             for fam in menu_families()]
    return head + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""


def template_pgd():
    head = NOTE + r"""\begin{table}[!ht]
\scriptsize\centering
\caption{The sensitivity estimator on Camelyon17 hospital~1, $5{,}000$ query
points, three pools, at the deployed constants. Random draws and projected
gradient ascent act on the same $\ell_\infty$ pixel ball of radius
$\varepsilon=0.03$, so the fourth column is the mean ratio by which the maximum over
$P$ random draws falls short of the worst case it estimates. Projected gradient
ascent in the parameter space of the stain family is not implemented, which is why
the comparison uses a pixel-ball family.}
\label{tab:sens-estimator}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{lccccc}
\toprule
estimator & cost per point & mean $s_i$ & $s_i$ over PGD & member cov. & accuracy \\
\midrule
"""
    lines = [
        _row("est:random=1", ["random, $P=1$", "$1$ forward", PH, PH, PH, PH]),
        _row("est:random=3", ["random, $P=3$", "$3$ forward", PH, PH, PH, PH]),
        _row("est:random=10", ["random, $P=10$", "$10$ forward", PH, PH, PH, PH]),
        _row("est:pgd", ["PGD, $10$ steps", "$10$ forward and backward",
                         PH, "$1$", PH, PH]),
    ]
    return head + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""


MENU_BENCH_LABEL = {"camelyon17": "Camelyon17, h.~2", "rxrx1": "RxRx1",
                    "iwildcam": "iWildCam", "kather": "Kather"}


def template_menu():
    head = NOTE + r"""\begin{table}[!ht]
\scriptsize\centering
\caption{Every family on the menu, on the official test split of every benchmark,
at the deployed constants, mean over three pools. The stability share
$\mathrm{sh}(f)$ is \eqref{eq:share}, $\min_{\wv}J_f$ is the certified optimum by
which the label-free rule ranks families, and falsified is the number of member
pairs Proposition~\ref{prop:falsify} rejects out of $780$; these three use no
label and are computed on the whole query subset, as the rule computes them.
Accuracy is that of the committee the method returns when made to use that
family, with its spread across pools, and uses target labels for analysis only.
The last column counts the pools on which the label-free rule of
Section~\ref{sec:choose-family} selects the family, and bold marks the most
accurate family on each benchmark.}
\label{tab:family-menu}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccc}
\toprule
benchmark & family & $\mathrm{sh}(f)$ & $\min_{\wv}J_f$ & falsified & accuracy & chosen \\
\midrule
"""
    lines = []
    for n, name in enumerate(TEST_COLUMNS):
        if n:
            lines.append(r"\midrule")
        for i, fam in enumerate(menu_families()):
            label = MENU_BENCH_LABEL[name] if i == 0 else ""
            lines.append(_row(f"menu:{name}:{fam}",
                              [label, family_label(fam)] + [PH] * 5))
    return head + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""


TEMPLATES = {H1_FILE: template_h1, TEST_FILE: template_test,
             SHARE_FILE: template_share, PGD_FILE: template_pgd,
             MENU_FILE: template_menu}


def ensure_tables(folder):
    """Create any table that does not exist yet; never overwrite one that does."""
    os.makedirs(folder, exist_ok=True)
    for name, build in TEMPLATES.items():
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(build())


def fill(path, row_id, updates):
    """Rewrite some cells of one tagged row, leaving every other row untouched."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    marker = r"\\ % @" + row_id
    for i, line in enumerate(lines):
        if line.rstrip().endswith("% @" + row_id) and marker in line:
            cells = [c.strip() for c in line.rsplit(marker, 1)[0].split("&")]
            for j, text in updates.items():
                cells[j] = text
            lines[i] = _row(row_id, cells)
            break
    else:
        raise SystemExit(f"row {row_id!r} not found in {path}; "
                         f"was the table edited by hand?")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# running the method
# --------------------------------------------------------------------------- #
def _member_scores(clean, pert, mode):
    """Stability scores one member at a time, to cap memory at one member."""
    return np.concatenate([CERT.stability_score(clean[i:i + 1], pert[i:i + 1],
                                                mode=mode)
                           for i in range(len(clean))])


class FamilyChooser:
    """What the label-free family choice reads, loaded once per pool.

    The choice depends on c, the estimator and the threshold, so a sweep over any
    of them repeats it at every setting.  What it reads does not change, so the
    stability scores of every family under both estimators are computed once,
    a family at a time so that memory holds one family's outputs at most.
    """

    def __init__(self, pool_dir, tag):
        pool = Pool(pool_dir)
        self.families = list(pool.stability_index(tag)["families"])
        preds, labels, _ = pool.load_clean(splits=["source_val"])
        val_y = labels["source_val"].astype(int)
        temps = calibrate.fit_pool(preds["source_val"], val_y)
        val_p = calibrate.apply_pool(preds["source_val"], temps)
        self.val_brier = np.array([brier(val_p[i], val_y) for i in range(len(val_p))])
        del preds, val_p
        self.scores, self.G = {}, None
        for fam in self.families:
            clean, pert, _ = pool.load_stability(tag, families=[fam])
            clean = calibrate.apply_pool(clean, temps)
            if self.G is None:
                self.G = COM.gram(clean)
            p = calibrate.apply_pool(pert[fam], temps)
            del pert
            for mode in ("worst", "mean"):
                self.scores[(fam, mode)] = _member_scores(clean, p, mode)
            del clean, p
            gc.collect()

    def rank(self, c, mode, min_share):
        radii, shares = {}, {}
        for fam in self.families:
            s = self.scores[(fam, mode)]
            radii[fam] = CERT.certified_radii(self.val_brier, s, c=c)
            shares[fam] = CERT.stability_share(self.val_brier, s, c=c)
        return CERT.select_family(self.G, radii, shares, min_share=min_share)


def run_setting(pool, tag, family, overrides, max_batches):
    """The method at the deployed constants with some overridden, on one pool."""
    s = dict(DEPLOYED)
    s.update(overrides)
    args = Namespace(pool=pool, tag=tag, family=family, k=s["k"], n_q=s["n_q"],
                     period=s["period"], mode=s["mode"], rule=s["rule"], c=s["c"],
                     eta=s["eta"], lam=s["lam"],
                     max_class_share=s["max_class_share"], allow_repeats=False,
                     no_calibrate=False, seed=s["seed"], max_batches=max_batches,
                     out=None, quiet=True, lean=True, draws=s["draws"])
    agg, diag, _, extra = run_stacc.run(args)
    resel = [d for d in diag if d["batch"] % s["period"] == 0]
    return dict(
        acc=float(np.mean(agg["stacc"])),
        brier=float(np.mean(extra["brier"]["stacc"])),
        full_acc=float(np.mean(agg["full_pool"])),
        full_brier=float(np.mean(extra["brier"]["full_pool"])),
        J=float(np.mean([d["J"] for d in diag])),
        gap=float(np.mean([d["gap"] for d in diag])),
        member_cov=float(np.mean([d["member_coverage"] for d in diag])),
        committee_cov=float(np.mean([d["committee_covered"] for d in diag])),
        excluded=float(np.mean([d["excluded"] for d in resel])) if resel else float("nan"),
    )


def free():
    run_stacc._CACHE.clear()
    run_stacc._VAL_CACHE.clear()
    gc.collect()


def has_cache(pool, tag):
    return os.path.exists(os.path.join(pool, "stability", tag, "index.json"))


def need_cache(pool, tag, job):
    if not has_cache(pool, tag):
        hint = f"\n  run first:  python sensitivity.py {job}" if job else ""
        raise SystemExit(f"no cached run {tag!r} under {pool}{hint}")


def only_family(pool, tag):
    return list(Pool(pool).stability_index(tag)["families"])[0]


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
def job_h1(axis, o):
    name = "camelyon17"
    b = B.BENCHMARKS[name]
    base = B.tag(b["ood_val"], b["limit"])
    tag = base + "_P10" if axis == "P" else base
    key, values, _ = AXES[axis]
    results = {v: [] for v in values}
    for seed in o.seeds:
        pool = B.pool_dir(name, seed)
        need_cache(pool, tag, "cache-h1-P10" if axis == "P" else None)
        chooser = None if axis == "P" else FamilyChooser(pool, tag)
        memo = {}
        for v in values:
            s = dict(DEPLOYED)
            s[key] = v
            if chooser is None:
                chosen = fam = only_family(pool, tag)
            else:
                chosen, rows = chooser.rank(s["c"], s["mode"], s["min_share"])
                fam = chosen or min(rows, key=lambda r: r["J_star"])["family"]
            # the threshold changes only the family, so equal families share a run
            mkey = (fam, None if key == "min_share" else v)
            if mkey not in memo:
                memo[mkey] = run_setting(pool, tag, fam, {key: v}, o.max_batches)
            r = dict(memo[mkey])
            r["family"] = chosen
            results[v].append(r)
            print(f"  {seed:8} {axis}={vid(v):>6}  family {chosen or 'abstain':16}"
                  f"  acc {r['acc']:.4f}  J {r['J']:.3f}", flush=True)
        del chooser
        free()

    path = os.path.join(o.tables, H1_FILE)
    for v in values:
        rs = results[v]
        fams = list(dict.fromkeys(family_label(r["family"]) for r in rs))
        live = [r for r in rs if r["family"] is not None]
        fill(path, f"h1:{axis}={vid(v)}", {
            2: ", ".join(fams),
            3: mean_sd([r["acc"] if r["family"] else r["full_acc"] for r in rs]),
            4: num([r["brier"] if r["family"] else r["full_brier"] for r in rs]),
            5: num([r["J"] for r in live]),
            6: num([r["gap"] for r in live]),
            7: num([r["member_cov"] for r in live], 2),
            8: num([r["committee_cov"] for r in live], 2),
            9: num([r["excluded"] for r in live], 1),
        })
    print(f"filled {len(values)} rows of {path}")


def job_test(axis, o):
    key, values, deployed = AXES[axis]
    path = os.path.join(o.tables, TEST_FILE)
    column = {n: i + 2 for i, n in enumerate(TEST_COLUMNS)}
    names = [n for n in (o.benchmarks or TEST_COLUMNS) if n in TEST_COLUMNS]
    for name in names:
        b = B.BENCHMARKS[name]
        tag = B.tag(b["ood_test"], b["limit"]) + ("_P10" if axis == "P" else "")
        pools = [B.pool_dir(name, s) for s in o.seeds]
        if axis == "P" and not all(has_cache(p, tag) for p in pools):
            for v in values:
                fill(path, f"test:{axis}={vid(v)}", {column[name]: "--"})
            print(f"{name}: no 10-draw cache, its P cells are marked --; to fill "
                  f"them run  python sensitivity.py cache-{name}-P10  then this job")
            continue
        for p in pools:
            need_cache(p, tag, None)
        results = {v: [] for v in values}
        for seed, pool in zip(o.seeds, pools):
            fam = only_family(pool, tag) if axis == "P" else TEST_FAMILY
            for v in values:
                r = run_setting(pool, tag, fam, {key: v}, o.max_batches)
                results[v].append(r)
                print(f"  {name:10} {seed:8} {axis}={vid(v):>6}  acc {r['acc']:.4f}"
                      f"  J {r['J']:.3f}", flush=True)
            free()
        for v in values:
            acc = float(np.mean([r["acc"] for r in results[v]]))
            J = float(np.mean([r["J"] for r in results[v]]))
            fill(path, f"test:{axis}={vid(v)}",
                 {column[name]: f"${acc:.3f}$\\,/\\,${J:.3f}$"})
            if v == deployed and not o.max_batches:
                ra, rj = REFERENCE[name]
                same = abs(acc - ra) <= 0.0015 and abs(J - rj) <= 0.002
                print(f"  {name}: deployed row {acc:.3f} / {J:.3f} against Table 4 "
                      f"{ra:.3f} / {rj:.3f}: "
                      + ("reproduces it" if same else "WARNING, does not reproduce it"))
        print(f"filled the {name} column of {path}")


def job_shares(o):
    path = os.path.join(o.tables, SHARE_FILE)
    column = {n: i + 1 for i, n in enumerate(SHARE_COLUMNS)}
    names = [n for n in (o.benchmarks or SHARE_COLUMNS) if n in SHARE_COLUMNS]
    menu = menu_families()
    for col in names:
        name = "camelyon17" if col == "h1" else col
        b = B.BENCHMARKS[name]
        tag = B.tag(b["ood_val"] if col == "h1" else b["ood_test"], b["limit"])
        per = {f: {"share": [], "J": [], "fals": []} for f in menu}
        for seed in o.seeds:
            pool = B.pool_dir(name, seed)
            need_cache(pool, tag, None)
            chooser = FamilyChooser(pool, tag)
            chosen, rows = chooser.rank(DEPLOYED["c"], DEPLOYED["mode"],
                                        DEPLOYED["min_share"])
            for r in rows:
                if r["family"] in per:
                    per[r["family"]]["share"].append(r["stability_share"])
                    per[r["family"]]["J"].append(r["J_star"])
                    per[r["family"]]["fals"].append(r["falsified_pairs"])
            print(f"  {col:10} {seed:8} chosen at the deployed constants: "
                  f"{chosen or 'abstain'}", flush=True)
            del chooser
            gc.collect()
        ok = [f for f in menu if per[f]["fals"] and max(per[f]["fals"]) == 0]
        order = sorted(ok, key=lambda f: float(np.mean(per[f]["J"])))
        for fam in menu:
            if not per[fam]["share"]:
                fill(path, f"share:{fam}", {column[col]: "--"})
                continue
            rank = f"${order.index(fam) + 1}$" if fam in ok else "fals."
            fill(path, f"share:{fam}",
                 {column[col]: f"${np.mean(per[fam]['share']):.2f}$~({rank})"})
        print(f"filled the {col} column of {path}")


def job_pgd(o):
    name = "camelyon17"
    path = os.path.join(o.tables, PGD_FILE)
    s_rand = {P: [] for P in (1, 3, 10)}
    ratio = {P: [] for P in (1, 3, 10)}
    runs = {P: [] for P in (1, 3, 10)}
    s_pgd, runs_pgd = [], []
    for seed in o.seeds:
        pool = B.pool_dir(name, seed)
        need_cache(pool, PGD_TAG, "cache-pgd")
        rand = run_stacc.load_everything(pool, PGD_TAG, "linf@0.03")
        adv = run_stacc.load_everything(pool, PGD_TAG, "adv@0.03")
        sp = _member_scores(adv["clean"], adv["perturbed"], "worst")
        s_pgd.append(float(sp.mean()))
        for P in (1, 3, 10):
            sr = _member_scores(rand["clean"], rand["perturbed"][:, :P], "worst")
            s_rand[P].append(float(sr.mean()))
            ratio[P].append(float(np.mean(sr / np.maximum(sp, 1e-12))))
            runs[P].append(run_setting(pool, PGD_TAG, "linf@0.03", {"draws": P},
                                       o.max_batches))
            print(f"  {seed:8} random P={P:<2}  mean s {s_rand[P][-1]:.4f}  "
                  f"ratio to PGD {ratio[P][-1]:.3f}", flush=True)
        runs_pgd.append(run_setting(pool, PGD_TAG, "adv@0.03", {}, o.max_batches))
        print(f"  {seed:8} PGD          mean s {s_pgd[-1]:.4f}", flush=True)
        del rand, adv
        free()
    for P in (1, 3, 10):
        fill(path, f"est:random={P}", {
            2: num(s_rand[P]), 3: num(ratio[P], 2),
            4: num([r["member_cov"] for r in runs[P]], 2),
            5: mean_sd([r["acc"] for r in runs[P]]),
        })
    fill(path, "est:pgd", {
        2: num(s_pgd), 4: num([r["member_cov"] for r in runs_pgd], 2),
        5: mean_sd([r["acc"] for r in runs_pgd]),
    })
    print(f"filled {path}")


def job_menu(o):
    """Every family on the menu, its label-free statistics and its accuracy.

    The statistics are exactly those the rule reads, computed once per pool on the
    whole query subset.  The accuracy is that of the committee the method returns
    when made to use the family, over the same query stream as Table 4.  Families
    are run one at a time and released, so memory holds one family's outputs.
    """
    path = os.path.join(o.tables, MENU_FILE)
    menu = menu_families()
    names = [n for n in (o.benchmarks or TEST_COLUMNS) if n in TEST_COLUMNS]
    for name in names:
        b = B.BENCHMARKS[name]
        tag = B.tag(b["ood_test"], b["limit"])
        per = {f: {"share": [], "J": [], "fals": [], "acc": [], "chosen": 0}
               for f in menu}
        for seed in o.seeds:
            pool = B.pool_dir(name, seed)
            need_cache(pool, tag, None)
            chooser = FamilyChooser(pool, tag)
            available = list(chooser.families)
            chosen, rows = chooser.rank(DEPLOYED["c"], DEPLOYED["mode"],
                                        DEPLOYED["min_share"])
            del chooser
            gc.collect()
            for r in rows:
                if r["family"] in per:
                    per[r["family"]]["share"].append(r["stability_share"])
                    per[r["family"]]["J"].append(r["J_star"])
                    per[r["family"]]["fals"].append(r["falsified_pairs"])
            if chosen in per:
                per[chosen]["chosen"] += 1
            for fam in menu:
                if fam not in available:
                    continue
                r = run_setting(pool, tag, fam, {}, o.max_batches)
                per[fam]["acc"].append(r["acc"])
                mark = "   <- chosen" if fam == chosen else ""
                print(f"  {name:10} {seed:8} {fam:16} acc {r['acc']:.4f}{mark}",
                      flush=True)
                free()
        means = {f: float(np.mean(per[f]["acc"])) for f in menu if per[f]["acc"]}
        best = max(means, key=means.get) if means else None
        for fam in menu:
            q = per[fam]
            fill(path, f"menu:{name}:{fam}", {
                2: num(q["share"], 2),
                3: num(q["J"]),
                4: num(q["fals"], 1),
                5: mean_sd(q["acc"], bold=(fam == best)),
                6: f"${q['chosen']}/{len(o.seeds)}$",
            })
        if means:
            worst = min(means, key=means.get)
            picked = [f for f in menu if per[f]["chosen"] and f in means]
            for f in picked:
                print(f"  {name}: chosen {f} on {per[f]['chosen']}/{len(o.seeds)} "
                      f"pools, accuracy {means[f]:.3f}; best in the menu {best} "
                      f"{means[best]:.3f}, worst {worst} {means[worst]:.3f}; "
                      f"price of choosing without labels {means[best] - means[f]:.3f}")
            if not picked:
                print(f"  {name}: the rule abstains on every pool")
            if TEST_FAMILY in means and not o.max_batches:
                ra = REFERENCE[name][0]
                same = abs(means[TEST_FAMILY] - ra) <= 0.0015
                print(f"  {name}: {TEST_FAMILY} accuracy {means[TEST_FAMILY]:.3f} "
                      f"against Table 4 {ra:.3f}: "
                      + ("reproduces it" if same else "WARNING, does not reproduce it"))
        print(f"filled the {name} rows of {path}")


# --------------------------------------------------------------------------- #
# caching jobs, which need the GPU
# --------------------------------------------------------------------------- #
def _precompute(args):
    cmd = [sys.executable, os.path.join(_HERE, "precompute_stability.py")] + args
    rc = subprocess.call(cmd, cwd=_HERE)
    if rc != 0:
        raise SystemExit(f"precompute_stability.py exited with {rc}")


def job_cache_p10(target, o):
    """A 10-draw cache of the family in use, on one split.

    Draws are seeded by member, family and draw index, and the caching batch is
    the one the main cache used, so the first three draws here are exactly the
    main cache's three and the P=3 row reproduces the deployed run.
    """
    name = "camelyon17" if target == "h1" else target
    b = B.BENCHMARKS[name]
    split = b["ood_val"] if target == "h1" else b["ood_test"]
    base = B.tag(split, b["limit"])
    tag = base + "_P10"
    for seed in o.seeds:
        pool = B.pool_dir(name, seed)
        if has_cache(pool, tag):
            print(f"{pool}: {tag} already cached")
            continue
        if target == "h1":
            need_cache(pool, base, None)
            chooser = FamilyChooser(pool, base)
            fam, _ = chooser.rank(DEPLOYED["c"], DEPLOYED["mode"], DEPLOYED["min_share"])
            del chooser
            gc.collect()
            if fam is None:
                raise SystemExit(f"the rule abstains on {pool} at the deployed "
                                 f"constants, so there is no family to cache")
        else:
            fam = TEST_FAMILY
        fname, eps = fam.split("@")
        print(f"\n{pool}: caching {fam} with 10 draws as {tag}", flush=True)
        _precompute(["--pool", pool, "--data", b["data"], "--split", split,
                     "--families", f"{fname}:{eps}:10", "--limit", str(b["limit"]),
                     "--batch", str(b["batch"]), "--tag", tag, "--workers", "4"])


def job_cache_pgd(o):
    name = "camelyon17"
    b = B.BENCHMARKS[name]
    for seed in o.seeds:
        pool = B.pool_dir(name, seed)
        if has_cache(pool, PGD_TAG):
            print(f"{pool}: {PGD_TAG} already cached")
            continue
        print(f"\n{pool}: caching random and PGD estimators as {PGD_TAG}", flush=True)
        _precompute(["--pool", pool, "--data", b["data"], "--split", b["ood_val"],
                     "--families", "linf:0.03:10,adv:0.03:1", "--adv-steps", "10",
                     "--limit", "5000", "--batch", "256", "--tag", PGD_TAG,
                     "--workers", "4"])


# --------------------------------------------------------------------------- #
def build_jobs():
    jobs = {}
    for axis in H1_AXES:
        jobs[f"h1-{axis}"] = (lambda o, ax=axis: job_h1(ax, o),
                              f"hospital 1, sweep {axis}",
                              [("camelyon17", "ood_val", "_P10" if axis == "P" else "")])
    for axis in TEST_AXES:
        jobs[f"test-{axis}"] = (lambda o, ax=axis: job_test(ax, o),
                                f"four test columns, sweep {axis}",
                                [(n, "ood_test", "_P10" if axis == "P" else "")
                                 for n in TEST_COLUMNS])
    jobs["shares"] = (job_shares, "stability shares of every family, five splits",
                      [("camelyon17", "ood_val", "")]
                      + [(n, "ood_test", "") for n in TEST_COLUMNS])
    jobs["menu"] = (job_menu, "every family: share, J*, falsified, accuracy",
                    [(n, "ood_test", "") for n in TEST_COLUMNS])
    jobs["pgd"] = (job_pgd, "random draws against projected gradient ascent",
                   [("camelyon17", None, PGD_TAG)])
    jobs["cache-h1-P10"] = (lambda o: job_cache_p10("h1", o),
                            "GPU: 10-draw cache on hospital 1", [])
    for n in TEST_COLUMNS:
        jobs[f"cache-{n}-P10"] = (lambda o, t=n: job_cache_p10(t, o),
                                  f"GPU: 10-draw cache on the {n} test split", [])
    jobs["cache-pgd"] = (job_cache_pgd, "GPU: estimator cache on hospital 1", [])
    return jobs


def job_list(jobs, o):
    print(f"tables: {o.tables}\n")
    for name, (_, desc, needs) in jobs.items():
        state = ""
        if needs:
            missing = 0
            for bench, which, suffix in needs:
                b = B.BENCHMARKS[bench]
                tag = PGD_TAG if which is None else B.tag(b[which], b["limit"]) + suffix
                missing += sum(not has_cache(B.pool_dir(bench, s), tag) for s in o.seeds)
            state = "ready" if not missing else f"missing {missing} cache(s)"
        print(f"  {name:22} {desc:48} {state}")


def main(argv=None):
    jobs = build_jobs()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job", help="list, init, or one of the jobs `list` prints")
    p.add_argument("--benchmarks", default=None,
                   help="restrict a test or shares job, e.g. rxrx1 or h1,kather")
    p.add_argument("--seeds", default=None, help="restrict to some pools")
    p.add_argument("--tables", default=TABLES, help="folder holding the tables")
    p.add_argument("--max-batches", type=int, default=0, help="debugging only")
    o = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True, errors="replace")
    except Exception:
        pass
    o.seeds = [s.strip() for s in o.seeds.split(",")] if o.seeds else list(B.SEEDS)
    o.benchmarks = ([s.strip() for s in o.benchmarks.split(",")]
                    if o.benchmarks else None)
    ensure_tables(o.tables)

    if o.job == "init":
        print(f"tables ready in {o.tables}")
        return 0
    if o.job == "list":
        job_list(jobs, o)
        return 0
    if o.job not in jobs:
        print(f"unknown job {o.job!r}\n")
        job_list(jobs, o)
        return 2
    jobs[o.job][0](o)
    return 0


if __name__ == "__main__":
    sys.exit(main())

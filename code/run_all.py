"""Run the diagnostics and the method on every cached pool, in one pass.

This is the whole experimental phase after the caches exist, and it touches no
GPU.  For each pool it does two things in order.

First ``diagnose.py``, which measures the precondition, the per-family signal and
the certificate diagnostics, and which chooses the perturbation family without
labels by the rule of Section 5.4.  That choice is the input to the second step,
so the family is never picked by hand and never picked with target labels.

Then ``run_stacc.py`` on the chosen family, which runs the method and every
comparator over the query stream and writes the per-batch scores.

Both write JSON under ``../results`` at the paths ``benchmarks.result_path``
defines, which is where ``make_tables.py`` reads them from.

    python run_all.py --list
    python run_all.py --benchmarks camelyon17
    python run_all.py                       # everything cached, in registry order
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import benchmarks as B


def jobs(names=None, seeds=None, splits=("ood_test",)):
    """One entry per cached (benchmark, pool, split), skipping what has no cache."""
    out = []
    for j in B.cache_jobs(names, seeds, splits):
        index = os.path.join(j["pool"], "stability", j["tag"], "index.json")
        j["cached"] = os.path.exists(index)
        j["n_q"] = B.BENCHMARKS[j["benchmark"]]["n_q"]
        j["diag_out"] = B.result_path(j["benchmark"], j["seed"], j["which"], "diag")
        j["main_out"] = B.result_path(j["benchmark"], j["seed"], j["which"], "main")
        out.append(j)
    return out


def run(cmd, cwd):
    print("  $ " + " ".join(os.path.basename(c) if c.endswith(".py") else c
                            for c in cmd[1:]), flush=True)
    return subprocess.call(cmd, cwd=cwd)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", default=None)
    p.add_argument("--seeds", default=None)
    p.add_argument("--splits", default="ood_test")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--c", type=float, default=0.2)
    p.add_argument("--period", type=int, default=20)
    p.add_argument("--mode", default="worst")
    p.add_argument("--rule", default="line")
    p.add_argument("--min-share", type=float, default=0.6)
    p.add_argument("--max-class-share", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--list", action="store_true")
    p.add_argument("--redo", action="store_true",
                   help="rerun even where the result JSON already exists")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    names = [s.strip() for s in a.benchmarks.split(",")] if a.benchmarks else None
    seeds = [s.strip() for s in a.seeds.split(",")] if a.seeds else None
    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    todo = jobs(names, seeds, splits)

    if a.list:
        for j in todo:
            state = "cached  " if j["cached"] else "NO CACHE"
            done = "done" if os.path.exists(j["main_out"]) else "    "
            print(f"{state} {done} {j['benchmark']:11} {j['seed']:7} "
                  f"{j['which']:8} {j['tag']}")
        n = sum(1 for j in todo if j["cached"])
        print(f"\n{n} of {len(todo)} have caches")
        return 0

    os.makedirs(B.RESULTS, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    failures = []
    for n, j in enumerate([x for x in todo if x["cached"]], 1):
        label = f"{j['benchmark']} {j['seed']} {j['which']}"
        if os.path.exists(j["main_out"]) and not a.redo:
            print(f"[{n}] {label}: already done")
            continue
        print(f"\n=== [{n}] {label} ===", flush=True)
        t0 = time.time()

        rc = run([sys.executable, os.path.join(here, "diagnose.py"),
                  "--pool", j["pool"], "--tag", j["tag"],
                  "--mode", a.mode, "--c", str(a.c),
                  "--min-share", str(a.min_share),
                  "--out", j["diag_out"]], here)
        if rc != 0:
            failures.append(f"diagnose {label}")
            continue

        diag = json.load(open(j["diag_out"], encoding="utf-8"))
        family = diag.get("label_free_family")
        if family is None:
            # No family clears the guard of Proposition 16, so the method
            # abstains here.  The comparators still have to be run, since the
            # table reports what the abstention defers to, and the family with
            # the tightest certificate is the one whose diagnostics are quoted.
            ranking = sorted(diag.get("family_ranking", []),
                             key=lambda r: r["J_star"])
            if not ranking:
                failures.append(f"no families at all for {label}")
                continue
            family = ranking[0]["family"]
            print(f"  abstains; running comparators on {family} anyway")

        rc = run([sys.executable, os.path.join(here, "run_stacc.py"),
                  "--pool", j["pool"], "--tag", j["tag"], "--family", family,
                  "--k", str(a.k), "--n-q", str(j["n_q"]),
                  "--period", str(a.period), "--mode", a.mode,
                  "--rule", a.rule, "--c", str(a.c),
                  "--max-class-share", str(a.max_class_share),
                  "--seed", str(a.seed),
                  "--out", j["main_out"]], here)
        if rc != 0:
            failures.append(f"run_stacc {label}")
        print(f"  {(time.time() - t0) / 60:.1f} min", flush=True)

    print("")
    for f in failures:
        print("  failed: " + f)
    print(f"{len(failures)} failure(s)")
    print("ALL RUNS COMPLETE")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

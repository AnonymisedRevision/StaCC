"""Cache every perturbation run the experiments need, one job at a time.

The jobs come from ``benchmarks.py`` rather than from a list kept here, so
adding a benchmark or a pool seed changes one file and this driver follows.  A
job is one pool, one split, the whole family menu.

They run one after another rather than in parallel: two jobs sharing one GPU
finish no sooner together than apart, and risk running it out of memory.

Written in Python rather than as a shell script for a mundane but real reason.
The data lives under a path containing non-ASCII characters, and PowerShell
reads a script file as ANSI unless it carries a byte-order mark, which silently
corrupts such paths into a parse error.  Python source is UTF-8 by definition,
so the same string survives.

    python cache_queue.py --list
    python cache_queue.py --benchmarks camelyon17 --seeds seed_1
    python cache_queue.py                       # everything, in registry order
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import benchmarks as B


def wait_for(path, marker, poll=30):
    """Block until an earlier job's log reports it finished."""
    print(f"waiting for {path} to report completion", flush=True)
    while True:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                if marker in f.read():
                    break
        except OSError:
            pass
        time.sleep(poll)
    print("prerequisite finished, starting the queue", flush=True)


def describe(job):
    return (f"{job['benchmark']:11} {job['seed']:7} {job['which']:8} "
            f"{job['split']:22} n={job['limit']:>6} tag={job['tag']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", default=None,
                   help="comma-separated subset of the registry")
    p.add_argument("--seeds", default=None, help="comma-separated pool seeds")
    p.add_argument("--splits", default="ood_test")
    p.add_argument("--stage", choices=["clean", "stability", "all"],
                   default="all",
                   help="'clean' caches source_val per pool, which the source "
                        "term of the radius needs; 'stability' caches the query "
                        "subsets clean and perturbed")
    p.add_argument("--clean-limit", type=int, default=10000,
                   help="rows of source_val to score per pool")
    p.add_argument("--with-sweep", action="store_true",
                   help="also cache the dense radius grid the figure needs")
    p.add_argument("--list", action="store_true",
                   help="print the queue and exit without running anything")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="leave a cached run alone if its index is already there")
    p.add_argument("--force", action="store_true",
                   help="recache even where an index already exists")
    p.add_argument("--wait-for", default=None,
                   help="a log file whose completion gates the queue")
    p.add_argument("--marker", default="cached 40 members")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    names = [s.strip() for s in a.benchmarks.split(",")] if a.benchmarks else None
    seeds = [s.strip() for s in a.seeds.split(",")] if a.seeds else None
    splits = tuple(s.strip() for s in a.splits.split(",") if s.strip())
    jobs = B.cache_jobs(names, seeds, splits)
    if a.with_sweep:
        jobs.append(B.sweep_job())

    if a.list:
        for j in jobs:
            done = os.path.exists(os.path.join(j["pool"], "stability", j["tag"],
                                               "index.json"))
            print(("done " if done else "     ") + describe(j))
        print(f"\n{len(jobs)} jobs")
        return 0

    if a.wait_for:
        wait_for(a.wait_for, a.marker)

    here = os.path.dirname(os.path.abspath(__file__))
    failures, skipped = [], 0

    if a.stage in ("clean", "all"):
        pools = []
        for j in jobs:                      # one entry per pool, in queue order
            key = (j["pool"], j["data"])
            if key not in pools:
                pools.append(key)
        for n, (pool, data) in enumerate(pools, 1):
            print(f"\n=== clean [{n}/{len(pools)}] {os.path.basename(pool)} ===",
                  flush=True)
            cmd = [sys.executable, os.path.join(here, "precompute_clean.py"),
                   "--pool", pool, "--data", data,
                   "--limit", str(a.clean_limit), "--batch", "256",
                   "--workers", "4"]
            if a.force:
                cmd.append("--force")
            rc = subprocess.call(cmd, cwd=here)
            if rc != 0:
                failures.append(f"clean {os.path.basename(pool)}")
        if a.stage == "clean":
            print("")
            for f in failures:
                print("  failed: " + f)
            print("QUEUE COMPLETE")
            return 1 if failures else 0

    for n, j in enumerate(jobs, 1):
        index = os.path.join(j["pool"], "stability", j["tag"], "index.json")
        if os.path.exists(index) and not a.force:
            print(f"[{n}/{len(jobs)}] skip, already cached: {describe(j)}",
                  flush=True)
            skipped += 1
            continue
        print(f"\n=== [{n}/{len(jobs)}] {describe(j)} ===", flush=True)
        started = time.time()
        cmd = [sys.executable, os.path.join(here, "precompute_stability.py"),
               "--pool", j["pool"], "--data", j["data"], "--split", j["split"],
               "--families", j["families"], "--limit", str(j["limit"]),
               "--tag", j["tag"], "--batch", str(j["batch"]), "--workers", "4"]
        if a.force:
            cmd.append("--force")
        rc = subprocess.call(cmd, cwd=here)
        print(f"[{n}/{len(jobs)}] exit {rc} after "
              f"{(time.time() - started) / 60:.1f} min", flush=True)
        if rc != 0:
            failures.append(describe(j))

    print("")
    print(f"{len(jobs) - len(failures) - skipped} run, {skipped} already cached, "
          f"{len(failures)} failed")
    for f in failures:
        print("  failed: " + f)
    print("QUEUE COMPLETE")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

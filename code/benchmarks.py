"""The benchmarks, their pools, and what to cache for each.

One place that knows the four datasets, which split of each is the official
out-of-distribution test, which is the outer validation split settings may be
tuned on, and which pool seeds exist.  Every driver imports from here so that
adding a benchmark or a seed is a change in one file.

The split that matters is named ``ood_test`` in every entry, and nothing is ever
tuned on it.  Every constant of the method is fixed once, on Camelyon17's
hospital 1, the out-of-distribution validation domain WILDS designates, and is
used unchanged on all four benchmarks.  The ``tune`` entry names the split a
per-benchmark tuning would use; the reported pipeline never reads it.

Query-subset sizes differ by design rather than by accident.  A cached run holds
one array of (subset x classes) per member per draw, so cost scales with the
label cardinality; at 1,139 classes a subset the size of Camelyon's would be two
orders of magnitude larger on disk for no gain in what the batches can show.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
# Packed datasets and pools sit at the repository root by default; set
# STACC_DATA or STACC_POOLS to keep either somewhere else.
DATA = os.environ.get("STACC_DATA", os.path.join(_ROOT, "data"))
POOLS = os.environ.get("STACC_POOLS", os.path.join(_ROOT, "pools"))

# The perturbation menu, fixed in advance and identical on every benchmark.
# Three draws per family, which is what the worst-case estimator of Section 3.2
# takes its maximum over and what the cost of eq. (24) is quoted at.
FAMILIES = "stain:0.15:3,stain:0.35:3,photometric:0.4:3,photometric:0.8:3,gauss:0.25:3"
DRAWS = 3

# A denser grid of radii, cached once on the tuning split only, for the figure
# that shows what eps trades off.  It is a diagnostic curve rather than a result,
# so it runs on a smaller subset: twelve settings at three draws each is
# thirty-seven passes per member against the menu's sixteen.
SWEEP_FAMILIES = ("gauss:0.03:3,gauss:0.10:3,gauss:0.25:3,gauss:0.50:3,"
                  "stain:0.05:3,stain:0.15:3,stain:0.35:3,stain:0.60:3,"
                  "photometric:0.15:3,photometric:0.4:3,photometric:0.8:3,"
                  "photometric:1.2:3")
SWEEP = dict(benchmark="camelyon17", seed="seed_1", which="ood_val",
             limit=5000, tag="radius_sweep_n5000")

SEEDS = ["seed_1", "seed_2", "seed3"]

BENCHMARKS = {
    "camelyon17": dict(
        data=os.path.join(DATA, "camelyon17"),
        pool_prefix="camelyon17",
        classes=2,
        ood_val="ood_val_hospital1",
        ood_test="ood_test_hospital2",
        tune="ood_val_hospital1",
        limit=20000,
        batch=512,
        n_q=500,
        label="Camelyon17",
        task="tumour vs.\\ normal in lymph-node histopathology patches",
        shift="hospital, dominated by staining protocol",
    ),
    "rxrx1": dict(
        data=os.path.join(DATA, "rxrx1"),
        pool_prefix="rxrx1",
        classes=1139,
        ood_val="ood_val",
        ood_test="ood_test",
        tune="ood_val",
        limit=3000,
        batch=256,
        n_q=500,
        label="RxRx1",
        task="siRNA treatment from fluorescence microscopy of cells",
        shift="imaging experiment, an illumination and staining effect",
    ),
    "iwildcam": dict(
        data=os.path.join(DATA, "iwildcam"),
        pool_prefix="iwildcam",
        classes=182,
        ood_val="ood_val",
        ood_test="ood_test",
        tune="ood_val",
        limit=8000,
        batch=256,
        n_q=500,
        label="iWildCam",
        task="species recognition in camera-trap photographs",
        shift="camera location, with background and framing",
    ),
    "kather": dict(
        data=os.path.join(DATA, "kather"),
        pool_prefix="kather",
        classes=9,
        ood_val=None,
        ood_test="ood_test_crc_val_7k",
        tune="source_val",
        limit=7180,
        batch=256,
        n_q=500,
        label="Kather",
        task="colorectal tissue type in histopathology patches",
        shift="patient cohort and institution",
    ),
}


RESULTS = os.path.normpath(os.path.join(_HERE, "..", "results"))


def pool_dir(name, seed):
    """Directory of one pool, e.g. camelyon17 at seed_2."""
    return os.path.join(POOLS, f"{BENCHMARKS[name]['pool_prefix']}_{seed}")


def result_path(name, seed, which, kind):
    """Where one run's JSON lives.

    One convention, used by the driver that writes these and by the script that
    reads them into tables, so neither has to be told where the other put things
    and a missing file is a missing run rather than a typo in a manifest.
    """
    return os.path.join(RESULTS, f"{name}_{seed}_{which}_{kind}.json")


def tag(split, limit):
    """Cache tag for one split at one subset size."""
    return f"{split}_n{limit}"


def tune_tag(name):
    """Cache tag of the split a benchmark's settings are fixed on."""
    b = BENCHMARKS[name]
    return tag(b["tune"], b["limit"])


def cache_jobs(names=None, seeds=None, splits=("ood_val", "ood_test")):
    """Every caching job the experiments need, as plain dicts.

    One job is one pool, one split, the whole family menu.  Benchmarks without
    an out-of-distribution validation split simply contribute fewer jobs.
    """
    out, seen = [], set()
    for name in (names or list(BENCHMARKS)):
        b = BENCHMARKS[name]
        for seed in (seeds or SEEDS):
            pool = pool_dir(name, seed)
            if not os.path.isdir(pool):
                continue
            for which in splits:
                split = b.get(which)
                if not split:
                    continue
                # ``tune`` names the same split as ``ood_val`` on three of the
                # four benchmarks, so one cache serves both roles and asking for
                # both must not queue the work twice.
                key = (pool, split)
                if key in seen:
                    continue
                seen.add(key)
                out.append(dict(
                    benchmark=name, seed=seed, which=which, pool=pool,
                    data=b["data"], split=split, limit=b["limit"],
                    batch=b["batch"], families=FAMILIES,
                    tag=tag(split, b["limit"]),
                ))
    return out


def sweep_job():
    """The one extra caching job the radius figure needs."""
    s = SWEEP
    b = BENCHMARKS[s["benchmark"]]
    return dict(benchmark=s["benchmark"], seed=s["seed"], which=s["which"],
                pool=pool_dir(s["benchmark"], s["seed"]), data=b["data"],
                split=b[s["which"]], limit=s["limit"], batch=b["batch"],
                families=SWEEP_FAMILIES, tag=s["tag"])


def describe():
    """One line per benchmark, for a sanity check before a long run."""
    lines = []
    for name, b in BENCHMARKS.items():
        seeds = [s for s in SEEDS if os.path.isdir(pool_dir(name, s))]
        lines.append(
            f"{b['label']:12} C={b['classes']:>5}  test={b['ood_test']:<22}"
            f" val={str(b['ood_val']):<20} limit={b['limit']:>6}"
            f"  pools={len(seeds)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    jobs = cache_jobs()
    print(f"\n{len(jobs)} caching jobs")
    for j in jobs[:6]:
        print(f"  {j['benchmark']:11} {j['seed']:7} {j['split']}")
    if len(jobs) > 6:
        print(f"  ... and {len(jobs) - 6} more")

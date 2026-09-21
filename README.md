# StaCC

Code for *Stability-Certified Committees: Label-Free Ensemble Selection under
Distribution Shift*.

Given a pool of trained models and an unlabeled, shifted query stream, StaCC
turns each model's sensitivity to input perturbation into a certified bound on
its target Brier risk, minimizes the resulting committee bound by Frank–Wolfe,
and returns a weighted committee of `k` members together with the number
bounding its risk. The perturbation family is chosen without labels, and the
method abstains when no family passes its checks.

## Install

```bash
pip install -r requirements.txt
```

A GPU is needed to train pools and to cache model outputs. Everything after the
caches runs on numpy in seconds to minutes. All commands below run from `code/`.

```bash
cd code
python selftest.py          # every identity of the paper, checked numerically
```

## 1. Data

Four benchmarks, packed once into memory-mapped arrays under `data/`.

```bash
# Camelyon17 (WILDS, official splits recovered from the HuggingFace mirror)
python fetch_camelyon_hf.py --root ../data/camelyon17_hf
python pack_camelyon.py --src ../data/camelyon17_hf --dst ../data/camelyon17

# RxRx1 and iWildCam (WILDS)
python fetch_wilds.py --task rxrx1 --root ../data/wilds
python pack_wilds.py  --task rxrx1 --src ../data/wilds --dst ../data/rxrx1
python fetch_wilds.py --task iwildcam --root ../data/wilds
python pack_wilds.py  --task iwildcam --src ../data/wilds --dst ../data/iwildcam

# Kather: NCT-CRC-HE-100K.zip and CRC-VAL-HE-7K.zip from Zenodo record 1214456
python pack_kather.py --src ../data/kather_raw --dst ../data/kather
```

## 2. Pools

Three pools of 40 members per benchmark, each member drawn independently from a
fixed hyperparameter range and kept if it clears 1.25 times chance on source
validation. No target data enters pool construction.

**Trained Camelyon17 pools (model weights): [LINK TO BE ADDED]**

Unpack them into `pools/` and skip to step 3. Each pool directory holds the
members (`models/`), the exact invocation (`config.json`) and every member's
hyperparameter draw (`manifest.jsonl`). To build any pool from scratch:

```bash
python train_pool.py --benchmark camelyon17 --data ../data/camelyon17 \
    --pool-size 40 --image-size 96 --pool-seed 1 --out ../pools/camelyon17_seed_1
```

Repeat per benchmark (`camelyon17`, `rxrx1`, `iwildcam`, `kather`) with
`--pool-seed 2` and `3` for the other two pools. Directories must be named
`<benchmark>_seed_1`, `<benchmark>_seed_2` and `<benchmark>_seed3`, which is
what `benchmarks.py` expects. Runs are resumable.

## 3. Cache model outputs (GPU, once)

```bash
python cache_queue.py --stage clean
python cache_queue.py --stage stability
python cache_queue.py --stage stability --benchmarks camelyon17 --splits ood_val
python sensitivity.py cache-h1-P10
```

The first caches each member's source-validation outputs, the second each
member's outputs on the official test split, clean and under every family in the
menu. The last two are needed only for the sensitivity table on Camelyon17's
hospital 1. Cached outputs are not released; they are regenerated here.

## 4. Results (no GPU)

| paper | command |
| --- | --- |
| Tables 1, 3, 4, 5, 6 | `python run_all.py` then `python make_tables.py` |
| Table 7 (constants, hospital 1) | `python sensitivity.py h1-c`, and likewise `h1-k`, `h1-T`, `h1-P`, `h1-tol`, `h1-varsigma`, `h1-mode` |
| Table 8 (family menu) | `python sensitivity.py menu` |
| Figure 2 | `python figures.py --out ../figures` (writes `selection.pdf`) |

`run_all.py` chooses the perturbation family on each pool without labels
(`diagnose.py`), then runs the method and every comparator on it
(`run_stacc.py`) at the constants of Table 2, writing per-batch results to
`results/`. `make_tables.py` prints every table body from those files, with the
spread across pools and paired bootstrap intervals. The sensitivity jobs write
their tables to `sensitivity_tables/`.

## Layout

```
code/stacc/certificate.py   stability scores, certified radii, falsification, family choice
code/stacc/committee.py     the objective J, Frank–Wolfe, duality gap
code/stacc/perturb.py       perturbation families
code/stacc/baselines.py     every comparator
code/stacc/pool.py          the randomised pool trainer
code/benchmarks.py          benchmarks, splits, pools and cache tags, in one place
```

## License

MIT, see [LICENSE](LICENSE).

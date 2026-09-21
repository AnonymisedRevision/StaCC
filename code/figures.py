"""Generate the paper's empirical figures from the caches and result JSONs.

Four figures, each answering one question a reviewer will ask and none of them
decorative.

  certificate.pdf   Does the bound actually hold, and is it tight enough to say
                    anything?  Certified risk against realised risk, per member.
  premise.pdf       Does stability carry the signal validation misses?  The same
                    pool ranked by each, against target score.
  radius.pdf        Why the family and the radius matter, and what they trade
                    off against each other.
  selection.pdf     What the selection rule is actually choosing between, step
                    by step, on one batch.
  localisation.pdf  The geometry of the conceptual figure, measured: how far the
                    truth is confined, and how much the committee tightens that
                    over the best single certificate.

Style follows the paper rather than a dashboard.  Marks are thin, grids are
recessive, every series is distinguishable by shape and dash as well as by hue,
and the hues are the Okabe-Ito set, which is colour-vision-deficiency safe and
survives greyscale printing.  Colour never carries identity alone.

    python figures.py --out ../figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

import benchmarks as B
from stacc import calibrate
from stacc import certificate as CERT
from stacc import committee as COM
from stacc.metrics import accuracy, brier
from stacc.registry import Pool

# Okabe-Ito, validated for CVD separation and for contrast against a white
# surface.  Order is fixed and never cycled.
BLUE, VERMILLION, GREEN, ORANGE, PURPLE = (
    "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7")
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d9d9d9"

# The deployed settings of Table 2, so the figure draws the rule as it is
# actually run rather than a smaller illustration of it.
C_DEPLOYED, K, SEED = 0.2, 10, 1


def panels(which="ood_test", seed="seed_1", names=None):
    """One panel per benchmark: pool, cached run, chosen family, display name.

    The family is read from the diagnostics rather than named here, so the
    figures show the family the label-free procedure actually selected and
    change with it.  One pool per benchmark, since a scatter of forty members is
    already the unit the panel draws; the tables are what average over pools.
    """
    out = []
    for name in (names or list(B.BENCHMARKS)):
        b = B.BENCHMARKS[name]
        if not b.get(which):
            continue
        pool = B.pool_dir(name, seed)
        tag = B.tag(b[which], b["limit"])
        if not os.path.exists(os.path.join(pool, "stability", tag, "index.json")):
            continue
        diag_path = B.result_path(name, seed, which, "diag")
        family = None
        if os.path.exists(diag_path):
            d = json.load(open(diag_path, encoding="utf-8"))
            family = d.get("label_free_family")
            if family is None:
                ranking = sorted(d.get("family_ranking", []),
                                 key=lambda r: r["J_star"])
                family = ranking[0]["family"] if ranking else None
        if family is None:
            family = B.FAMILIES.split(",")[0].replace(":", "@").rsplit("@", 1)[0]
        out.append((pool, tag, family, b["label"], b["n_q"]))
    return out


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(pool_dir, tag, family, n_q=500):
    """Per-member quantities for one benchmark, averaged over query batches."""
    pool = Pool(pool_dir)
    clean, pert, idx = pool.load_stability(tag, families=[family])
    P = pert[family]
    y = np.load(os.path.join(pool.stability_dir(tag), "labels.npz"))["labels"].astype(int)
    preds, labels, _ = pool.load_clean(splits=["source_val"])
    val_p, val_y = preds["source_val"], labels["source_val"].astype(int)

    T = calibrate.fit_pool(val_p, val_y)
    val_p = calibrate.apply_pool(val_p, T)
    clean = calibrate.apply_pool(clean, T)
    P = calibrate.apply_pool(P, T)

    M, N, C = clean.shape
    val_brier = np.array([brier(val_p[i], val_y) for i in range(M)])
    val_score = np.array([accuracy(val_p[i], val_y) for i in range(M)])
    target_score = np.array([accuracy(clean[i], y) for i in range(M)])

    rng = np.random.default_rng(SEED)
    order = rng.permutation(N)
    U, E, S, chosen = [], [], [], np.zeros(M, dtype=bool)
    best_member, committee_radius, realised = [], [], []
    com = None
    for b in range(N // n_q):
        sel = order[b * n_q:(b + 1) * n_q]
        batch, pb, yb = clean[:, sel, :], P[:, :, sel, :], y[sel]
        G = COM.gram(batch)
        s = CERT.stability_score(batch, pb)
        rho = CERT.certified_radii(val_brier, s, c=C_DEPLOYED)
        U.append(rho ** 2)
        E.append(COM.realised_risks(G, batch, yb))
        S.append(s)
        if b % 10 == 0:
            dead = CERT.degenerate_members(batch)
            com = COM.frank_wolfe(G, rho ** 2, K, exclude=dead)
            chosen[com.members] = True
        # the three radii of figure 1, in signature units: the tightest single
        # certificate, the committee's certified radius, and the distance the
        # committee prediction actually sits from the truth
        best_member.append(float(rho.min()))
        committee_radius.append(float(np.sqrt(max(com.J, 0.0))))
        realised.append(float(np.sqrt(max(
            COM.weighted_risk(G, batch, yb, com.weights), 0.0))))
    return dict(U=np.mean(U, axis=0), E=np.mean(E, axis=0), s=np.mean(S, axis=0),
                val_score=val_score, target=target_score, chosen=chosen,
                best_member=np.array(best_member),
                committee_radius=np.array(committee_radius),
                realised=np.array(realised))


# --------------------------------------------------------------------------- #
# figure 1: does the bound hold?
# --------------------------------------------------------------------------- #
def fig_certificate(data, out):
    # Each panel keeps its own scale, since the benchmarks differ by a factor of
    # two in risk, so the panels need room for their own tick labels.
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.05), sharey=False)
    fig.subplots_adjust(wspace=0.42)
    for ax, (d, name) in zip(axes, data):
        lo = 0.0
        hi = max(d["U"].max(), d["E"].max()) * 1.06
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls="--", zorder=1)
        ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=BLUE, alpha=0.06,
                        lw=0, zorder=0)
        free = ~d["chosen"]
        ax.scatter(d["E"][free], d["U"][free], s=13, facecolors="none",
                   edgecolors=BLUE, linewidths=0.8, zorder=3)
        ax.scatter(d["E"][d["chosen"]], d["U"][d["chosen"]], s=26,
                   marker="D", facecolors=VERMILLION, edgecolors="white",
                   linewidths=0.6, zorder=4)
        cov = float(np.mean(d["U"] >= d["E"]))
        ax.set_title(name)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.locator_params(axis="both", nbins=4)
        ax.text(0.95, 0.06, f"covered {cov:.0%}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                          alpha=0.85))
        ax.set_xlabel(r"realised risk $E(\sigma_i)$")
    axes[0].set_ylabel(r"certified risk $U_i$")
    handles = [
        Line2D([], [], ls="none", marker="o", mfc="none", mec=BLUE, mew=0.8,
               ms=4.5, label="pool member"),
        Line2D([], [], ls="none", marker="D", color=VERMILLION, ms=4.5,
               label="selected into a committee"),
        Line2D([], [], ls="--", color=INK, lw=0.8, label="certificate is tight"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    fig.savefig(os.path.join(out, "certificate.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 2: does stability carry what validation misses?
# --------------------------------------------------------------------------- #
def fig_premise(data, out):
    fig, axes = plt.subplots(2, 4, figsize=(6.6, 3.5), sharey="row")
    for j, (d, name) in enumerate(data):
        for i, (x, xlabel, colour) in enumerate([
                (d["val_score"], "source-validation score", MUTED),
                (-d["s"], r"negated stability $-s_i$", BLUE)]):
            ax = axes[i, j]
            ax.scatter(x, d["target"], s=12, facecolors="none",
                       edgecolors=colour, linewidths=0.8)
            r = float(np.corrcoef(x, d["target"])[0, 1])
            good = r >= 0.3
            ax.text(0.05, 0.06, f"$r={r:+.2f}$", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=7.5,
                    color=GREEN if good else VERMILLION,
                    fontweight="bold" if good else "normal",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2,
                              alpha=0.85))
            if np.ptp(x) > 0:
                b, a = np.polyfit(x, d["target"], 1)
                xs = np.linspace(x.min(), x.max(), 2)
                ax.plot(xs, a + b * xs, color=colour, lw=0.9, ls="-", alpha=0.7)
            ax.set_xlabel(xlabel, fontsize=7)
            if i == 0:
                ax.set_title(name)
    axes[0, 0].set_ylabel("target score")
    axes[1, 0].set_ylabel("target score")
    axes[0, 0].annotate("ranked by\nvalidation", xy=(-0.62, 0.5),
                        xycoords="axes fraction", ha="center", va="center",
                        rotation=90, fontsize=7.5, color=MUTED)
    axes[1, 0].annotate("ranked by\nstability", xy=(-0.62, 0.5),
                        xycoords="axes fraction", ha="center", va="center",
                        rotation=90, fontsize=7.5, color=BLUE)
    fig.savefig(os.path.join(out, "premise.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 3: the radius trades validity against information
# --------------------------------------------------------------------------- #
def fig_radius(sweep_path, out):
    rows = json.load(open(sweep_path, encoding="utf-8"))["rows"]
    fams = {}
    for r in rows:
        base = r["family"].split("@")[0]
        fams.setdefault(base, []).append(r)
    order = ["stain", "photometric", "gauss"]
    style_of = {"stain": (BLUE, "o", "-"), "photometric": (GREEN, "s", "--"),
                "gauss": (VERMILLION, "^", ":")}
    label_of = {"stain": "stain jitter", "photometric": "photometric",
                "gauss": "Gaussian noise"}

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    for base in order:
        rs = sorted(fams[base], key=lambda r: r["applied"])
        x = [r["applied"] for r in rs]
        col, mk, ls = style_of[base]
        axes[0].plot(x, [r["pearson_stability"] for r in rs], color=col,
                     marker=mk, ls=ls, lw=1.4, ms=4.5, label=label_of[base])
        axes[1].plot(x, [r["coverage"] for r in rs], color=col, marker=mk,
                     ls=ls, lw=1.4, ms=4.5)
    axes[0].axhline(0.0, color=INK, lw=0.8, ls="-", alpha=0.6)
    axes[0].set_ylabel("correlation of $-s_i$\nwith target score")
    axes[0].set_ylim(-1.12, 1.05)
    axes[0].text(0.985, 0.96, "carries signal", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=7, color=GREEN)
    axes[0].text(0.985, 0.04, "actively misleading", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=7, color=VERMILLION)
    axes[1].axhline(1.0, color=INK, lw=0.8, ls="--", alpha=0.6)
    axes[1].set_ylabel("member coverage")
    axes[1].set_xlabel(r"applied pixel displacement (radius $\varepsilon$)")
    axes[1].set_ylim(-0.05, 1.12)
    axes[1].text(0.02, 0.955, "certificate holds", transform=axes[1].transAxes,
                 ha="left", va="top", fontsize=7, color=MUTED)
    axes[0].legend(frameon=False, loc="center left", handlelength=2.4,
                   bbox_to_anchor=(0.02, 0.32))
    fig.savefig(os.path.join(out, "radius.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 4: the construction on one batch
# --------------------------------------------------------------------------- #
def fig_selection(pool_dir, tag, family, out, batch_index=0, n_q=500,
                  max_panels=3):
    """The Frank-Wolfe step, drawn in the two quantities it trades off.

    Theorem 17(i) says the member admitted at each step minimises certified risk
    minus squared distance from what the committee already predicts.  Those are
    two axes, the rule is a family of parallel lines across them, and the member
    it picks is the one the sweeping line reaches first.  Drawing it makes the
    phrase "jointly most certified and most novel" a picture rather than a
    slogan, and shows that neither axis alone would have chosen the same member.
    """
    pool = Pool(pool_dir)
    clean, pert, _ = pool.load_stability(tag, families=[family])
    P = pert[family]
    y = np.load(os.path.join(pool.stability_dir(tag), "labels.npz"))["labels"].astype(int)
    preds, labels, _ = pool.load_clean(splits=["source_val"])
    val_p, val_y = preds["source_val"], labels["source_val"].astype(int)
    T = calibrate.fit_pool(val_p, val_y)
    val_p, clean, P = (calibrate.apply_pool(val_p, T), calibrate.apply_pool(clean, T),
                       calibrate.apply_pool(P, T))
    M, N, C = clean.shape
    val_brier = np.array([brier(val_p[i], val_y) for i in range(M)])

    rng = np.random.default_rng(SEED)
    sel = rng.permutation(N)[batch_index * n_q:(batch_index + 1) * n_q]
    batch, pb, yb = clean[:, sel, :], P[:, :, sel, :], y[sel]
    G = COM.gram(batch)
    rho2 = CERT.certified_radii(
        val_brier, CERT.stability_score(batch, pb), c=C_DEPLOYED) ** 2
    scores = np.array([accuracy(batch[i], yb) for i in range(M)])
    oracle = int(np.argmax(scores))
    g = np.diag(G)

    # replay the greedy steps so each panel is one admission
    w = np.zeros(M)
    w[int(np.argmin(rho2))] = 1.0
    order = [int(np.argmin(rho2))]
    panels = []
    for t in range(K - 1):
        Gw = G @ w
        wGw = float(w @ Gw)
        dist2 = np.maximum(g - 2.0 * Gw + wGw, 0.0)
        cand = rho2 - dist2
        masked = cand.copy()
        masked[np.asarray(order, dtype=int)] = np.inf
        pick = int(np.argmin(masked))
        panels.append((rho2.copy(), dist2.copy(), list(order), pick))
        grad = (rho2 - g) + 2.0 * Gw
        d = -w.copy()
        d[pick] += 1.0
        quad = float(d @ G @ d)
        gamma = 1.0 if quad <= 1e-300 else float(np.clip(-(grad @ d) / (2 * quad), 0, 1))
        w = w + gamma * d
        order.append(pick)

    # A committee of ten has nine admissions and nine panels would be
    # unreadable at this width, so the first few are drawn and the rest are
    # summarised in the caption.  The rule is identical at every step.
    shown = panels[:max_panels]
    fig, axes = plt.subplots(1, len(shown), figsize=(6.6, 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    ytop = max(max(Y.max() for _, Y, _, _ in shown) * 1.25, 0.3)
    ybot = -0.06 * ytop
    for ax, (X, Y, have, pick) in zip(axes, shown):
        lo, hi = X.min() - 0.05, X.max() + 0.05
        # iso-lines of the objective: certified risk minus novelty.  The winner
        # is the point the sweeping line reaches first, so everything else sits
        # below the dashed line through it.
        for off in np.arange(np.floor(lo - ytop), np.ceil(hi - ybot), 0.15):
            ax.plot([lo, hi], [lo - off, hi - off], color=GRID, lw=0.5, zorder=0)
        ax.set_ylim(ybot, ytop)
        best = X[pick] - Y[pick]
        ax.plot([lo, hi], [lo - best, hi - best], color=VERMILLION, lw=1.2,
                ls="--", zorder=2)
        rest = np.setdiff1d(np.arange(M), np.array(have + [pick]))
        ax.scatter(X[rest], Y[rest], s=14, facecolors="none", edgecolors=MUTED,
                   linewidths=0.7, zorder=3)
        ax.scatter(X[have], Y[have], s=30, marker="D", facecolors="white",
                   edgecolors=VERMILLION, linewidths=1.0, zorder=4)
        ax.scatter(X[pick], Y[pick], s=44, marker="D", facecolors=VERMILLION,
                   edgecolors="white", linewidths=0.7, zorder=6)
        if oracle not in have + [pick]:
            ax.scatter(X[oracle], Y[oracle], s=95, marker="*",
                       facecolors="none", edgecolors=ORANGE, linewidths=1.1,
                       zorder=5)
        else:
            ax.scatter(X[oracle], Y[oracle], s=95, marker="*",
                       facecolors=ORANGE, edgecolors=INK, linewidths=0.5,
                       zorder=7)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"certified risk $\rho_i^2$")
        ax.set_title(f"step {len(have)}: admit member {pick}", fontsize=8)
    axes[0].set_ylabel(r"novelty $\|\sigma_i-\sigma_{w}\|^2$")
    handles = [
        Line2D([], [], ls="none", marker="o", mfc="none", mec=MUTED, ms=4.5,
               label="available member"),
        Line2D([], [], ls="none", marker="D", mfc="white", mec=VERMILLION,
               mew=1.0, ms=4.5, label="already in the committee"),
        Line2D([], [], ls="none", marker="D", color=VERMILLION, ms=4.5,
               label="admitted at this step"),
        Line2D([], [], ls="none", marker="*", mfc="none", mec=ORANGE, mew=1.1,
               ms=9, label="oracle member"),
        Line2D([], [], ls="--", color=VERMILLION, lw=1.2,
               label=r"level set through the winner"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.20))
    fig.savefig(os.path.join(out, "selection.pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# figure 5: the localisation of figure 1, measured
# --------------------------------------------------------------------------- #
def fig_localisation(data, out):
    """Figure 1's geometry, in numbers rather than in a drawing.

    The conceptual figure cannot be drawn from the data, because the certified
    radii are several times the distances between members, so every ball would
    cover every other and the picture would be one blob.  What can be drawn is
    the quantity that picture is about, namely how far the truth is confined,
    and how much the committee tightens the confinement over the best single
    certificate.  Everything here is in signature units, the square root of a
    Brier risk, so it is a distance in the space Figure 1 draws.
    """
    short = {"Camelyon17 h1": "Camelyon h1", "Camelyon17 h2": "Camelyon h2"}
    names = [short.get(n, n) for _, n in data]
    x = np.arange(len(names))
    single = np.array([d["best_member"].mean() for d, _ in data])
    committee = np.array([d["committee_radius"].mean() for d, _ in data])
    actual = np.array([d["realised"].mean() for d, _ in data])

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ax.vlines(x, actual, single, color=GRID, lw=6, zorder=0)
    ax.plot(x, single, ls="none", marker="o", ms=7, mfc="white", mec=BLUE,
            mew=1.5, zorder=3, label=r"tightest single certificate $\min_i\rho_i$")
    ax.plot(x, committee, ls="none", marker="D", ms=7, color=VERMILLION,
            zorder=4, label=r"committee's certified radius $\sqrt{J(w)}$")
    ax.plot(x, actual, ls="none", marker="X", ms=8, color=INK, zorder=5,
            label=r"where the truth actually is $\|\sigma_w-\mathbf{y}\|$")
    for xi, a, c in zip(x, actual, committee):
        ax.annotate("", xy=(xi + 0.16, a), xytext=(xi + 0.16, c),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8,
                                    shrinkA=1, shrinkB=1))
        ax.text(xi + 0.21, 0.5 * (a + c), f"slack {c - a:+.2f}", fontsize=6.5,
                color=MUTED, va="center")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_xlim(-0.45, len(names) - 0.35)
    ax.set_ylabel("distance in signature space")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=1, handletextpad=0.4)
    fig.savefig(os.path.join(out, "localisation.pdf"))
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="../figures")
    p.add_argument("--which", default="ood_test", choices=["ood_test", "ood_val"])
    p.add_argument("--seed", default="seed_1", help="which pool the panels draw")
    p.add_argument("--sweep", default="../results/radius_sweep.json",
                   help="a diagnose.py JSON over the dense radius grid, which "
                        "is what the radius figure plots")
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    style()

    cols = panels(a.which, a.seed)
    if not cols:
        print("no caches found; run cache_queue.py first")
        return 1

    print("loading per-member quantities")
    data = []
    for pool_dir, tag, family, name, n_q in cols:
        print(f"  {name}  ({family}, batches of {n_q})")
        data.append((load(pool_dir, tag, family, n_q), name))

    print("certificate.pdf");  fig_certificate(data, a.out)
    print("premise.pdf");      fig_premise(data, a.out)
    if os.path.exists(a.sweep):
        print("radius.pdf");   fig_radius(a.sweep, a.out)
    else:
        print(f"  skipped radius.pdf, {a.sweep} not found")
    pool_dir, tag, family, _, n_q = cols[0]
    print("selection.pdf");   fig_selection(pool_dir, tag, family, a.out, n_q=n_q)
    print("localisation.pdf"); fig_localisation(data, a.out)
    print(f"\nwrote figures to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

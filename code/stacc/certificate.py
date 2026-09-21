"""Stability scores, certified radii, and the label-free falsification test.

The certificate of the paper is

    sqrt(R_T(h))  <=  sqrt(R_S(h))  +  S_T(h, eps)  +  2 sqrt(2 beta(eps)),

so on a query batch each member gets a certified radius

    rho_i = sqrt(Rhat_S(h_i)) + s_i(Q, eps) + c

with s_i the displacement of the member's signature under perturbation of the
batch and c a constant common to the pool.  Since ||sigma_i - y|| = sqrt(E(sigma_i)),
the certificate says exactly that the unknown label signature lies in the ball
of radius rho_i around member i's prediction, so the pool confines it to an
intersection of M balls.  When two of those balls fail to meet the intersection
is empty and the transport assumption is refuted, which is the one diagnostic in
this framework that needs neither labels nor a committee.

Two estimators of s_i are provided and they answer different questions.  The
worst-case estimator takes a maximum over draws and approaches the supremum in
the definition of the sensitivity from below, which is what Theorem 2 asks for.
The average-case estimator takes a mean and is unbiased for the matched-kernel
quantity of Theorem 19, which is the right object when the perturbation family is
believed to generate the shift.  The gap between them is itself a diagnostic,
since it widens when the family is mismatched.
"""
from __future__ import annotations

import numpy as np

from .committee import min_quadratic_simplex, objective, squared_distances

# --------------------------------------------------------------------------- #
# stability scores
# --------------------------------------------------------------------------- #


def per_sample_sensitivity(clean, perturbed, mode="worst"):
    """g_i(x) for every member and query point.

    ``clean`` is (M, n, C) and ``perturbed`` is (M, P, n, C), the outputs under
    P draws from the perturbation family.  Returns (M, n), each entry in [0, 2].

    ``worst`` takes the maximum over draws, the estimator of Theorem 2, which
    underestimates the supremum and therefore yields a radius that is optimistic
    by however much the search missed.  ``mean`` takes the average over draws,
    which is unbiased for the matched-kernel sensitivity of Theorem 19.
    """
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    if perturbed.ndim != 4:
        raise ValueError(f"expected (M, P, n, C) perturbed outputs, got {perturbed.shape}")
    if clean.shape[0] != perturbed.shape[0] or clean.shape[1:] != perturbed.shape[2:]:
        raise ValueError(
            f"shape mismatch: clean {clean.shape} against perturbed {perturbed.shape}"
        )
    d2 = ((perturbed - clean[:, None, :, :]) ** 2).sum(axis=3)  # (M, P, n)
    if mode == "worst":
        return d2.max(axis=1)
    if mode == "mean":
        return d2.mean(axis=1)
    raise ValueError(f"unknown mode {mode!r}; expected 'worst' or 'mean'")


def stability_score(clean, perturbed, mode="worst"):
    """s_i = sqrt( mean_x g_i(x) ), the signature displacement of member i.

    This is the batch-normalised norm of the difference between the member's
    signature on the perturbed batch and on the actual one, which is the
    quantity the informal principle of the paper refers to.
    """
    return np.sqrt(per_sample_sensitivity(clean, perturbed, mode).mean(axis=1))


def flip_rate(clean, perturbed):
    """The eps-flip rate of Remark 24, the decision-level stability score.

    Fraction of query points at which some draw changes the predicted class.
    Label-free, invariant to temperature, and the natural score when the
    reported metric is a zero-one quantity rather than the Brier risk.
    """
    clean = np.asarray(clean)
    perturbed = np.asarray(perturbed)
    base = clean.argmax(axis=2)  # (M, n)
    moved = perturbed.argmax(axis=3) != base[:, None, :]  # (M, P, n)
    return moved.any(axis=1).mean(axis=1)


# --------------------------------------------------------------------------- #
# certified radii
# --------------------------------------------------------------------------- #


def concentration_slack(n, n_source, M, eta=0.1):
    """The computable part of the constant c of Corollary 4.

    Returns the sum of the three Hoeffding terms.  The remaining piece of c is
    2 sqrt(2 beta(eps)), the transport defect, which is not computable from data
    and is left to the caller as the conservativeness dial of Remark 10.
    """
    log_term = np.log(6.0 * max(M, 1) / max(eta, 1e-12))
    a_n = 2.0 * np.sqrt(log_term / (2.0 * max(n, 1)))
    a_s = 2.0 * np.sqrt(log_term / (2.0 * max(n_source, 1)))
    return float(np.sqrt(a_s) + 2.0 * np.sqrt(a_n))


def certified_radii(source_risk, stability, c=0.0):
    """rho_i = sqrt(Rhat_S(h_i)) + s_i + c, the radius of Corollary 4.

    ``source_risk`` is the empirical source-validation Brier risk of each
    member, or a single float to use the validation floor of Remark 8, under
    which validation enters only as a filter and the ranking is by stability
    alone.
    """
    s = np.asarray(stability, dtype=np.float64)
    r = np.asarray(source_risk, dtype=np.float64)
    if r.ndim == 0:
        r = np.full_like(s, float(r))
    if r.shape != s.shape:
        raise ValueError(f"source risk shape {r.shape} does not match stability {s.shape}")
    return np.sqrt(np.maximum(r, 0.0)) + s + float(c)


def predictive_dispersion(batch):
    """How spread a member's predicted classes are over the query batch.

    Returns, per member, the fraction of the batch assigned to its most frequent
    predicted class, so 1.0 means the member predicts one class everywhere.

    This exists because Proposition 7 has a hole that only shows up in practice.
    The proposition rules out a predictor that is constant, and its repair is the
    source-risk term, which works for a member that is degenerate on the source
    because such a member cannot pass a validation floor.  A member can instead
    be perfectly reasonable on the source and collapse to one class on the
    shifted target.  It is then constant exactly where the stability score is
    measured, so its score is as small as a score can be, and any rule that
    ranks by stability alone puts it first.  Its source risk is what excludes it,
    which is why the per-member variant survives this and the floor variant,
    which discards that information, does not.

    The check is label-free and costs one argmax over the batch.
    """
    pred = np.asarray(batch).argmax(axis=2)  # (M, n)
    M, n = pred.shape
    out = np.empty(M)
    for i in range(M):
        counts = np.bincount(pred[i])
        out[i] = counts.max() / max(n, 1)
    return out


def degenerate_members(batch, max_share=0.99):
    """Members that predict a single class on essentially the whole batch.

    The threshold is deliberately lenient.  A target batch genuinely dominated by one
    class can push a good model close to one class, and the intent is to exclude only
    an outright collapse, not to second-guess a confident model.
    """
    return np.flatnonzero(predictive_dispersion(batch) >= max_share)


def constant_predictor_floor(class_prior):
    """The source risk below which no constant predictor can rank, Prop. 8(ii).

    Every constant predictor has source Brier risk at least 1 - ||prior||^2, so
    a member whose certified base radius is below the square root of this bound
    is certified above every degenerate predictor.  This is the structural
    repair of the false direction of the stability hypothesis.
    """
    pi = np.asarray(class_prior, dtype=np.float64)
    pi = pi / max(pi.sum(), 1e-300)
    return float(1.0 - (pi ** 2).sum())


# --------------------------------------------------------------------------- #
# falsification
# --------------------------------------------------------------------------- #


def falsified_pairs(G, rho):
    """Pairs whose certified balls cannot meet, Proposition 6.

    ``G`` is the signature Gram matrix.  A pair with
    ||sigma_i - sigma_j|| > rho_i + rho_j makes the localisation set empty, so
    either the transport assumption fails at the chosen radius or the
    concentration event did not occur.  Costs O(M^2) and reads no label.
    """
    rho = np.asarray(rho, dtype=np.float64)
    D2 = squared_distances(np.asarray(G, dtype=np.float64))
    budget = (rho[:, None] + rho[None, :]) ** 2
    bad = np.triu(D2 > budget, k=1)
    return np.argwhere(bad)


def stability_share(source_risk, stability, c=0.0):
    """What fraction of the certified radius the stability term contributes.

    The radius is rho_i = sqrt(Rhat_S(h_i)) + s_i + c, so when s_i is small next
    to the source term the ordering of rho is essentially the ordering of source
    validation risk.  That matters because source validation is exactly the
    signal the method exists to replace: if the certificate reduces to it, the
    method inherits its failure rather than repairing it, and it does so while
    looking like it is doing something else.

    The share is label-free and it separates the cases sharply in practice.  It
    sits near 0.85 where the method works and near 0.4 where it does not, so it
    is the natural precondition to check before deploying at all.
    """
    s = np.asarray(stability, dtype=np.float64)
    rho = certified_radii(source_risk, s, c=c)
    return float(s.mean() / max(rho.mean(), 1e-12))


def select_family(G, radii_by_family, shares_by_family=None, min_share=0.6):
    """Choose a perturbation family without labels, or abstain.

    Each family f yields its own certified radii and hence its own localisation
    set K_f, and on the intersection of the corresponding events the truth lies
    in every one of them at once.  Any single family's bound is therefore valid,
    the tightest is valid too, and

        f* = argmin_f  min_w J_f(w)

    picks a family from quantities the method already computes.  The union bound
    over families enters only the shared constant, not the ranking.

    Two guards keep the criterion from choosing a family that merely fails to
    perturb.  A family whose certificates are inconsistent, meaning the
    falsification test fires, is discarded.  And a family whose stability term
    contributes less than ``min_share`` of the certified radius is discarded
    too, because its bound is tight only by restating the source risk.  If no
    family survives, the answer is to abstain rather than to pick the least bad
    one, and the caller should fall back to averaging the whole pool.

    Returns the chosen key, or None to abstain, together with the full ranking.
    """
    G = np.asarray(G, dtype=np.float64)
    rows = []
    for key, rho in radii_by_family.items():
        rho = np.asarray(rho, dtype=np.float64)
        falsified = len(falsified_pairs(G, rho))
        loc = localisation_bound(G, rho)
        share = None if shares_by_family is None else float(shares_by_family[key])
        rows.append({
            "family": key,
            "J_star": float(loc["J_star"]),
            "diam_K_upper": float(loc["diam_K_upper"]),
            "falsified_pairs": int(falsified),
            "stability_share": share,
            "admissible": (falsified == 0 and loc["J_star"] >= 0.0
                           and (share is None or share >= min_share)),
        })
    admissible = [r for r in rows if r["admissible"]]
    if not admissible:
        return None, rows
    return sorted(admissible, key=lambda r: r["J_star"])[0]["family"], rows




def localisation_bound(G, rho, w=None):
    """An upper bound on diam(K), the quantity Proposition 20 shows is decisive.

    Every weighting gives K inside a ball of squared radius J(w), so
    diam(K) <= 2 sqrt(min_w J(w)); the single best member gives
    diam(K) <= 2 min_i rho_i.  The smaller of the two is returned, together with
    both, since Proposition 20 makes this the measure of how much any label-free
    rule could possibly win before one is run.

    The minimax floor of Proposition 20 is diam(K)^2 / 4, which is not computable
    exactly because only an upper bound on the diameter is available; what is
    returned is therefore an upper bound on that floor, and it never exceeds
    J_star, consistently with equation (28).
    """
    G = np.asarray(G, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    rho2 = rho ** 2
    if w is None:
        c = rho2 - np.diag(G)
        w = min_quadratic_simplex(G, c)
    J_star = objective(G, rho2, w)
    from_committee = 2.0 * np.sqrt(max(J_star, 0.0))
    from_member = 2.0 * float(rho.min())
    diam_upper = float(min(from_committee, from_member))
    return {
        "diam_K_upper": diam_upper,
        "from_committee": float(from_committee),
        "from_member": float(from_member),
        "J_star": float(J_star),
        "minimax_floor_upper": float(diam_upper ** 2 / 4.0),
    }

"""The certified committee objective and the Frank-Wolfe selection rule.

Everything in this module operates on prediction signatures.  A pool of M
members evaluated on a query batch of n samples over C classes gives an
(M, n, C) array; flattened it is an (M, D) matrix with D = nC, and the paper's
inner product is the batch-normalised one,

    <a, b> = (1/n) a^T b,

so that every quantity below is a per-sample average and the Brier risk of a
signature is exactly ||sigma - y||^2.

The whole method is second-order in the signatures, so it factors through the
Gram matrix G_ij = <sigma_i, sigma_j>.  That matters in practice: G is M x M and
is formed once per reselection, after which the objective, its gradient, the
Frank-Wolfe steps, the duality gap, the pool diameter and the falsification test
are all O(M^2) and independent of the batch size.

Objective (Theorem 9 of the paper).  With rho_i the certified radius of member
i, the certified committee risk is

    J(w) = sum_i w_i rho_i^2  -  A(w),        A(w) = sum_i w_i ||sigma_i - sigma_w||^2,

which expands to the convex quadratic  J(w) = c^T w + w^T G w  with
c = rho^2 - diag(G).  Its meaning is geometric rather than merely algebraic:
the unknown label signature is provably inside the ball of squared radius J(w)
centred at the committee prediction sigma_w, so J(w) bounds the committee's
risk and minimising it tightens the guarantee.

No labels enter any function here.  Labels appear only in the separate
diagnostic helpers at the end, and only to score what the label-free rule has
already chosen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# signature geometry
# --------------------------------------------------------------------------- #


def flatten(probs):
    """(M, n, C) probabilities -> (M, D) signatures, D = nC."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 3:
        raise ValueError(f"expected (M, n, C) probabilities, got shape {probs.shape}")
    return probs.reshape(probs.shape[0], -1)


def gram(probs):
    """Gram matrix of the signatures under the batch-normalised inner product.

    Returns G with G_ij = (1/n) sum_x <h_i(x), h_j(x)>.  Symmetry is imposed
    explicitly because the product of a large matrix with its transpose is only
    symmetric to floating-point tolerance, and the eigen-solvers downstream are
    happier with an exactly symmetric input.
    """
    F = flatten(probs)
    n = probs.shape[1]
    G = (F @ F.T) / float(n)
    return 0.5 * (G + G.T)


def squared_distances(G):
    """Pairwise ||sigma_i - sigma_j||^2 from the Gram matrix."""
    g = np.diag(G)
    D2 = g[:, None] + g[None, :] - 2.0 * G
    np.fill_diagonal(D2, 0.0)
    return np.maximum(D2, 0.0)


def pool_diameter(G):
    """diam = max_ij ||sigma_i - sigma_j||, the curvature scale of Theorem 17."""
    return float(np.sqrt(squared_distances(G).max()))


def ambiguity(G, w):
    """A(w) = sum_i w_i ||sigma_i - sigma_w||^2, label-free."""
    w = np.asarray(w, dtype=np.float64)
    return float(w @ np.diag(G) - w @ G @ w)


# --------------------------------------------------------------------------- #
# the objective
# --------------------------------------------------------------------------- #


def linear_term(G, rho2):
    """The vector c with J(w) = c^T w + w^T G w."""
    return np.asarray(rho2, dtype=np.float64) - np.diag(G)


def objective(G, rho2, w):
    """J(w), the certified committee risk."""
    w = np.asarray(w, dtype=np.float64)
    return float(linear_term(G, rho2) @ w + w @ G @ w)


def gradient(G, rho2, w):
    """grad J(w) = c + 2 G w."""
    w = np.asarray(w, dtype=np.float64)
    return linear_term(G, rho2) + 2.0 * (G @ w)


def duality_gap(G, rho2, w):
    """Frank-Wolfe gap, an upper bound on J(w) - min J that costs O(M^2).

    This is the second certificate of Theorem 17, eq. (22).  It is computable without
    labels, so the method reports both a bound on its own risk and a bound on
    how far its committee is from the best weighting of the entire pool.
    """
    g = gradient(G, rho2, w)
    w = np.asarray(w, dtype=np.float64)
    return float(g @ w - g.min())


# --------------------------------------------------------------------------- #
# the solver
# --------------------------------------------------------------------------- #
@dataclass
class Committee:
    """The result of a selection, with everything the diagnostics need."""

    weights: np.ndarray  # (M,) sparse, on the pool simplex
    members: np.ndarray  # indices of the support, in the order selected
    J: float  # certified committee risk, the guarantee
    gap: float  # Frank-Wolfe duality gap at the returned iterate
    ambiguity: float  # A(w), the diversity term actually achieved
    certified_mean: float  # sum_i w_i rho_i^2, the stability term
    diam: float
    steps: int
    meta: dict = field(default_factory=dict)

    @property
    def size(self):
        return int(len(self.members))


def _step_size(rule, t, G, grad, w, e_i):
    """Frank-Wolfe step size for the three rules of Theorem 17."""
    d = e_i - w
    if rule == "uniform":
        # gamma_0 = 1, gamma_t = 1/(t+1): the iterate stays the uniform average
        # of the selected vertices, which is Theorem 17(iv).
        return 1.0 if t == 0 else 1.0 / (t + 1.0)
    if rule == "fw":
        # gamma_t = 2/(t+2), the rule Theorem 17(iii) states the rate for.
        return 2.0 / (t + 2.0)
    if rule == "line":
        # Exact line search.  J is quadratic, so J(w + gamma d) = J(w) +
        # gamma <grad, d> + gamma^2 d^T G d and the minimiser is closed form.
        # This never does worse than the fixed step at any iteration, so the
        # rate of Theorem 17(iii) still applies.
        quad = float(d @ G @ d)
        if quad <= 1e-300:
            return 0.0
        return float(np.clip(-(grad @ d) / (2.0 * quad), 0.0, 1.0))
    raise ValueError(f"unknown step rule {rule!r}; expected fw, line or uniform")


def frank_wolfe(G, rho2, k, rule="line", injective=True, w0=None, tol=0.0,
                exclude=None):
    """Select a committee of at most k members by Frank-Wolfe on J.

    The linear minimisation step is

        i_t = argmin_i { rho_i^2 - ||sigma_i - sigma_{w_t}||^2 },

    which is Theorem 17(i): the member that enters is the one jointly most
    certified and most novel relative to what the committee already predicts.
    Because every step moves toward a single vertex of the simplex, the iterate
    is k-sparse by construction and no cardinality constraint is imposed.

    ``injective`` forbids re-selecting a member already in the support.  A
    repeat means the certificate would rather double a member's weight than
    admit a new one, which is legitimate for the objective but yields a
    committee of fewer than k distinct models; the injective variant is the one
    a practitioner deploys, and it is reported alongside the guaranteed one.

    ``exclude`` removes members from consideration entirely.  It is a mask
    rather than an infinite radius on purpose: an infinite entry multiplied by
    a zero weight is not a large number but a NaN, which then propagates
    silently through the objective and the line search and returns a committee
    that looks well formed and is not.
    """
    G = np.asarray(G, dtype=np.float64)
    rho2 = np.asarray(rho2, dtype=np.float64)
    M = G.shape[0]
    if rho2.shape != (M,):
        raise ValueError(f"rho2 has shape {rho2.shape}, expected ({M},)")
    if not 1 <= k <= M:
        raise ValueError(f"committee size k={k} must lie in [1, {M}]")
    if not np.isfinite(rho2).all():
        raise ValueError("rho2 contains non-finite entries; use `exclude` to "
                         "drop members rather than giving them infinite radii")

    banned = np.zeros(M, dtype=bool)
    if exclude is not None:
        banned[np.asarray(exclude, dtype=int)] = True
    if banned.all():
        raise ValueError("every member was excluded")

    g_diag = np.diag(G)
    c = rho2 - g_diag

    if w0 is None:
        # Start at the vertex the rule itself prefers, which is the member of
        # lowest certified risk: at w = e_i the objective is exactly rho_i^2.
        allowed = np.flatnonzero(~banned)
        start = int(allowed[np.argmin(rho2[allowed])])
        w = np.zeros(M)
        w[start] = 1.0
        order = [start]
    else:
        w = np.asarray(w0, dtype=np.float64).copy()
        order = list(np.flatnonzero(w > 0))

    steps = 0
    for t in range(k - 1):
        grad = c + 2.0 * (G @ w)
        cand = grad.copy()
        cand[banned] = np.inf
        if injective:
            cand[np.asarray(order, dtype=int)] = np.inf
        if not np.isfinite(cand).any():
            break
        i = int(np.argmin(cand))
        gap = float(grad @ w - grad[i])
        if gap <= tol:
            break
        e_i = np.zeros(M)
        e_i[i] = 1.0
        gamma = _step_size(rule, t, G, grad, w, e_i)
        if gamma <= 0.0:
            break
        w = (1.0 - gamma) * w + gamma * e_i
        if i not in order:
            order.append(i)
        steps += 1

    w = np.maximum(w, 0.0)
    w /= max(w.sum(), 1e-300)
    support = np.array([i for i in order if w[i] > 1e-12], dtype=int)
    if len(support) == 0:  # numerically degenerate, fall back to the argmax
        support = np.array([int(np.argmax(w))], dtype=int)

    return Committee(
        weights=w,
        members=support,
        J=objective(G, rho2, w),
        gap=duality_gap(G, rho2, w),
        ambiguity=ambiguity(G, w),
        certified_mean=float(rho2 @ w),
        diam=pool_diameter(G),
        steps=steps,
        meta={"rule": rule, "injective": bool(injective)},
    )


def min_quadratic_simplex(G, c, iters=2000, tol=1e-12):
    """Minimise c^T w + w^T G w over the probability simplex.

    Frank-Wolfe with exact line search, run to convergence rather than stopped
    at k steps.  Used for the dense reference points, namely the minimum of J
    over the whole pool and, with the labelled linear term, the oracle stacking
    weights of Theorem 17(vi).
    """
    G = np.asarray(G, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    M = G.shape[0]
    w = np.full(M, 1.0 / M)
    for _ in range(iters):
        grad = c + 2.0 * (G @ w)
        i = int(np.argmin(grad))
        d = -w.copy()
        d[i] += 1.0
        gap = float(-(grad @ d))
        if gap <= tol:
            break
        quad = float(d @ G @ d)
        gamma = 1.0 if quad <= 1e-300 else float(np.clip(-(grad @ d) / (2.0 * quad), 0.0, 1.0))
        if gamma <= 0.0:
            break
        w = w + gamma * d
    w = np.maximum(w, 0.0)
    return w / max(w.sum(), 1e-300)


# --------------------------------------------------------------------------- #
# the two degenerate limits, kept here because they are ablations of J itself
# --------------------------------------------------------------------------- #


def topk_certified(rho2, k):
    """Stability-only limit: the k members of lowest certified risk.

    This is what the method degenerates to when the ambiguity term is deleted,
    and Proposition 13 gives a pool on which it is worse than the optimum by
    the entire certified radius.
    """
    return np.argsort(np.asarray(rho2, dtype=np.float64))[:k]


def max_ambiguity(G, k):
    """Diversity-only limit: greedily maximise A over uniform committees.

    This is what the method degenerates to when the certified-risk term is
    deleted, or equivalently when every certificate is equal.  Greedy max-sum
    diversification, the standard tractable stand-in for the NP-hard problem.
    """
    D2 = squared_distances(G)
    M = D2.shape[0]
    i, j = np.unravel_index(int(np.argmax(D2)), D2.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < min(k, M):
        rest = [m for m in range(M) if m not in chosen]
        if not rest:
            break
        chosen.append(int(rest[int(np.argmax(D2[np.ix_(rest, chosen)].sum(axis=1)))]))
    return np.array(chosen[:k], dtype=int)


# --------------------------------------------------------------------------- #
# label-using diagnostics, never consulted by the selection
# --------------------------------------------------------------------------- #


def label_inner_products(probs, labels):
    """b_i = <sigma_i, y>, the only place labels touch the Gram algebra."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n = probs.shape[1]
    return probs[:, np.arange(n), labels].mean(axis=1)


def realised_risks(G, probs, labels):
    """E(sigma_i) = ||sigma_i - y||^2 for every member, from the Gram matrix.

    Equal to the multiclass Brier score of each member on the batch; computing
    it this way keeps the identity ||y||^2 = 1 explicit, which is what makes
    the slack rho_i^2 - E(sigma_i) exactly comparable to J.
    """
    b = label_inner_products(probs, labels)
    return np.diag(G) - 2.0 * b + 1.0


def weighted_risk(G, probs, labels, w):
    """E(sigma_w) for a weighting, again through the Gram matrix."""
    w = np.asarray(w, dtype=np.float64)
    b = label_inner_products(probs, labels)
    return float(w @ G @ w - 2.0 * (b @ w) + 1.0)


def oracle_stacking(G, probs, labels, iters=2000):
    """The Brier-optimal convex combination of the pool, using target labels.

    Unreachable by any label-free rule; it is the tightest of the three
    comparators in Theorem 17(vi) and marks the ceiling of what weighting alone
    can deliver on a batch.
    """
    b = label_inner_products(probs, labels)
    return min_quadratic_simplex(G, -2.0 * b, iters=iters)

"""Baselines, all consuming the same cached predictions as the method.

Three groups, and the grouping matters more than the individual entries.

Reference points fix the range of outcomes: the full pool, the single model a
practitioner would deploy, and the unreachable oracle.

Label-free rules see what StaCC sees on the target, namely member outputs on
an unlabelled query batch.  These are the honest comparisons.

Source-supervised rules rank or combine members by labelled source-domain
validation data alone.  They are the ones to beat wherever validation transfers,
and the ones expected to fail where it does not.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _entropy(probs, eps=1e-12):
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def uniform(probs, idx=None):
    """Average the given members' probabilities."""
    A = probs if idx is None else probs[np.asarray(idx)]
    return A.mean(axis=0)


def weighted(probs, w, idx=None):
    A = probs if idx is None else probs[np.asarray(idx)]
    w = np.asarray(w, dtype=np.float64)
    w = w / max(w.sum(), 1e-300)
    return np.tensordot(w, A, axes=(0, 0))


# --------------------------------------------------------------------------- #
# label-free selection
# --------------------------------------------------------------------------- #
def lowest_entropy(batch, k):
    """The k members whose own predictions on this batch are most confident."""
    return np.argsort([_entropy(batch[i]).mean() for i in range(len(batch))])[:k]


def random_committee(M, k, rng):
    return rng.choice(M, size=k, replace=False)


def k_medoids(d2, k, rng, iters=50):
    """k-medoids on signature distances, the deterministic diverse-subset rule.

    Takes the matrix of squared distances between members rather than the
    signatures themselves.  Forming ``sig[:, None, :] - sig[None, :, :]`` here
    allocated an (M, M, nC) array, which is twelve megabytes on a two-class
    benchmark and seven gigabytes on a thousand-class one, per query batch.  The
    Gram matrix the caller already holds gives the same distances for nothing.
    """
    d2 = np.asarray(d2, dtype=np.float64)
    M = len(d2)
    dist = np.sqrt(np.maximum(d2, 0.0))
    med = rng.choice(M, size=k, replace=False)
    for _ in range(iters):
        assign = np.argmin(dist[:, med], axis=1)
        new = med.copy()
        for j in range(k):
            members = np.flatnonzero(assign == j)
            if len(members):
                new[j] = members[np.argmin(dist[np.ix_(members, members)].sum(axis=1))]
        if np.array_equal(np.sort(new), np.sort(med)):
            break
        med = new
    return med


def dpp_greedy(d2, k, gamma=1.0):
    """Greedy MAP for a determinantal point process on a signature RBF kernel.

    The canonical probabilistic tool for diverse subset selection, and the main
    competitor to a geometric design.  Greedy MAP is the standard tractable
    stand-in for sampling from the DPP itself.

    Takes squared distances between members, for the reason given in
    :func:`k_medoids`.  The kernel is built on distances rescaled by the largest
    signature norm, which the caller supplies through ``d2`` already in those
    units, so the scaling here divides by the largest pairwise distance instead;
    both put the kernel on a comparable scale and neither reads a label.
    """
    d2 = np.asarray(d2, dtype=np.float64)
    scale = max(float(d2.max()), 1e-12)
    L = np.exp(-gamma * d2 / scale)
    M = len(L)
    chosen, avail = [], list(range(M))
    C = np.zeros((0, M))
    d = np.diag(L).astype(np.float64).copy()
    for _ in range(k):
        j = max(avail, key=lambda i: d[i])
        chosen.append(j)
        avail.remove(j)
        if not avail:
            break
        e = (L[j] - (C[:, j] @ C if len(C) else 0)) / max(np.sqrt(d[j]), 1e-12)
        C = np.vstack([C, e])
        d = np.maximum(d - e ** 2, 1e-12)
    return np.array(chosen)


# --------------------------------------------------------------------------- #
# label-free weighting over the whole pool
# --------------------------------------------------------------------------- #
def spectral_meta_learner(batch, eps=1e-9):
    """Member reliabilities from the covariance of their predictions.

    For a binary task, map each member's output to a +/-1 vote; under a
    conditional-independence model the leading eigenvector of the off-diagonal
    covariance is proportional to the members' reliabilities, which is the
    spectral meta-learner of Parisi et al.  Weights are the positive part of
    that eigenvector.
    """
    M = len(batch)
    votes = np.where(batch.argmax(axis=2) == 1, 1.0, -1.0) if batch.shape[2] == 2 \
        else (batch.max(axis=2) * 2 - 1)
    V = votes - votes.mean(axis=1, keepdims=True)
    Q = (V @ V.T) / max(V.shape[1], 1)
    np.fill_diagonal(Q, 0.0)                       # off-diagonal only
    vals, vecs = np.linalg.eigh(Q)
    lead = vecs[:, int(np.argmax(vals))]
    if lead.sum() < 0:
        lead = -lead
    w = np.clip(lead, 0.0, None)
    return w / max(w.sum(), eps) if w.sum() > eps else np.full(M, 1.0 / M)


def atc_fit(val_probs, val_labels):
    """The source-side quantities ATC and DoC need, computed once.

    Nothing here depends on the query batch, so it is kept out of the part that
    does. Recomputing it inside the stream loop meant rescanning the whole
    validation split once per member per batch, which on a long stream of small
    batches dominated everything else the loop did.
    """
    M = len(val_probs)
    thresholds, accs, mean_conf = np.empty(M), np.empty(M), np.empty(M)
    for i in range(M):
        conf = val_probs[i].max(axis=1)
        acc = float((val_probs[i].argmax(axis=1) == val_labels).mean())
        accs[i] = acc
        mean_conf[i] = float(conf.mean())
        # threshold t with mean(conf > t) == acc
        thresholds[i] = np.quantile(conf, 1.0 - np.clip(acc, 0.0, 1.0))
    return {"thresholds": thresholds, "source_acc": accs, "mean_conf": mean_conf}


def atc_weights(fit, batch, eps=1e-9):
    """Average thresholded confidence: label-free target-accuracy estimates.

    For each member a threshold is fitted on source validation so that the
    fraction of confident predictions matches its validation accuracy; the same
    threshold applied to the target batch estimates target accuracy without
    labels. Those estimates become the weights. ``fit`` comes from
    :func:`atc_fit`.
    """
    batch = np.asarray(batch)
    M = len(batch)
    est = (batch.max(axis=2) > fit["thresholds"][:, None]).mean(axis=1)
    w = np.clip(est, 0.0, None)
    return w / max(w.sum(), eps) if w.sum() > eps else np.full(M, 1.0 / M)


def confidence_ensemble(batch, quantile=0.5):
    """Weight members by their confidence on this batch, above the median."""
    conf = batch.max(axis=2).mean(axis=1)
    thresh = np.quantile(conf, quantile)
    w = np.where(conf >= thresh, conf, 0.0)
    return w / max(w.sum(), 1e-300)


def difference_of_confidences(fit, batch):
    """DoC: source accuracy minus the drop in mean confidence under the shift.

    \\citet{guillory2021predicting} estimate target accuracy as the source
    accuracy less the difference between mean source confidence and mean target
    confidence, the intuition being that a shift a model notices shows up in its
    confidence before it shows up in anything else observable.  One estimate per
    member, so it is a ranking rule of the same kind as our certified radius and
    it enters the comparison as such.

    ``fit`` comes from :func:`atc_fit`, which already holds both source-side
    quantities this needs. Returns the per-member estimate, higher meaning
    better, so a committee is the top k of ``argsort(-doc)``.
    """
    drop = fit["mean_conf"] - np.asarray(batch).max(axis=2).mean(axis=1)
    return fit["source_acc"] - drop


def augmentation_consistency(batch, perturbed):
    """Agreement between a member's clean and perturbed decisions on the batch.

    The estimator of \\citet{deng2021does} in the form this cache supports: rank
    members by the fraction of query points whose predicted class is unchanged
    under the same perturbation the certificate uses.  Their construction uses a
    self-supervised transformation task, but the mechanism it exploits is the
    one measured here, namely that a model whose decisions survive a
    semantics-preserving change of the input is a model that is right more often.

    It is deliberately the closest comparator in the table to our own criterion,
    and the difference is exactly the paper's point.  This is a decision-level
    agreement score with no bound attached, whereas \\eqref{eq:sensitivity} is a
    displacement in probability space that enters a certified radius alongside a
    source term.  Comparing the two isolates what the certificate adds over the
    consistency signal it is built on.

    ``batch`` is (M, n, C) and ``perturbed`` is (M, P, n, C) for one family,
    which is the layout the caches use.  Returns the per-member agreement,
    higher meaning better.
    """
    clean = np.asarray(batch).argmax(axis=2)                # (M, n)
    pert = np.asarray(perturbed).argmax(axis=3)             # (M, P, n)
    return (pert == clean[:, None, :]).mean(axis=(1, 2))


# --------------------------------------------------------------------------- #
# source-supervised selection
# --------------------------------------------------------------------------- #
def best_by_source(val_scores, k=1):
    """The k members ranked highest on labelled source validation."""
    return np.argsort(-np.asarray(val_scores))[:k]


def greedy_selection(val_probs, val_labels, k, score_fn, with_replacement=True):
    """Caruana-style forward selection, maximising a source-validation score."""
    M = len(val_probs)
    chosen = []
    for _ in range(k):
        gains = []
        for i in range(M):
            if not with_replacement and i in chosen:
                gains.append(-np.inf)
                continue
            cand = chosen + [i]
            gains.append(score_fn(val_probs[cand].mean(axis=0), val_labels))
        chosen.append(int(np.argmax(gains)))
    # report distinct members, padded if selection repeated one
    seen = list(dict.fromkeys(chosen))
    while len(seen) < k:
        for i in np.argsort([-score_fn(val_probs[i], val_labels) for i in range(M)]):
            if i not in seen:
                seen.append(int(i))
                break
    return np.array(seen[:k])


# --------------------------------------------------------------------------- #
# additions for the certified-committee study
# --------------------------------------------------------------------------- #
def agreement_weights(batch, eps=1e-9):
    """Agreement-on-the-line: weight a member by how often the pool agrees with it.

    Agreement between pairs of models on unlabelled target data tracks their
    accuracy closely enough to predict it, so the mean agreement of a member
    with the rest of the pool is a label-free reliability estimate.  It is a
    point estimate, not a bound, which is exactly the gap the certificate of
    this paper is meant to close.
    """
    preds = np.asarray(batch).argmax(axis=2)
    M = len(preds)
    agree = np.zeros(M)
    for i in range(M):
        others = [j for j in range(M) if j != i]
        agree[i] = float(np.mean([np.mean(preds[i] == preds[j]) for j in others]))
    w = np.clip(agree - agree.min(), 0.0, None)
    return w / max(w.sum(), eps) if w.sum() > eps else np.full(M, 1.0 / M)


def l2_model_averaging(val_probs, val_labels, lam=1.0, iters=500):
    """Model averaging with an L2 penalty toward uniform weights.

    Weights minimise the source-validation Brier risk of the aggregate plus
    lam * ||w - uniform||^2, then are applied unchanged to the target batch.
    Source-supervised, and the natural weighting counterpart to the selection
    baselines that rank on validation.
    """
    V = np.asarray(val_probs)
    y = np.asarray(val_labels, dtype=int)
    M, n, C = V.shape

    # Accumulated over row blocks rather than by flattening the whole validation
    # split.  A float64 copy of (M, n, C) is three and a half gigabytes at
    # RxRx1's class count, and the one-hot matrix would be another; both are
    # avoidable, since G and b are sums over rows.
    G = np.zeros((M, M))
    b = np.zeros(M)
    step_rows = max(1, int(4e7 // max(C, 1)))
    for s in range(0, n, step_rows):
        block = np.asarray(V[:, s:s + step_rows, :], dtype=np.float64)
        F = block.reshape(M, -1)
        G += F @ F.T
        # <sigma_i, y> over this block is the sum of each member's probability
        # on the true class, so the one-hot matrix never has to be built.
        rows = np.arange(block.shape[1])
        b += block[:, rows, y[s:s + step_rows]].sum(axis=1)
    G /= n
    b /= n
    u = np.full(M, 1.0 / M)
    w = u.copy()
    step = 1.0 / max(2.0 * (np.linalg.eigvalsh(G).max() + lam), 1e-6)
    for _ in range(iters):
        grad = 2.0 * (G @ w - b) + 2.0 * lam * (w - u)
        w = _project_simplex(w - step * grad)
    return w


def soft_neighbourhood_density(batch, temperature=0.05, max_points=512, rng=None):
    """SND, an unsupervised domain-adaptation validation criterion.

    Entropy of the row-wise softmax of the similarity matrix between a member's
    output vectors on the target batch.  A member whose target predictions form
    dense, well-separated neighbourhoods scores high, and the criterion needs no
    labels.  Computed on the prediction matrix rather than on features, so it
    applies to a heterogeneous pool whose members share no representation.
    """
    A = np.asarray(batch, dtype=np.float64)
    M, n, _ = A.shape
    # the similarity matrix is n x n per member, so a large query batch is
    # subsampled rather than paid for in full
    if n > max_points:
        rng = np.random.default_rng(0) if rng is None else rng
        A = A[:, rng.choice(n, size=max_points, replace=False), :]
    out = np.empty(M)
    for i in range(M):
        X = A[i] / np.linalg.norm(A[i], axis=1, keepdims=True).clip(1e-12)
        S = X @ X.T
        np.fill_diagonal(S, -np.inf)
        S = S / temperature
        S -= S.max(axis=1, keepdims=True)
        P = np.exp(S)
        P /= P.sum(axis=1, keepdims=True).clip(1e-300)
        out[i] = float(-(P * np.log(np.clip(P, 1e-12, 1.0))).sum(axis=1).mean())
    return out


def _project_simplex(w):
    """Euclidean projection onto the probability simplex."""
    w = np.asarray(w, dtype=np.float64)
    u = np.sort(w)[::-1]
    css = np.cumsum(u)
    idx = np.arange(1, len(w) + 1)
    cond = u - (css - 1) / idx > 0
    r = idx[cond][-1]
    theta = (css[cond][-1] - 1) / r
    return np.maximum(w - theta, 0.0)

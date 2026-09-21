"""Temperature scaling, fitted and applied entirely on cached probabilities.

Output-space stability favours members whose softmax saturates, because a
saturated softmax is locally flat.  That is the same confound that turns
confidence-based selection into a preference for overconfidence, and it would
let a badly calibrated member buy a small stability score without being robust.
Temperature scaling removes the route: it leaves every decision unchanged,
lowers the source Brier risk, and makes the sensitivity reflect the decision
boundary rather than the output scale.

Nothing here needs a GPU or the original logits.  Softmax is invariant to a
constant shift of its argument, so taking the logarithm of a cached probability
vector recovers the logits up to exactly such a shift, and re-scaling by a
temperature is therefore exact rather than approximate.

The same temperature must be applied to a member's perturbed outputs as to its
clean ones, otherwise the displacement being measured is partly a change of
scale rather than a change of prediction.
"""
from __future__ import annotations

import numpy as np


def _logits_from_probs(probs, eps=1e-12):
    """Recover logits up to a per-sample constant, which softmax discards."""
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0))


def apply_temperature(probs, T, eps=1e-12):
    """Rescale cached probabilities by a temperature, exactly."""
    T = float(T)
    if T <= 0:
        raise ValueError(f"temperature must be positive, got {T}")
    z = _logits_from_probs(probs, eps) / T
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _nll(probs, labels, eps=1e-12):
    n = probs.shape[0]
    return float(-np.log(np.clip(probs[np.arange(n), labels], eps, 1.0)).mean())


def fit_temperature(val_probs, val_labels, grid=None, refine=True):
    """Fit one temperature per member by minimising validation NLL.

    A coarse logarithmic grid followed by a golden-section refinement, which is
    both faster and more robust than gradient descent on a single scalar, and
    avoids a torch dependency in what is otherwise a numpy-only pipeline.
    """
    val_probs = np.asarray(val_probs, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=int)
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 33))

    def loss(T):
        return _nll(apply_temperature(val_probs, T), labels)

    losses = [loss(T) for T in grid]
    best = int(np.argmin(losses))
    if not refine:
        return float(grid[best])

    lo = float(grid[max(best - 1, 0)])
    hi = float(grid[min(best + 1, len(grid) - 1)])
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(40):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = loss(d)
        if abs(b - a) < 1e-4:
            break
    return float(0.5 * (a + b))


def fit_pool(val_probs, val_labels):
    """Temperatures for a whole pool, ``val_probs`` of shape (M, n, C)."""
    val_probs = np.asarray(val_probs, dtype=np.float64)
    return np.array(
        [fit_temperature(val_probs[i], val_labels) for i in range(val_probs.shape[0])]
    )


def apply_pool(probs, temperatures, eps=1e-12, dtype=np.float32):
    """Apply per-member temperatures to (M, ..., C) outputs.

    Handles both the clean (M, n, C) and the perturbed (M, P, n, C) layouts, so
    a member's draws are always rescaled with the member's own temperature.

    Written as a power rather than as log, divide, subtract, exponentiate and
    normalise, because

        softmax(log(p) / T)_c = p_c^(1/T) / sum_j p_j^(1/T)

    exactly.  The two forms agree to machine precision and the second needs one
    pass and one temporary instead of six.  That is invisible on a two-class
    benchmark and decisive on a thousand-class one: the perturbed array of a
    single RxRx1 pool holds 410 million numbers, so the long form allocated
    roughly twenty gigabytes of intermediates to do what this does in one and a
    half.

    ``dtype`` is single precision for the same reason.  The cached outputs this
    reads are stored at half precision, so working in float32 loses nothing that
    was ever there, and the quantities the method compares differ by three
    orders of magnitude more than the rounding.
    """
    probs = np.asarray(probs)
    T = np.asarray(temperatures, dtype=np.float64)
    if T.shape[0] != probs.shape[0]:
        raise ValueError(f"{T.shape[0]} temperatures for {probs.shape[0]} members")
    shape = (T.shape[0],) + (1,) * (probs.ndim - 1)
    inv = (1.0 / T).reshape(shape).astype(dtype)

    out = np.clip(probs.astype(dtype, copy=True), np.asarray(eps, dtype), 1.0)
    np.power(out, inv, out=out)
    out /= np.maximum(out.sum(axis=-1, keepdims=True), np.asarray(eps, dtype))
    return out

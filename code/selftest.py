"""Numerical verification of the paper's identities and constructions.

Every claim in the theory that is an identity should hold to machine precision,
and every claim that is an inequality should never be violated on random inputs.
Verifying that in code is cheap and catches the two kinds of error that matter,
namely a wrong formula and a right formula implemented wrongly.  The two
separation constructions are reproduced from their stated numbers, so that if a
proof is later edited the code disagrees loudly.

    python selftest.py
"""
from __future__ import annotations

import sys

import numpy as np

from stacc import baselines as B, calibrate, certificate as C, committee as K
from stacc.metrics import brier

FAILURES = []


def check(name, ok, detail=""):
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def random_pool(M=12, n=40, C_=3, seed=0):
    """A pool of random probability signatures and a random label signature."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(M, n, C_)) * 2.0
    e = np.exp(logits - logits.max(axis=2, keepdims=True))
    probs = e / e.sum(axis=2, keepdims=True)
    labels = rng.integers(0, C_, size=n)
    return probs, labels


# --------------------------------------------------------------------------- #
def test_gram_and_risk():
    print("\nsignature algebra")
    probs, labels = random_pool()
    G = K.gram(probs)

    risks_gram = K.realised_risks(G, probs, labels)
    risks_direct = np.array([brier(probs[i], labels) for i in range(len(probs))])
    check(
        "Brier risk from the Gram matrix equals the direct computation",
        np.allclose(risks_gram, risks_direct, atol=1e-12),
        f"max err {np.abs(risks_gram - risks_direct).max():.2e}",
    )

    w = np.random.default_rng(1).dirichlet(np.ones(len(probs)))
    sig_w = np.tensordot(w, probs, axes=(0, 0))
    check(
        "weighted risk from the Gram matrix equals the direct computation",
        abs(K.weighted_risk(G, probs, labels, w) - brier(sig_w, labels)) < 1e-12,
    )

    D2 = K.squared_distances(G)
    F = probs.reshape(len(probs), -1)
    n = probs.shape[1]
    direct = ((F[:, None, :] - F[None, :, :]) ** 2).sum(axis=2) / n
    check(
        "pairwise squared distances match the flattened computation",
        np.allclose(D2, direct, atol=1e-12),
    )


def test_ambiguity_decomposition():
    print("\nambiguity decomposition, Theorem 9(i)")
    probs, labels = random_pool(seed=2)
    G = K.gram(probs)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        w = rng.dirichlet(np.ones(len(probs)) * rng.uniform(0.2, 3.0))
        lhs = K.weighted_risk(G, probs, labels, w)
        rhs = w @ K.realised_risks(G, probs, labels) - K.ambiguity(G, w)
        worst = max(worst, abs(lhs - rhs))
    check(
        "E(sigma_w) = sum_i w_i E(sigma_i) - A(w) exactly",
        worst < 1e-12,
        f"max err over 200 weightings {worst:.2e}",
    )


def test_certified_objective():
    print("\ncertified committee risk, Theorem 9(ii) and (iii)")
    probs, labels = random_pool(seed=4)
    G = K.gram(probs)
    risks = K.realised_risks(G, probs, labels)
    # valid certificates: radius at least the true distance to the truth
    rng = np.random.default_rng(5)
    rho = np.sqrt(risks) * (1.0 + rng.uniform(0.0, 0.4, size=len(risks)))
    rho2 = rho ** 2

    worst_slack, worst_neg, worst_ball = 0.0, 0.0, 0.0
    for _ in range(200):
        w = rng.dirichlet(np.ones(len(probs)))
        J = K.objective(G, rho2, w)
        E = K.weighted_risk(G, probs, labels, w)
        worst_slack = max(worst_slack, abs(J - E - w @ (rho2 - risks)))
        worst_neg = min(worst_neg, J)
        worst_ball = max(worst_ball, E - J)
    check("J(w) = E(sigma_w) + sum_i w_i slack_i", worst_slack < 1e-12,
          f"max err {worst_slack:.2e}")
    check("J(w) >= 0 whenever the certificates hold", worst_neg >= -1e-12,
          f"min J {worst_neg:.2e}")
    check("E(sigma_w) <= J(w), the committee bound", worst_ball <= 1e-12,
          f"max violation {worst_ball:.2e}")

    # localisation: the truth is inside the ball of squared radius J(w)
    y = np.zeros_like(probs[0])
    y[np.arange(probs.shape[1]), labels] = 1.0
    w = rng.dirichlet(np.ones(len(probs)))
    sig_w = np.tensordot(w, probs, axes=(0, 0))
    dist2 = float(((sig_w - y) ** 2).sum() / probs.shape[1])
    check(
        "the truth lies within sqrt(J(w)) of the committee prediction",
        dist2 <= K.objective(G, rho2, w) + 1e-12,
    )


def test_gradient_and_step():
    print("\ngradient and the Frank-Wolfe step, Theorem 17(i)")
    probs, _ = random_pool(seed=6)
    G = K.gram(probs)
    rng = np.random.default_rng(7)
    rho2 = rng.uniform(0.1, 1.0, size=len(probs))
    w = rng.dirichlet(np.ones(len(probs)))

    g = K.gradient(G, rho2, w)
    num = np.empty_like(g)
    h = 1e-7
    for i in range(len(g)):
        e = np.zeros_like(w)
        e[i] = h
        num[i] = (K.objective(G, rho2, w + e) - K.objective(G, rho2, w - e)) / (2 * h)
    check("analytic gradient matches finite differences", np.allclose(g, num, atol=1e-5),
          f"max err {np.abs(g - num).max():.2e}")

    # the selection rule is exactly argmin of the gradient
    sig_w = np.tensordot(w, probs, axes=(0, 0))
    n = probs.shape[1]
    score = np.array(
        [rho2[i] - ((probs[i] - sig_w) ** 2).sum() / n for i in range(len(probs))]
    )
    check(
        "argmin_i {rho_i^2 - ||sigma_i - sigma_w||^2} equals argmin_i dJ/dw_i",
        int(np.argmin(score)) == int(np.argmin(g)),
    )


def test_frank_wolfe_rate():
    print("\nFrank-Wolfe sparsity, rate and duality gap, Theorem 17")
    probs, _ = random_pool(M=30, n=60, seed=8)
    G = K.gram(probs)
    rng = np.random.default_rng(9)
    rho2 = rng.uniform(0.05, 0.6, size=len(probs))
    diam2 = K.pool_diameter(G) ** 2
    J_star = K.objective(G, rho2, K.min_quadratic_simplex(G, rho2 - np.diag(G)))

    ok_rate, ok_sparse, ok_gap = True, True, True
    for k in (2, 3, 5, 8):
        for rule in ("fw", "line", "uniform"):
            c = K.frank_wolfe(G, rho2, k, rule=rule, injective=False)
            ok_sparse &= c.size <= k
            ok_gap &= c.J - J_star <= c.gap + 1e-9
            if rule in ("fw", "line"):
                ok_rate &= c.J - J_star <= 4.0 * diam2 / (k + 2) + 1e-9
            else:
                ok_rate &= c.J - J_star <= diam2 * (1.0 + np.log(k)) / k + 1e-9
    check("committee is k-sparse", ok_sparse)
    check("certified suboptimality respects the stated rate", ok_rate)
    check("duality gap upper-bounds the true suboptimality", ok_gap)

    c = K.frank_wolfe(G, rho2, 3, rule="line", injective=True)
    check("injective variant returns distinct members",
          len(set(c.members.tolist())) == c.size)
    check("J at a vertex equals that member's certified risk",
          abs(K.objective(G, rho2, np.eye(len(probs))[3]) - rho2[3]) < 1e-12)


def test_separation_constructions():
    print("\nthe two separation constructions, Propositions 15 and 20")

    # Proposition 12: maximising ambiguity alone, C = 2, n = 1, y = e_1
    a = np.array([0.0, 0.5, 0.9, 0.9, 0.9])
    probs = np.stack([np.array([[x, 1.0 - x]]) for x in a])
    labels = np.array([0])
    G = K.gram(probs)
    pair = K.max_ambiguity(G, 2)
    got = K.weighted_risk(G, probs, labels, np.eye(5)[pair].mean(axis=0))
    full = K.weighted_risk(G, probs, labels, np.full(5, 0.2))
    best = K.weighted_risk(G, probs, labels, np.eye(5)[[2, 3]].mean(axis=0))
    check("max-ambiguity pair is {0, 0.9}", set(pair.tolist()) == {0, 2 if 2 in pair else 4}
          or set(a[pair]) == {0.0, 0.9})
    check("its risk is 0.605", abs(got - 0.605) < 1e-9, f"got {got:.6f}")
    check("full-pool risk is 0.2592", abs(full - 0.2592) < 1e-9, f"got {full:.6f}")
    check("best pair risk is 0.02", abs(best - 0.02) < 1e-9, f"got {best:.6f}")
    check("diversity alone is worse than the full pool here", got > full)

    # Proposition 13: top-k by certificate against the optimum
    rho, tau = 0.4, 0.05
    D = 6
    y = np.zeros(D)
    u, v = np.zeros(D), np.zeros(D)
    u[0], v[1] = 1.0, 1.0
    S = np.stack([rho * u, rho * u, rho * (1 + tau) * v, -rho * (1 + tau) * v])
    G4 = S @ S.T
    rho4 = np.array([rho, rho, rho * (1 + tau), rho * (1 + tau)])
    top2 = K.topk_certified(rho4 ** 2, 2)
    w_top = np.eye(4)[top2].mean(axis=0)
    w_alt = np.eye(4)[[2, 3]].mean(axis=0)
    check("top-2 by certificate picks the duplicated pair", set(top2.tolist()) == {0, 1})
    check("its certified risk is rho^2",
          abs(K.objective(G4, rho4 ** 2, w_top) - rho ** 2) < 1e-12)
    check("the spread pair certifies zero",
          abs(K.objective(G4, rho4 ** 2, w_alt)) < 1e-12)


def test_falsification_and_localisation():
    print("\nfalsification and localisation, Propositions 6 and 20")
    probs, labels = random_pool(seed=10)
    G = K.gram(probs)
    risks = K.realised_risks(G, probs, labels)

    valid = np.sqrt(risks) * 1.05
    check("valid certificates are not falsified",
          len(C.falsified_pairs(G, valid)) == 0)
    tiny = np.sqrt(risks) * 0.05
    check("certificates that are far too small are falsified",
          len(C.falsified_pairs(G, tiny)) > 0)

    loc = C.localisation_bound(G, valid)
    check("localisation bound from the committee never exceeds the single-member one"
          " when the committee helps",
          loc["diam_K_upper"] <= loc["from_member"] + 1e-12)
    check("the reported minimax floor never exceeds J*",
          loc["minimax_floor_upper"] <= loc["J_star"] + 1e-12)


def test_stability_and_calibration():
    print("\nstability scores and temperature scaling")
    rng = np.random.default_rng(11)
    M, P, n, Cc = 5, 4, 30, 3
    clean, _ = random_pool(M=M, n=n, C_=Cc, seed=12)
    pert = np.clip(clean[:, None] + rng.normal(scale=0.02, size=(M, P, n, Cc)), 1e-6, 1)
    pert /= pert.sum(axis=3, keepdims=True)

    s_worst = C.stability_score(clean, pert, "worst")
    s_mean = C.stability_score(clean, pert, "mean")
    check("worst-case stability is at least the average-case", np.all(s_worst >= s_mean - 1e-12))
    check("stability of an unperturbed pool is zero",
          abs(C.stability_score(clean, clean[:, None].repeat(P, 1)).max()) < 1e-12)

    # a member constant on the batch has zero stability whatever the family
    const = np.tile(np.array([0.7, 0.2, 0.1]), (1, n, 1))
    check("a constant predictor has stability exactly zero",
          abs(C.stability_score(const, const[:, None].repeat(P, 1))[0]) < 1e-15)
    prior = np.array([0.5, 0.3, 0.2])
    check("constant-predictor risk floor is 1 - ||prior||^2",
          abs(C.constant_predictor_floor(prior) - (1 - (prior ** 2).sum())) < 1e-15)

    probs, labels = random_pool(M=3, n=200, C_=3, seed=13)
    check("temperature 1 is the identity",
          np.allclose(calibrate.apply_temperature(probs[0], 1.0), probs[0], atol=1e-12))
    T = calibrate.fit_pool(probs, labels)
    scaled = calibrate.apply_pool(probs, T)
    check("temperature scaling leaves every decision unchanged",
          np.array_equal(scaled.argmax(axis=2), probs.argmax(axis=2)))
    check("fitted temperature does not increase validation NLL",
          all(
              -np.log(calibrate.apply_temperature(probs[i], T[i])[np.arange(200), labels]).mean()
              <= -np.log(probs[i][np.arange(200), labels]).mean() + 1e-9
              for i in range(3)
          ))


def test_family_choice_and_guards():
    print("\nchoosing a family without labels, and the two guards")
    probs, labels = random_pool(M=10, n=50, C_=3, seed=20)
    G = K.gram(probs)
    risks = K.realised_risks(G, probs, labels)
    val = risks * 0.6  # a plausible source risk, smaller than the target risk

    rng = np.random.default_rng(21)
    # an informative family: stability spread across members, large enough that
    # the certificates hold; and an uninformative one that barely perturbs
    s_good = np.sqrt(np.maximum(risks - val, 0)) * 1.2 + 0.02
    s_tiny = np.full(len(probs), 1e-4)
    radii = {"good": C.certified_radii(val, s_good),
             "tiny": C.certified_radii(val, s_tiny)}
    shares = {k_: C.stability_share(val, s) for k_, s in
              (("good", s_good), ("tiny", s_tiny))}

    check("an uninformative family has a small stability share",
          shares["tiny"] < 0.1 < shares["good"],
          f"tiny {shares['tiny']:.3f} vs good {shares['good']:.3f}")

    # threshold set between the two shares, so the guard has to discriminate
    # rather than reject both and pass the check by accident
    thresh = 0.5 * (shares["tiny"] + shares["good"])
    chosen, ranking = C.select_family(G, radii, shares, min_share=thresh)
    check("the share guard keeps the informative family and drops the other",
          chosen == "good", f"chose {chosen!r} at threshold {thresh:.3f}")
    chosen_all, _ = C.select_family(G, radii, shares, min_share=0.0)
    check("without the guard the uninformative family wins on J* alone",
          chosen_all == "tiny",
          "which is exactly why the guard exists")
    none_chosen, _ = C.select_family(G, {"tiny": radii["tiny"]},
                                     {"tiny": shares["tiny"]}, min_share=thresh)
    check("with no admissible family the procedure abstains", none_chosen is None)

    # the degeneracy guard: a member predicting one class everywhere
    batch = probs.copy()
    batch[3] = 0.0
    batch[3, :, 1] = 1.0
    dead = C.degenerate_members(batch, max_share=0.99)
    check("a member constant on the batch is flagged degenerate",
          3 in dead.tolist() and len(dead) == 1, f"flagged {dead.tolist()}")
    # and it is exactly the member a stability-only rule would rank first,
    # which is Proposition 7(iv): perturbing it cannot move a constant output
    pert = np.clip(batch[:, None].repeat(3, 1)
                   + rng.normal(scale=0.05, size=(len(batch), 3) + batch.shape[1:]),
                   1e-6, 1)
    pert /= pert.sum(axis=3, keepdims=True)
    pert[3] = batch[3][None]  # the collapsed member does not move
    s_all = C.stability_score(batch, pert)
    check("the collapsed member has the smallest raw stability score",
          int(np.argmin(s_all)) == 3,
          f"argmin is {int(np.argmin(s_all))}, s={s_all[3]:.2e}")

    # excluding it must not produce NaN, which an infinite radius would
    rho2 = C.certified_radii(val, s_good) ** 2
    Gb = K.gram(batch)
    c_ok = K.frank_wolfe(Gb, rho2, 3, exclude=dead)
    check("excluding a member keeps the objective finite",
          np.isfinite(c_ok.J) and np.isfinite(c_ok.weights).all())
    check("an excluded member receives no weight",
          c_ok.weights[3] == 0.0)
    try:
        bad = rho2.copy()
        bad[3] = np.inf
        K.frank_wolfe(Gb, bad, 3)
        check("infinite radii are rejected rather than silently producing NaN", False)
    except ValueError:
        check("infinite radii are rejected rather than silently producing NaN", True)


def test_perturbation_families():
    print("\nperturbation families actually perturb")
    try:
        import torch
    except ImportError:
        print("  [skip] torch not available")
        return
    from stacc import perturb as PB

    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32)
    gen = torch.Generator().manual_seed(1)
    for name, eps in (("gauss", 0.03), ("linf", 0.03), ("photometric", 0.2),
                      ("stain", 0.12)):
        fam = PB.build_family(name, eps)
        xp = fam.draw(x, generator=gen)
        moved = PB.realised_displacement(x, xp)
        check(f"{name} moves the batch", moved > 1e-4, f"displacement {moved:.4f}")
        check(f"{name} stays in the pixel cube",
              bool((xp >= 0).all() and (xp <= 1).all()))

    # The adversarial search is the one that failed silently in an earlier
    # version: at delta = 0 the objective and its gradient both vanish, so
    # ascent from the origin returns the clean input and every member looks
    # perfectly robust.  A constant model is the sharpest probe, since then no
    # perturbation changes the output and only the input displacement can show
    # that the search moved at all.
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 32 * 32, 3)

        def forward(self, z):
            return self.fc(z.flatten(1))

    m = Tiny()
    xa = PB.adversarial_displacement(m, x, eps=0.05, steps=3, seed=2)
    moved = PB.realised_displacement(x, xa)
    check("adversarial search leaves the origin", moved > 1e-3,
          f"displacement {moved:.4f}")
    check("adversarial search respects the radius",
          float((xa - x).abs().max()) <= 0.05 + 1e-6,
          f"max |delta| {float((xa - x).abs().max()):.4f}")
    check("adversarial search stays in the pixel cube",
          bool((xa >= 0).all() and (xa <= 1).all()))


def test_transport_certificate():
    print("\nthe stability certificate itself, Theorem 2, on a synthetic shift")
    # Build a source and a target that are an exact eps-transport of each other:
    # each target point is a source point moved within the ball, labels kept.
    rng = np.random.default_rng(14)
    n, Cc = 400, 3
    labels = rng.integers(0, Cc, size=n)

    worst_violation = -np.inf
    for trial in range(30):
        zs = rng.normal(size=(n, Cc))
        zt = zs + rng.normal(scale=0.5, size=(n, Cc))  # the "moved" inputs
        soft = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(
            z - z.max(1, keepdims=True)
        ).sum(1, keepdims=True)
        ps, pt = soft(zs), soft(zt)
        R_S = brier(ps, labels)
        R_T = brier(pt, labels)
        S_T = np.sqrt(((pt - ps) ** 2).sum(axis=1).mean())
        worst_violation = max(worst_violation, np.sqrt(R_T) - (np.sqrt(R_S) + S_T))
    check(
        "sqrt(R_T) <= sqrt(R_S) + S_T under an exact transport",
        worst_violation <= 1e-12,
        f"worst margin {worst_violation:.2e}",
    )

    # tightness: displacement aligned with the residual attains equality
    p_s = np.tile(np.array([[0.7, 0.3]]), (1, 1))
    p_t = np.array([[0.6, 0.4]])
    lab = np.array([0])
    lhs = np.sqrt(brier(p_t, lab))
    rhs = np.sqrt(brier(p_s, lab)) + np.sqrt(((p_t - p_s) ** 2).sum(axis=1).mean())
    check("the bound is attained when the displacement is aligned",
          abs(lhs - rhs) < 1e-12, f"gap {lhs - rhs:.2e}")


# --------------------------------------------------------------------------- #
def test_estimator_baselines():
    """The two label-free accuracy estimators used as ranking comparators."""
    print()
    print("label-free accuracy estimators used as comparators")
    rng = np.random.default_rng(3)
    M, n, C_ = 6, 200, 4
    probs, labels = random_pool(M, n, C_, seed=3)
    val_p, val_y = random_pool(M, n, C_, seed=4)

    # A member perfectly consistent under perturbation must score 1, and one
    # whose perturbed decisions are a fixed permutation of its clean ones must
    # score 0.  Both are exact, so any indexing slip shows immediately.
    pert = np.repeat(probs[:, None], 3, axis=1)
    agree = B.augmentation_consistency(probs, pert)
    check("augmentation consistency: identical perturbation scores 1",
          np.allclose(agree, 1.0), f"min {agree.min():.6f}")
    rolled = np.roll(probs, 1, axis=2)
    disagree = B.augmentation_consistency(probs, np.repeat(rolled[:, None], 3, axis=1))
    check("augmentation consistency: rotated decisions score 0",
          np.allclose(disagree, 0.0), f"max {disagree.max():.6f}")

    # DoC reduces to source accuracy when the target batch is the source batch,
    # because the confidence drop is then identically zero.
    fit = B.atc_fit(val_p, val_y)
    doc = B.difference_of_confidences(fit, val_p)
    acc = np.array([(val_p[i].argmax(1) == val_y).mean() for i in range(M)])
    check("difference of confidences: no shift recovers source accuracy",
          np.allclose(doc, acc), f"max dev {np.abs(doc - acc).max():.2e}")
    doc_shift = B.difference_of_confidences(fit, probs)
    check("difference of confidences: finite under shift",
          np.all(np.isfinite(doc_shift)))

    # k-medoids and the DPP now take member-to-member squared distances rather
    # than the signatures, which is a refactor for memory rather than a change
    # of rule, so both must still return k distinct members and must agree with
    # the distances the Gram matrix gives.
    G = K.gram(probs)
    d2 = K.squared_distances(G)
    direct = ((probs.reshape(M, -1)[:, None, :]
               - probs.reshape(M, -1)[None, :, :]) ** 2).sum(axis=2) / probs.shape[1]
    check("squared distances from the Gram matrix match the direct form",
          np.allclose(d2, direct, atol=1e-12),
          f"max err {np.abs(d2 - direct).max():.2e}")
    med = B.k_medoids(d2, 3, np.random.default_rng(0))
    dpp = B.dpp_greedy(d2, 3)
    check("k-medoids returns k distinct members",
          len(set(med.tolist())) == 3 and med.max() < M)
    check("greedy DPP returns k distinct members",
          len(set(dpp.tolist())) == 3 and dpp.max() < M)

    # ATC's fitted threshold reproduces each member's source accuracy when the
    # batch is the source split, which is the property the threshold is chosen
    # for and which a refactor of the fit could silently break.
    w = B.atc_weights(fit, val_p)
    est = (val_p.max(axis=2) > fit["thresholds"][:, None]).mean(axis=1)
    check("ATC: fitted threshold reproduces source accuracy",
          np.allclose(est, acc, atol=1e-9), f"max dev {np.abs(est - acc).max():.2e}")
    check("ATC: weights are a probability vector",
          abs(w.sum() - 1.0) < 1e-12 and (w >= 0).all())



def main():
    np.seterr(all="raise")
    print("StaCC self-test: identities to machine precision, inequalities never violated")
    test_gram_and_risk()
    test_ambiguity_decomposition()
    test_certified_objective()
    test_gradient_and_step()
    test_frank_wolfe_rate()
    test_separation_constructions()
    test_falsification_and_localisation()
    test_stability_and_calibration()
    test_family_choice_and_guards()
    test_perturbation_families()
    test_transport_certificate()
    test_estimator_baselines()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

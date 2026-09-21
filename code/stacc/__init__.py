"""StaCC: stability-certified committee selection under distribution shift.

The package splits along the line the paper draws.  ``certificate`` turns a
member's response to input perturbation into a certified radius and localises
the unknown label signature; ``committee`` aggregates those radii into the
convex objective J and selects a committee by Frank-Wolfe; ``perturb`` supplies
the perturbation families, the one design axis the theory leaves open;
``calibrate`` removes the overconfidence confound; ``baselines`` holds the
comparators; ``registry`` reads pools from disk.

The remaining modules build and score pools: ``data``, ``models`` and
``augment`` are shared by the trainer and the caching scripts, ``pool`` is the
randomised trainer with ``train_metrics`` its checkpoint-selection metrics, and
``metrics`` scores the method and the comparators.
"""
from __future__ import annotations

from . import baselines, calibrate, certificate, committee, metrics, perturb, registry

__all__ = [
    "baselines",
    "calibrate",
    "certificate",
    "committee",
    "metrics",
    "perturb",
    "registry",
]

"""Perturbation families, the one design axis the certificate leaves open.

Theorem 2 holds for every metric d on inputs and every radius eps.  What varies
with the choice is whether the transport assumption is nearly true, and only the
data can say which family makes it so.  The families below are therefore
interchangeable at the interface and deliberately different in what they do.

All of them act on a batch of images in [0, 1] pixel space, before ImageNet
normalisation, so that eps is interpretable as a radius in pixel units and the
same number means the same thing across benchmarks.  The pipeline a model sees
is resize, to-tensor, perturb, normalise, which differs from the ordinary
evaluation pipeline only by the inserted step.

Every family reports the realised displacement it applied, because eps is a
nominal radius and a photometric operator only respects it approximately.  The
reported number is the one to quote, not the requested one.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .augment import IMAGENET_MEAN, IMAGENET_STD

# --------------------------------------------------------------------------- #
# normalisation, kept explicit so perturbation happens in pixel space
# --------------------------------------------------------------------------- #


def normalise(x):
    """[0, 1] pixel tensor -> ImageNet-normalised tensor the backbones expect."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _l2_per_image(delta):
    """Root-mean-square displacement per image, comparable across resolutions."""
    return delta.flatten(1).pow(2).mean(dim=1).sqrt()


# --------------------------------------------------------------------------- #
# the families
# --------------------------------------------------------------------------- #
@dataclass
class Family:
    """A named random perturbation of a [0, 1] pixel batch."""

    name: str
    eps: float
    adversarial: bool = False

    def draw(self, x, generator=None):
        raise NotImplementedError

    def describe(self):
        return f"{self.name}(eps={self.eps:g})"


class GaussianPixel(Family):
    """F1a: isotropic Gaussian noise of scale eps, clipped to the pixel cube.

    The literal reading of the theorem in the l2 metric, the cheapest family to
    evaluate, and the one least likely to match the mechanism of a real shift.
    """

    def __init__(self, eps=0.03):
        super().__init__(name="gauss", eps=eps)

    def draw(self, x, generator=None):
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return (x + self.eps * noise).clamp_(0.0, 1.0)


class UniformLinf(Family):
    """F1b: uniform noise in the l-infinity ball of radius eps."""

    def __init__(self, eps=0.03):
        super().__init__(name="linf", eps=eps)

    def draw(self, x, generator=None):
        u = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return (x + self.eps * (2.0 * u - 1.0)).clamp_(0.0, 1.0)


class Photometric(Family):
    """F3: brightness, contrast, saturation and blur, jittered by eps.

    The family most likely to make the transport assumption nearly true on the
    medical benchmarks, where the shift between sites is largely one of staining
    and exposure rather than of content.  eps scales the jitter range, so
    eps = 0.1 means each factor is drawn from [0.9, 1.1].
    """

    def __init__(self, eps=0.15, blur=True):
        super().__init__(name="photometric", eps=eps)
        self.blur = blur

    def draw(self, x, generator=None):
        b = x.shape[0]

        def factors():
            u = torch.rand(b, device=x.device, dtype=x.dtype, generator=generator)
            return 1.0 + self.eps * (2.0 * u - 1.0)

        # torchvision's functional adjusters take one image at a time, so the
        # batch is handled by explicit broadcasting rather than a Python loop.
        fb = factors().view(b, 1, 1, 1)
        out = (x * fb).clamp(0.0, 1.0)

        fc = factors().view(b, 1, 1, 1)
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        out = ((out - mean) * fc + mean).clamp(0.0, 1.0)

        fs = factors().view(b, 1, 1, 1)
        grey = out.mean(dim=1, keepdim=True)
        out = ((out - grey) * fs + grey).clamp(0.0, 1.0)

        if self.blur and x.shape[-1] >= 16:
            k = torch.tensor(
                [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                device=x.device,
                dtype=x.dtype,
            )
            k = (k / k.sum()).view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
            blurred = F.conv2d(F.pad(out, (1, 1, 1, 1), mode="reflect"), k, groups=x.shape[1])
            mix = torch.rand(b, 1, 1, 1, device=x.device, dtype=x.dtype, generator=generator)
            out = (1.0 - mix) * out + mix * blurred
        return out.clamp(0.0, 1.0)


class StainJitter(Family):
    """F3b: per-channel gain and bias, the crude stain model for histopathology.

    Camelyon17's hospital shift is dominated by staining protocol, which acts
    close to an affine map in colour space, so this is the family for which the
    transport assumption is most plausible on that benchmark.
    """

    def __init__(self, eps=0.12):
        super().__init__(name="stain", eps=eps)

    def draw(self, x, generator=None):
        b, c = x.shape[0], x.shape[1]
        shape = (b, c, 1, 1)
        gain = 1.0 + self.eps * (
            2.0 * torch.rand(shape, device=x.device, dtype=x.dtype, generator=generator) - 1.0
        )
        bias = self.eps * (
            2.0 * torch.rand(shape, device=x.device, dtype=x.dtype, generator=generator) - 1.0
        )
        return (x * gain + bias).clamp(0.0, 1.0)


FAMILIES = {
    "gauss": GaussianPixel,
    "linf": UniformLinf,
    "photometric": Photometric,
    "stain": StainJitter,
}


def build_family(name, eps):
    if name not in FAMILIES:
        raise ValueError(f"unknown family {name!r}; expected one of {sorted(FAMILIES)}")
    return FAMILIES[name](eps=eps)


# --------------------------------------------------------------------------- #
# F2: adversarial search, the tightest estimate of the worst case
# --------------------------------------------------------------------------- #


def adversarial_displacement(model, x, eps, steps=5, norm="linf", seed=None):
    """Maximise ||h(x') - h(x)||^2 over the eps-ball, by projected ascent.

    This approaches the supremum in the definition of the sensitivity from
    below, so it yields a larger and therefore more honest certified radius than
    random draws.  It is the only family that needs gradients, and it costs a
    forward and a backward pass per step per member.

    The objective is the squared displacement of the member's own probability
    output, not any loss against a label, so no label is read here either.

    The random start is not optional here, and the reason is specific to this
    objective rather than borrowed from the usual practice.  At delta = 0 the
    prediction equals its own reference, so the objective is exactly zero and,
    being a sum of squares at its minimum, so is its gradient.  Ascent from the
    origin therefore never leaves it, and the family silently returns the clean
    predictions with a stability score of zero for every member, which looks
    like perfect robustness rather than like a failure.  Starting from a random
    point inside the ball breaks the stationarity.
    """
    model.eval()
    with torch.no_grad():
        p0 = torch.softmax(model(normalise(x)), dim=1)

    if seed is not None:
        torch.manual_seed(seed)
    if norm == "linf":
        delta = (2.0 * torch.rand_like(x) - 1.0) * eps
    else:
        delta = torch.randn_like(x)
        dn = delta.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1)
        delta = delta / dn * eps * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
    delta = ((x + delta).clamp(0.0, 1.0) - x).detach().requires_grad_(True)
    alpha = 2.5 * eps / max(steps, 1)

    for _ in range(steps):
        p = torch.softmax(model(normalise((x + delta).clamp(0.0, 1.0))), dim=1)
        loss = ((p - p0) ** 2).sum(dim=1).sum()
        (grad,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            if norm == "linf":
                delta += alpha * grad.sign()
                delta.clamp_(-eps, eps)
            else:
                g = grad.flatten(1)
                g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
                delta += alpha * g.view_as(delta)
                dn = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
                scale = (eps / dn).clamp(max=1.0).view(-1, 1, 1, 1)
                delta *= scale
            delta.clamp_(-1.0, 1.0)
        delta.requires_grad_(True)

    return (x + delta.detach()).clamp(0.0, 1.0)


def realised_displacement(x, x_pert):
    """Mean per-image root-mean-square pixel displacement actually applied."""
    return float(_l2_per_image(x_pert - x).mean().item())

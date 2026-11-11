"""Radially weighted FFT amplitude loss (ZoomSR "RadialFreqLoss" methodology).

A plain full-band amplitude L1 is redundant with the pixel L1 (Parseval), so the amplitude difference is weighted by a
soft radial curve that is ~0 at low frequencies and rises to 1 at Nyquist:

    w(r) = clamp((r / R_nyq - r_lo) / (1 - r_lo), 0, 1) ** power
    L    = mean(w(r) * | |F(pred)| - |F(target)| |)

``r_lo`` is the normalised radius below which nothing is supervised (left to the pixel loss), ``power`` the convexity
of the ramp (0 = truncated full-band L1, larger = only the highest frequencies). The spectrum is taken with
``torch.fft.fft2`` on the luma (Rec.601) of the RGB tensors by default (``channel='rgb'`` uses every channel) with
``norm='ortho'`` so the amplitude has pixel scale; an optional ``valid_mask`` (BCHW, >0.5 = valid) is multiplied into
both images first so unsupervised pixels do not enter the spectrum. ``return_details=True`` also returns the
high-frequency energy ratio (energy with r > ``hf_cut`` over total energy) of prediction and target, which is the
"high-frequency gap" diagnostic of the methodology.
"""
from __future__ import annotations

import torch
from torch import nn

from basicsr.utils.registry import LOSS_REGISTRY

_NORMS = ("ortho", "backward", "forward")


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    """@luna-comment Input: BCHW RGB. Output: Rec601 Y. Key Steps: fixed weighted sum.
    Key Params: .299/.587/.114. State/Dependencies: differentiable tensor arithmetic only."""
    return 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def normalised_radius(height: int, width: int, device=None, dtype=None) -> torch.Tensor:
    """@luna-comment Input: spectrum height/width. Output: [H, W] radius with 1 at an axis Nyquist.
    Key Steps: fftfreq on both axes (cycles/pixel, |f| <= 0.5), Euclidean radius, divide by 0.5.
    Key Params: fft2 layout (no fftshift), corners reach sqrt(2). State/Dependencies: torch.fft only."""
    fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.fftfreq(width, d=1.0, device=device, dtype=dtype)
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / 0.5


def radial_weight(height: int, width: int, r_lo: float, power: float, device=None, dtype=None) -> torch.Tensor:
    """@luna-comment Input: spectrum size, cut-off r_lo in [0, 1), exponent power >= 0. Output: [H, W] weights in [0, 1].
    Key Steps: clamp((r - r_lo) / (1 - r_lo), 0, 1) ** power on the normalised radius.
    Key Params: r_lo masks the low-frequency disc, power sets the convexity. State/Dependencies: normalised_radius."""
    r = normalised_radius(height, width, device, dtype)
    ramp = ((r - float(r_lo)) / (1.0 - float(r_lo))).clamp(0.0, 1.0)
    if float(power) == 0.0:
        return (ramp > 0).to(ramp.dtype)
    return ramp ** float(power)


@LOSS_REGISTRY.register()
class RadialFreqLoss(nn.Module):
    """@luna-comment Input: pred/target BCHW RGB (+ optional valid mask). Output: weighted amplitude-spectrum L1.
    Key Steps: luma (or all channels), mask, fft2 amplitude, radial weight, mean. Key Params: power, r_lo, channel,
    norm, hf_cut for the diagnostic ratio. State/Dependencies: caches the weight per (H, W, device, dtype)."""

    def __init__(self, loss_weight: float = 1.0, power: float = 2.0, r_lo: float = 0.1, channel: str = "luma",
                 norm: str = "ortho", hf_cut: float = 0.5):
        super().__init__()
        if float(power) < 0.0:
            raise ValueError("power must be non-negative")
        if not 0.0 <= float(r_lo) < 1.0:
            raise ValueError("r_lo must be in [0, 1)")
        if channel not in ("luma", "rgb"):
            raise ValueError("channel must be 'luma' or 'rgb'")
        if norm not in _NORMS:
            raise ValueError(f"norm must be one of {_NORMS}")
        if not 0.0 < float(hf_cut) < 1.5:
            raise ValueError("hf_cut must be in (0, 1.5)")
        self.loss_weight = float(loss_weight)
        self.power = float(power)
        self.r_lo = float(r_lo)
        self.channel = channel
        self.norm = norm
        self.hf_cut = float(hf_cut)
        self._cache: dict = {}

    def _weights(self, height: int, width: int, device, dtype):
        key = (height, width, str(device), dtype)
        if key not in self._cache:
            w = radial_weight(height, width, self.r_lo, self.power, device, dtype)
            hf = (normalised_radius(height, width, device, dtype) > self.hf_cut).to(dtype)
            self._cache[key] = (w, hf)
        return self._cache[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None,
                return_details: bool = False, **kwargs):
        if pred.ndim != 4 or pred.shape != target.shape:
            raise ValueError("pred and target must be BCHW tensors of the same shape")
        if self.channel == "luma":
            if pred.shape[1] != 3:
                raise ValueError("channel='luma' expects 3-channel RGB inputs")
            x, y = _luma(pred), _luma(target)
        else:
            x, y = pred, target
        if valid_mask is not None:
            if valid_mask.ndim != 4 or valid_mask.shape[0] != x.shape[0] or valid_mask.shape[-2:] != x.shape[-2:]:
                raise ValueError("valid_mask must be BCHW with the batch and spatial size of pred")
            mask = (valid_mask[:, :1] > 0.5).to(x.dtype)
            x, y = x * mask, y * mask
        amp_x = torch.abs(torch.fft.fft2(x, norm=self.norm))
        amp_y = torch.abs(torch.fft.fft2(y, norm=self.norm))
        w, hf = self._weights(x.shape[-2], x.shape[-1], x.device, amp_x.dtype)
        loss = (w[None, None] * (amp_x - amp_y).abs()).mean()
        if not return_details:
            return self.loss_weight * loss
        energy_x, energy_y = amp_x.detach() ** 2, amp_y.detach() ** 2
        eps = energy_x.new_tensor(1e-12)
        details = {
            "hf_ratio_pred": ((energy_x * hf).sum((-2, -1)) / (energy_x.sum((-2, -1)) + eps)).mean(),
            "hf_ratio_gt": ((energy_y * hf).sum((-2, -1)) / (energy_y.sum((-2, -1)) + eps)).mean(),
            "w_mean": w.mean(),
        }
        return self.loss_weight * loss, details

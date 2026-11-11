"""RadialFreqLoss: the radial weight is 0 inside r_lo and 1 at Nyquist with a power-controlled ramp, identical
images give zero loss, a blurred prediction gives a positive differentiable loss with a lower high-frequency ratio,
the valid mask excludes unsupervised pixels, power=0 is the truncated full-band L1, and the registry builds it."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_v3m_block256 import bootstrap  # noqa: E402  same bootstrap as the other kit tests (binds package roots)

bootstrap(str(ROOT))
from basicsr.losses import build_loss  # noqa: E402
from realesrgan.losses.radial_freq_loss import RadialFreqLoss, normalised_radius, radial_weight  # noqa: E402


def _text_like(seed: int = 0, size: int = 64) -> torch.Tensor:
    """@luna-comment Input: seed/size. Output: [1,3,H,W] RGB with sharp strokes on a flat background.
    Key Steps: seeded random rectangles. Key Params: 6 strokes. State/Dependencies: torch RNG only."""
    g = torch.Generator().manual_seed(seed)
    img = torch.full((1, 3, size, size), 0.85)
    for _ in range(6):
        y, x = torch.randint(0, size - 8, (2,), generator=g).tolist()
        h, w = torch.randint(2, 6, (2,), generator=g).tolist()
        img[:, :, y:y + h, x:x + w] = 0.1
    return img


def _blur(img: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.avg_pool2d(img, 5, stride=1, padding=2, count_include_pad=False)


def test_weight_curve_shape():
    """@luna-comment Input: 64x64 spectrum. Output: assertions on w(r). Key Steps: DC, inside r_lo, Nyquist, power order.
    Key Params: r_lo 0.1, power 1/2/3. State/Dependencies: radial_weight / normalised_radius."""
    r = normalised_radius(64, 64)
    w1, w2, w3 = (radial_weight(64, 64, 0.1, p) for p in (1.0, 2.0, 3.0))
    assert w2[0, 0] == 0 and (w2[r <= 0.1] == 0).all()
    assert torch.isclose(w2[0, 32], torch.tensor(1.0)) and torch.isclose(w2[32, 0], torch.tensor(1.0))
    inner = (r > 0.1) & (r < 1.0)
    assert (w3[inner] <= w2[inner]).all() and (w2[inner] <= w1[inner]).all() and (w3[inner] < w1[inner]).any()
    w0 = radial_weight(64, 64, 0.1, 0.0)
    assert set(w0.unique().tolist()) == {0.0, 1.0} and (w0[r > 0.1] == 1).all()


def test_zero_blur_gradient_and_ratio():
    """@luna-comment Input: synthetic strokes and their blur. Output: zero / positive loss, gradient, hf ratio order.
    Key Steps: forward twice, backward on the blurred copy. Key Params: power 2, r_lo 0.1. State/Dependencies: CPU torch."""
    gt = _text_like()
    crit = RadialFreqLoss(power=2.0, r_lo=0.1)
    assert crit(gt.clone(), gt).item() == 0.0
    pred = _blur(gt).requires_grad_(True)
    loss, details = crit(pred, gt, return_details=True)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0.0
    assert details["hf_ratio_pred"].item() < details["hf_ratio_gt"].item()
    assert 0.0 < details["w_mean"].item() < 1.0


def test_valid_mask_excludes_unsupervised_pixels():
    """@luna-comment Input: pred differing from gt only where the mask is 0. Output: ~zero loss with the mask, positive without.
    Key Steps: corrupt the masked-out quadrant. Key Params: mask top-left quadrant 0. State/Dependencies: CPU torch."""
    gt = _text_like(1)
    pred = gt.clone()
    pred[:, :, :32, :32] = torch.rand(1, 3, 32, 32)
    mask = torch.ones(1, 1, 64, 64)
    mask[:, :, :32, :32] = 0.0
    crit = RadialFreqLoss()
    assert crit(pred, gt, valid_mask=mask).item() < 1e-6
    assert crit(pred, gt).item() > 1e-3


def test_power_zero_is_truncated_full_band_l1_and_rgb_channel():
    """@luna-comment Input: strokes vs blur. Output: power 0 equals mean over r>r_lo of |dA|; rgb channel runs.
    Key Steps: manual spectrum difference. Key Params: norm ortho. State/Dependencies: torch.fft."""
    gt, pred = _text_like(2), _blur(_text_like(2))
    crit = RadialFreqLoss(power=0.0, r_lo=0.1, norm="ortho")
    y_gt = 0.299 * gt[:, :1] + 0.587 * gt[:, 1:2] + 0.114 * gt[:, 2:3]
    y_pr = 0.299 * pred[:, :1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
    d = (torch.fft.fft2(y_pr, norm="ortho").abs() - torch.fft.fft2(y_gt, norm="ortho").abs()).abs()
    manual = (d * (normalised_radius(64, 64) > 0.1)).mean()
    assert torch.isclose(crit(pred, gt), manual)
    assert RadialFreqLoss(channel="rgb")(pred, gt).item() > 0.0


def test_registry_build_and_validation():
    """@luna-comment Input: BasicSR loss config. Output: RadialFreqLoss instance; bad params rejected.
    Key Steps: build_loss by type name. Key Params: power/r_lo/channel/norm. State/Dependencies: LOSS_REGISTRY."""
    crit = build_loss({"type": "RadialFreqLoss", "loss_weight": 1.0, "power": 3, "r_lo": 0.15, "channel": "luma", "norm": "ortho"})
    assert isinstance(crit, RadialFreqLoss) and crit.power == 3.0 and crit.r_lo == 0.15
    for bad in (dict(power=-1), dict(r_lo=1.0), dict(channel="y"), dict(norm="none")):
        try:
            RadialFreqLoss(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")
